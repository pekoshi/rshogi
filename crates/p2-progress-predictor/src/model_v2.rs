//! V2 Progress 推論用 MLP (48 dim 入力対応、 case b 採用)
//!
//! Architecture (case b: 容量と推論レイテンシのバランス):
//! ```text
//! input  [48] → Linear(48→64) + ReLU →
//!               Linear(64→32) + ReLU →
//!               Linear(32→16) + ReLU →
//!               Linear(16→ 1) + sigmoid
//! ```
//!
//! ## params 数
//!
//! ```text
//! 48*64 + 64 + 64*32 + 32 + 32*16 + 16 + 16*1 + 1
//!  = 3072 + 64 + 2048 + 32 + 512 + 16 + 16 + 1
//!  = 5761
//! ```
//!
//! V1 (1601 params) と比較して 3.6x の容量増。
//!
//! ## ホットパス制約
//!
//! - heap allocation 0 (重みは const-sized array、 中間 buffer も stack)
//! - 推論レイテンシ目標: < 300 ns / call on AVX2 + FMA (V1 比 ~3x)
//! - V1 (`MlProgressPredictor`) と並立、 既存 public API 破壊なし

use crate::features_v2::{compute_features_v2, ProgressFeaturesV2, FEATURE_DIM_V2};
use rshogi_core::position::Position;

/// 第 1 隠れ層次元
pub const HIDDEN1_V2: usize = 64;
/// 第 2 隠れ層次元
pub const HIDDEN2_V2: usize = 32;
/// 第 3 隠れ層次元
pub const HIDDEN3_V2: usize = 16;
/// 出力次元 (=1、 progress scalar)
pub const OUTPUT_V2: usize = 1;

/// 全 weight + bias を持つ POD struct (V2)。
///
/// row-major、 `w_i[out, in]` 形式 (out が外側 = MLP の各 neuron 単位で連続)。
#[derive(Debug, Clone)]
pub struct WeightsV2 {
    /// Layer 1 weight `[HIDDEN1_V2][FEATURE_DIM_V2]` (out, in)
    pub w1: [[f32; FEATURE_DIM_V2]; HIDDEN1_V2],
    pub b1: [f32; HIDDEN1_V2],
    /// Layer 2 weight `[HIDDEN2_V2][HIDDEN1_V2]`
    pub w2: [[f32; HIDDEN1_V2]; HIDDEN2_V2],
    pub b2: [f32; HIDDEN2_V2],
    /// Layer 3 weight `[HIDDEN3_V2][HIDDEN2_V2]`
    pub w3: [[f32; HIDDEN2_V2]; HIDDEN3_V2],
    pub b3: [f32; HIDDEN3_V2],
    /// Layer 4 weight `[OUTPUT_V2][HIDDEN3_V2]`
    pub w4: [[f32; HIDDEN3_V2]; OUTPUT_V2],
    pub b4: [f32; OUTPUT_V2],
}

impl WeightsV2 {
    /// 全重み 0 / bias 0 → forward 結果は sigmoid(0)=0.5 が常に返る。
    pub fn zeros() -> Box<Self> {
        // 5KB スタック展開を避けるため Box 経由で alloc
        Box::new(Self {
            w1: [[0.0; FEATURE_DIM_V2]; HIDDEN1_V2],
            b1: [0.0; HIDDEN1_V2],
            w2: [[0.0; HIDDEN1_V2]; HIDDEN2_V2],
            b2: [0.0; HIDDEN2_V2],
            w3: [[0.0; HIDDEN2_V2]; HIDDEN3_V2],
            b3: [0.0; HIDDEN3_V2],
            w4: [[0.0; HIDDEN3_V2]; OUTPUT_V2],
            b4: [0.0; OUTPUT_V2],
        })
    }

    /// Xavier (Glorot uniform) 初期化。 訓練前の random init として利用。
    ///
    /// 各 layer で `±sqrt(6 / (fan_in + fan_out))` の uniform 分布。
    pub fn from_random(seed: u64) -> Box<Self> {
        let mut w = Self::zeros();
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        // splitmix64 風の next_f32 (-0.5, 0.5)
        let mut next_uniform = || -> f32 {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            ((z >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5
        };

        // Xavier uniform: ±sqrt(6 / (fan_in + fan_out))
        let xavier =
            |fan_in: usize, fan_out: usize| -> f32 { (6.0 / (fan_in + fan_out) as f32).sqrt() };

        let lim1 = xavier(FEATURE_DIM_V2, HIDDEN1_V2);
        let lim2 = xavier(HIDDEN1_V2, HIDDEN2_V2);
        let lim3 = xavier(HIDDEN2_V2, HIDDEN3_V2);
        let lim4 = xavier(HIDDEN3_V2, OUTPUT_V2);

        for row in w.w1.iter_mut() {
            for v in row.iter_mut() {
                *v = next_uniform() * 2.0 * lim1;
            }
        }
        for row in w.w2.iter_mut() {
            for v in row.iter_mut() {
                *v = next_uniform() * 2.0 * lim2;
            }
        }
        for row in w.w3.iter_mut() {
            for v in row.iter_mut() {
                *v = next_uniform() * 2.0 * lim3;
            }
        }
        for row in w.w4.iter_mut() {
            for v in row.iter_mut() {
                *v = next_uniform() * 2.0 * lim4;
            }
        }
        // bias は 0 init
        w
    }
}

/// V2 推論器本体。
///
/// AVX2 が利用可能な x86_64 では `avx2_v2_weights` shadow copy を利用する。
pub struct MlProgressPredictorV2 {
    pub weights: Box<WeightsV2>,
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    avx2_weights: Box<crate::avx2_v2::Avx2WeightsV2>,
}

impl std::fmt::Debug for MlProgressPredictorV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlProgressPredictorV2")
            .field("feature_dim", &FEATURE_DIM_V2)
            .field("hidden1", &HIDDEN1_V2)
            .field("hidden2", &HIDDEN2_V2)
            .field("hidden3", &HIDDEN3_V2)
            .finish()
    }
}

impl MlProgressPredictorV2 {
    /// 訓練済み重みでインスタンス化。
    pub fn from_weights(weights: Box<WeightsV2>) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        let avx2_weights = Box::new(crate::avx2_v2::Avx2WeightsV2::from_weights(&weights));
        Self {
            weights,
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            avx2_weights,
        }
    }

    /// Xavier random init で 1 つ作る。 訓練前の sanity check / unit test 用。
    pub fn from_random(seed: u64) -> Self {
        Self::from_weights(WeightsV2::from_random(seed))
    }

    /// position から直接 progress を推定する (V2 features 構築込み)。
    ///
    /// nodchip soft channel に `progress_weights` を渡したい場合は
    /// `predict_features` を直接呼ぶこと。
    #[inline]
    pub fn predict(&self, pos: &Position) -> f32 {
        let feats = compute_features_v2(pos, 0, None);
        self.predict_features(&feats)
    }

    /// 事前計算済み V2 feature から推論。 ホットパス。
    #[inline]
    pub fn predict_features(&self, feats: &ProgressFeaturesV2) -> f32 {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: 直前で AVX2 + FMA を runtime 検出済。 `avx2_weights` は
                // `from_weights` で初期化済 (32-byte aligned)。
                return unsafe {
                    crate::avx2_v2::predict_features_v2_avx2(&self.avx2_weights, feats)
                };
            }
        }
        self.predict_features_scalar(feats)
    }

    /// scalar fallback path。 AVX2 が利用不可な環境、 および test で AVX2 結果と
    /// 突き合わせるために公開。
    pub fn predict_features_scalar(&self, feats: &ProgressFeaturesV2) -> f32 {
        let w = &self.weights;
        let x = feats.as_slice();

        // Layer 1: input 48 → hidden 64 + ReLU
        let mut h1 = [0.0f32; HIDDEN1_V2];
        for (j, h) in h1.iter_mut().enumerate() {
            let mut acc = w.b1[j];
            let row = &w.w1[j];
            for k in 0..FEATURE_DIM_V2 {
                acc += row[k] * x[k];
            }
            *h = if acc > 0.0 { acc } else { 0.0 };
        }

        // Layer 2: 64 → 32 + ReLU
        let mut h2 = [0.0f32; HIDDEN2_V2];
        for (j, h) in h2.iter_mut().enumerate() {
            let mut acc = w.b2[j];
            let row = &w.w2[j];
            for k in 0..HIDDEN1_V2 {
                acc += row[k] * h1[k];
            }
            *h = if acc > 0.0 { acc } else { 0.0 };
        }

        // Layer 3: 32 → 16 + ReLU
        let mut h3 = [0.0f32; HIDDEN3_V2];
        for (j, h) in h3.iter_mut().enumerate() {
            let mut acc = w.b3[j];
            let row = &w.w3[j];
            for k in 0..HIDDEN2_V2 {
                acc += row[k] * h2[k];
            }
            *h = if acc > 0.0 { acc } else { 0.0 };
        }

        // Layer 4: 16 → 1 + sigmoid
        let mut acc = w.b4[0];
        for (k, h3k) in h3.iter().enumerate() {
            acc += w.w4[0][k] * h3k;
        }
        sigmoid(acc)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // 数値安定な分岐版
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::position::Position;

    #[test]
    fn dims_match_expected() {
        assert_eq!(FEATURE_DIM_V2, 48);
        assert_eq!(HIDDEN1_V2, 64);
        assert_eq!(HIDDEN2_V2, 32);
        assert_eq!(HIDDEN3_V2, 16);
        assert_eq!(OUTPUT_V2, 1);
    }

    #[test]
    fn total_param_count_is_5761() {
        // 48*64 + 64 + 64*32 + 32 + 32*16 + 16 + 16*1 + 1 = 5761
        let count = FEATURE_DIM_V2 * HIDDEN1_V2
            + HIDDEN1_V2
            + HIDDEN1_V2 * HIDDEN2_V2
            + HIDDEN2_V2
            + HIDDEN2_V2 * HIDDEN3_V2
            + HIDDEN3_V2
            + HIDDEN3_V2 * OUTPUT_V2
            + OUTPUT_V2;
        assert_eq!(count, 5761);
    }

    #[test]
    fn zero_weights_yield_half() {
        let p = MlProgressPredictorV2::from_weights(WeightsV2::zeros());
        let feats = ProgressFeaturesV2::default();
        let y = p.predict_features(&feats);
        assert!((y - 0.5).abs() < 1e-6, "got {}", y);
    }

    #[test]
    fn random_predict_in_unit_range() {
        let p = MlProgressPredictorV2::from_random(42);
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let y = p.predict(&pos);
        assert!((0.0..=1.0).contains(&y), "progress {} out of [0,1]", y);
    }

    #[test]
    fn scalar_runs_on_random_feats() {
        let p = MlProgressPredictorV2::from_random(7);
        let mut feats = ProgressFeaturesV2::default();
        // 適当な input
        for (i, v) in feats.data.iter_mut().enumerate() {
            *v = (i as f32 / 48.0) * 0.5;
        }
        let y = p.predict_features_scalar(&feats);
        assert!((0.0..=1.0).contains(&y), "scalar predict {} out of range", y);
    }
}
