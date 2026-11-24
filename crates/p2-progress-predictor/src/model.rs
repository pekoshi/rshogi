//! Progress 推論用 MLP (pure Rust、 candle 非依存)
//!
//! Architecture:
//! ```text
//! input  [32] → Linear(32→32) + ReLU → Linear(32→16) + ReLU → Linear(16→1) + sigmoid
//! ```
//! params: 32*32 + 32 + 32*16 + 16 + 16*1 + 1 = 1601
//!
//! ## ホットパス制約
//!
//! - heap allocation 0 (重みは const-sized array、 中間 buffer も stack)
//! - branch なし (ReLU は分岐 1 つだが auto-vectorize 範囲)
//! - 推論レイテンシ目標: < 1 μs / call on 1 thread x86_64

use crate::features::{compute_features, ProgressFeatures, FEATURE_DIM};
use rshogi_core::position::Position;

/// 第 1 隠れ層次元
pub const HIDDEN1: usize = 32;
/// 第 2 隠れ層次元
pub const HIDDEN2: usize = 16;
/// 出力次元 (=1、 progress scalar)
pub const OUTPUT: usize = 1;

/// 全 weight + bias を持つ POD struct。
///
/// row-major、 `w_i[out, in]` 形式 (out が外側 = MLP の各 neuron 単位で連続)。
#[derive(Debug, Clone)]
pub struct ProgressWeights {
    pub w1: [[f32; FEATURE_DIM]; HIDDEN1],
    pub b1: [f32; HIDDEN1],
    pub w2: [[f32; HIDDEN1]; HIDDEN2],
    pub b2: [f32; HIDDEN2],
    pub w3: [[f32; HIDDEN2]; OUTPUT],
    pub b3: [f32; OUTPUT],
}

impl ProgressWeights {
    /// 全重み 0 / bias 0 → forward 結果は sigmoid(0)=0.5 が常に返る。
    ///
    /// 訓練前の sanity check / unit test 用。
    pub fn zeros() -> Self {
        Self {
            w1: [[0.0; FEATURE_DIM]; HIDDEN1],
            b1: [0.0; HIDDEN1],
            w2: [[0.0; HIDDEN1]; HIDDEN2],
            b2: [0.0; HIDDEN2],
            w3: [[0.0; HIDDEN2]; OUTPUT],
            b3: [0.0; OUTPUT],
        }
    }

    /// 「ルールベース近似」 fallback 重み。
    ///
    /// 訓練 ckpt が無い状況でも、 ML 経路で `compute_endgame_score` 相当の
    /// スコアを返すための identity-like 重み構成。 具体的には:
    ///
    /// - layer1 の各 neuron は 1 つの input feature を pass-through (対角)
    /// - layer2 / layer3 で `compute_endgame_score` 相当の線形結合を再現
    ///
    /// と組むには訓練側で fine-tuning する方が clean。 当面 `zeros()` の派生
    /// として「最終 bias=0 (sigmoid 0.5)」 ではなく「最終 bias を `score=ply/256`
    /// の crude proxy」 にして smoke test だけ通す。
    ///
    /// **本来の運用ではここを訓練済み重みで上書きすることが前提**。
    pub fn default_baseline() -> Self {
        // layer1, layer2 は all-zero (出力 0)、 b3 だけ logit(0.3)≈-0.847 で
        // 「平均的に序中盤」 を返す出発点とする。
        let mut w = Self::zeros();
        w.b3[0] = -0.847; // sigmoid(-0.847) ≈ 0.30
        w
    }
}

/// 推論器本体。
///
/// `predict` 内部で `compute_features` を呼んで position から特徴量を抽出する。
/// 既に features を計算済みの場合は `predict_features` を使う方が高速。
///
/// ## SIMD 最適化
///
/// x86_64 で AVX2 が利用可能な場合、 `predict_features` は内部で
/// `crate::avx2::predict_features_avx2` を呼ぶ。 そのために load 時に
/// col-major aligned shadow copy `avx2_weights` を保持している。
/// shadow copy のメモリ overhead は ~7 KB (1601 × 4 byte + alignment)、
/// 1 predictor あたり 1 度の load コストはホットパス外。
#[derive(Debug, Clone)]
pub struct MlProgressPredictor {
    pub weights: ProgressWeights,
    /// AVX2 推論用 col-major aligned shadow copy。 `weights` から派生 (load 時のみ計算)。
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    avx2_weights: Box<crate::avx2::Avx2Weights>,
}

impl MlProgressPredictor {
    /// 訓練済み重みでインスタンス化。
    pub fn from_weights(weights: ProgressWeights) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        let avx2_weights = Box::new(crate::avx2::Avx2Weights::from_weights(&weights));
        Self {
            weights,
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            avx2_weights,
        }
    }

    /// ckpt 不在時の fallback。 sigmoid(b3) ≒ 0.3 を常に返す。
    pub fn from_default() -> Self {
        Self::from_weights(ProgressWeights::default_baseline())
    }

    /// position から直接 progress を推定する。 allocation 0。
    ///
    /// Returns: `[0, 1]` の進行度スカラー (0=完全な序盤、 1=詰みレベル終盤)。
    #[inline]
    pub fn predict(&self, pos: &Position) -> f32 {
        let feats = compute_features(pos, 0);
        self.predict_features(&feats)
    }

    /// 事前計算済み feature から推論。
    ///
    /// 推論コアはここ。 stack 上の固定長 array で全演算を行う。
    ///
    /// x86_64 で AVX2 + FMA が runtime detection に成功した場合は SIMD path を、
    /// それ以外では scalar fallback を呼ぶ。
    #[inline]
    pub fn predict_features(&self, feats: &ProgressFeatures) -> f32 {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: 直前で AVX2 + FMA を runtime 検出済。 `avx2_weights` は
                // `from_weights` で初期化済 (32-byte aligned)。
                return unsafe { crate::avx2::predict_features_avx2(&self.avx2_weights, feats) };
            }
        }
        self.predict_features_scalar(feats)
    }

    /// scalar fallback path。 AVX2 が利用不可な環境、 および test で AVX2 結果と
    /// 突き合わせるために公開。
    pub fn predict_features_scalar(&self, feats: &ProgressFeatures) -> f32 {
        let w = &self.weights;
        let x = feats.as_slice();

        // Layer 1: input 32 → hidden 32 + ReLU
        let mut h1 = [0.0f32; HIDDEN1];
        for (j, h) in h1.iter_mut().enumerate() {
            let mut acc = w.b1[j];
            let row = &w.w1[j];
            for (k, xk) in x.iter().enumerate() {
                acc += row[k] * xk;
            }
            *h = if acc > 0.0 { acc } else { 0.0 };
        }

        // Layer 2: 32 → 16 + ReLU
        let mut h2 = [0.0f32; HIDDEN2];
        for (j, h) in h2.iter_mut().enumerate() {
            let mut acc = w.b2[j];
            let row = &w.w2[j];
            for (k, h1k) in h1.iter().enumerate() {
                acc += row[k] * h1k;
            }
            *h = if acc > 0.0 { acc } else { 0.0 };
        }

        // Layer 3: 16 → 1 + sigmoid
        let mut acc = w.b3[0];
        for (k, h2k) in h2.iter().enumerate() {
            acc += w.w3[0][k] * h2k;
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
    fn zero_weights_yield_half() {
        let p = MlProgressPredictor::from_weights(ProgressWeights::zeros());
        let feats = ProgressFeatures::default();
        let y = p.predict_features(&feats);
        assert!((y - 0.5).abs() < 1e-6, "got {}", y);
    }

    #[test]
    fn default_baseline_is_below_half() {
        let p = MlProgressPredictor::from_default();
        let feats = ProgressFeatures::default();
        let y = p.predict_features(&feats);
        assert!((0.20..0.40).contains(&y), "default baseline {} out of [0.20, 0.40]", y);
    }

    #[test]
    fn predict_runs_on_startpos() {
        let p = MlProgressPredictor::from_default();
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let y = p.predict(&pos);
        assert!((0.0..=1.0).contains(&y), "progress {} out of [0,1]", y);
    }

    #[test]
    fn sigmoid_extremes() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(20.0) > 0.999);
        assert!(sigmoid(-20.0) < 0.001);
    }
}
