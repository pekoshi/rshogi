//! AVX2 microkernel for `MlProgressPredictorV2` (V2 = 48→64→32→16→1)。
//!
//! ## 設計
//!
//! V1 (`avx2.rs`) と同じ「input scalar broadcast + col-major weight × 8 lane FMA」
//! を踏襲し、 V2 の 4 段 MLP に拡張。
//!
//! ## Register allocation (Skylake 16 ymm 利用想定)
//!
//! - **Layer 1 (48→64)**: 出力 64 = 8 col-block × 8 lane → 8 ymm acc 保持
//!   - + 1 ymm: input scalar broadcast (xi)
//!   - + 1 ymm: weight 1 行 1 block (使い捨て)
//!   - 計 ~10 ymm 同時使用。 残り 6 で zero / 他演算
//! - **Layer 2 (64→32)**: 4 ymm acc + 1 broadcast + 1 weight = 6 ymm
//! - **Layer 3 (32→16)**: 2 ymm acc + 1 broadcast + 1 weight = 4 ymm
//! - **Layer 4 (16→1)**: 2 ymm dot → hsum → scalar sigmoid
//!
//! いずれも 16 ymm を超えない設計。
//!
//! ## Weights layout
//!
//! 訓練側は row-major `[out, in]` で持つ。 AVX2 は col-major `[in, out]` の方が
//! 「input scalar × out block」 broadcast に向くため、 load 時に 1 度だけ転置。
//!
//! - `w1_col`: `[FEATURE_DIM_V2][HIDDEN1_V2]` = `[48][64]` f32 (12 KB)
//! - `w2_col`: `[HIDDEN1_V2][HIDDEN2_V2]`     = `[64][32]` f32 ( 8 KB)
//! - `w3_col`: `[HIDDEN2_V2][HIDDEN3_V2]`     = `[32][16]` f32 ( 2 KB)
//! - `w4`   : `[HIDDEN3_V2]` row-major そのまま (16 f32)
//!
//! 各 col-block 行は 8 lane の倍数で、 32-byte aligned。
//!
//! ## Safety
//!
//! 各 `unsafe` ブロックには:
//! - AVX2 + FMA を `is_x86_feature_detected!` で確認済 (caller 側)
//! - `_mm256_load_ps` の 32-byte alignment は `#[repr(C, align(32))]` で保証

#![cfg(any(target_arch = "x86_64", target_arch = "x86"))]

use crate::features_v2::{ProgressFeaturesV2, FEATURE_DIM_V2};
use crate::model_v2::{WeightsV2, HIDDEN1_V2, HIDDEN2_V2, HIDDEN3_V2};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// compile-time invariant check
const _: () = {
    assert!(HIDDEN1_V2.is_multiple_of(8));
    assert!(HIDDEN2_V2.is_multiple_of(8));
    assert!(HIDDEN3_V2.is_multiple_of(8));
};

/// AVX2 推論用に col-major 再配置した V2 重み (32-byte aligned)。
#[repr(C, align(32))]
pub struct Avx2WeightsV2 {
    /// w1 col-major: `w1_col[in_idx][out_idx]` = `weights.w1[out_idx][in_idx]`
    pub(crate) w1_col: [[f32; HIDDEN1_V2]; FEATURE_DIM_V2],
    pub(crate) b1: [f32; HIDDEN1_V2],
    /// w2 col-major: `w2_col[in_idx][out_idx]` = `weights.w2[out_idx][in_idx]`
    pub(crate) w2_col: [[f32; HIDDEN2_V2]; HIDDEN1_V2],
    pub(crate) b2: [f32; HIDDEN2_V2],
    /// w3 col-major: `w3_col[in_idx][out_idx]` = `weights.w3[out_idx][in_idx]`
    pub(crate) w3_col: [[f32; HIDDEN3_V2]; HIDDEN2_V2],
    pub(crate) b3: [f32; HIDDEN3_V2],
    /// w4 は出力 1 のみなので row-major のまま (16 f32)
    pub(crate) w4: [f32; HIDDEN3_V2],
    pub(crate) b4: f32,
}

impl std::fmt::Debug for Avx2WeightsV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Avx2WeightsV2")
            .field("layers", &"48->64->32->16->1 (col-major shadow)")
            .field("b4", &self.b4)
            .finish()
    }
}

impl Avx2WeightsV2 {
    /// row-major `WeightsV2` から col-major aligned copy を作る。
    /// load 時に 1 回だけ呼ばれる (ホットパス外)。
    pub fn from_weights(w: &WeightsV2) -> Self {
        let mut out = Self::zeros();
        // w1: weights.w1[out][in] → out.w1_col[in][out]
        for o in 0..HIDDEN1_V2 {
            for i in 0..FEATURE_DIM_V2 {
                out.w1_col[i][o] = w.w1[o][i];
            }
        }
        out.b1 = w.b1;
        // w2: weights.w2[out][in] → out.w2_col[in][out]
        for o in 0..HIDDEN2_V2 {
            for i in 0..HIDDEN1_V2 {
                out.w2_col[i][o] = w.w2[o][i];
            }
        }
        out.b2 = w.b2;
        // w3: weights.w3[out][in] → out.w3_col[in][out]
        for o in 0..HIDDEN3_V2 {
            for i in 0..HIDDEN2_V2 {
                out.w3_col[i][o] = w.w3[o][i];
            }
        }
        out.b3 = w.b3;
        out.w4 = w.w4[0];
        out.b4 = w.b4[0];
        out
    }

    fn zeros() -> Self {
        Self {
            w1_col: [[0.0; HIDDEN1_V2]; FEATURE_DIM_V2],
            b1: [0.0; HIDDEN1_V2],
            w2_col: [[0.0; HIDDEN2_V2]; HIDDEN1_V2],
            b2: [0.0; HIDDEN2_V2],
            w3_col: [[0.0; HIDDEN3_V2]; HIDDEN2_V2],
            b3: [0.0; HIDDEN3_V2],
            w4: [0.0; HIDDEN3_V2],
            b4: 0.0,
        }
    }
}

/// AVX2 path 推論本体 (V2)。 caller 側で AVX2 + FMA 検出済みを保証すること。
///
/// # Safety
///
/// - 呼び出し前に `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`
///   を確認している必要がある
/// - `weights` は `#[repr(C, align(32))]` で 32-byte aligned
/// - `feats.data` は 4-byte aligned のみ → input は `_mm256_loadu_ps`
///
/// 守るべき不変条件:
/// - `HIDDEN1_V2 % 8 == 0`, `HIDDEN2_V2 % 8 == 0`, `HIDDEN3_V2 % 8 == 0` (const assert で保証)
/// - 各 layer 出力は ReLU 後 [0, +∞) で finite
#[target_feature(enable = "avx2,fma")]
pub unsafe fn predict_features_v2_avx2(weights: &Avx2WeightsV2, feats: &ProgressFeaturesV2) -> f32 {
    let x = &feats.data;
    let zero = _mm256_setzero_ps();

    // ============================================================
    // Layer 1: 48 → 64 + ReLU
    // ============================================================
    //
    // 出力 64 を 8 ymm acc (h1_0..h1_7) に保持。 初期値は b1。
    let mut h1_0 = _mm256_loadu_ps(weights.b1.as_ptr());
    let mut h1_1 = _mm256_loadu_ps(weights.b1.as_ptr().add(8));
    let mut h1_2 = _mm256_loadu_ps(weights.b1.as_ptr().add(16));
    let mut h1_3 = _mm256_loadu_ps(weights.b1.as_ptr().add(24));
    let mut h1_4 = _mm256_loadu_ps(weights.b1.as_ptr().add(32));
    let mut h1_5 = _mm256_loadu_ps(weights.b1.as_ptr().add(40));
    let mut h1_6 = _mm256_loadu_ps(weights.b1.as_ptr().add(48));
    let mut h1_7 = _mm256_loadu_ps(weights.b1.as_ptr().add(56));

    // 48 input を scalar broadcast し、 8 col-block を FMA で蓄積
    for (xi_scalar, w_row) in x.iter().zip(weights.w1_col.iter()) {
        let xi = _mm256_set1_ps(*xi_scalar);
        // w1_col[i] は 64 f32 で、 align(32) の親 struct 内、
        // row stride = 64 * 4 = 256 byte → 32-byte aligned 保証
        let w_row = w_row.as_ptr();
        let w0 = _mm256_load_ps(w_row);
        let w1 = _mm256_load_ps(w_row.add(8));
        let w2 = _mm256_load_ps(w_row.add(16));
        let w3 = _mm256_load_ps(w_row.add(24));
        let w4 = _mm256_load_ps(w_row.add(32));
        let w5 = _mm256_load_ps(w_row.add(40));
        let w6 = _mm256_load_ps(w_row.add(48));
        let w7 = _mm256_load_ps(w_row.add(56));
        h1_0 = _mm256_fmadd_ps(xi, w0, h1_0);
        h1_1 = _mm256_fmadd_ps(xi, w1, h1_1);
        h1_2 = _mm256_fmadd_ps(xi, w2, h1_2);
        h1_3 = _mm256_fmadd_ps(xi, w3, h1_3);
        h1_4 = _mm256_fmadd_ps(xi, w4, h1_4);
        h1_5 = _mm256_fmadd_ps(xi, w5, h1_5);
        h1_6 = _mm256_fmadd_ps(xi, w6, h1_6);
        h1_7 = _mm256_fmadd_ps(xi, w7, h1_7);
    }

    // ReLU
    h1_0 = _mm256_max_ps(h1_0, zero);
    h1_1 = _mm256_max_ps(h1_1, zero);
    h1_2 = _mm256_max_ps(h1_2, zero);
    h1_3 = _mm256_max_ps(h1_3, zero);
    h1_4 = _mm256_max_ps(h1_4, zero);
    h1_5 = _mm256_max_ps(h1_5, zero);
    h1_6 = _mm256_max_ps(h1_6, zero);
    h1_7 = _mm256_max_ps(h1_7, zero);

    // h1 を stack 上の 32-byte aligned buffer に書き戻す (layer 2 で scalar broadcast 用)
    #[repr(align(32))]
    struct Aligned32<const N: usize>([f32; N]);
    let mut h1 = Aligned32::<HIDDEN1_V2>([0.0; HIDDEN1_V2]);
    _mm256_store_ps(h1.0.as_mut_ptr(), h1_0);
    _mm256_store_ps(h1.0.as_mut_ptr().add(8), h1_1);
    _mm256_store_ps(h1.0.as_mut_ptr().add(16), h1_2);
    _mm256_store_ps(h1.0.as_mut_ptr().add(24), h1_3);
    _mm256_store_ps(h1.0.as_mut_ptr().add(32), h1_4);
    _mm256_store_ps(h1.0.as_mut_ptr().add(40), h1_5);
    _mm256_store_ps(h1.0.as_mut_ptr().add(48), h1_6);
    _mm256_store_ps(h1.0.as_mut_ptr().add(56), h1_7);

    // ============================================================
    // Layer 2: 64 → 32 + ReLU
    // ============================================================
    //
    // 出力 32 = 4 col-block × 8 lane を 4 ymm acc に保持
    let mut h2_0 = _mm256_loadu_ps(weights.b2.as_ptr());
    let mut h2_1 = _mm256_loadu_ps(weights.b2.as_ptr().add(8));
    let mut h2_2 = _mm256_loadu_ps(weights.b2.as_ptr().add(16));
    let mut h2_3 = _mm256_loadu_ps(weights.b2.as_ptr().add(24));

    for (hi_scalar, w_row) in h1.0.iter().zip(weights.w2_col.iter()) {
        let hi = _mm256_set1_ps(*hi_scalar);
        // w2_col[i] は 32 f32、 row stride = 32 * 4 = 128 byte → 32-byte aligned 保証
        let w_row = w_row.as_ptr();
        let w0 = _mm256_load_ps(w_row);
        let w1 = _mm256_load_ps(w_row.add(8));
        let w2 = _mm256_load_ps(w_row.add(16));
        let w3 = _mm256_load_ps(w_row.add(24));
        h2_0 = _mm256_fmadd_ps(hi, w0, h2_0);
        h2_1 = _mm256_fmadd_ps(hi, w1, h2_1);
        h2_2 = _mm256_fmadd_ps(hi, w2, h2_2);
        h2_3 = _mm256_fmadd_ps(hi, w3, h2_3);
    }

    h2_0 = _mm256_max_ps(h2_0, zero);
    h2_1 = _mm256_max_ps(h2_1, zero);
    h2_2 = _mm256_max_ps(h2_2, zero);
    h2_3 = _mm256_max_ps(h2_3, zero);

    let mut h2 = Aligned32::<HIDDEN2_V2>([0.0; HIDDEN2_V2]);
    _mm256_store_ps(h2.0.as_mut_ptr(), h2_0);
    _mm256_store_ps(h2.0.as_mut_ptr().add(8), h2_1);
    _mm256_store_ps(h2.0.as_mut_ptr().add(16), h2_2);
    _mm256_store_ps(h2.0.as_mut_ptr().add(24), h2_3);

    // ============================================================
    // Layer 3: 32 → 16 + ReLU
    // ============================================================
    //
    // 出力 16 = 2 col-block × 8 lane を 2 ymm acc に保持
    let mut h3_0 = _mm256_loadu_ps(weights.b3.as_ptr());
    let mut h3_1 = _mm256_loadu_ps(weights.b3.as_ptr().add(8));

    for (hi_scalar, w_row) in h2.0.iter().zip(weights.w3_col.iter()) {
        let hi = _mm256_set1_ps(*hi_scalar);
        // w3_col[i] は 16 f32、 row stride = 16 * 4 = 64 byte → 32-byte aligned 保証
        let w_row = w_row.as_ptr();
        let w0 = _mm256_load_ps(w_row);
        let w1 = _mm256_load_ps(w_row.add(8));
        h3_0 = _mm256_fmadd_ps(hi, w0, h3_0);
        h3_1 = _mm256_fmadd_ps(hi, w1, h3_1);
    }

    h3_0 = _mm256_max_ps(h3_0, zero);
    h3_1 = _mm256_max_ps(h3_1, zero);

    // ============================================================
    // Layer 4: 16 → 1 + sigmoid
    // ============================================================
    //
    // dot product + hsum + sigmoid scalar
    // weights.w4 (16 f32) は align 4 → unaligned load
    let w4_0 = _mm256_loadu_ps(weights.w4.as_ptr());
    let w4_1 = _mm256_loadu_ps(weights.w4.as_ptr().add(8));
    let p0 = _mm256_mul_ps(h3_0, w4_0);
    let p1 = _mm256_mul_ps(h3_1, w4_1);
    let sum = _mm256_add_ps(p0, p1);

    // horizontal sum (8 → 1)
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps(sum, 1);
    let q = _mm_add_ps(low, high);
    let shuf = _mm_movehdup_ps(q);
    let sums = _mm_add_ps(q, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let dot = _mm_add_ss(sums, shuf2);
    let dot_scalar = _mm_cvtss_f32(dot);

    let logit = dot_scalar + weights.b4;
    sigmoid_scalar(logit)
}

/// Scalar sigmoid (V1 と同じ実装、 出力 1 つだけなので AVX2 化のメリットなし)
#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
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
    use crate::model_v2::MlProgressPredictorV2;

    /// random-ish weights を作って scalar と AVX2 が一致することを確認。
    fn make_random_weights(seed: u32) -> Box<WeightsV2> {
        let mut w = WeightsV2::zeros();
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (((s >> 8) & 0xffff) as f32 / 65536.0) - 0.5
        };
        for j in 0..HIDDEN1_V2 {
            for k in 0..FEATURE_DIM_V2 {
                w.w1[j][k] = next() * 0.5;
            }
            w.b1[j] = next() * 0.1;
        }
        for j in 0..HIDDEN2_V2 {
            for k in 0..HIDDEN1_V2 {
                w.w2[j][k] = next() * 0.5;
            }
            w.b2[j] = next() * 0.1;
        }
        for j in 0..HIDDEN3_V2 {
            for k in 0..HIDDEN2_V2 {
                w.w3[j][k] = next() * 0.5;
            }
            w.b3[j] = next() * 0.1;
        }
        for k in 0..HIDDEN3_V2 {
            w.w4[0][k] = next();
        }
        w.b4[0] = next() * 0.1;
        w
    }

    fn make_random_feats(seed: u32) -> ProgressFeaturesV2 {
        let mut f = ProgressFeaturesV2::default();
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(99);
        for v in f.data.iter_mut() {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((s >> 8) & 0xffff) as f32 / 65536.0;
        }
        f
    }

    #[test]
    fn avx2_v2_matches_scalar_within_tolerance() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("avx2/fma not available on this machine, skipping");
            return;
        }
        for seed in 0..16u32 {
            let weights = make_random_weights(seed);
            let aw = Avx2WeightsV2::from_weights(&weights);
            let predictor = MlProgressPredictorV2::from_weights(weights);

            for fs in 0..4u32 {
                let feats = make_random_feats(seed ^ (fs.wrapping_mul(17)));
                let scalar = predictor.predict_features_scalar(&feats);
                // SAFETY: avx2/fma detection succeeded above
                let avx = unsafe { predict_features_v2_avx2(&aw, &feats) };
                let err = (scalar - avx).abs();
                assert!(
                    err < 1e-5,
                    "seed={} fs={}: scalar={} avx={} err={}",
                    seed,
                    fs,
                    scalar,
                    avx,
                    err
                );
            }
        }
    }

    #[test]
    fn from_random_dispatch_matches_scalar() {
        // MlProgressPredictorV2::predict_features (runtime dispatch) と
        // predict_features_scalar が一致する (random feats 8 case)
        let p = MlProgressPredictorV2::from_random(123);
        for fs in 0..8u32 {
            let feats = make_random_feats(fs);
            let scalar = p.predict_features_scalar(&feats);
            let dispatch = p.predict_features(&feats);
            let err = (scalar - dispatch).abs();
            assert!(err < 1e-5, "fs={}: scalar={} dispatch={} err={}", fs, scalar, dispatch, err);
        }
    }
}
