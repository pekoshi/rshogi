//! AVX2 microkernel for `MlProgressPredictor`.
//!
//! ## 設計
//!
//! - **方針**: input scalar broadcast + col-major weight × 8 lane FMA accumulate
//!   - hsum を 1 度も使わない (broadcast 方式)
//!   - 32→32: 4 col-block (× 8 lane) を 32 input でループ → 128 FMA / layer
//!   - 32→16: 2 col-block (× 8 lane) を 32 input でループ → 64 FMA / layer
//!   - 16→1: row-major dot product + hsum (どうせ 1 出力なので scalar fallback で十分)
//! - **weights layout**: load 時に col-major `[in][out_block]` に再配置:
//!   - `Avx2W1`: `[FEATURE_DIM][4 × __m256]` = 32×32 f32
//!   - `Avx2W2`: `[FEATURE_DIM][2 × __m256]` = 32×16 f32
//!   - `Avx2W3`: scalar row そのまま (16 f32)
//!   - bias は出力 col-block 単位の `__m256` で保持
//! - **hidden register hold**: layer 1 → layer 2 へ register pass。 layer 2 → 3 は
//!   一旦 stack に書き戻す (16 → 1 は scalar の方が速い)
//!
//! ## Safety
//!
//! - 各 unsafe ブロックは AVX2 detection 後にのみ実行される。 caller (`predict_features`)
//!   が `is_x86_feature_detected!("avx2")` をチェックする。
//! - `Avx2Weights` は `#[repr(C, align(32))]` で 32-byte aligned。 `_mm256_load_ps`
//!   は 32-byte aligned 必須 (unaligned だと #GP)。
//! - `__m256` field を `Copy + Clone` 経由で渡しても OK (POD だが、 `derive(Clone)` は
//!   `__m256` に対しては手動実装が必要)。

#![cfg(any(target_arch = "x86_64", target_arch = "x86"))]

use crate::features::{ProgressFeatures, FEATURE_DIM};
use crate::model::{ProgressWeights, HIDDEN1, HIDDEN2};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// HIDDEN1=32 → 4 col-block × 8 lane、 HIDDEN2=16 → 2 col-block × 8 lane。
// compile-time check のみ。 コード本体は const に依存せず直書きしている。
const _: () = {
    assert!(HIDDEN1.is_multiple_of(8));
    assert!(HIDDEN2.is_multiple_of(8));
};

/// AVX2 推論用に col-major 再配置した重み (32-byte aligned)。
///
/// `Copy` ではなく `Clone` のみ。 32 byte alignment は `_mm256_load_ps` 用。
#[repr(C, align(32))]
#[derive(Debug)]
pub struct Avx2Weights {
    /// w1 col-major: `w1_col[in_idx][block][lane]` = `weights.w1[block*8+lane][in_idx]`
    ///
    /// f32 で flatten すると `[FEATURE_DIM][H1_BLOCKS*8]` = `[32][32]`。
    pub(crate) w1_col: [[f32; HIDDEN1]; FEATURE_DIM],
    /// b1 を col-block 単位で並べた layout
    pub(crate) b1: [f32; HIDDEN1],
    /// w2 col-major: `w2_col[in_idx][block][lane]` = `weights.w2[block*8+lane][in_idx]`
    pub(crate) w2_col: [[f32; HIDDEN2]; HIDDEN1],
    pub(crate) b2: [f32; HIDDEN2],
    /// w3 row-major そのまま (16 → 1)
    pub(crate) w3: [f32; HIDDEN2],
    pub(crate) b3: f32,
}

impl Avx2Weights {
    /// row-major `ProgressWeights` から col-major aligned copy を作る。
    /// load 時に 1 回だけ呼ばれる (ホットパス外)。
    pub fn from_weights(w: &ProgressWeights) -> Self {
        let mut out = Self::zeros();
        // w1: weights.w1[out][in] → out.w1_col[in][out]
        for o in 0..HIDDEN1 {
            for i in 0..FEATURE_DIM {
                out.w1_col[i][o] = w.w1[o][i];
            }
        }
        out.b1 = w.b1;
        // w2: weights.w2[out][in] → out.w2_col[in][out]
        for o in 0..HIDDEN2 {
            for i in 0..HIDDEN1 {
                out.w2_col[i][o] = w.w2[o][i];
            }
        }
        out.b2 = w.b2;
        out.w3 = w.w3[0];
        out.b3 = w.b3[0];
        out
    }

    fn zeros() -> Self {
        Self {
            w1_col: [[0.0; HIDDEN1]; FEATURE_DIM],
            b1: [0.0; HIDDEN1],
            w2_col: [[0.0; HIDDEN2]; HIDDEN1],
            b2: [0.0; HIDDEN2],
            w3: [0.0; HIDDEN2],
            b3: 0.0,
        }
    }
}

impl Clone for Avx2Weights {
    fn clone(&self) -> Self {
        Self {
            w1_col: self.w1_col,
            b1: self.b1,
            w2_col: self.w2_col,
            b2: self.b2,
            w3: self.w3,
            b3: self.b3,
        }
    }
}

/// AVX2 path 推論本体。 caller 側で AVX2 検出済みを保証すること。
///
/// # Safety
///
/// - 呼び出し前に `is_x86_feature_detected!("avx2")` で AVX2 サポートを確認している必要がある
/// - `weights` の `w1_col` / `w2_col` は `#[repr(align(32))]` で 32-byte aligned。
///   各行 `&w1_col[i][0]` も自動的に 32-byte aligned (FEATURE_DIM * f32 = 128 byte で
///   alignment が保たれる)
/// - `feats.data` は 4-byte aligned (no specific guarantee for 32-byte)、 そのため input
///   は `_mm256_loadu_ps` (unaligned) を使用
///
/// 守るべき不変条件:
/// - `HIDDEN1 % 8 == 0` かつ `HIDDEN2 % 8 == 0` (compile time const なので保証済)
/// - layer 1 出力 / layer 2 出力は ReLU 後 [0, +∞) で finite (NaN なし)
#[target_feature(enable = "avx2,fma")]
pub unsafe fn predict_features_avx2(weights: &Avx2Weights, feats: &ProgressFeatures) -> f32 {
    let x = &feats.data;

    // ========== Layer 1: 32 → 32 + ReLU ==========
    //
    // h1 を 4 個の __m256 (合計 32 lane) に保持。 初期値は b1。
    //
    // bias は b1 を unaligned load (b1 は align 4 のみ)
    let mut h1_0 = _mm256_loadu_ps(weights.b1.as_ptr());
    let mut h1_1 = _mm256_loadu_ps(weights.b1.as_ptr().add(8));
    let mut h1_2 = _mm256_loadu_ps(weights.b1.as_ptr().add(16));
    let mut h1_3 = _mm256_loadu_ps(weights.b1.as_ptr().add(24));

    // 32 input をスカラ broadcast、 col-row (= 32 出力) を FMA で蓄積
    for (i, &xi_scalar) in x.iter().enumerate() {
        let xi = _mm256_set1_ps(xi_scalar);
        // w1_col[i] は 32 f32、 col-major で 4 block (8 lane × 4)
        // w1_col は align(32) なので各 row も 32-byte aligned (FEATURE_DIM=32 boundary)
        let w_row = weights.w1_col[i].as_ptr();
        let w0 = _mm256_load_ps(w_row);
        let w1 = _mm256_load_ps(w_row.add(8));
        let w2 = _mm256_load_ps(w_row.add(16));
        let w3 = _mm256_load_ps(w_row.add(24));
        h1_0 = _mm256_fmadd_ps(xi, w0, h1_0);
        h1_1 = _mm256_fmadd_ps(xi, w1, h1_1);
        h1_2 = _mm256_fmadd_ps(xi, w2, h1_2);
        h1_3 = _mm256_fmadd_ps(xi, w3, h1_3);
    }

    // ReLU: max(0, x)
    let zero = _mm256_setzero_ps();
    h1_0 = _mm256_max_ps(h1_0, zero);
    h1_1 = _mm256_max_ps(h1_1, zero);
    h1_2 = _mm256_max_ps(h1_2, zero);
    h1_3 = _mm256_max_ps(h1_3, zero);

    // h1 を stack に書き出す (layer 2 入力で再 broadcast するため scalar アクセスが必要)
    // 16-byte aligned で十分だが 32-byte aligned にして store_ps を使う
    #[repr(align(32))]
    struct Aligned32<const N: usize>([f32; N]);
    let mut h1 = Aligned32::<HIDDEN1>([0.0; HIDDEN1]);
    _mm256_store_ps(h1.0.as_mut_ptr(), h1_0);
    _mm256_store_ps(h1.0.as_mut_ptr().add(8), h1_1);
    _mm256_store_ps(h1.0.as_mut_ptr().add(16), h1_2);
    _mm256_store_ps(h1.0.as_mut_ptr().add(24), h1_3);

    // ========== Layer 2: 32 → 16 + ReLU ==========
    //
    // h2 を 2 個の __m256 (合計 16 lane) に保持。
    let mut h2_0 = _mm256_loadu_ps(weights.b2.as_ptr());
    let mut h2_1 = _mm256_loadu_ps(weights.b2.as_ptr().add(8));

    for i in 0..HIDDEN1 {
        let hi = _mm256_set1_ps(h1.0[i]);
        // w2_col[i] は 16 f32 で 32-byte 単位の連続 (HIDDEN2=16 で 64 byte = 2 × 32 byte)
        // ただし w2_col の row stride は 16 f32 = 64 byte、 各 row の先頭は 64-byte 境界
        // → 32-byte aligned 保証あり
        let w_row = weights.w2_col[i].as_ptr();
        let w0 = _mm256_load_ps(w_row);
        let w1 = _mm256_load_ps(w_row.add(8));
        h2_0 = _mm256_fmadd_ps(hi, w0, h2_0);
        h2_1 = _mm256_fmadd_ps(hi, w1, h2_1);
    }

    h2_0 = _mm256_max_ps(h2_0, zero);
    h2_1 = _mm256_max_ps(h2_1, zero);

    // ========== Layer 3: 16 → 1 + sigmoid ==========
    //
    // 1 出力なので dot product + hsum。 hsum を 1 回行う。
    //
    // weights.w3 (16 f32) は align 4 → unaligned load
    let w3_0 = _mm256_loadu_ps(weights.w3.as_ptr());
    let w3_1 = _mm256_loadu_ps(weights.w3.as_ptr().add(8));
    let p0 = _mm256_mul_ps(h2_0, w3_0);
    let p1 = _mm256_mul_ps(h2_1, w3_1);
    let sum = _mm256_add_ps(p0, p1);

    // horizontal sum (8 → 1)
    // sum = [s0, s1, s2, s3, s4, s5, s6, s7]
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps(sum, 1);
    let q = _mm_add_ps(low, high); // [s0+s4, s1+s5, s2+s6, s3+s7]
    let shuf = _mm_movehdup_ps(q); // [q1, q1, q3, q3]
    let sums = _mm_add_ps(q, shuf); // [q0+q1, _, q2+q3, _]
    let shuf2 = _mm_movehl_ps(shuf, sums); // [q2+q3, q3, _, _]
    let dot = _mm_add_ss(sums, shuf2);
    let dot_scalar = _mm_cvtss_f32(dot);

    let logit = dot_scalar + weights.b3;
    sigmoid_scalar(logit)
}

/// Scalar sigmoid。 output は 1 つだけなので AVX2 化のメリットなし。
#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
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
    use crate::model::ProgressWeights;

    /// random-ish weights を作って scalar と AVX2 が一致することを確認。
    fn make_random_weights(seed: u32) -> ProgressWeights {
        let mut w = ProgressWeights::zeros();
        // 線形合同 RNG (test 専用、 簡素)
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            // [-0.5, 0.5)
            (((s >> 8) & 0xffff) as f32 / 65536.0) - 0.5
        };
        for j in 0..HIDDEN1 {
            for k in 0..FEATURE_DIM {
                w.w1[j][k] = next();
            }
            w.b1[j] = next() * 0.1;
        }
        for j in 0..HIDDEN2 {
            for k in 0..HIDDEN1 {
                w.w2[j][k] = next();
            }
            w.b2[j] = next() * 0.1;
        }
        for k in 0..HIDDEN2 {
            w.w3[0][k] = next();
        }
        w.b3[0] = next() * 0.1;
        w
    }

    fn make_random_feats(seed: u32) -> ProgressFeatures {
        let mut f = ProgressFeatures::default();
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(99);
        for v in f.data.iter_mut() {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((s >> 8) & 0xffff) as f32 / 65536.0;
        }
        f
    }

    #[test]
    fn avx2_matches_scalar_within_tolerance() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("avx2/fma not available on this machine, skipping");
            return;
        }
        for seed in 0..16u32 {
            let weights = make_random_weights(seed);
            let aw = Avx2Weights::from_weights(&weights);
            let predictor = crate::model::MlProgressPredictor::from_weights(weights);

            for fs in 0..8u32 {
                let feats = make_random_feats(seed ^ (fs.wrapping_mul(17)));
                let scalar = predictor.predict_features_scalar(&feats);
                // SAFETY: avx2/fma detection succeeded above
                let avx = unsafe { predict_features_avx2(&aw, &feats) };
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
}
