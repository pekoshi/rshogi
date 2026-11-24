//! 共有 SIMD math primitives (exp, tanh)。
//!
//! AVX2/FMA を要求する関数の集まり。 softmax / gelu などから利用。

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// AVX2 8-wide exp。 相対誤差 ~1e-6 (cephes-style 5 次多項式)。
///
/// # Safety
/// AVX2/FMA を要求。 入力は ±88 程度で saturate。
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp256_ps(x: __m256) -> __m256 {
    // SAFETY: target_feature により AVX2/FMA 有効。 すべての intrinsics は副作用なし。
    let exp_hi = _mm256_set1_ps(88.376_26);
    let exp_lo = _mm256_set1_ps(-88.376_26);
    let log2ef = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let c1 = _mm256_set1_ps(0.693_359_4_f32);
    let c2 = _mm256_set1_ps(-2.121_944_4e-4_f32);
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);

    let mut xx = _mm256_min_ps(x, exp_hi);
    xx = _mm256_max_ps(xx, exp_lo);

    let fx = _mm256_fmadd_ps(xx, log2ef, half);
    let fx_floor = _mm256_floor_ps(fx);
    let z = _mm256_fnmadd_ps(fx_floor, c1, xx);
    let z = _mm256_fnmadd_ps(fx_floor, c2, z);
    let zz = _mm256_mul_ps(z, z);

    let p0 = _mm256_set1_ps(1.987_569e-4_f32);
    let p1 = _mm256_set1_ps(1.398_199_3e-3_f32);
    let p2 = _mm256_set1_ps(8.333_452e-3_f32);
    let p3 = _mm256_set1_ps(4.166_580_4e-2_f32);
    let p4 = _mm256_set1_ps(1.666_666_7e-1_f32);
    let p5 = _mm256_set1_ps(5e-1_f32);

    let mut y = p0;
    y = _mm256_fmadd_ps(y, z, p1);
    y = _mm256_fmadd_ps(y, z, p2);
    y = _mm256_fmadd_ps(y, z, p3);
    y = _mm256_fmadd_ps(y, z, p4);
    y = _mm256_fmadd_ps(y, z, p5);
    y = _mm256_fmadd_ps(y, zz, z);
    y = _mm256_add_ps(y, one);

    let emm0 = _mm256_cvttps_epi32(fx_floor);
    let bias = _mm256_set1_epi32(0x7f);
    let emm = _mm256_add_epi32(emm0, bias);
    let emm = _mm256_slli_epi32::<23>(emm);
    let pow2n = _mm256_castsi256_ps(emm);
    _mm256_mul_ps(y, pow2n)
}

/// AVX2 8-wide tanh: `tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)`、 精度 exp 由来。
///
/// # Safety
/// AVX2/FMA を要求。
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh256_ps(x: __m256) -> __m256 {
    // SAFETY: target_feature により AVX2/FMA 有効。
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    // tanh(x) ≈ tanh(±10) ≈ ±1 で saturate
    let bound = _mm256_set1_ps(10.0);
    let neg_bound = _mm256_set1_ps(-10.0);
    let x = _mm256_max_ps(_mm256_min_ps(x, bound), neg_bound);
    let e2x = exp256_ps(_mm256_mul_ps(two, x));
    _mm256_div_ps(_mm256_sub_ps(e2x, one), _mm256_add_ps(e2x, one))
}
