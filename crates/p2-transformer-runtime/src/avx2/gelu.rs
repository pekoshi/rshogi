//! AVX2 GELU activation。
//!
//! candle の `tensor.gelu()` は erf based の精密 GELU。 推論では tanh 近似で
//! 十分なため、 `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` を採用。

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::simd_math::tanh256_ps;

/// In-place GELU on slice (AVX2)。
///
/// # Safety
/// AVX2 必須。 slice 長制限なし。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gelu_inplace_avx2(data: &mut [f32]) {
    // SAFETY: target_feature により AVX2/FMA 有効。
    let n = data.len();
    let n_aligned = n & !7;
    let kappa = _mm256_set1_ps(0.797_884_6_f32); // sqrt(2/pi)
    let coeff = _mm256_set1_ps(0.044_715_f32);
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);

    let ptr = data.as_mut_ptr();
    let mut i = 0usize;
    while i < n_aligned {
        let x = _mm256_loadu_ps(ptr.add(i));
        let xx = _mm256_mul_ps(x, x);
        let xxx = _mm256_mul_ps(xx, x);
        let inner = _mm256_fmadd_ps(coeff, xxx, x);
        let inner = _mm256_mul_ps(kappa, inner);
        let t = tanh256_ps(inner);
        let one_plus = _mm256_add_ps(one, t);
        let y = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus));
        _mm256_storeu_ps(ptr.add(i), y);
        i += 8;
    }
    // scalar tail
    for j in n_aligned..n {
        let x = *ptr.add(j);
        let inner = 0.797_884_6_f32 * (x + 0.044_715_f32 * x * x * x);
        *ptr.add(j) = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// scalar GELU (テスト / fallback)。
pub fn gelu_inplace_scalar(data: &mut [f32]) {
    for x in data.iter_mut() {
        let v = *x;
        let inner = 0.797_884_6_f32 * (v + 0.044_715_f32 * v * v * v);
        *x = 0.5 * v * (1.0 + inner.tanh());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let mut data_ref: Vec<f32> = (0..17).map(|i| (i as f32) * 0.2 - 1.5).collect();
        let mut data_avx2 = data_ref.clone();
        gelu_inplace_scalar(&mut data_ref);
        unsafe {
            gelu_inplace_avx2(&mut data_avx2);
        }
        for i in 0..data_ref.len() {
            assert!(
                (data_ref[i] - data_avx2[i]).abs() < 2e-3,
                "i={i} ref={} avx2={}",
                data_ref[i],
                data_avx2[i]
            );
        }
    }
}
