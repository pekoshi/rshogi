//! AVX2 LayerNorm。
//!
//! 各行 (`[M, D]` の D 次元) について mean / variance を計算し、 (x - mean) / sqrt(var + eps)
//! を gamma, beta で affine する。

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// In-place LayerNorm。 入力 `x: [M, D]`、 出力同じ buffer に書き戻す。
///
/// # Safety
/// AVX2 必須。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn layer_norm_avx2(
    x: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    m: usize,
    d: usize,
    eps: f32,
) {
    debug_assert!(x.len() >= m * d);
    debug_assert!(gamma.len() >= d);
    debug_assert!(beta.len() >= d);
    // SAFETY: target_feature により AVX2/FMA 有効。 ポインタ算術範囲は assert で保証。

    let inv_d = 1.0 / d as f32;
    let d_aligned = d & !7;
    let ptr = x.as_mut_ptr();
    let gptr = gamma.as_ptr();
    let bptr = beta.as_ptr();

    for row in 0..m {
        let row_ptr = ptr.add(row * d);

        // 1. mean
        let mut sum_vec = _mm256_setzero_ps();
        let mut i = 0usize;
        while i < d_aligned {
            sum_vec = _mm256_add_ps(sum_vec, _mm256_loadu_ps(row_ptr.add(i)));
            i += 8;
        }
        let sum_arr = std::mem::transmute::<__m256, [f32; 8]>(sum_vec);
        let mut sum_scalar = sum_arr.iter().copied().sum::<f32>();
        for j in d_aligned..d {
            sum_scalar += *row_ptr.add(j);
        }
        let mean = sum_scalar * inv_d;
        let mean_vec = _mm256_set1_ps(mean);

        // 2. variance
        let mut var_vec = _mm256_setzero_ps();
        let mut i = 0usize;
        while i < d_aligned {
            let v = _mm256_sub_ps(_mm256_loadu_ps(row_ptr.add(i)), mean_vec);
            var_vec = _mm256_fmadd_ps(v, v, var_vec);
            i += 8;
        }
        let var_arr = std::mem::transmute::<__m256, [f32; 8]>(var_vec);
        let mut var_scalar = var_arr.iter().copied().sum::<f32>();
        for j in d_aligned..d {
            let v = *row_ptr.add(j) - mean;
            var_scalar += v * v;
        }
        let var = var_scalar * inv_d;
        let inv_std = 1.0 / (var + eps).sqrt();
        let inv_std_vec = _mm256_set1_ps(inv_std);

        // 3. normalize + affine
        let mut i = 0usize;
        while i < d_aligned {
            let v = _mm256_loadu_ps(row_ptr.add(i));
            let n = _mm256_mul_ps(_mm256_sub_ps(v, mean_vec), inv_std_vec);
            let g = _mm256_loadu_ps(gptr.add(i));
            let b = _mm256_loadu_ps(bptr.add(i));
            let y = _mm256_fmadd_ps(n, g, b);
            _mm256_storeu_ps(row_ptr.add(i), y);
            i += 8;
        }
        for j in d_aligned..d {
            let v = (*row_ptr.add(j) - mean) * inv_std;
            *row_ptr.add(j) = v * *gptr.add(j) + *bptr.add(j);
        }
    }
}

/// scalar LayerNorm (テスト / fallback)。
pub fn layer_norm_scalar(x: &mut [f32], gamma: &[f32], beta: &[f32], m: usize, d: usize, eps: f32) {
    let inv_d = 1.0 / d as f32;
    for row in 0..m {
        let base = row * d;
        let mut sum = 0.0f32;
        for j in 0..d {
            sum += x[base + j];
        }
        let mean = sum * inv_d;
        let mut var = 0.0f32;
        for j in 0..d {
            let v = x[base + j] - mean;
            var += v * v;
        }
        let inv_std = 1.0 / (var * inv_d + eps).sqrt();
        for j in 0..d {
            let v = (x[base + j] - mean) * inv_std;
            x[base + j] = v * gamma[j] + beta[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let m = 3;
        let d = 17;
        let mut data_ref: Vec<f32> = (0..m * d).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let mut data_avx2 = data_ref.clone();
        let gamma: Vec<f32> = (0..d).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let beta: Vec<f32> = (0..d).map(|i| -0.1 + (i as f32) * 0.02).collect();
        layer_norm_scalar(&mut data_ref, &gamma, &beta, m, d, 1e-5);
        unsafe {
            layer_norm_avx2(&mut data_avx2, &gamma, &beta, m, d, 1e-5);
        }
        for i in 0..m * d {
            assert!(
                (data_ref[i] - data_avx2[i]).abs() < 1e-3,
                "i={i} ref={} avx2={}",
                data_ref[i],
                data_avx2[i]
            );
        }
    }
}
