//! AVX2 row-wise softmax (mask 込み)。
//!
//! 入力 `[M, N]` row-major、 各行 `n` 個の値に対し softmax を in-place 計算する。
//! Mask は scalar `f32` で、 `mask[n] = -inf` の列を除外できる (attention の PAD 用)。

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::simd_math::exp256_ps;

/// 単一行に対する softmax (M=1 のスカラー実装)。
///
/// `row.len()` <= `valid_len` の要素のみ softmax 対象、 残りは 0 を書く。
pub fn softmax_row_with_mask(row: &mut [f32], valid_len: usize) {
    let n = row.len();
    let v = valid_len.min(n);
    if v == 0 {
        for x in row.iter_mut() {
            *x = 0.0;
        }
        return;
    }
    // max
    let mut max = row[0];
    for &x in &row[1..v] {
        if x > max {
            max = x;
        }
    }
    // exp & sum
    let mut sum = 0.0f32;
    for x in &mut row[..v] {
        let e = (*x - max).exp();
        *x = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for x in &mut row[..v] {
        *x *= inv;
    }
    for x in &mut row[v..] {
        *x = 0.0;
    }
}

/// 複数行に対する softmax (`[M, N]` row-major、 各行 `valid_len` 列のみ有効)。
///
/// # Safety
///
/// AVX2 を要求。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn softmax_rows_avx2(data: &mut [f32], m: usize, n: usize, valid_len: usize) {
    debug_assert!(data.len() >= m * n);
    debug_assert!(valid_len <= n);
    // SAFETY: 全 SIMD intrinsics は target_feature により有効。 ポインタ算術は
    // 上記 debug_assert で範囲保証。

    let v = valid_len;
    for row in 0..m {
        let base = row * n;
        let row_ptr = data.as_mut_ptr().add(base);

        // 1. row max (valid 範囲)
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        let v_aligned = v & !7;
        let mut i = 0usize;
        while i < v_aligned {
            let x = _mm256_loadu_ps(row_ptr.add(i));
            max_vec = _mm256_max_ps(max_vec, x);
            i += 8;
        }
        // horizontal max
        let max_arr = std::mem::transmute::<__m256, [f32; 8]>(max_vec);
        let mut max_scalar = max_arr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for j in v_aligned..v {
            let x = *row_ptr.add(j);
            if x > max_scalar {
                max_scalar = x;
            }
        }

        // 2. exp(x - max) と sum
        let max_vec = _mm256_set1_ps(max_scalar);
        let mut sum_vec = _mm256_setzero_ps();
        let mut i = 0usize;
        while i < v_aligned {
            let x = _mm256_loadu_ps(row_ptr.add(i));
            let xs = _mm256_sub_ps(x, max_vec);
            let e = exp256_ps(xs);
            _mm256_storeu_ps(row_ptr.add(i), e);
            sum_vec = _mm256_add_ps(sum_vec, e);
            i += 8;
        }
        // horizontal sum
        let sum_arr = std::mem::transmute::<__m256, [f32; 8]>(sum_vec);
        let mut sum_scalar = sum_arr.iter().copied().sum::<f32>();
        for j in v_aligned..v {
            let e = (*row_ptr.add(j) - max_scalar).exp();
            *row_ptr.add(j) = e;
            sum_scalar += e;
        }

        // 3. div
        let inv = 1.0 / sum_scalar;
        let inv_vec = _mm256_set1_ps(inv);
        let mut i = 0usize;
        while i < v_aligned {
            let e = _mm256_loadu_ps(row_ptr.add(i));
            _mm256_storeu_ps(row_ptr.add(i), _mm256_mul_ps(e, inv_vec));
            i += 8;
        }
        for j in v_aligned..v {
            *row_ptr.add(j) *= inv;
        }

        // 4. mask tail (n - v) を 0 で埋める
        for j in v..n {
            *row_ptr.add(j) = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_row_basic() {
        let mut row = vec![1.0_f32, 2.0, 3.0, 0.0];
        softmax_row_with_mask(&mut row, 3);
        let sum: f32 = row[..3].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(row[3], 0.0);
    }

    #[test]
    fn softmax_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let m = 4;
        let n = 13;
        let v = 10;
        let mut data_ref: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let mut data_avx2 = data_ref.clone();

        for row in 0..m {
            softmax_row_with_mask(&mut data_ref[row * n..(row + 1) * n], v);
        }
        unsafe {
            softmax_rows_avx2(&mut data_avx2, m, n, v);
        }
        for i in 0..m * n {
            assert!(
                (data_ref[i] - data_avx2[i]).abs() < 1e-4,
                "i={i} ref={} avx2={}",
                data_ref[i],
                data_avx2[i]
            );
        }
    }
}
