//! AVX2 行列積 microkernel。
//!
//! 入力レイアウト (PyTorch 互換):
//! - `a`: `[M, K]` row-major   (= activations)
//! - `b_packed`: `[K, N]` row-major (= weights を `pack_weight_for_gemm` で transpose 済み)
//! - `c`: `[M, N]` row-major   (出力)
//! - `bias`: `Option<&[N]>`    (None なら zero、 fused add)
//!
//! 計算: `C[m,n] = sum_k A[m,k] * B_packed[k,n] + bias[n]`
//!
//! ## カーネル設計
//!
//! - 出力 N 方向を 16 並列で進める (2 ymm = 16 f32)
//! - 出力 M 方向を 4 行同時 (8 acc ymm = 4 行 × 2 col group)
//! - K 方向は 1 step ずつ
//! - 4×16 micro-tile を `M`/4、 `N`/16 でループする
//!
//! ## レジスタ割り当て (4M × 16N micro-kernel)
//!
//! ```text
//! ymm0..ymm7 : C 4 acc × 2 col group (M=0..3、 N=n..n+15)
//! ymm8..ymm9 : B_packed[k, n..n+15]  (2 ymm = 16 col)
//! ymm10..13  : A[m, k] broadcast (4 行ぶん)
//! ```
//!
//! 合計 14 ymm。 i7-7700 (16 ymm available) でかろうじて収まる。
//!
//! ## エッジ処理
//!
//! - `M % 4 != 0`、 `N % 16 != 0` の残差は 1×8 / scalar tail。

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// AVX2 GEMM: `C = A * B_packed (+ bias)`、 row-major、 f32。
///
/// # 引数
///
/// - `a`: `[M, K]` row-major activations
/// - `b_packed`: `[K, N]` row-major (= PyTorch Linear.weight `[N, K]` を pack_weight_for_gemm で転置)
/// - `c`: `[M, N]` row-major 出力 (上書き)
/// - `bias`: `None` または `[N]` (各 col に加算)
///
/// # Safety
///
/// `unsafe fn` の理由: 呼び出し元で AVX2 サポートを保証する責務がある。
///
/// 守るべき不変条件:
/// - CPU が AVX2/FMA を実装している (`is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")` で事前確認)
/// - スライス長: `a.len() >= m*k`、 `b_packed.len() >= k*n`、 `c.len() >= m*n`、 `bias` あれば `bias.len() >= n`
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_a_bt(
    a: &[f32],
    b_packed: &[f32],
    c: &mut [f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b_packed.len() >= n * k);
    debug_assert!(c.len() >= m * n);
    if let Some(bs) = bias {
        debug_assert!(bs.len() >= n);
    }

    // SAFETY: 関数全体が target_feature(avx2,fma) を持つため intrinsics が有効。
    // ポインタ算術範囲は上記 debug_assert で保証。

    let a_ptr = a.as_ptr();
    let b_ptr = b_packed.as_ptr();
    let c_ptr = c.as_mut_ptr();
    let bias_ptr = bias.map(|bs| bs.as_ptr()).unwrap_or(std::ptr::null());
    let has_bias = !bias_ptr.is_null();

    let m_tail_start = m & !3; // 4 の倍数まで
    let n_main_end = n & !15; // 16 の倍数
    let n_8_end = n & !7; // 8 の倍数

    // ループ順: N outer × M inner で、 各 N panel (16 col × K row = 16K floats) を L1 に
    // 載せて全 M 行で再利用する。 K=192 で 12 KB / panel、 K=768 で 49 KB / panel。
    //
    // 初期化を避けるため、 N tile loop の最初の M=0 反復で C を bias で初期化する。
    // しかし K outer / N inner / M inner の通常順序のほうが C 書き込みコストが減る。
    // ここでは「N outer × M inner」で C は bias 込みで毎 (n,m) 一括書き出し。
    let mut n0 = 0usize;
    while n0 < n_main_end {
        let b_panel_base = b_ptr.add(n0); // [K, 16]
        let (bias_lo, bias_hi) = if has_bias {
            (_mm256_loadu_ps(bias_ptr.add(n0)), _mm256_loadu_ps(bias_ptr.add(n0 + 8)))
        } else {
            (_mm256_setzero_ps(), _mm256_setzero_ps())
        };

        // M を 4 行ずつ
        let mut m0 = 0usize;
        while m0 < m_tail_start {
            let mut acc0_lo = bias_lo;
            let mut acc0_hi = bias_hi;
            let mut acc1_lo = bias_lo;
            let mut acc1_hi = bias_hi;
            let mut acc2_lo = bias_lo;
            let mut acc2_hi = bias_hi;
            let mut acc3_lo = bias_lo;
            let mut acc3_hi = bias_hi;

            let mut b_row = b_panel_base;
            let mut a0p = a_ptr.add(m0 * k);
            let mut a1p = a_ptr.add((m0 + 1) * k);
            let mut a2p = a_ptr.add((m0 + 2) * k);
            let mut a3p = a_ptr.add((m0 + 3) * k);
            for _ in 0..k {
                let b_lo = _mm256_loadu_ps(b_row);
                let b_hi = _mm256_loadu_ps(b_row.add(8));
                let av0 = _mm256_broadcast_ss(&*a0p);
                let av1 = _mm256_broadcast_ss(&*a1p);
                let av2 = _mm256_broadcast_ss(&*a2p);
                let av3 = _mm256_broadcast_ss(&*a3p);
                acc0_lo = _mm256_fmadd_ps(av0, b_lo, acc0_lo);
                acc0_hi = _mm256_fmadd_ps(av0, b_hi, acc0_hi);
                acc1_lo = _mm256_fmadd_ps(av1, b_lo, acc1_lo);
                acc1_hi = _mm256_fmadd_ps(av1, b_hi, acc1_hi);
                acc2_lo = _mm256_fmadd_ps(av2, b_lo, acc2_lo);
                acc2_hi = _mm256_fmadd_ps(av2, b_hi, acc2_hi);
                acc3_lo = _mm256_fmadd_ps(av3, b_lo, acc3_lo);
                acc3_hi = _mm256_fmadd_ps(av3, b_hi, acc3_hi);
                b_row = b_row.add(n);
                a0p = a0p.add(1);
                a1p = a1p.add(1);
                a2p = a2p.add(1);
                a3p = a3p.add(1);
            }

            let c_row0_base = c_ptr.add(m0 * n);
            let c_row1_base = c_ptr.add((m0 + 1) * n);
            let c_row2_base = c_ptr.add((m0 + 2) * n);
            let c_row3_base = c_ptr.add((m0 + 3) * n);
            _mm256_storeu_ps(c_row0_base.add(n0), acc0_lo);
            _mm256_storeu_ps(c_row0_base.add(n0 + 8), acc0_hi);
            _mm256_storeu_ps(c_row1_base.add(n0), acc1_lo);
            _mm256_storeu_ps(c_row1_base.add(n0 + 8), acc1_hi);
            _mm256_storeu_ps(c_row2_base.add(n0), acc2_lo);
            _mm256_storeu_ps(c_row2_base.add(n0 + 8), acc2_hi);
            _mm256_storeu_ps(c_row3_base.add(n0), acc3_lo);
            _mm256_storeu_ps(c_row3_base.add(n0 + 8), acc3_hi);

            m0 += 4;
        }

        // M tail (m_tail_start..m)、 1 行ずつ
        while m0 < m {
            let mut acc_lo = bias_lo;
            let mut acc_hi = bias_hi;
            let mut b_row = b_panel_base;
            let mut ap = a_ptr.add(m0 * k);
            for _ in 0..k {
                let b_lo = _mm256_loadu_ps(b_row);
                let b_hi = _mm256_loadu_ps(b_row.add(8));
                let av = _mm256_broadcast_ss(&*ap);
                acc_lo = _mm256_fmadd_ps(av, b_lo, acc_lo);
                acc_hi = _mm256_fmadd_ps(av, b_hi, acc_hi);
                b_row = b_row.add(n);
                ap = ap.add(1);
            }
            let c_row_base = c_ptr.add(m0 * n);
            _mm256_storeu_ps(c_row_base.add(n0), acc_lo);
            _mm256_storeu_ps(c_row_base.add(n0 + 8), acc_hi);
            m0 += 1;
        }

        n0 += 16;
    }

    // N=8 tail (n_main_end..n_8_end)
    while n0 < n_8_end {
        let b_panel_base = b_ptr.add(n0);
        let bias_v = if has_bias {
            _mm256_loadu_ps(bias_ptr.add(n0))
        } else {
            _mm256_setzero_ps()
        };
        let mut m0 = 0usize;
        while m0 < m_tail_start {
            let mut acc0 = bias_v;
            let mut acc1 = bias_v;
            let mut acc2 = bias_v;
            let mut acc3 = bias_v;
            let mut b_row = b_panel_base;
            let mut a0p = a_ptr.add(m0 * k);
            let mut a1p = a_ptr.add((m0 + 1) * k);
            let mut a2p = a_ptr.add((m0 + 2) * k);
            let mut a3p = a_ptr.add((m0 + 3) * k);
            for _ in 0..k {
                let b_vec = _mm256_loadu_ps(b_row);
                let av0 = _mm256_broadcast_ss(&*a0p);
                let av1 = _mm256_broadcast_ss(&*a1p);
                let av2 = _mm256_broadcast_ss(&*a2p);
                let av3 = _mm256_broadcast_ss(&*a3p);
                acc0 = _mm256_fmadd_ps(av0, b_vec, acc0);
                acc1 = _mm256_fmadd_ps(av1, b_vec, acc1);
                acc2 = _mm256_fmadd_ps(av2, b_vec, acc2);
                acc3 = _mm256_fmadd_ps(av3, b_vec, acc3);
                b_row = b_row.add(n);
                a0p = a0p.add(1);
                a1p = a1p.add(1);
                a2p = a2p.add(1);
                a3p = a3p.add(1);
            }
            _mm256_storeu_ps(c_ptr.add(m0 * n + n0), acc0);
            _mm256_storeu_ps(c_ptr.add((m0 + 1) * n + n0), acc1);
            _mm256_storeu_ps(c_ptr.add((m0 + 2) * n + n0), acc2);
            _mm256_storeu_ps(c_ptr.add((m0 + 3) * n + n0), acc3);
            m0 += 4;
        }
        while m0 < m {
            let mut acc = bias_v;
            let mut b_row = b_panel_base;
            let mut ap = a_ptr.add(m0 * k);
            for _ in 0..k {
                let bv = _mm256_loadu_ps(b_row);
                let av = _mm256_broadcast_ss(&*ap);
                acc = _mm256_fmadd_ps(av, bv, acc);
                b_row = b_row.add(n);
                ap = ap.add(1);
            }
            _mm256_storeu_ps(c_ptr.add(m0 * n + n0), acc);
            m0 += 1;
        }
        n0 += 8;
    }

    // N scalar tail (n_8_end..n)
    while n0 < n {
        let bias_s = if has_bias { *bias_ptr.add(n0) } else { 0.0 };
        for mm in 0..m {
            let mut s = bias_s;
            let mut b_row = b_ptr.add(n0);
            let mut ap = a_ptr.add(mm * k);
            for _ in 0..k {
                s += *ap * *b_row;
                b_row = b_row.add(n);
                ap = ap.add(1);
            }
            *c_ptr.add(mm * n + n0) = s;
        }
        n0 += 1;
    }
}

/// 重みを `[N, K]` (PyTorch Linear) から `[K, N]` (gemm 内ループ連続 read 用) に
/// pack する。 init 時 1 度だけ実行 (hot path 外)。
pub fn pack_weight_for_gemm(src: &[f32], dst: &mut [f32], n: usize, k: usize) {
    debug_assert_eq!(src.len(), n * k);
    debug_assert_eq!(dst.len(), n * k);
    for kk in 0..k {
        for nn in 0..n {
            dst[kk * n + nn] = src[nn * k + kk];
        }
    }
}

/// scalar gemm (テスト / fallback 用)。
///
/// `c = a * b_packed^T + bias`、 a: `[M,K]`、 b_packed: `[K,N]`、 c: `[M,N]`。
/// (test 用、 packing 済み B を取る)
pub fn gemm_a_bt_scalar_packed(
    a: &[f32],
    b_packed: &[f32],
    c: &mut [f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) {
    for mm in 0..m {
        for nn in 0..n {
            let mut s = bias.map(|bs| bs[nn]).unwrap_or(0.0);
            for kk in 0..k {
                s += a[mm * k + kk] * b_packed[kk * n + nn];
            }
            c[mm * n + nn] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_then_gemm_matches_scalar() {
        let m = 5;
        let n = 17;
        let k = 13;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.03 - 0.7).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();

        // pack first
        let mut b_packed = vec![0f32; n * k];
        pack_weight_for_gemm(&b, &mut b_packed, n, k);

        let mut c_ref = vec![0f32; m * n];
        gemm_a_bt_scalar_packed(&a, &b_packed, &mut c_ref, Some(&bias), m, n, k);

        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let mut c_avx2 = vec![0f32; m * n];
        unsafe {
            gemm_a_bt(&a, &b_packed, &mut c_avx2, Some(&bias), m, n, k);
        }

        for i in 0..m * n {
            assert!(
                (c_ref[i] - c_avx2[i]).abs() < 1e-4,
                "i={i} ref={} avx2={}",
                c_ref[i],
                c_avx2[i]
            );
        }
    }

    #[test]
    fn gemm_no_bias_aligned() {
        let m = 4;
        let n = 16;
        let k = 32;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.02).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.05).collect();
        let mut b_packed = vec![0f32; n * k];
        pack_weight_for_gemm(&b, &mut b_packed, n, k);

        let mut c_ref = vec![0f32; m * n];
        gemm_a_bt_scalar_packed(&a, &b_packed, &mut c_ref, None, m, n, k);

        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let mut c_avx2 = vec![0f32; m * n];
        unsafe {
            gemm_a_bt(&a, &b_packed, &mut c_avx2, None, m, n, k);
        }
        for i in 0..m * n {
            assert!((c_ref[i] - c_avx2[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn gemm_large_rectangular() {
        // 主要 transformer 形状 (small版) を test
        let m = 12;
        let n = 24; // mod 16 = 8、 tail-8 path をテスト
        let k = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.027).cos()).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 0.4).collect();

        let mut b_packed = vec![0f32; n * k];
        pack_weight_for_gemm(&b, &mut b_packed, n, k);

        let mut c_ref = vec![0f32; m * n];
        gemm_a_bt_scalar_packed(&a, &b_packed, &mut c_ref, Some(&bias), m, n, k);

        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let mut c_avx2 = vec![0f32; m * n];
        unsafe {
            gemm_a_bt(&a, &b_packed, &mut c_avx2, Some(&bias), m, n, k);
        }
        for i in 0..m * n {
            assert!(
                (c_ref[i] - c_avx2[i]).abs() < 1e-3,
                "i={i} ref={} avx2={}",
                c_ref[i],
                c_avx2[i]
            );
        }
    }
}
