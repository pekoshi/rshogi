//! AVX2 Multi-head attention (single-batch、 fused、 mask 込み)。
//!
//! Q, K, V は `[T, d_model]` row-major で渡される (gemm 出力直結を想定)。
//! head split は仮想 (physical transpose せず stride 読み)。
//!
//! ## アルゴリズム
//!
//! 各 head 独立に:
//! 1. Q [T, d_head]、 K [T, d_head] (head ごと stride 読み)
//! 2. S = Q @ Kᵀ = [T, T]、 scale
//! 3. softmax (row-wise、 valid_len ≤ T)
//! 4. O[T, d_head] = S @ V[T, d_head]
//! 5. O を out[T, d_model] の head 位置に書き戻す
//!
//! ステップ 2、 4 は AVX2 SIMD で 1 ymm = 8 f32 単位で計算。

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::softmax::softmax_rows_avx2;

/// Multi-head attention の forward (batch=1)。
///
/// 入力:
/// - `q`, `k`, `v`: `[T, d_model]` row-major (= Q/K/V proj 出力)
/// - `t_total`: 物理 token 長 (PAD 含む)
/// - `t_valid`: 有効 token 数 (PAD 除外)
/// - `n_heads`, `d_head`: head 設定 (d_model = n_heads * d_head)
///
/// 出力:
/// - `out`: `[T, d_model]` row-major
/// - `attn_scratch`: `[n_heads * t_total * t_total]` (各 head 専用領域)
///
/// # Safety
/// AVX2 + FMA を要求。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mha_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    attn_scratch: &mut [f32],
    t_total: usize,
    t_valid: usize,
    n_heads: usize,
    d_head: usize,
) {
    let d_model = n_heads * d_head;
    debug_assert!(q.len() >= t_total * d_model);
    debug_assert!(k.len() >= t_total * d_model);
    debug_assert!(v.len() >= t_total * d_model);
    debug_assert!(out.len() >= t_total * d_model);
    debug_assert!(attn_scratch.len() >= n_heads * t_total * t_total);
    debug_assert_eq!(d_head % 8, 0, "d_head must be a multiple of 8 for AVX2 path");

    // SAFETY: 全 SIMD intrinsics は target_feature(avx2,fma) で有効。 ポインタ算術は
    // debug_assert で範囲保証。

    let scale = 1.0_f32 / (d_head as f32).sqrt();
    let scale_vec = _mm256_set1_ps(scale);

    let q_ptr = q.as_ptr();
    let k_ptr = k.as_ptr();
    let v_ptr = v.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let attn_ptr = attn_scratch.as_mut_ptr();

    for h in 0..n_heads {
        let head_off = h * d_head;
        let attn_base = h * t_total * t_total;

        // 1. S[t1, t2] = scale * sum_d Q[t1, h, d] * K[t2, h, d]
        //
        //    inner loop: d ∈ [0..d_head)、 d_head は 8 の倍数を仮定 (debug_assert)。
        //    1 ymm = 8 f32 で d 全体を 1 ymm 累積 → horizontal sum (毎 (t1,t2) ペア)。
        //
        //    最適化候補: t2 を 4 列同時処理して S[t1, t2..t2+4] を一括計算する
        //    (Q 1 行 × K 4 行 同時 dot product)。 現状は 1 t2 / iter の素直版。
        for t1 in 0..t_valid {
            let q_row = q_ptr.add(t1 * d_model + head_off);
            // t2 を 4 並列に: 4 dot product 同時
            let mut t2 = 0usize;
            while t2 + 4 <= t_valid {
                let k0 = k_ptr.add(t2 * d_model + head_off);
                let k1 = k_ptr.add((t2 + 1) * d_model + head_off);
                let k2 = k_ptr.add((t2 + 2) * d_model + head_off);
                let k3 = k_ptr.add((t2 + 3) * d_model + head_off);
                let mut s0 = _mm256_setzero_ps();
                let mut s1 = _mm256_setzero_ps();
                let mut s2 = _mm256_setzero_ps();
                let mut s3 = _mm256_setzero_ps();
                let mut d = 0usize;
                while d < d_head {
                    let qv = _mm256_loadu_ps(q_row.add(d));
                    s0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0.add(d)), s0);
                    s1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k1.add(d)), s1);
                    s2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k2.add(d)), s2);
                    s3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k3.add(d)), s3);
                    d += 8;
                }
                // horizontal sum of 4 ymm → 4 scalars
                let r0 = horizontal_sum_avx2(s0) * scale;
                let r1 = horizontal_sum_avx2(s1) * scale;
                let r2 = horizontal_sum_avx2(s2) * scale;
                let r3 = horizontal_sum_avx2(s3) * scale;
                let dst = attn_ptr.add(attn_base + t1 * t_total + t2);
                *dst = r0;
                *dst.add(1) = r1;
                *dst.add(2) = r2;
                *dst.add(3) = r3;
                t2 += 4;
            }
            // tail
            while t2 < t_valid {
                let kp = k_ptr.add(t2 * d_model + head_off);
                let mut acc = _mm256_setzero_ps();
                let mut d = 0usize;
                while d < d_head {
                    acc = _mm256_fmadd_ps(
                        _mm256_loadu_ps(q_row.add(d)),
                        _mm256_loadu_ps(kp.add(d)),
                        acc,
                    );
                    d += 8;
                }
                let r = horizontal_sum_avx2(acc) * scale;
                *attn_ptr.add(attn_base + t1 * t_total + t2) = r;
                t2 += 1;
            }
        }

        // 2. softmax (rows = valid t1)
        let attn_slice = std::slice::from_raw_parts_mut(attn_ptr.add(attn_base), t_total * t_total);
        softmax_rows_avx2(attn_slice, t_valid, t_total, t_valid);

        // 3. O[t1, h, d] = sum_t2 S[t1, t2] * V[t2, h, d]
        //
        //    最適化: t1 を 4 行ずつ batch して V[t2, :] を 4 行分の dot に再利用。
        //    各 t2 で V load 1 回、 S 重みは 4 つの broadcast。
        //    d_head <= 64 (= 8 ymm)、 t1 block size = 4 → 4 * 4 = 16 acc が想定 (d_head=32 で十分)。
        debug_assert!(d_head <= 64);
        let n_d_vec = d_head / 8;

        let t1_block_end = t_valid & !3;
        let mut t1 = 0usize;
        while t1 < t1_block_end {
            // 4 行 × n_d_vec 列 のアキュムレータ
            let mut acc: [[__m256; 8]; 4] = [[_mm256_setzero_ps(); 8]; 4];
            let s_row0 = attn_ptr.add(attn_base + t1 * t_total);
            let s_row1 = attn_ptr.add(attn_base + (t1 + 1) * t_total);
            let s_row2 = attn_ptr.add(attn_base + (t1 + 2) * t_total);
            let s_row3 = attn_ptr.add(attn_base + (t1 + 3) * t_total);
            for t2 in 0..t_valid {
                let w0 = _mm256_broadcast_ss(&*s_row0.add(t2));
                let w1 = _mm256_broadcast_ss(&*s_row1.add(t2));
                let w2 = _mm256_broadcast_ss(&*s_row2.add(t2));
                let w3 = _mm256_broadcast_ss(&*s_row3.add(t2));
                let v_row = v_ptr.add(t2 * d_model + head_off);
                for di in 0..n_d_vec {
                    let vv = _mm256_loadu_ps(v_row.add(di * 8));
                    acc[0][di] = _mm256_fmadd_ps(w0, vv, acc[0][di]);
                    acc[1][di] = _mm256_fmadd_ps(w1, vv, acc[1][di]);
                    acc[2][di] = _mm256_fmadd_ps(w2, vv, acc[2][di]);
                    acc[3][di] = _mm256_fmadd_ps(w3, vv, acc[3][di]);
                }
            }
            for r in 0..4 {
                let out_row = out_ptr.add((t1 + r) * d_model + head_off);
                for di in 0..n_d_vec {
                    _mm256_storeu_ps(out_row.add(di * 8), acc[r][di]);
                }
            }
            t1 += 4;
        }
        // t1 tail (t1_block_end..t_valid)
        while t1 < t_valid {
            let mut acc: [__m256; 8] = [_mm256_setzero_ps(); 8];
            let s_row = attn_ptr.add(attn_base + t1 * t_total);
            for t2 in 0..t_valid {
                let w = _mm256_broadcast_ss(&*s_row.add(t2));
                let v_row = v_ptr.add(t2 * d_model + head_off);
                for di in 0..n_d_vec {
                    let vv = _mm256_loadu_ps(v_row.add(di * 8));
                    acc[di] = _mm256_fmadd_ps(w, vv, acc[di]);
                }
            }
            let out_row = out_ptr.add(t1 * d_model + head_off);
            for di in 0..n_d_vec {
                _mm256_storeu_ps(out_row.add(di * 8), acc[di]);
            }
            t1 += 1;
        }

        // PAD 行 (t1 >= t_valid) の out を 0 にする (forward の次層が読まない前提だが防御的に)
        for t1 in t_valid..t_total {
            let out_row = out_ptr.add(t1 * d_model + head_off);
            for d in 0..d_head {
                *out_row.add(d) = 0.0;
            }
        }
    }

    let _ = scale_vec;
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // SAFETY: target_feature(avx2,fma) により AVX2 有効。
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_mha(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        t_total: usize,
        t_valid: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Vec<f32> {
        let d_model = n_heads * d_head;
        let scale = 1.0_f32 / (d_head as f32).sqrt();
        let mut out = vec![0f32; t_total * d_model];
        for h in 0..n_heads {
            let mut attn = vec![0f32; t_valid * t_valid];
            for t1 in 0..t_valid {
                for t2 in 0..t_valid {
                    let mut s = 0f32;
                    for d in 0..d_head {
                        s += q[t1 * d_model + h * d_head + d] * k[t2 * d_model + h * d_head + d];
                    }
                    attn[t1 * t_valid + t2] = s * scale;
                }
            }
            for t1 in 0..t_valid {
                let row = &mut attn[t1 * t_valid..(t1 + 1) * t_valid];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for x in row.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                let inv = 1.0 / sum;
                for x in row.iter_mut() {
                    *x *= inv;
                }
            }
            for t1 in 0..t_valid {
                for d in 0..d_head {
                    let mut s = 0f32;
                    for t2 in 0..t_valid {
                        s += attn[t1 * t_valid + t2] * v[t2 * d_model + h * d_head + d];
                    }
                    out[t1 * d_model + h * d_head + d] = s;
                }
            }
        }
        out
    }

    #[test]
    fn mha_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let t_total = 16;
        let t_valid = 12;
        let n_heads = 2;
        let d_head = 8;
        let d_model = n_heads * d_head;
        let q: Vec<f32> = (0..t_total * d_model).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..t_total * d_model).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let v: Vec<f32> = (0..t_total * d_model).map(|i| (i as f32) * 0.03 + 0.05).collect();

        let ref_out = scalar_mha(&q, &k, &v, t_total, t_valid, n_heads, d_head);

        let mut out = vec![0f32; t_total * d_model];
        let mut scratch = vec![0f32; n_heads * t_total * t_total];
        unsafe {
            mha_avx2(&q, &k, &v, &mut out, &mut scratch, t_total, t_valid, n_heads, d_head);
        }
        for t1 in 0..t_valid {
            for d in 0..d_model {
                let idx = t1 * d_model + d;
                assert!(
                    (ref_out[idx] - out[idx]).abs() < 1e-3,
                    "t1={t1} d={d} ref={} avx2={}",
                    ref_out[idx],
                    out[idx]
                );
            }
        }
    }

    #[test]
    fn mha_avx2_t_aligned() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        // t_valid = 116、 n_heads=6、 d_head=32 (PUCT model 実構成)
        let t_total = 116;
        let t_valid = 116;
        let n_heads = 6;
        let d_head = 32;
        let d_model = n_heads * d_head;
        let q: Vec<f32> = (0..t_total * d_model).map(|i| ((i as f32) * 0.0017).sin()).collect();
        let k: Vec<f32> = (0..t_total * d_model).map(|i| ((i as f32) * 0.0023).cos()).collect();
        let v: Vec<f32> = (0..t_total * d_model).map(|i| ((i as f32) * 0.0031).sin()).collect();

        let ref_out = scalar_mha(&q, &k, &v, t_total, t_valid, n_heads, d_head);

        let mut out = vec![0f32; t_total * d_model];
        let mut scratch = vec![0f32; n_heads * t_total * t_total];
        unsafe {
            mha_avx2(&q, &k, &v, &mut out, &mut scratch, t_total, t_valid, n_heads, d_head);
        }
        let mut max_err = 0f32;
        for t1 in 0..t_valid {
            for d in 0..d_model {
                let idx = t1 * d_model + d;
                let e = (ref_out[idx] - out[idx]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
        }
        assert!(max_err < 1e-3, "max_err = {max_err}");
    }
}
