//! AVX2 ベース INT8 dot product / linear primitive (P4 骨格).
//!
//! NNUE 標準の量子化:
//!   - 重み w_f32 → w_i8 = round(w / s_w)、 s_w = max(|w|) / 127 (per-tensor)
//!   - 活性 a_f32 → a_i8 = round(a / s_a)、 ClippedReLU/SCReLU 後は [0, 1] → [0, 127]
//!   - i8 × i8 → i32 で累積、 dequant: y_f32 = sum_i32 * s_w * s_a + bias_f32
//!
//! AVX2 命令の主要パターン (Stockfish/dlshogi 流):
//!   _mm256_maddubs_epi16(a_u8, b_i8): u8 × i8 → i16、 隣接 2 要素ペアの 2-tap 部分和
//!   _mm256_madd_epi16(c_i16, d_i16):  i16 × i16 → i32 ペア足し合わせ
//!   合わせて 32 個の i8×i8 を 1 命令ペアで i32 8 個に縮約
//!
//! ただし入力 a が u8 (= clipped relu 後 [0, 127]) のときに maddubs が使える。
//! a が i8 (= 通常の符号付き) なら別経路 (saturate or signed madd)。
//!
//! 本ファイルは骨格。 ホットパスの fallback (scalar) と AVX2 ファストパス両方を提供。

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 量子化パラメータ (per-tensor)
#[derive(Debug, Clone, Copy)]
pub struct QuantParams {
    /// f32 重みを i8 に量子化したスケール (s_w = max|w| / 127)
    pub w_scale: f32,
    /// activation のスケール (s_a = max|a| / 127、 clipped activation は 1/127)
    pub a_scale: f32,
}

impl QuantParams {
    /// f32 重みベクトルから per-tensor scale を決定
    pub fn from_weights(weights: &[f32]) -> Self {
        let max_abs = weights.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
        let w_scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        Self {
            w_scale,
            a_scale: 1.0 / 127.0, // clipped activation 用 default
        }
    }

    /// f32 重みを i8 に量子化
    pub fn quantize_weights(weights: &[f32], scale: f32) -> Vec<i8> {
        weights
            .iter()
            .map(|&w| {
                let q = (w / scale).round();
                q.clamp(-127.0, 127.0) as i8
            })
            .collect()
    }
}

/// scalar dot product (i8 × i8 → i32) — fallback / 検証用
pub fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// AVX2 dot product (u8 × i8 → i32). a は [0, 127] の clipped activation を想定.
///
/// 32 要素単位で処理、 余りは scalar fallback。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_u8i8_avx2(a: &[u8], b: &[i8]) -> i32 {
    unsafe {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let aligned_n = n & !31; // 32 の倍数
        let mut acc = _mm256_setzero_si256();

        let mut i = 0;
        while i < aligned_n {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            // u8 × i8 → i16 ペア (隣接 2 要素を加算)
            let prod = _mm256_maddubs_epi16(va, vb);
            // i16 → i32 ペア加算 (×1 で signed widen)
            let one = _mm256_set1_epi16(1);
            let widened = _mm256_madd_epi16(prod, one);
            acc = _mm256_add_epi32(acc, widened);
            i += 32;
        }

        // 横加算 (8 lane → 1 i32)
        let lo = _mm256_extracti128_si256(acc, 0);
        let hi = _mm256_extracti128_si256(acc, 1);
        let sum128 = _mm_add_epi32(lo, hi);
        let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0x4E)); // [2,3,0,1]
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0xB1)); // [1,0,3,2]
        let mut total = _mm_cvtsi128_si32(sum32);

        // 余り (32 未満) を scalar
        for j in i..n {
            total += a[j] as i32 * b[j] as i32;
        }
        total
    }
}

/// Linear (output = W * input + bias_f32) — i8 weight × u8 activation、 i32 累積、 f32 dequant
///
/// shapes:
///   weight: (out_dim, in_dim) row-major i8
///   input: (in_dim,) u8 (clipped activation [0, 127])
///   bias: (out_dim,) f32
///   output: (out_dim,) f32
///   scale = w_scale * a_scale (前段の activation scale を含む合成スケール)
pub fn linear_int8(
    input: &[u8],
    weight: &[i8],
    bias: &[f32],
    out_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let in_dim = input.len();
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(bias.len(), out_dim);

    let mut out = vec![0.0f32; out_dim];

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

    for o in 0..out_dim {
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        let sum_i32 = if has_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                dot_u8i8_avx2(input, row)
            }
            #[cfg(not(target_arch = "x86_64"))]
            dot_i8_scalar_unsigned(input, row)
        } else {
            dot_i8_scalar_unsigned(input, row)
        };
        out[o] = sum_i32 as f32 * scale + bias[o];
    }
    out
}

/// scalar fallback for u8 × i8 (= signed dot product where input ≥ 0)
fn dot_i8_scalar_unsigned(a: &[u8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// f32 → u8 quantization (clipped activation [0, 1] → [0, 127])
///
/// SCReLU 後は [0, 1]^2 = [0, 1]、 ClippedReLU 後は [0, 1]、 どちらも u8 [0, 127] に量子化。
pub fn quantize_activations_u8(x: &[f32]) -> Vec<u8> {
    x.iter()
        .map(|&v| {
            let c = v.clamp(0.0, 1.0);
            (c * 127.0).round() as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_i8_scalar_basic() {
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        assert_eq!(dot_i8_scalar(&a, &b), 4 + 10 + 18);
    }

    #[test]
    fn dot_u8i8_scalar_unsigned() {
        let a = vec![10u8, 20, 30];
        let b = vec![1i8, -2, 3];
        assert_eq!(dot_i8_scalar_unsigned(&a, &b), 10 - 40 + 90);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dot_u8i8_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 unavailable, skipping");
            return;
        }
        let n = 64;
        let a: Vec<u8> = (0..n).map(|i| (i % 100) as u8).collect();
        let b: Vec<i8> = (0..n).map(|i| ((i - 50) % 100) as i8).collect();
        let scalar = dot_i8_scalar_unsigned(&a, &b);
        let avx = unsafe { dot_u8i8_avx2(&a, &b) };
        assert_eq!(avx, scalar, "AVX2 result mismatch");
    }

    #[test]
    fn linear_int8_matches_f32_baseline() {
        // simple verification: 量子化後の linear が f32 baseline と近い
        let in_dim = 32;
        let out_dim = 4;
        let f32_w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32 - 64.0) / 128.0).collect();
        let f32_a: Vec<f32> = (0..in_dim).map(|i| (i as f32) / 100.0).collect();
        let bias: Vec<f32> = vec![0.5, -0.3, 0.0, 0.1];

        // f32 baseline
        let mut f32_out = bias.clone();
        for o in 0..out_dim {
            for i in 0..in_dim {
                f32_out[o] += f32_w[o * in_dim + i] * f32_a[i];
            }
        }

        // INT8 quantize
        let qp = QuantParams::from_weights(&f32_w);
        let w_q = QuantParams::quantize_weights(&f32_w, qp.w_scale);
        let a_q = quantize_activations_u8(&f32_a);
        let scale = qp.w_scale * (1.0 / 127.0);
        let int8_out = linear_int8(&a_q, &w_q, &bias, out_dim, scale);

        // 量子化誤差 < 5% を許容
        for (i, (&a, &b)) in f32_out.iter().zip(int8_out.iter()).enumerate() {
            let abs_err = (a - b).abs();
            let rel = if a.abs() > 1e-3 {
                abs_err / a.abs()
            } else {
                abs_err
            };
            assert!(rel < 0.05, "out[{}]: f32={} int8={} rel_err={}", i, a, b, rel);
        }
    }
}
