//! NNUE ネットワーク層の実装
//!
//! - `InputAffineTransform`: i32 入力 + i16 重み → i32 出力 (Layer 1 専用)
//! - `AffineTransform`: u8 入力 + i8 重み → i32 出力 (Layer 2/3)
//! - `ClippedReLU`: i32 → u8 (>> WEIGHT_SCALE_BITS, clamp [0, 127])
//!
//! AVX2 が利用可能な場合は SIMD 最適化パスを使用する。

use std::io::{self, Read, Write};

use crate::align::AlignedBox;
use crate::types::WEIGHT_SCALE_BITS;

// ─── AVX2 SIMD ヘルパー関数 ───

/// AVX2 水平加算 (i32×8 → i32)
///
/// 上位/下位 128bit を加算し、さらに 64bit, 32bit と畳み込む。
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn hsum_i32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    // SAFETY: 呼び出し側が avx2 フィーチャを保証する
    unsafe {
        use std::arch::x86_64::*;

        let hi = _mm256_extracti128_si256(v, 1);
        let lo = _mm256_castsi256_si128(v);
        let sum128 = _mm_add_epi32(lo, hi);

        let hi64 = _mm_unpackhi_epi64(sum128, sum128);
        let sum64 = _mm_add_epi32(sum128, hi64);

        let hi32 = _mm_shuffle_epi32(sum64, 1);
        let sum32 = _mm_add_epi32(sum64, hi32);

        _mm_cvtsi128_si32(sum32)
    }
}

/// AVX2 u8×i8→i32 積和演算
///
/// `maddubs_epi16` (u8×i8→i16 飽和加算) + `madd_epi16` (i16→i32 積和) で
/// 32 バイトの dot product を i32×8 のアキュムレータに加算する。
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn m256_add_dpbusd_epi32(
    acc: &mut std::arch::x86_64::__m256i,
    a: std::arch::x86_64::__m256i,
    b: std::arch::x86_64::__m256i,
) {
    // SAFETY: 呼び出し側が avx2 フィーチャを保証する
    unsafe {
        use std::arch::x86_64::*;
        let product = _mm256_maddubs_epi16(a, b);
        let product32 = _mm256_madd_epi16(product, _mm256_set1_epi16(1));
        *acc = _mm256_add_epi32(*acc, product32);
    }
}

// ─── InputAffineTransform (Layer 1) ───

/// Layer 1 専用: i32 入力 + i16 重み → i32 出力
///
/// weights は行優先 `[OUT][padded(IN)]` 配置。
pub struct InputAffineTransform<const IN: usize, const OUT: usize> {
    pub biases: Box<[i32; OUT]>,
    pub weights: AlignedBox<i16>,
}

impl<const IN: usize, const OUT: usize> InputAffineTransform<IN, OUT> {
    /// パディング後の入力次元数 (i16 で 32B 境界 = 16 要素単位)
    pub const PADDED_IN: usize = IN.div_ceil(16) * 16;

    /// 全ゼロで初期化
    pub fn new_zeroed() -> Self {
        Self {
            biases: Box::new([0i32; OUT]),
            weights: AlignedBox::new(OUT * Self::PADDED_IN),
        }
    }

    /// 順伝播: output[j] = bias[j] + Σ_i weights[j][i] * input[i]
    ///
    /// AVX2 利用可能時は `cvtepi16_epi32` + `mullo_epi32` の行ベクトル方式で高速化。
    pub fn propagate(&self, input: &[i32; IN], output: &mut [i32; OUT]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY:
            // - weights.len() >= OUT * PADDED_IN (構造上保証)
            // - PADDED_IN は 16 の倍数 (i16×16 = 32B AVX2 レジスタ幅)
            // - weights は AlignedBox<i16> で 64B アライン
            // - input は 2048 i32 (IN=2048) で unaligned load 使用
            unsafe {
                use std::arch::x86_64::*;

                let weights_ptr = self.weights.as_ptr();
                let input_ptr = input.as_ptr();
                // 8 i16 = 128bit = __m128i、cvtepi16_epi32 で 8 i32 = __m256i に拡張
                let num_chunks = IN / 8;

                for (j, out) in output.iter_mut().enumerate() {
                    let mut acc = _mm256_setzero_si256();
                    let row_offset = j * Self::PADDED_IN;

                    for k in 0..num_chunks {
                        let offset = k * 8;
                        // 8 i16 を読み込み、i32 に符号拡張
                        let w_i16 =
                            _mm_loadu_si128(weights_ptr.add(row_offset + offset) as *const __m128i);
                        let w_i32 = _mm256_cvtepi16_epi32(w_i16);
                        // 8 i32 入力を読み込み
                        let x_i32 = _mm256_loadu_si256(input_ptr.add(offset) as *const __m256i);
                        // 要素ごと乗算 + 累積
                        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w_i32, x_i32));
                    }

                    let mut scalar_acc = self.biases[j] + hsum_i32_avx2(acc);

                    // 残り要素 (IN % 8) をスカラーで処理
                    let row = &self.weights[row_offset..row_offset + Self::PADDED_IN];
                    for i in (num_chunks * 8)..IN {
                        scalar_acc += row[i] as i32 * input[i];
                    }

                    *out = scalar_acc;
                }
            }
            return;
        }

        // スカラーフォールバック
        #[allow(unreachable_code)]
        for (j, out) in output.iter_mut().enumerate() {
            let row = &self.weights[j * Self::PADDED_IN..(j + 1) * Self::PADDED_IN];
            let mut acc = self.biases[j];
            for i in 0..IN {
                acc += row[i] as i32 * input[i];
            }
            *out = acc;
        }
    }

    /// バイナリストリームから読み込み
    ///
    /// フォーマット: biases (i32 LE × OUT), weights (i16 LE × OUT × IN)
    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut result = Self::new_zeroed();

        // biases
        let mut buf = vec![0u8; OUT * 4];
        reader.read_exact(&mut buf)?;
        for j in 0..OUT {
            result.biases[j] =
                i32::from_le_bytes([buf[j * 4], buf[j * 4 + 1], buf[j * 4 + 2], buf[j * 4 + 3]]);
        }

        // weights: ファイル上は [OUT][IN]、メモリ上は [OUT][PADDED_IN]
        let mut wbuf = vec![0u8; OUT * IN * 2];
        reader.read_exact(&mut wbuf)?;
        for j in 0..OUT {
            for i in 0..IN {
                let offset = (j * IN + i) * 2;
                let val = i16::from_le_bytes([wbuf[offset], wbuf[offset + 1]]);
                result.weights[j * Self::PADDED_IN + i] = val;
            }
            // パディング部分はゼロのまま
        }

        Ok(result)
    }

    /// バイナリストリームに書き出し
    ///
    /// フォーマット: biases (i32 LE × OUT), weights (i16 LE × OUT × IN)
    /// メモリ上 `[OUT][PADDED_IN]` からパディング部分を除いた IN 列のみを書き出す。
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // biases
        for j in 0..OUT {
            writer.write_all(&self.biases[j].to_le_bytes())?;
        }

        // weights: PADDED_IN からパディングを除いた IN 列を書き出す
        for j in 0..OUT {
            let row_offset = j * Self::PADDED_IN;
            for i in 0..IN {
                writer.write_all(&self.weights[row_offset + i].to_le_bytes())?;
            }
        }

        Ok(())
    }
}

// ─── AffineTransform (Layer 2/3) ───

/// Layer 2/3: u8 入力 + i8 重み → i32 出力
///
/// AVX2 かつ `OUT % 8 == 0` の場合はスクランブル形式重みを使用し、
/// ループ逆転最適化版の propagate を実行する。
/// それ以外は行優先 `[OUT][padded(IN)]` のスカラー or 行ベクトル SIMD。
pub struct AffineTransform<const IN: usize, const OUT: usize> {
    pub biases: Box<[i32; OUT]>,
    pub weights: AlignedBox<i8>,
}

impl<const IN: usize, const OUT: usize> AffineTransform<IN, OUT> {
    /// パディング後の入力次元数 (i8 で 32B 境界 = 32 要素単位)
    pub const PADDED_IN: usize = IN.div_ceil(32) * 32;

    /// チャンクサイズ (u8×4 = i32 として読む単位)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    const CHUNK_SIZE: usize = 4;

    /// 入力チャンク数 (ループ逆転最適化用)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    const NUM_INPUT_CHUNKS: usize = Self::PADDED_IN / Self::CHUNK_SIZE;

    /// スクランブル形式のウェイトを使用するか
    ///
    /// AVX2: `OUT % 8 == 0` の場合に使用
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    pub(crate) const fn should_use_scrambled_weights() -> bool {
        OUT.is_multiple_of(8) && OUT > 0
    }

    /// 重みインデックスのスクランブル変換
    ///
    /// 行優先 `weights[output][input]` から列優先 `weights[input_chunk][output][4]` に変換。
    /// ループ逆転最適化で入力チャンクをブロードキャストし全出力に同時適用する形式。
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    pub(crate) const fn get_weight_index_scrambled(i: usize) -> usize {
        (i / Self::CHUNK_SIZE) % (Self::PADDED_IN / Self::CHUNK_SIZE) * OUT * Self::CHUNK_SIZE
            + i / Self::PADDED_IN * Self::CHUNK_SIZE
            + i % Self::CHUNK_SIZE
    }

    /// 全ゼロで初期化
    pub fn new_zeroed() -> Self {
        Self {
            biases: Box::new([0i32; OUT]),
            weights: AlignedBox::new(OUT * Self::PADDED_IN),
        }
    }

    /// 順伝播: output[j] = bias[j] + Σ_i weights[j][i] * input[i]
    ///
    /// AVX2 + `OUT % 8 == 0`: スクランブル重み + ループ逆転最適化
    /// AVX2 + `OUT % 8 != 0`: 行ベクトル方式 (maddubs + hsum)
    /// それ以外: スカラーフォールバック
    pub fn propagate(&self, input: &[u8], output: &mut [i32; OUT]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY:
            // - input.len() >= IN (呼び出し側で保証)
            // - weights.len() >= OUT * PADDED_IN (構造上保証)
            // - weights は AlignedBox<i8> で 64B アライン
            // - PADDED_IN は 32 の倍数なのでオフセットは常に 32B 境界
            unsafe {
                use std::arch::x86_64::*;

                // OUT % 8 == 0: ループ逆転最適化版 (スクランブル重み使用)
                #[allow(clippy::needless_range_loop)]
                if Self::should_use_scrambled_weights() {
                    const MAX_REGS: usize = 128;
                    let num_regs = OUT / 8;
                    debug_assert!(num_regs <= MAX_REGS);

                    // アキュムレータをバイアスで初期化
                    let mut acc = [_mm256_setzero_si256(); MAX_REGS];
                    let bias_ptr = self.biases.as_ptr() as *const __m256i;
                    for k in 0..num_regs {
                        acc[k] = _mm256_loadu_si256(bias_ptr.add(k));
                    }

                    let input32 = input.as_ptr() as *const i32;
                    let weights_ptr = self.weights.as_ptr();

                    // 外側: 入力チャンク (4 バイト = 1 i32)
                    for i in 0..Self::NUM_INPUT_CHUNKS {
                        let in_val = _mm256_set1_epi32(*input32.add(i));

                        // スクランブル形式: weights[input_chunk][output][4]
                        let col = weights_ptr.add(i * OUT * Self::CHUNK_SIZE) as *const __m256i;

                        // 内側: 全出力レジスタに積和演算
                        for k in 0..num_regs {
                            m256_add_dpbusd_epi32(
                                &mut acc[k],
                                in_val,
                                _mm256_load_si256(col.add(k)),
                            );
                        }
                    }

                    // 結果を出力
                    let out_ptr = output.as_mut_ptr() as *mut __m256i;
                    for k in 0..num_regs {
                        _mm256_storeu_si256(out_ptr.add(k), acc[k]);
                    }
                    return;
                }

                // OUT % 8 != 0: 行ベクトル方式 (L3: 256→1 等)
                let num_chunks = Self::PADDED_IN / 32;
                let one = _mm256_set1_epi16(1);
                let input_ptr = input.as_ptr();
                let weights_ptr = self.weights.as_ptr();

                for (j, (out, &bias)) in output.iter_mut().zip(&*self.biases).enumerate() {
                    let mut acc = _mm256_setzero_si256();
                    let weight_row_offset = j * Self::PADDED_IN;

                    for k in 0..num_chunks {
                        let offset = k * 32;
                        let in_vec = _mm256_loadu_si256(input_ptr.add(offset) as *const __m256i);
                        let w_vec = _mm256_load_si256(
                            weights_ptr.add(weight_row_offset + offset) as *const __m256i
                        );
                        let prod16 = _mm256_maddubs_epi16(in_vec, w_vec);
                        let prod32 = _mm256_madd_epi16(prod16, one);
                        acc = _mm256_add_epi32(acc, prod32);
                    }

                    *out = bias + hsum_i32_avx2(acc);
                }
            }
            return;
        }

        // スカラーフォールバック
        #[allow(unreachable_code)]
        for (j, out) in output.iter_mut().enumerate() {
            let row = &self.weights[j * Self::PADDED_IN..(j + 1) * Self::PADDED_IN];
            let mut acc = self.biases[j];
            for i in 0..IN {
                acc += row[i] as i32 * input[i] as i32;
            }
            *out = acc;
        }
    }

    /// バイナリストリームから読み込み
    ///
    /// AVX2 かつ `OUT % 8 == 0` の場合はスクランブル形式で重みを格納する。
    ///
    /// フォーマット: biases (i32 LE × OUT), weights (i8 × OUT × IN)
    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut result = Self::new_zeroed();

        // biases
        let mut buf = vec![0u8; OUT * 4];
        reader.read_exact(&mut buf)?;
        for j in 0..OUT {
            result.biases[j] =
                i32::from_le_bytes([buf[j * 4], buf[j * 4 + 1], buf[j * 4 + 2], buf[j * 4 + 3]]);
        }

        // weights: ファイル上は [OUT][IN]、メモリ上は [OUT][PADDED_IN]
        let weight_size = OUT * Self::PADDED_IN;
        let mut wbuf = vec![0u8; OUT * IN];
        reader.read_exact(&mut wbuf)?;

        // AVX2 + スクランブル対象: 列優先 weights[input_chunk][output][4] に変換
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if Self::should_use_scrambled_weights() {
                for j in 0..OUT {
                    for i in 0..IN {
                        let linear_idx = j * Self::PADDED_IN + i;
                        let scrambled_idx = Self::get_weight_index_scrambled(linear_idx);
                        debug_assert!(scrambled_idx < weight_size);
                        result.weights[scrambled_idx] = wbuf[j * IN + i] as i8;
                    }
                }
                return Ok(result);
            }
        }

        // 非スクランブル: 行優先のまま格納
        let _ = weight_size;
        for j in 0..OUT {
            for i in 0..IN {
                result.weights[j * Self::PADDED_IN + i] = wbuf[j * IN + i] as i8;
            }
            // パディング部分はゼロのまま
        }

        Ok(result)
    }

    /// バイナリストリームに書き出し
    ///
    /// フォーマット: biases (i32 LE × OUT), weights (i8 × OUT × IN)
    /// AVX2 + `OUT % 8 == 0` の場合、メモリ上のスクランブル配置を逆変換して
    /// 行優先順序で書き出す。
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // biases
        for j in 0..OUT {
            writer.write_all(&self.biases[j].to_le_bytes())?;
        }

        // weights: スクランブル逆変換して行優先で書き出す
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if Self::should_use_scrambled_weights() {
                for j in 0..OUT {
                    for i in 0..IN {
                        let linear_idx = j * Self::PADDED_IN + i;
                        let scrambled_idx = Self::get_weight_index_scrambled(linear_idx);
                        writer.write_all(&[self.weights[scrambled_idx] as u8])?;
                    }
                }
                return Ok(());
            }
        }

        // 非スクランブル: 行優先のまま、パディングを除いて書き出す
        for j in 0..OUT {
            let row_offset = j * Self::PADDED_IN;
            for i in 0..IN {
                writer.write_all(&[self.weights[row_offset + i] as u8])?;
            }
        }

        Ok(())
    }
}

// ─── ClippedReLU ───

/// ClippedReLU: i32 → u8
///
/// `output[i] = clamp(input[i] >> WEIGHT_SCALE_BITS, 0, 127)`
///
/// AVX2 利用可能時は 32 要素/イテレーションで処理。
pub struct ClippedReLU;

impl ClippedReLU {
    /// i32 配列を ClippedReLU で u8 配列に変換
    pub fn propagate<const DIM: usize>(input: &[i32; DIM], output: &mut [u8; DIM]) {
        let mut processed: usize = 0;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let num_chunks = DIM / 32;
            if num_chunks > 0 {
                // SAFETY:
                // - num_chunks > 0 を確認済み
                // - loadu/storeu を使用するためアライメント不要
                // - DIM >= 32 なのでバッファ範囲内
                unsafe {
                    use std::arch::x86_64::*;

                    let zero = _mm256_setzero_si256();
                    let offsets = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

                    let in_ptr = input.as_ptr() as *const __m256i;
                    let out_ptr = output.as_mut_ptr() as *mut __m256i;

                    for i in 0..num_chunks {
                        // 4 × __m256i (4×8 = 32 i32 要素) を読み込み
                        let in0 = _mm256_loadu_si256(in_ptr.add(i * 4));
                        let in1 = _mm256_loadu_si256(in_ptr.add(i * 4 + 1));
                        let in2 = _mm256_loadu_si256(in_ptr.add(i * 4 + 2));
                        let in3 = _mm256_loadu_si256(in_ptr.add(i * 4 + 3));

                        // i32 → i16 (飽和パック) → 右シフト
                        let words0 = _mm256_srai_epi16(
                            _mm256_packs_epi32(in0, in1),
                            WEIGHT_SCALE_BITS as i32,
                        );
                        let words1 = _mm256_srai_epi16(
                            _mm256_packs_epi32(in2, in3),
                            WEIGHT_SCALE_BITS as i32,
                        );

                        // i16 → i8 (飽和パック) → max(0) でクランプ
                        let bytes = _mm256_max_epi8(_mm256_packs_epi16(words0, words1), zero);

                        // レーン並び替え (packs のクロスレーン問題を修正)
                        let result = _mm256_permutevar8x32_epi32(bytes, offsets);

                        _mm256_storeu_si256(out_ptr.add(i), result);
                    }
                }
                processed = num_chunks * 32;
            }
        }

        // スカラーフォールバック (残り要素)
        for i in processed..DIM {
            let shifted = input[i] >> WEIGHT_SCALE_BITS;
            output[i] = shifted.clamp(0, 127) as u8;
        }
    }
}

// ─── スカラー参照実装 (テスト用) ───

/// テスト用: 必ずスカラーで計算する参照実装
#[cfg(test)]
pub(crate) mod scalar_reference {
    use crate::types::WEIGHT_SCALE_BITS;

    /// InputAffineTransform のスカラー参照実装
    pub fn input_affine_propagate(
        biases: &[i32],
        weights: &[i16],
        padded_in: usize,
        input: &[i32],
        output: &mut [i32],
    ) {
        for (j, out) in output.iter_mut().enumerate() {
            let row = &weights[j * padded_in..(j + 1) * padded_in];
            let mut acc = biases[j];
            for (i, &x) in input.iter().enumerate() {
                acc += row[i] as i32 * x;
            }
            *out = acc;
        }
    }

    /// AffineTransform のスカラー参照実装
    pub fn affine_propagate(
        biases: &[i32],
        weights: &[i8],
        padded_in: usize,
        input: &[u8],
        output: &mut [i32],
    ) {
        for (j, out) in output.iter_mut().enumerate() {
            // スクランブルされていない行優先レイアウトを使用
            let row = &weights[j * padded_in..(j + 1) * padded_in];
            let mut acc = biases[j];
            for (i, &x) in input.iter().enumerate() {
                acc += row[i] as i32 * x as i32;
            }
            *out = acc;
        }
    }

    /// ClippedReLU のスカラー参照実装
    pub fn clipped_relu(input: &[i32], output: &mut [u8]) {
        for (i, &x) in input.iter().enumerate() {
            let shifted = x >> WEIGHT_SCALE_BITS;
            output[i] = shifted.clamp(0, 127) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_affine_transform_small() {
        // 4→2 の小さな変換で手計算と照合
        let mut layer = InputAffineTransform::<4, 2>::new_zeroed();
        // biases = [10, 20]
        layer.biases[0] = 10;
        layer.biases[1] = 20;
        // weights[0] = [1, 2, 3, 4], weights[1] = [5, 6, 7, 8]
        let padded = InputAffineTransform::<4, 2>::PADDED_IN;
        // row 0
        layer.weights[0] = 1;
        layer.weights[1] = 2;
        layer.weights[2] = 3;
        layer.weights[3] = 4;
        // row 1
        layer.weights[padded] = 5;
        layer.weights[padded + 1] = 6;
        layer.weights[padded + 2] = 7;
        layer.weights[padded + 3] = 8;

        let input = [1i32, 2, 3, 4];
        let mut output = [0i32; 2];
        layer.propagate(&input, &mut output);

        // output[0] = 10 + 1*1 + 2*2 + 3*3 + 4*4 = 10 + 1+4+9+16 = 40
        assert_eq!(output[0], 40);
        // output[1] = 20 + 5*1 + 6*2 + 7*3 + 8*4 = 20 + 5+12+21+32 = 90
        assert_eq!(output[1], 90);
    }

    #[test]
    fn test_affine_transform_small() {
        // 4→2 の小さな変換で手計算と照合
        let mut layer = AffineTransform::<4, 2>::new_zeroed();
        layer.biases[0] = 100;
        layer.biases[1] = 200;
        let padded = AffineTransform::<4, 2>::PADDED_IN;
        // row 0
        layer.weights[0] = 1;
        layer.weights[1] = -2;
        layer.weights[2] = 3;
        layer.weights[3] = -4;
        // row 1
        layer.weights[padded] = 5;
        layer.weights[padded + 1] = 6;
        layer.weights[padded + 2] = -7;
        layer.weights[padded + 3] = 8;

        let input: [u8; 4] = [10, 20, 30, 40];
        let mut output = [0i32; 2];
        layer.propagate(&input, &mut output);

        // output[0] = 100 + 1*10 + (-2)*20 + 3*30 + (-4)*40 = 100 + 10 - 40 + 90 - 160 = 0
        assert_eq!(output[0], 0);
        // output[1] = 200 + 5*10 + 6*20 + (-7)*30 + 8*40 = 200 + 50 + 120 - 210 + 320 = 480
        assert_eq!(output[1], 480);
    }

    #[test]
    fn test_clipped_relu() {
        let input = [
            -100i32,   // 負 → 0
            0,         // 0 → 0
            64,        // 64 >> 6 = 1
            127 * 64,  // 127
            128 * 64,  // 128 → clamp 127
            1000 * 64, // 大きい → 127
        ];
        let mut output = [0u8; 6];
        ClippedReLU::propagate(&input, &mut output);

        assert_eq!(output[0], 0);
        assert_eq!(output[1], 0);
        assert_eq!(output[2], 1);
        assert_eq!(output[3], 127);
        assert_eq!(output[4], 127);
        assert_eq!(output[5], 127);
    }
}
