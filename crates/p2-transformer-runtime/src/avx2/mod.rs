//! AVX2 microkernels for the CPU Transformer inference path.
//!
//! このサブモジュールは batch=1 の単発推論を高速化するための AVX2 (256-bit f32)
//! microkernel 群を提供する。 候補 token 数 (`T <= 128`) と d_model=192 の小サイズ
//! 行列積を、 candle の pure-Rust gemm に依存せず手書き AVX2 で計算する。
//!
//! ## ホット path 設計
//!
//! - すべての `unsafe` ブロックは SIMD intrinsics 専用。 安全性コメントを必ず付ける。
//! - heap allocation はゼロ。 呼び出し側で `Scratch` を確保し、 内部 buffer を再利用する。
//! - 各 kernel は AVX2 を仮定する。 caller 側で `is_x86_feature_detected!("avx2")`
//!   を確認した上で呼び出すこと。 `unsafe fn` にしているのは「AVX2 必須」の契約を
//!   型レベルで明示するため。
//!
//! ## モジュール構成
//!
//! - `matmul`: 行列積 `C = A * Bᵀ + bias` (AVX2 8-wide column micro-kernel)
//! - `softmax`: 行ごとの softmax (PAD mask 込み)
//! - `gelu`: GELU activation (近似 tanh 形)
//! - `layer_norm`: LayerNorm (mean + variance を AVX2 reduction)
//! - `attention`: Multi-head attention の融合 forward
//!
//! ## 数値精度
//!
//! 算術順序は candle の `f32` 経路と完全には一致しないが、 `abs_err < 1e-3`
//! を目標とする。 統合 test `avx2_matches_candle_within_tolerance` で確認。

#![cfg(target_arch = "x86_64")]
// SIMD intrinsics は全て unsafe fn だが、 ここでは関数全体が target_feature(avx2,fma)
// で AVX2 を要求しており、 安全性は「呼び出し元が AVX2 detection を行うこと」 と
// 「slice 長を debug_assert で検査すること」 に集約される。 個別の intrinsic call ごとに
// unsafe ブロックを書くのは可読性を著しく損なうため、 本モジュールでは
// `unsafe_op_in_unsafe_fn` lint を allow する (cargo policy の例外として、 関数単位の
// safety コメントで保証する)。
#![allow(unsafe_op_in_unsafe_fn)]
// SIMD microkernel では固定長配列 (acc[0..n_d_vec]) への index アクセスが自然なため、
// iterator/enumerate への書換は可読性を下げるだけで安全性向上に寄与しない。
#![allow(clippy::needless_range_loop)]

pub mod attention;
pub mod gelu;
pub mod layer_norm;
pub mod matmul;
pub mod simd_math;
pub mod softmax;
