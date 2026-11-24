//! P2 特徴量抽出モジュール
//!
//! NNUE 入力用の 2048 次元特徴量を Position から生成する。
//! 既存 NNUE (HalfKP) とは別フォーマット。

pub mod delta;
pub mod extract;
pub mod feature_index;
pub mod feature_types;

#[cfg(test)]
mod tests;

pub use delta::{apply_delta, compute_delta};
pub use extract::extract_features;
pub use feature_index::*;
pub use feature_types::*;
