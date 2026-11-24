//! Transformer 入力 token 構築モジュール。
//!
//! - [`vocab`]: vocab id の定数と range 定義。
//! - [`board`]: 盤上駒 → vocab id 変換 (lookup table)。
//! - [`hand`]: 手駒 count-aware vocab。
//! - [`king_relative`]: 玉相対座標 → 289 bucket。
//! - [`summary`]: phase / material / ply bucket utilities。
//! - [`convert`]: `Position` → [`convert::TransformerTokens`] 構築。
//! - [`psv`]: PSV (40 byte fixed record) → token record 化 skeleton。

pub mod board;
pub mod convert;
pub mod hand;
pub mod king_relative;
pub mod psv;
pub mod summary;
pub mod vocab;
