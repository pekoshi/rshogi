//! CUDA バックエンド
//!
//! GPU での NNUE バッチ推論を提供する。
//! `cuda` feature 有効時のみコンパイルされる。

pub(crate) mod ffi;
pub(crate) mod memory;
