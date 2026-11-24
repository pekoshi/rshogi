//! Runtime 共通エラー型。

use thiserror::Error;

/// 推論時に発生する可能性のあるエラー。
#[derive(Debug, Error)]
pub enum InferenceError {
    /// 入出力 buffer のサイズ不整合。
    #[error(
        "buffer size mismatch: batch_len={batch}, out_values={values}, out_policies={policies}"
    )]
    ShapeMismatch {
        batch: usize,
        values: usize,
        policies: usize,
    },
    /// batch サイズが backend の最大値を超過。
    #[error("batch size {got} exceeds backend maximum {max}")]
    BatchTooLarge { got: usize, max: usize },
    /// candle backend からのエラー。
    #[error("candle backend error: {0}")]
    Candle(String),
    /// model ckpt 読込エラー。
    #[error("model load error: {0}")]
    ModelLoad(String),
    /// 該当 backend が build されていない (例: feature 無効)。
    #[error("backend unavailable: {0}")]
    BackendUnavailable(&'static str),
}

impl From<candle_core::Error> for InferenceError {
    fn from(e: candle_core::Error) -> Self {
        InferenceError::Candle(e.to_string())
    }
}

impl From<anyhow::Error> for InferenceError {
    fn from(e: anyhow::Error) -> Self {
        InferenceError::Candle(e.to_string())
    }
}
