//! Backend 共通 trait。
//!
//! `TransformerRuntime` を満たす実装が CPU / CUDA / Hybrid に存在する。 PUCT
//! 側からは `Box<dyn TransformerRuntime>` で持つことを想定するが、 hot path
//! の batch loop 内では dynamic dispatch を多重呼びしない (1 回の `eval` で
//! batch 全件を回す)。

use p2_transformer::TransformerTokens;

use crate::buffer::PolicyBuffer;
use crate::config::MatryoshkaLevel;
use crate::error::InferenceError;

/// Backend 共通 trait。
///
/// 入力 `batch.len()` と出力 buffer 長 (`out_value.len()`, `out_policy.len()`)
/// は等しいことを呼び出し側で保証する。 不一致時は `ShapeMismatch` を返す。
pub trait TransformerRuntime: Send + Sync {
    /// `MatryoshkaLevel::Full` で評価。
    fn eval(
        &mut self,
        batch: &[TransformerTokens],
        out_value: &mut [f32],
        out_policy: &mut [PolicyBuffer],
    ) -> Result<(), InferenceError>;

    /// `MatryoshkaLevel` 指定付き評価。 デフォルト実装は Full にフォールバック。
    fn eval_with_level(
        &mut self,
        batch: &[TransformerTokens],
        level: MatryoshkaLevel,
        out_value: &mut [f32],
        out_policy: &mut [PolicyBuffer],
    ) -> Result<(), InferenceError> {
        let _ = level; // Light/Medium は TODO、 現状 Full にフォールバック
        self.eval(batch, out_value, out_policy)
    }

    /// backend の最大 batch (これを超えると `BatchTooLarge`)。
    fn max_batch(&self) -> usize;

    /// human-readable backend 名 (ログ / metrics 用)。
    fn name(&self) -> &'static str;
}

/// shape 検査の共通ヘルパ。
#[inline]
pub(crate) fn check_shapes(
    batch: &[TransformerTokens],
    out_value: &[f32],
    out_policy: &[PolicyBuffer],
    max_batch: usize,
) -> Result<(), InferenceError> {
    if batch.len() != out_value.len() || batch.len() != out_policy.len() {
        return Err(InferenceError::ShapeMismatch {
            batch: batch.len(),
            values: out_value.len(),
            policies: out_policy.len(),
        });
    }
    if batch.len() > max_batch {
        return Err(InferenceError::BatchTooLarge {
            got: batch.len(),
            max: max_batch,
        });
    }
    Ok(())
}
