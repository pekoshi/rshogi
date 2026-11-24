//! Reusable な出力 buffer 型。

use crate::config::RUNTIME_POLICY_SIZE;

/// policy logits 用の reusable buffer (`POLICY_SIZE = 2187`)。
///
/// PUCT 側で予め `vec![PolicyBuffer::default(); batch]` を確保し、 毎回同じ
/// buffer を渡すことで heap allocation を排除する。
#[repr(C, align(64))]
#[derive(Clone)]
pub struct PolicyBuffer {
    /// 2187 個の logits (raw、 softmax は呼び出し側で実施)。
    pub logits: [f32; RUNTIME_POLICY_SIZE],
}

impl PolicyBuffer {
    /// すべて 0 で初期化。
    #[inline]
    pub fn zeroed() -> Self {
        Self {
            logits: [0.0f32; RUNTIME_POLICY_SIZE],
        }
    }

    /// `logits` を 0 にリセット (再利用前の念のため)。
    #[inline]
    pub fn clear(&mut self) {
        self.logits.fill(0.0);
    }
}

impl Default for PolicyBuffer {
    fn default() -> Self {
        Self::zeroed()
    }
}
