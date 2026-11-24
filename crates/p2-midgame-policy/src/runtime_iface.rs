//! Policy 推論 backend の共通 trait。
//!
//! 並行 Agent B (`p2-transformer-runtime`) の `TransformerRuntime` API が未確定
//! のため、 本クレートは **自前で最小限の trait を切る + adapter pattern** で
//! 進める。 trait sig が固まったら `transformer-runtime` feature の adapter
//! 経由で接続できる。
//!
//! ## 設計理由
//!
//! - αβ ordering は **batch 1 forward** が支配的 (root で 1 回 / 中盤 hook で
//!   1 node あたり 1 回)。 batch スケジューラは MCTS と異なり不要。
//! - `Box<dyn PolicyEvaluator>` を `Search` から `&mut` 経由で呼ぶ。 hot path
//!   の TT 手反復より粒度が粗いため dynamic dispatch コストは無視可能。
//! - 推論成功時は **borrowed slice を返さず**、 呼び出し側が確保した buffer
//!   (`&mut [f32]`) に書き込む形にする。 cache 側の固定 storage と一貫した
//!   パターンになる。

use crate::move_lookup::POLICY_SIZE;
use rshogi_core::position::Position;

/// 推論失敗時のエラー型。
///
/// 依存関係を最小にするため `thiserror` は使わず手書き。
#[derive(Debug)]
pub enum InferenceError {
    /// 出力 buffer のサイズが [`POLICY_SIZE`] と異なる。
    InvalidBufferSize { expected: usize, got: usize },
    /// backend 固有エラー (ckpt load 失敗、 device alloc 失敗、 etc.)。
    Backend(String),
    /// runtime 未設定 (`MidgamePolicyModel` が空、 model load 失敗等)。
    NotConfigured,
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBufferSize { expected, got } => {
                write!(f, "invalid policy buffer size: expected {expected}, got {got}")
            }
            Self::Backend(msg) => write!(f, "backend error: {msg}"),
            Self::NotConfigured => write!(f, "policy runtime not configured"),
        }
    }
}

impl std::error::Error for InferenceError {}

/// Policy 推論 backend の共通 trait。
///
/// **batch 1 専用** にして API を最小化する。 batch スケジューラが必要に
/// なったら別 trait を切る (本クレートでは現状 root + future midgame hook の
/// いずれも単局面なので不要)。
pub trait PolicyEvaluator: Send {
    /// 局面に対する policy logits を `out_logits[..POLICY_SIZE]` に書き込む。
    ///
    /// - `out_logits` は **raw logits** を返す (softmax / temperature は呼び
    ///   出し側の責任)。
    /// - 失敗時は [`InferenceError`] を返す。 呼び出し側は policy を使わず
    ///   通常 ordering に fallback する。
    fn evaluate(&mut self, pos: &Position, out_logits: &mut [f32]) -> Result<(), InferenceError>;

    /// human-readable backend 名 (ログ用)。
    fn name(&self) -> &'static str;
}

/// 検査ヘルパ: 出力 buffer 長が [`POLICY_SIZE`] と一致するかを確認。
#[inline]
pub fn check_logits_buffer(out_logits: &[f32]) -> Result<(), InferenceError> {
    if out_logits.len() != POLICY_SIZE {
        return Err(InferenceError::InvalidBufferSize {
            expected: POLICY_SIZE,
            got: out_logits.len(),
        });
    }
    Ok(())
}

/// テスト / smoke 用の固定 logits を返す dummy evaluator。
///
/// hook の wiring を検証する用途。 実 model 無しでも cargo test を通せる。
pub struct ConstantPolicyEvaluator {
    /// 全 logits に書き込む定数値。
    pub fill: f32,
}

impl ConstantPolicyEvaluator {
    /// 全 0 logits を返す evaluator。
    pub fn zeros() -> Self {
        Self { fill: 0.0 }
    }
}

impl PolicyEvaluator for ConstantPolicyEvaluator {
    fn evaluate(&mut self, _pos: &Position, out_logits: &mut [f32]) -> Result<(), InferenceError> {
        check_logits_buffer(out_logits)?;
        out_logits.fill(self.fill);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "constant"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_evaluator_fills_buffer() {
        let mut buf = vec![0f32; POLICY_SIZE];
        let pos = Position::new();
        let mut ev = ConstantPolicyEvaluator { fill: 0.5 };
        ev.evaluate(&pos, &mut buf).unwrap();
        assert!(buf.iter().all(|&x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn buffer_size_mismatch_is_caught() {
        let mut buf = vec![0f32; 10];
        let pos = Position::new();
        let mut ev = ConstantPolicyEvaluator { fill: 0.0 };
        let err = ev.evaluate(&pos, &mut buf).unwrap_err();
        match err {
            InferenceError::InvalidBufferSize { expected, got } => {
                assert_eq!(expected, POLICY_SIZE);
                assert_eq!(got, 10);
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
