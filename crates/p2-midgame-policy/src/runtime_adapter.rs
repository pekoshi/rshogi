//! 並行 Agent B が提供する `p2-transformer-runtime::TransformerRuntime` を、
//! 本クレートの [`PolicyEvaluator`] trait に変換する adapter。
//!
//! - `transformer-runtime` feature を有効化したときのみ build される。
//! - candle / Transformer model 依存を引き込むため、 default では本ファイル
//!   は build されない (`cpu` feature と排他的)。
//!
//! ## 連携モデル
//!
//! `TransformerRuntime::eval(batch, value, policy)` は batch 形式だが本 hook
//! の現状用途は batch=1 のみ。 単 1 局面の `Position` → `TransformerTokens`
//! 変換 → `[batch_size=1]` で eval → `PolicyBuffer.logits[..2187]` を取り出す。
//!
//! ## 既知の制約
//!
//! - `p2_transformer::POLICY_SIZE` と本クレートの [`POLICY_SIZE`] は等しい
//!   ことを debug_assert で確認する。 異なる場合は build 時にバージョン整合性
//!   崩れている。
//! - `Send` 制約は満たすが `Sync` ではない (`TransformerRuntime: Send + Sync`
//!   だが eval は `&mut self`)。 単一 thread 専用。
//!
//! [`POLICY_SIZE`]: crate::move_lookup::POLICY_SIZE

use p2_transformer::{TransformerTokens, position_to_tokens};
use p2_transformer_runtime::{PolicyBuffer, TransformerRuntime};
use rshogi_core::position::Position;

use crate::move_lookup::POLICY_SIZE;
use crate::runtime_iface::{InferenceError, PolicyEvaluator, check_logits_buffer};

/// `TransformerRuntime` を `PolicyEvaluator` に橋渡しする。
///
/// 内部に reusable `TransformerTokens` 1 個 + `PolicyBuffer` 1 個 + value
/// scratch を持ち、 hot path の heap allocation を回避する。
pub struct TransformerRuntimeAdapter<R: TransformerRuntime> {
    runtime: R,
    /// 入力 token batch (max_batch=1 想定)。
    tokens: [TransformerTokens; 1],
    /// 推論結果 policy (batch=1)。 `[0].logits` を copy する。
    policy: [PolicyBuffer; 1],
    /// 推論結果 value (未使用だが API 上 buffer が必要)。
    value: [f32; 1],
}

impl<R: TransformerRuntime> TransformerRuntimeAdapter<R> {
    /// Adapter を作成。
    pub fn new(runtime: R) -> Self {
        Self {
            runtime,
            tokens: [TransformerTokens::default(); 1],
            policy: [PolicyBuffer::zeroed()],
            value: [0.0],
        }
    }

    /// 内部 runtime への immutable 参照。
    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    /// 内部 runtime への mutable 参照 (advanced use)。
    pub fn runtime_mut(&mut self) -> &mut R {
        &mut self.runtime
    }
}

impl<R: TransformerRuntime + Send> PolicyEvaluator for TransformerRuntimeAdapter<R> {
    fn evaluate(&mut self, pos: &Position, out_logits: &mut [f32]) -> Result<(), InferenceError> {
        check_logits_buffer(out_logits)?;
        debug_assert_eq!(
            POLICY_SIZE,
            p2_transformer::POLICY_SIZE,
            "POLICY_SIZE mismatch between p2-midgame-policy and p2-transformer"
        );

        // Position → tokens
        self.tokens[0] = position_to_tokens(pos);

        // 推論 (batch=1)
        self.runtime
            .eval(&self.tokens, &mut self.value, &mut self.policy)
            .map_err(|e| InferenceError::Backend(format!("{e:?}")))?;

        // logits を out にコピー
        out_logits.copy_from_slice(&self.policy[0].logits);
        Ok(())
    }

    fn name(&self) -> &'static str {
        self.runtime.name()
    }
}

// Adapter (テストは feature gate のため重い candle ビルドが必要、 本 crate は
// 単体テストとして実装しない。 統合テストは p2-transformer-runtime 側の bench /
// rshogi-usi smoke test で検証する想定)。
