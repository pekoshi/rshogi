//! PolicyValueNet を NnueEvaluator trait に統合する wrapper
//!
//! 既存 PUCT engine は `NnueEvaluator::evaluate(&P2NnueInput) -> i32` を呼ぶが、
//! PolicyValueNet は HalfKAv2_hm active indices を必要とする。
//!
//! 解決: P2NnueInput を「Position cache」 として使い、 PolicyValueEvaluator が
//! 内部で Position から active indices を取り直す代わりに、 PUCT 側で
//! 別 API (Position 直接渡し) を呼ぶ形を提供する。
//!
//! v2 統合 (PUCT engine 側修正): `evaluate_from_position(&Position) -> PolicyValueOutput`
//! を使う直接 path を提供。

use std::sync::Arc;

use rshogi_core::position::Position;

use crate::evaluator::NnueEvaluator;
use crate::features::P2NnueInput;
use crate::types::P2NnueOutput;

use super::adapter::position_to_active_indices;
use super::network::{PolicyValueNet, PolicyValueOutput};

/// PolicyValueNet を NnueEvaluator trait に薄く wrap する evaluator
///
/// 注意: 既存 trait の `evaluate(&P2NnueInput)` 経路は使えない (HalfKAv2_hm input が違うため)。
/// PUCT engine 側を修正して `evaluate_from_position(&Position)` を呼ぶ形に統合する必要あり。
///
/// 現状は **Position を直接受ける wrapper** として動作。
pub struct PolicyValueEvaluator {
    net: Arc<PolicyValueNet>,
}

impl PolicyValueEvaluator {
    pub fn new(net: PolicyValueNet) -> Self {
        Self { net: Arc::new(net) }
    }

    pub fn from_arc(net: Arc<PolicyValueNet>) -> Self {
        Self { net }
    }

    /// Position から forward (PUCT engine から直接呼ぶ用 API)
    pub fn evaluate_from_position(&self, pos: &Position) -> PolicyValueOutput {
        let (stm_idx, nstm_idx) = position_to_active_indices(pos);
        self.net.forward(&stm_idx, &nstm_idx)
    }

    /// PUCT engine が期待する i32 winrate-scaled value (VALUE_SCALE=32768)
    ///
    /// PUCT は `value_to_winrate(raw) = (raw / 32768).clamp(-1.0, 1.0)` で読む。
    /// value_raw (logit) → sigmoid → [-1,1] centered → 32768 倍。
    pub fn evaluate_value_cp(&self, pos: &Position) -> i32 {
        let out = self.evaluate_from_position(pos);
        let sig = 1.0 / (1.0 + (-out.value_raw).exp());
        let centered = (sig - 0.5) * 2.0; // -1..1
        (centered * 32768.0) as i32
    }
}

/// NnueEvaluator trait 実装 (互換性のため、 実用上は evaluate_from_position を使う)
///
/// 現状 P2NnueInput には Position が含まれていないため、 この経路は **panic** する。
/// PUCT engine 側で evaluate_from_position を直接呼ぶよう修正が必要。
impl NnueEvaluator for PolicyValueEvaluator {
    fn evaluate(&self, _input: &P2NnueInput) -> i32 {
        unimplemented!("PolicyValueEvaluator requires Position; use evaluate_from_position instead")
    }

    fn evaluate_full(&self, _input: &P2NnueInput) -> P2NnueOutput {
        unimplemented!("PolicyValueEvaluator requires Position; use evaluate_from_position instead")
    }

    fn backend_name(&self) -> &'static str {
        "policy_value_net_cpu_f32"
    }
}

/// GPU 推論 evaluator (feature `pv-cuda`)
#[cfg(feature = "pv-cuda")]
pub struct PolicyValueEvaluatorGpu {
    net: std::sync::Mutex<super::cuda::PolicyValueNetCuda>,
}

#[cfg(feature = "pv-cuda")]
impl PolicyValueEvaluatorGpu {
    pub fn new(net: super::cuda::PolicyValueNetCuda) -> Self {
        Self {
            net: std::sync::Mutex::new(net),
        }
    }

    /// Position から forward + value_raw → i32 winrate scale
    pub fn evaluate_value_cp(&self, pos: &rshogi_core::position::Position) -> i32 {
        let (stm, nstm) = position_to_active_indices(pos);
        let out = self.net.lock().unwrap().forward(&stm, &nstm).expect("gpu forward");
        let sig = 1.0 / (1.0 + (-out.value_raw).exp());
        let centered = (sig - 0.5) * 2.0;
        (centered * 32768.0) as i32
    }
}
