use std::sync::Arc;

use p2_nnue::evaluator::NnueEvaluator;
use p2_nnue::features::P2NnueInput;
use p2_nnue::policy_value_net::PolicyValueEvaluator;
#[cfg(feature = "pv-cuda")]
use p2_nnue::policy_value_net::PolicyValueEvaluatorGpu;
use p2_nnue::queue::types::{EvalRequest, EvalResult};
use p2_nnue::queue::{EvalRequestQueue, EvalResultQueue};
use rshogi_core::position::Position;

/// NNUE 評価ブリッジ
///
/// MCTS エンジンが局面評価を取得するための統一インターフェース。
/// - `Direct` / `Queued`: 既存 P2-NNUE (dlshogi features 9728dim) パス
/// - `PolicyValue`: bullet 訓練 NNUE (HalfKAv2_hm + SwiGLU policy)、 Position から直接 forward
pub enum EvalBridge {
    /// 同期的に NnueEvaluator::evaluate を直接呼ぶ
    Direct(Arc<dyn NnueEvaluator + Send + Sync>),
    /// 既存 lock-free queue 経由
    Queued {
        evaluator: Arc<dyn NnueEvaluator + Send + Sync>,
        request_queue: Arc<EvalRequestQueue>,
        result_queue: Arc<EvalResultQueue>,
    },
    /// Policy-Value NNUE (HalfKAv2_hm + SwiGLU)、 Position から直接 forward
    PolicyValue(Arc<PolicyValueEvaluator>),
    /// Policy-Value GPU 推論 (cudarc)、 feature `pv-cuda` で有効
    #[cfg(feature = "pv-cuda")]
    PolicyValueGpu(Arc<PolicyValueEvaluatorGpu>),
}

impl EvalBridge {
    /// 内部の NnueEvaluator への参照を取得 (Direct/Queued モード時のみ有効)
    ///
    /// PolicyValue モードでは panic する。 batch flush 経路は PolicyValue では使われない。
    pub fn evaluator(&self) -> &(dyn NnueEvaluator + Send + Sync) {
        match self {
            EvalBridge::Direct(eval) => eval.as_ref(),
            EvalBridge::Queued { evaluator, .. } => evaluator.as_ref(),
            EvalBridge::PolicyValue(_) => {
                panic!("EvalBridge::evaluator() not supported for PolicyValue mode")
            }
            #[cfg(feature = "pv-cuda")]
            EvalBridge::PolicyValueGpu(_) => {
                panic!("EvalBridge::evaluator() not supported for PolicyValueGpu mode")
            }
        }
    }

    /// 葉ノードの評価値を取得する
    ///
    /// - Direct: evaluator.evaluate(P2NnueInput) を直接呼ぶ
    /// - Queued: request push → inline drain（同期評価）→ result pop
    /// - PolicyValue: Position から直接 forward (input は無視)
    pub fn evaluate_leaf(
        &self,
        node_id: u32,
        input: &P2NnueInput,
        pos: &Position,
        request_id: u64,
    ) -> i32 {
        match self {
            EvalBridge::PolicyValue(evaluator) => evaluator.evaluate_value_cp(pos),
            #[cfg(feature = "pv-cuda")]
            EvalBridge::PolicyValueGpu(evaluator) => evaluator.evaluate_value_cp(pos),
            EvalBridge::Direct(evaluator) => evaluator.evaluate(input),
            EvalBridge::Queued {
                evaluator,
                request_queue,
                result_queue,
            } => {
                // リクエストを push
                // P2NnueInput はコピーする必要がある（queue が所有権を取る）
                let req = EvalRequest {
                    request_id,
                    node_id,
                    nnue_input: P2NnueInput {
                        piece_features: input.piece_features.clone(),
                        side_to_move: input.side_to_move,
                        game_phase: input.game_phase,
                        king_bucket: input.king_bucket,
                    },
                };

                // push（満杯時はインライン drain してリトライ）
                if let Err(p2_nnue::queue::QueueFull(req)) = request_queue.try_push(req) {
                    // fallback: 直接評価
                    return evaluator.evaluate(&req.nnue_input);
                }

                // インライン drain: キューからリクエストを取り出して評価
                while let Ok(pending) = request_queue.try_pop() {
                    let value = evaluator.evaluate(&pending.nnue_input);
                    let result = EvalResult {
                        request_id: pending.request_id,
                        node_id: pending.node_id,
                        value,
                    };
                    let _ = result_queue.try_push(result);
                }

                // 結果を pop（自分の request_id を探す）
                loop {
                    match result_queue.try_pop() {
                        Ok(result) if result.request_id == request_id => return result.value,
                        Ok(result) => {
                            // 他のリクエストの結果 → 戻す
                            let _ = result_queue.try_push(result);
                        }
                        Err(_) => {
                            // 結果キューが空 = まだ処理されていない → 直接評価 fallback
                            return evaluator.evaluate(input);
                        }
                    }
                }
            }
        }
    }
}
