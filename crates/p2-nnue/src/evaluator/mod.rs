//! NNUE 評価器の抽象化
//!
//! - `NnueEvaluator`: バッチ推論をサポートする trait
//! - `CpuEvaluator`: CPU (スカラー/SIMD) による実装
//! - `CudaEvaluator`: GPU CUDA による実装 (feature = "cuda")

mod cpu;
pub mod cpu_v3;
pub mod router;
pub mod simd_int8;
#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
pub mod gpu;

pub use cpu::CpuEvaluator;
pub use cpu_v3::{CpuEvaluatorV3, P2OutputV3};
pub use router::{HybridRouter, RouterStatsSnapshot};

use crate::features::P2NnueInput;
use crate::types::P2NnueOutput;

/// NNUE 評価器トレイト
///
/// CPU/GPU バックエンドを抽象化し、単一評価とバッチ評価の両方をサポートする。
pub trait NnueEvaluator: Send {
    /// 単一入力の評価
    fn evaluate(&self, input: &P2NnueInput) -> i32;

    /// バッチ評価
    ///
    /// デフォルト実装は `evaluate` を繰り返し呼ぶ。
    /// GPU バックエンドはカーネル起動でバッチ全体を一括処理する。
    fn evaluate_batch(&self, inputs: &[P2NnueInput], outputs: &mut [i32]) {
        assert_eq!(inputs.len(), outputs.len());
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            *output = self.evaluate(input);
        }
    }

    /// value + policy hint + uncertainty を返す完全評価
    ///
    /// デフォルト実装は evaluate() で value のみ返す。
    fn evaluate_full(&self, input: &P2NnueInput) -> P2NnueOutput {
        P2NnueOutput {
            value: self.evaluate(input),
            policy_hint: None,
            uncertainty: None,
        }
    }

    /// バックエンド名
    fn backend_name(&self) -> &'static str;
}
