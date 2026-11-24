//! CPU ベースの NNUE 評価器

use std::sync::Arc;

use super::NnueEvaluator;
use crate::features::P2NnueInput;
use crate::network::P2Network;
use crate::types::P2NnueOutput;

/// CPU (スカラー/SIMD) による NNUE 評価器
///
/// `P2Network::forward()` をラップする。
/// AVX2 利用可能時は自動的に SIMD 最適化パスが使われる。
pub struct CpuEvaluator {
    network: Arc<P2Network>,
}

impl CpuEvaluator {
    /// 共有ネットワークから評価器を作成
    pub fn new(network: Arc<P2Network>) -> Self {
        Self { network }
    }
}

impl NnueEvaluator for CpuEvaluator {
    fn evaluate(&self, input: &P2NnueInput) -> i32 {
        self.network.forward_value_only(input)
    }

    fn evaluate_full(&self, input: &P2NnueInput) -> P2NnueOutput {
        self.network.forward(input)
    }

    fn backend_name(&self) -> &'static str {
        if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
            "cpu-avx2"
        } else {
            "cpu-scalar"
        }
    }
}
