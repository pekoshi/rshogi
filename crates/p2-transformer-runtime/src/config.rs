//! Runtime 設定型。

use p2_transformer::POLICY_SIZE;

/// Matryoshka 切替 hint。 GPU 負荷 / 時間予算に応じて runtime が n_layer を選ぶ。
///
/// 現段階では Full のみ動作 (model ckpt が単一 depth のみ)。 Light / Medium
/// 用の branch は preserve しておき、 別 ckpt が出来次第有効化する。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MatryoshkaLevel {
    /// 浅い (例: n_layer=2)。 Bulk leaf 用の最速 path。 **TODO: 未実装**。
    Light,
    /// 中間 (例: n_layer=4)。 Normal leaf 用。 **TODO: 未実装**。
    Medium,
    /// フル深さ (現行 6 layer)。 PV/Urgent 用。
    #[default]
    Full,
}

/// 評価優先度。 scheduler が backend を選ぶ際の hint。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EvalPriority {
    /// root / PV / 詰み近傍など、 latency 重視 → CPU 即時。
    Urgent,
    /// 通常 leaf。 batch 集約後 hybrid 判定。
    #[default]
    Normal,
    /// Bulk leaf (rollout 等)。 GPU 優先。
    Bulk,
}

/// Hybrid runtime の routing 設定。
#[derive(Clone, Copy, Debug)]
pub struct HybridConfig {
    /// この値以上の batch は GPU、 未満は CPU。
    pub gpu_threshold: usize,
    /// scheduler 側で参照する最大 batch (informational)。
    pub max_batch: usize,
    /// scheduler 側 flush timeout (μs)。 GPU 待ち過ぎ防止。
    pub flush_timeout_us: u64,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            gpu_threshold: 32,
            max_batch: 128,
            flush_timeout_us: 200,
        }
    }
}

/// Policy buffer の固定長 (`POLICY_SIZE` から再 export)。
pub const RUNTIME_POLICY_SIZE: usize = POLICY_SIZE;
