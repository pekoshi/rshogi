//! Policy-Value NNUE (HalfKAv2_hm + L1=2048 + SwiGLU)
//!
//! PUCT で使う policy + value 推論 network。
//!
//! arch:
//! - 入力: HalfKAv2_hm 73,305 features (rshogi-core 既存 encoding 流用)
//! - FT: L1=2048 (stm + ntm = 4096)
//! - value head: 2*L1 → L2=32 → L3=96 → 1 (screlu)
//! - policy head: SwiGLU = SiLU(W1·x) * (W3·x) → W_o → 2187 logits
//!
//! variants (差は POLICY_HIDDEN と RMSNorm 有無):
//! - `Compact` : PH=1376、 RMSNorm なし
//! - `CompactRms`: PH=1376、 post-SCReLU RMSNorm あり
//! - `Wide`    : PH=2048、 RMSNorm なし
//!
//! ckpt は bullet 由来 raw.bin (f32 column-major)、 詳細 [[arch_reference_v11_v12_v13_p0_20260520]]
//!
//! USI option (rshogi-usi 拡張):
//!   PuctEvalFile    = raw.bin path
//!   PuctEvalVariant = compact | compact_rms | wide
//!
//! phase 切替は既存 `p2_search::phase::PhaseTracker` を流用。
//! 詳細 [[hybrid_usi_v11_integration_plan_20260520]]

pub mod adapter;
pub mod config;
pub mod evaluator;
pub mod loader;
pub mod network;

#[cfg(feature = "pv-cuda")]
pub mod cuda;

pub use adapter::position_to_active_indices;
pub use config::{
    NormalisationConfig, PolicyValueConfig, PolicyValueVariant, infer_variant_from_size,
};
pub use evaluator::PolicyValueEvaluator;

#[cfg(feature = "pv-cuda")]
pub use evaluator::PolicyValueEvaluatorGpu;
pub use loader::{PolicyValueWeights, load_raw_bin};
pub use network::{PolicyValueNet, PolicyValueOutput};

#[cfg(feature = "pv-cuda")]
pub use cuda::{PolicyValueCudaContext, PolicyValueNetCuda};
