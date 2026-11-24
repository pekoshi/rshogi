//! # p2-transformer-runtime
//!
//! 序中盤 Transformer + PUCT 用の hybrid inference runtime。
//!
//! ## 設計理念
//!
//! - **PUCT の本質は単局面速度ではなく leaf throughput**。 batch wait latency
//!   を最小化しつつ batch を集約して並列実行する。
//! - 小 batch (1〜数局面) → CPU 即時 (low latency)
//! - 大 batch (>= [`HybridConfig::gpu_threshold`]) → GPU (batch throughput)
//!
//! ## 層構造
//!
//! ```text
//!  MCTS scheduler
//!     │  TransformerTokens
//!     ▼
//!  TransformerRuntime (trait)
//!     ├─ CpuTransformerRuntime  (candle CPU、 単局面最速)
//!     ├─ CudaTransformerRuntime (candle CUDA、 persistent Tensor、 cfg(cuda))
//!     └─ HybridTransformerRuntime { cpu, [gpu] } ── routing ──┐
//!                                                              │ batch size /
//!                                                              │ priority
//!                                                              ▼
//!                                                       backend 選択
//! ```
//!
//! ## 主要 trait / struct
//!
//! - [`TransformerRuntime`]: backend 共通 trait (`eval`, `eval_with_level`)
//! - [`CpuTransformerRuntime`]: candle CPU backend
//! - [`CudaTransformerRuntime`]: candle CUDA backend (`feature = "cuda"`)
//! - [`HybridTransformerRuntime`]: routing wrapper
//! - [`HybridConfig`]: routing 設定
//! - [`MatryoshkaLevel`]: depth 切替 hint (Light / Medium / Full)
//! - [`EvalPriority`][]: 評価優先度 (Urgent / Normal / Bulk)
//! - [`PolicyBuffer`]: policy logits 用 reusable buffer
//! - [`InferenceError`]: backend 共通エラー型
//!
//! ## 既知制約
//!
//! - 現段階では Matryoshka は Full のみ動作。 Light / Medium は branch だけ
//!   用意し、 ckpt が出来次第 layer 数を切替える (`runtime.rs` の TODO 参照)。
//! - GPU backend は default features では build されない (Windows + CUDA 12.6
//!   の NVCC issue 回避)。 `--features cuda` で opt-in。
//! - fp16 / int8 quantization、 pinned memory、 TensorRT export は将来拡張。
//!
//! 詳細設計は `docs/transformer_inference_runtime_design.md` を参照。

pub mod backend;
pub mod buffer;
pub mod config;
pub mod error;
pub mod hybrid;
pub mod model;
pub mod runtime;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx2_forward;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use backend::TransformerRuntime;
pub use buffer::PolicyBuffer;
pub use config::{EvalPriority, HybridConfig, MatryoshkaLevel};
pub use error::InferenceError;
pub use hybrid::HybridTransformerRuntime;
pub use model::{ModelHandle, RuntimeModelConfig};
pub use runtime::CpuTransformerRuntime;

#[cfg(feature = "cuda")]
pub use cuda::{CudaBuffers, CudaTransformerRuntime};
