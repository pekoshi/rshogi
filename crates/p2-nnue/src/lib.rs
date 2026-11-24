//! P2 NNUE 評価ネットワーク
//!
//! 2048 次元特徴量を入力とし、`2048 → 512 → 256 → 1` の 3 層ネットワークで
//! evaluation value を返す。

pub mod align;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod evaluator;
pub mod features;
pub mod io;
pub mod layers;
pub mod network;
pub mod network_v3;
pub mod policy_value_net;
pub mod queue;
pub mod types;
pub mod types_v3;

pub use types::ModelConfig;
pub use types_v3::{ModelConfigV3, QuantFmt};

#[cfg(test)]
mod tests;
