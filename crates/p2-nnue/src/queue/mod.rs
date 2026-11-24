//! Lock-free MPMC bounded ring buffer queue
//!
//! MCTS 探索スレッド（プロデューサー）と NNUE 評価ワーカー（コンシューマー）間の
//! メッセージパッシングに使用する。

pub mod ring_buffer;
pub mod types;

#[cfg(test)]
mod tests;

pub use ring_buffer::{MpmcQueue, QueueEmpty, QueueFull};
pub use types::{EvalRequest, EvalResult};

/// NNUE 評価リクエスト用キュー
pub type EvalRequestQueue = MpmcQueue<EvalRequest>;
/// NNUE 評価結果用キュー
pub type EvalResultQueue = MpmcQueue<EvalResult>;
