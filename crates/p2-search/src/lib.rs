pub mod adapter;
pub mod batch_eval;
pub mod controller;
pub mod detector;
pub mod dfpn;
pub mod eval_cache;
pub mod hybrid;
pub mod mate_probe;
pub mod mate_solver_thread;
pub mod mcts;
pub mod phase;
pub mod puct;
pub mod report;
pub mod tactical;
pub mod tactical_cache;
pub mod types;

#[cfg(test)]
mod tests;

pub use controller::SearchController;
pub use detector::{EndgameDetector, EndgameReason, EndgameSignal};
pub use mcts::{PuctConfig, PuctEngine};
pub use phase::{EndgameFeatures, GamePhase, PhaseTracker, SearchMode};
pub use puct::{PuctBackend, PuctLimits, PuctResult};
pub use report::SearchReport;
pub use types::{SearchBackend, SearchScore, UnifiedSearchResult};
