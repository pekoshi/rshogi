use rshogi_core::position::Position;
use rshogi_core::types::Move;

use crate::phase::SearchMode;

/// PUCT 探索の制限
#[derive(Debug, Clone)]
pub struct PuctLimits {
    pub max_nodes: u64,
    pub max_time_ms: u64,
    /// 探索モード（終盤判定結果）
    pub search_mode: SearchMode,
    /// 既存 phase 判定結果 (hybrid 発動条件用、 None なら hybrid 経由で判定スキップ)
    pub game_phase: Option<crate::phase::GamePhase>,
    /// 既存 compute_endgame_score の結果 (hybrid 発動条件用、 default 0.0)
    pub endgame_score: f32,
}

impl Default for PuctLimits {
    fn default() -> Self {
        Self {
            max_nodes: 0,
            max_time_ms: 0,
            search_mode: SearchMode::Normal,
            game_phase: None,
            endgame_score: 0.0,
        }
    }
}

/// PUCT 探索結果
#[derive(Debug, Clone)]
pub struct PuctResult {
    pub bestmove: Move,
    pub value: f32,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

/// PUCT バックエンド trait
///
/// PUCT/MCTS 探索の実装が完了したらこの trait を実装する。
pub trait PuctBackend: Send {
    /// PUCT 探索を実行
    fn search(&mut self, pos: &Position, limits: &PuctLimits) -> PuctResult;
    /// 探索を中断
    fn stop(&self);
}
