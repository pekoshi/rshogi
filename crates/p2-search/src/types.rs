use rshogi_core::types::{Depth, Move, Value};

use crate::detector::EndgameReason;
use crate::puct::PuctResult;

/// 使用されたバックエンド
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchBackend {
    AlphaBeta,
    Puct,
}

/// 探索スコア（バックエンド別）
#[derive(Debug, Clone, Copy)]
pub enum SearchScore {
    /// αβ の centipawn 評価値
    Cp(Value),
    /// PUCT の勝率 [-1.0, 1.0]
    WinRate(f32),
}

/// 統一探索結果
#[derive(Debug)]
pub struct UnifiedSearchResult {
    pub bestmove: Move,
    pub ponder_move: Move,
    pub backend: SearchBackend,
    pub nodes: u64,
    pub score: SearchScore,
    pub pv: Vec<Move>,
    pub depth: Depth,
    pub endgame_reason: EndgameReason,
    pub stats_report: String,
}

impl UnifiedSearchResult {
    /// αβ 結果から変換
    pub fn from_ab(result: rshogi_core::search::SearchResult, reason: EndgameReason) -> Self {
        Self {
            bestmove: result.best_move,
            ponder_move: result.ponder_move,
            backend: SearchBackend::AlphaBeta,
            nodes: result.nodes,
            score: SearchScore::Cp(result.score),
            pv: result.pv,
            depth: result.depth,
            endgame_reason: reason,
            stats_report: result.stats_report,
        }
    }

    /// PUCT 結果から変換
    pub fn from_puct(result: PuctResult) -> Self {
        Self {
            bestmove: result.bestmove,
            ponder_move: Move::NONE,
            backend: SearchBackend::Puct,
            nodes: result.nodes,
            score: SearchScore::WinRate(result.value),
            pv: result.pv,
            depth: 0,
            endgame_reason: EndgameReason::None,
            stats_report: String::new(),
        }
    }

    /// USI 形式のスコア文字列
    pub fn score_string(&self) -> String {
        match self.score {
            SearchScore::Cp(v) => format!("score cp {}", v.to_cp()),
            SearchScore::WinRate(v) => format!("string puct value={v:.4}"),
        }
    }
}
