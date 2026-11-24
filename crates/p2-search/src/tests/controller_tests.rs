use rshogi_core::search::SearchResult;
use rshogi_core::types::{Move, Value};

use crate::detector::EndgameReason;
use crate::puct::PuctResult;
use crate::types::{SearchBackend, UnifiedSearchResult};

#[test]
fn from_ab_sets_backend_correctly() {
    let ab_result = SearchResult {
        best_move: Move::NONE,
        ponder_move: Move::NONE,
        score: Value::new(100),
        depth: 10,
        nodes: 5000,
        pv: vec![],
        stats_report: String::new(),
    };
    let unified = UnifiedSearchResult::from_ab(ab_result, EndgameReason::ProgressedPosition);
    assert_eq!(unified.backend, SearchBackend::AlphaBeta);
    assert_eq!(unified.endgame_reason, EndgameReason::ProgressedPosition);
    assert_eq!(unified.nodes, 5000);
    assert_eq!(unified.depth, 10);
}

#[test]
fn from_puct_sets_backend_correctly() {
    let puct_result = PuctResult {
        bestmove: Move::NONE,
        value: 0.75,
        nodes: 10000,
        pv: vec![],
    };
    let unified = UnifiedSearchResult::from_puct(puct_result);
    assert_eq!(unified.backend, SearchBackend::Puct);
    assert_eq!(unified.endgame_reason, EndgameReason::None);
    assert_eq!(unified.nodes, 10000);
}

#[test]
fn score_string_cp_format() {
    let ab_result = SearchResult {
        best_move: Move::NONE,
        ponder_move: Move::NONE,
        score: Value::new(90), // 90 / 90 * 100 = 100 cp
        depth: 1,
        nodes: 1,
        pv: vec![],
        stats_report: String::new(),
    };
    let unified = UnifiedSearchResult::from_ab(ab_result, EndgameReason::None);
    let s = unified.score_string();
    assert!(s.starts_with("score cp "), "unexpected: {s}");
}

#[test]
fn score_string_winrate_format() {
    let puct_result = PuctResult {
        bestmove: Move::NONE,
        value: 0.6234,
        nodes: 1,
        pv: vec![],
    };
    let unified = UnifiedSearchResult::from_puct(puct_result);
    let s = unified.score_string();
    assert_eq!(s, "string puct value=0.6234");
}
