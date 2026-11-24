use rshogi_core::position::Position;

use crate::phase::{
    GamePhase, PhaseTracker, SearchMode, compute_endgame_score, compute_features, determine_phase,
    determine_search_mode, progress_score, score_to_phase_immediate,
};

fn pos_from_sfen(sfen: &str) -> Position {
    let mut pos = Position::new();
    pos.set_sfen(sfen).unwrap();
    pos
}

#[test]
fn score_range_0_1() {
    let pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let features = compute_features(&pos, 0);
    let score = compute_endgame_score(&features);
    assert!(score >= 0.0, "score={score} < 0.0");
    assert!(score <= 1.0, "score={score} > 1.0");
}

#[test]
fn score_range_endgame() {
    // 王と金のみの終盤局面
    let pos = pos_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2G2g 1");
    let features = compute_features(&pos, 2500);
    let score = compute_endgame_score(&features);
    assert!(score >= 0.0, "score={score} < 0.0");
    assert!(score <= 1.0, "score={score} > 1.0");
    // 終盤局面なので score は高めになるはず
    assert!(score > 0.3, "終盤局面のスコアが低すぎる: {score}");
}

#[test]
fn phase_hysteresis_enter() {
    let mut tracker = PhaseTracker::new();
    // 0.65 以上で Endgame に遷移
    let phase = determine_phase(0.70, &mut tracker);
    assert_eq!(phase, GamePhase::Endgame);
}

#[test]
fn phase_hysteresis_exit() {
    let mut tracker = PhaseTracker::new();
    // Endgame に入れてから
    determine_phase(0.70, &mut tracker);
    assert_eq!(tracker.current_phase(), GamePhase::Endgame);

    // 0.45 未満で Middlegame に戻る
    let phase = determine_phase(0.40, &mut tracker);
    assert_eq!(phase, GamePhase::Middlegame);
}

#[test]
fn phase_hysteresis_hold() {
    let mut tracker = PhaseTracker::new();
    // Endgame に入れてから
    determine_phase(0.70, &mut tracker);

    // 0.45-0.65 の間では Endgame を維持
    let phase = determine_phase(0.50, &mut tracker);
    assert_eq!(phase, GamePhase::Endgame);
}

#[test]
fn search_mode_normal() {
    let mode = determine_search_mode(0.30, false, 100, 0.0, 0, 0.0);
    assert_eq!(mode, SearchMode::Normal);
}

#[test]
fn search_mode_endgame() {
    let mode = determine_search_mode(0.70, false, 100, 0.0, 0, 0.0);
    assert_eq!(mode, SearchMode::Endgame);
}

#[test]
fn search_mode_mate_threat() {
    let mode = determine_search_mode(0.70, false, 100, 0.4, 3, 0.0);
    assert_eq!(mode, SearchMode::MateThreat);
}

#[test]
fn search_mode_forced_mate() {
    let mode = determine_search_mode(0.85, true, 2500, 0.0, 0, 0.0);
    assert_eq!(mode, SearchMode::ForcedMate);
}

#[test]
fn features_hirate() {
    let pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let features = compute_features(&pos, 0);

    // 序盤: material_phase は低い（駒が多い = 終盤度低い）
    assert!(
        features.material_phase < 0.3,
        "平手の material_phase が高すぎる: {}",
        features.material_phase
    );

    // eval_magnitude は 0（評価値 0）
    assert!(
        features.eval_magnitude < 0.01,
        "eval_magnitude が 0 でない: {}",
        features.eval_magnitude
    );
}

#[test]
fn features_endgame() {
    // 王と金のみ
    let pos = pos_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2G2g 1");
    let features = compute_features(&pos, 1000);

    // 終盤: material_phase は高い（駒が少ない = 終盤度高い）
    assert!(
        features.material_phase > 0.7,
        "終盤の material_phase が低すぎる: {}",
        features.material_phase
    );

    // eval_magnitude はそれなりに高い
    assert!(
        features.eval_magnitude > 0.2,
        "eval_magnitude が低い: {}",
        features.eval_magnitude
    );
}

// ===== 進行度精度向上検証 (Opening / engagement / promotion) =====

#[test]
fn features_hirate_engagement_zero() {
    let pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let features = compute_features(&pos, 0);
    assert!(
        features.engagement < 0.01,
        "平手で engagement が 0 でない: {}",
        features.engagement
    );
    assert!(
        features.promotion_progress < 0.01,
        "平手で promotion が 0 でない: {}",
        features.promotion_progress
    );
}

#[test]
fn features_engagement_normalized() {
    // 王のみ + 互いに 3 駒ずつ持駒 (盤上駒減で 18 駒制約クリア)
    let pos = pos_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b 3P3p 1");
    let features = compute_features(&pos, 0);
    // 6 駒 → engagement = 6/10 = 0.6
    assert!(
        (features.engagement - 0.6).abs() < 0.05,
        "engagement 計算ずれ: {}",
        features.engagement
    );
}

#[test]
fn phase_immediate_opening() {
    assert_eq!(score_to_phase_immediate(0.05), GamePhase::Opening);
    assert_eq!(score_to_phase_immediate(0.15), GamePhase::Opening);
}

#[test]
fn phase_immediate_middlegame() {
    assert_eq!(score_to_phase_immediate(0.30), GamePhase::Middlegame);
    assert_eq!(score_to_phase_immediate(0.50), GamePhase::Middlegame);
}

#[test]
fn phase_immediate_endgame() {
    assert_eq!(score_to_phase_immediate(0.70), GamePhase::Endgame);
    assert_eq!(score_to_phase_immediate(0.90), GamePhase::Endgame);
}

#[test]
fn determine_phase_opening_to_middlegame() {
    let mut tracker = PhaseTracker::new();
    assert_eq!(tracker.current_phase(), GamePhase::Opening);
    // 0.20 で Middlegame 入り
    determine_phase(0.25, &mut tracker);
    assert_eq!(tracker.current_phase(), GamePhase::Middlegame);
}

#[test]
fn determine_phase_middlegame_back_to_opening() {
    let mut tracker = PhaseTracker::new();
    determine_phase(0.30, &mut tracker);
    assert_eq!(tracker.current_phase(), GamePhase::Middlegame);
    // 0.15 未満で Opening に戻る
    determine_phase(0.10, &mut tracker);
    assert_eq!(tracker.current_phase(), GamePhase::Opening);
}

#[test]
fn determine_phase_progression_exposed() {
    let mut tracker = PhaseTracker::new();
    determine_phase(0.42, &mut tracker);
    assert!((tracker.progression() - 0.42).abs() < 1e-6);
}

#[test]
fn progress_score_consistency() {
    let pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let features = compute_features(&pos, 0);
    let score_a = progress_score(&features);
    let score_b = compute_endgame_score(&features);
    assert!(
        (score_a - score_b).abs() < 1e-6,
        "progress_score と compute_endgame_score 不一致"
    );
}
