use rshogi_core::bitboard::king_effect;
use rshogi_core::position::Position;

use crate::phase::{
    self, ENDGAME_ENTER_THRESHOLD, PhaseTracker, SearchMode, compute_board_material,
    compute_endgame_score, compute_features,
};

/// 駒価値による絶対終盤判定の閾値（盤上 material total）
const ENDGAME_MATERIAL_THRESHOLD: i32 = 2000;

/// king safety で相手利き優勢と判定する閾値
const KING_SAFETY_EFFECT_DIFF: i32 = 5;

/// 終盤判定理由
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndgameReason {
    None,
    LowMaterial,
    /// 進行度スコア (compute_endgame_score) が閾値以上
    ///
    /// 旧 `HighPly` の代替。 駒・玉間距離・mobility・利き圧力などを総合した
    /// 連続値が ENDGAME_ENTER_THRESHOLD (0.65) を超えると発動する。
    ProgressedPosition,
    MateThreat,
    KingSafetyCritical,
}

/// 終盤判定結果
#[derive(Debug, Clone, Copy)]
pub struct EndgameSignal {
    pub use_alpha_beta: bool,
    pub reason: EndgameReason,
    /// compute_endgame_score (0.0..=1.0)。 mate / low material / king safety で
    /// 即決した場合は局所計算を省略するため 1.0 を返す。
    pub score: f32,
}

/// 終盤検出器
#[derive(Debug)]
pub struct EndgameDetector {
    pub material_threshold: i32,
    phase_tracker: PhaseTracker,
}

/// SearchMode → EndgameReason 変換
pub fn search_mode_to_reason(mode: SearchMode) -> EndgameReason {
    match mode {
        SearchMode::Normal => EndgameReason::None,
        SearchMode::Endgame => EndgameReason::LowMaterial,
        SearchMode::MateThreat => EndgameReason::MateThreat,
        SearchMode::ForcedMate => EndgameReason::MateThreat,
    }
}

impl EndgameDetector {
    pub fn new() -> Self {
        Self {
            material_threshold: ENDGAME_MATERIAL_THRESHOLD,
            phase_tracker: PhaseTracker::new(),
        }
    }

    /// 終盤判定
    ///
    /// 判定優先順位:
    /// 1. mate_1ply / 低 material / king safety (絶対終盤、 即決)
    /// 2. compute_endgame_score (自然な進行度判定、 ply に依存しない)
    ///
    /// 旧実装の `ply >= 80` ハードコードは廃止。 60 手で終盤になる急戦も、
    /// 100 手で序中盤的な局面も自然に扱える。
    pub fn evaluate(&self, pos: &mut Position) -> EndgameSignal {
        // 1. Mate threat (絶対終盤)
        if rshogi_core::mate::mate_1ply(pos).is_some() {
            return EndgameSignal {
                use_alpha_beta: true,
                reason: EndgameReason::MateThreat,
                score: 1.0,
            };
        }

        // 2. Material 閾値（極低駒数 = 絶対終盤）
        let board_material = compute_board_material(pos);
        if board_material <= self.material_threshold {
            return EndgameSignal {
                use_alpha_beta: true,
                reason: EndgameReason::LowMaterial,
                score: 1.0,
            };
        }

        // 3. King safety critical
        if self.is_king_safety_critical(pos) {
            return EndgameSignal {
                use_alpha_beta: true,
                reason: EndgameReason::KingSafetyCritical,
                score: 1.0,
            };
        }

        // 4. 連続スコアで自然な進行度判定 (ply 廃止)
        // eval_cp は detector に渡されないため 0 (= eval_magnitude=0 で計算)。
        // material/king_distance/mobility/attack_pressure の重みで十分に判定できる。
        let features = compute_features(pos, 0);
        let score = compute_endgame_score(&features);
        if score >= ENDGAME_ENTER_THRESHOLD {
            return EndgameSignal {
                use_alpha_beta: true,
                reason: EndgameReason::ProgressedPosition,
                score,
            };
        }

        EndgameSignal {
            use_alpha_beta: false,
            reason: EndgameReason::None,
            score,
        }
    }

    /// phase.rs ベースの SearchMode を計算
    pub fn compute_search_mode(&mut self, pos: &Position, eval_cp: i32) -> SearchMode {
        let features = compute_features(pos, eval_cp);
        let score = compute_endgame_score(&features);
        let _phase = phase::determine_phase(score, &mut self.phase_tracker);
        phase::determine_search_mode(score, pos.in_check(), eval_cp, 0.0, 0, 0.0)
    }

    /// PhaseTracker への参照
    pub fn phase_tracker_mut(&mut self) -> &mut PhaseTracker {
        &mut self.phase_tracker
    }

    /// 簡易 king safety 判定
    ///
    /// 手番側の玉の隣接マスにおける相手利き合計 - 自分利き合計 >= 閾値 の場合 true。
    fn is_king_safety_critical(&self, pos: &Position) -> bool {
        let us = pos.side_to_move();
        let them = !us;
        let king_sq = pos.king_square(us);
        let neighbors = king_effect(king_sq);

        let mut effect_diff: i32 = 0;
        for sq in neighbors.iter() {
            effect_diff += pos.board_effect(them, sq) as i32 - pos.board_effect(us, sq) as i32;
        }

        effect_diff >= KING_SAFETY_EFFECT_DIFF
    }
}

impl Default for EndgameDetector {
    fn default() -> Self {
        Self::new()
    }
}
