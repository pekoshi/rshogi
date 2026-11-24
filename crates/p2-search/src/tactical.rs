//! Tactical PUCT Stage 2: 戦術的 bonus / risk penalty 計算
//!
//! 通常 PUCT:   score = Q + c_puct * P * sqrt(N) / (1+n)
//! Tactical PUCT: score = Q + (上記 U) + tactical_bonus - risk_penalty
//!
//! tactical_bonus / risk_penalty は move 単位で計算 (Position は move 適用前の局面)。
//!
//! # 計算する特徴
//!
//! tactical_bonus:
//!  - 王手 (gives_check)
//!  - 駒取り (is_capture)
//!  - 成り (is_promotion)
//!  - 駒打ち (is_drop)
//!  - 敵玉周辺攻め (move target が敵玉から距離 ≤ 2)
//!
//! risk_penalty:
//!  - SEE bad (see_ge(threshold=0) が false の capture)
//!  - 自玉危険化 (将来拡張、 現状は SEE のみ)
//!
//! # 重み
//!
//! `TacticalWeights` 構造体に集約。 PuctConfig 経由で外から差し替え可能。

use rshogi_core::position::Position;
use rshogi_core::types::{Move, Value};

/// tactical bonus / risk penalty の重み
#[derive(Debug, Clone)]
pub struct TacticalWeights {
    pub check_bonus: f32,
    pub capture_bonus: f32,
    pub promotion_bonus: f32,
    pub drop_bonus: f32,
    pub king_attack_bonus: f32,
    pub see_bad_penalty: f32,
}

impl Default for TacticalWeights {
    fn default() -> Self {
        Self {
            check_bonus: 0.10,
            capture_bonus: 0.06,
            promotion_bonus: 0.05,
            drop_bonus: 0.03,
            king_attack_bonus: 0.05,
            see_bad_penalty: 0.20,
        }
    }
}

/// 1 手あたりの tactical bonus を計算 (PUCT score に加算)
///
/// `pos`: move 適用前の局面
/// `mv`: 候補手
/// `gives_check`: 既に判定済みなら渡す (None なら pos.gives_check で計算)
pub fn tactical_bonus(
    pos: &Position,
    mv: Move,
    weights: &TacticalWeights,
    gives_check: Option<bool>,
) -> f32 {
    if mv == Move::NONE || mv.is_pass() {
        return 0.0;
    }

    let mut bonus = 0.0_f32;

    // 王手
    let check = gives_check.unwrap_or_else(|| pos.gives_check(mv));
    if check {
        bonus += weights.check_bonus;
    }

    // 駒取り
    if pos.is_capture(mv) {
        bonus += weights.capture_bonus;
    }

    // 成り
    if mv.is_promotion() {
        bonus += weights.promotion_bonus;
    }

    // 駒打ち
    if mv.is_drop() {
        bonus += weights.drop_bonus;
    }

    // 敵玉周辺攻め: move target が敵玉から距離 ≤ 2 (Chebyshev)
    let to = mv.to();
    let opp = !pos.side_to_move();
    let opp_king = pos.king_square(opp);
    let file_diff = (to.file() as i32 - opp_king.file() as i32).abs();
    let rank_diff = (to.rank() as i32 - opp_king.rank() as i32).abs();
    let chebyshev = file_diff.max(rank_diff);
    if chebyshev <= 2 {
        bonus += weights.king_attack_bonus;
    }

    bonus
}

/// 1 手あたりの risk penalty を計算 (PUCT score から減算)
///
/// `pos`: move 適用前の局面
/// `mv`: 候補手
pub fn risk_penalty(pos: &Position, mv: Move, weights: &TacticalWeights) -> f32 {
    if mv == Move::NONE || mv.is_pass() {
        return 0.0;
    }

    let mut penalty = 0.0_f32;

    // SEE bad: capture (or drop) が SEE で負の場合 → ただ捨て候補
    if pos.is_capture(mv) && !pos.see_ge(mv, Value::ZERO) {
        penalty += weights.see_bad_penalty;
    }

    penalty
}

/// PUCT score への合算 (= tactical_bonus - risk_penalty)
#[inline]
pub fn tactical_delta(
    pos: &Position,
    mv: Move,
    weights: &TacticalWeights,
    gives_check: Option<bool>,
) -> f32 {
    tactical_bonus(pos, mv, weights, gives_check) - risk_penalty(pos, mv, weights)
}

// ────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hirate() -> Position {
        let mut p = Position::new();
        p.set_hirate();
        p
    }

    #[test]
    fn tactical_weights_default_sane() {
        let w = TacticalWeights::default();
        assert!(w.check_bonus > 0.0);
        assert!(w.capture_bonus > 0.0);
        assert!(w.promotion_bonus > 0.0);
        assert!(w.see_bad_penalty > 0.0);
    }

    #[test]
    fn pass_move_gives_zero_delta() {
        let pos = hirate();
        let weights = TacticalWeights::default();
        let d = tactical_delta(&pos, Move::NONE, &weights, None);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn hirate_legal_moves_no_capture_no_check() {
        // 平手初期局面の合法手は全て quiet (capture/check なし)
        // tactical_bonus は 0 か king_attack_bonus 程度のみ
        use rshogi_core::movegen::{MoveList, generate_legal};
        let pos = hirate();
        let mut moves = MoveList::new();
        generate_legal(&pos, &mut moves);
        let weights = TacticalWeights::default();
        for i in 0..moves.len() {
            let mv = moves.at(i);
            let d = tactical_delta(&pos, mv, &weights, None);
            // bonus は 0 〜 king_attack_bonus 範囲、 penalty は 0
            assert!(d >= 0.0);
            assert!(d <= weights.king_attack_bonus + 1e-6);
        }
    }
}
