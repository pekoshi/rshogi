//! Dual head 系 (= dual_psqt + dual_cross) の **真 accumulator (= 完全 delta)** 用の
//! 共通基盤 module。
//!
//! 1 手の do_move による「利き変化」 の影響範囲を 計算する helper を提供する。
//! `collect_dual_*_active_per_head` の per-move full scan を完全に置き換え、
//! 加減算 only の delta 更新を実現する。
//!
//! # 1 手の影響範囲
//!
//! ある手 `M: from → to` (= captured 駒 C もありうる) で利き / king zone が変化する駒の
//! 集合:
//!
//! 1. **moved piece itself** (= from 消滅、 to 出現)
//! 2. **captured piece** (= captured_sq 消滅、 = 相手駒)
//! 3. **influenced sliders** (= from / to / captured_sq の各 sq の ray 上に居る slider、
//!    = 飛車 / 角行 / 香車 / 馬 / 竜)
//! 4. **king moved** (= 玉が動いた場合、 king_zone 全体が shift → full refresh 必須)
//!
//! # 提供する型
//!
//! - [`MoveDelta`]: 1 手の delta source (= moved/captured/sliders)
//!
//! # 提供する関数
//!
//! - [`compute_move_delta`]: do_move 後の `pos` から `MoveDelta` を構築
//! - [`collect_sliders_from_sq_bb`]: 1 sq から ray-walk で影響 slider を bitboard で集約
//!
//! # 数学的観察
//!
//! 1 手で生じる active feature の変化は **常に有限 + 局所的** であり、 board 全体の
//! 再走査は不要。 影響範囲を正確に計算すれば、 各 dual head の active set を加減算
//! のみで更新可能。

use crate::bitboard::{Bitboard, bishop_effect, rook_effect};
use crate::position::Position;
use crate::types::{Color, PieceType, Square};

/// 1 手分の影響 slider 駒 (= 飛 / 角 / 香 / 馬 / 竜) の集合の上限。
///
/// 旧 `InfluencedSliderSet` 配列実装で使用していた上限定数。 Path D 以降は bitboard
/// 直接運用 (= `MoveDelta.influenced_sliders_bb`) に移行し、 配列上限は不要だが、
/// 後方互換 / test reference のため定数を維持する。
pub const MAX_INFLUENCED_SLIDERS: usize = 32;

/// 1 手の delta source (= 完全 delta 計算に必要な全情報)。
#[derive(Clone, Copy, Debug)]
pub struct MoveDelta {
    /// moved piece の from sq (= drop なら None)。
    pub moved_from: Option<Square>,
    /// moved piece の to sq。
    pub moved_to: Square,
    /// moved piece の color (= 手番側 = side_to_move 前の手番)。
    pub moved_color: Color,
    /// moved の **promote 後** piece_type (= 不成なら元と同じ)。
    pub moved_piece_after: PieceType,
    /// moved の **promote 前** piece_type (= 不成なら moved_piece_after と同じ)。
    /// promote 時のみ before != after。
    pub moved_piece_before: PieceType,

    /// captured 駒の (sq, color, piece_type)。 capture なしの手では None。
    /// captured_sq は moved_to と同じ。
    pub captured: Option<(Square, Color, PieceType)>,

    /// 影響を受ける slider 集合 (= from/to/captured_sq の ray-walk で発見)。
    /// Path D: 配列 `InfluencedSliderSet` から bitboard 直接運用に移行。
    /// 各 trigger sq の `rook_effect & (Rook|Dragon|Lance)` + `bishop_effect & (Bishop|Horse)` を
    /// **bitboard OR** で集約 (= push_unique の O(n) 線形検索を排除)。
    ///
    /// **重要**: 移動駒自身が slider の場合、 to_sq が この集合に含まれることがある (= 自己参照、
    /// caller 側で moved piece と重複しないよう除外)。 captured slider は そもそも盤面から
    /// 消えるので含まれない。
    pub influenced_sliders_bb: Bitboard,

    /// king が動いたかどうか (= king_moved なら caller 側で両 perspective full refresh)。
    pub king_moved_any: bool,
}

/// 与えられた `sq` から **rook / bishop ray** で見える slider 駒の bitboard を返す。
///
/// 対象 slider: 飛 / 角 / 香 / 馬 / 竜。
/// - 直線方向 (= rook ray): 飛 / 龍 / 香 のいずれか
/// - 斜方向 (= bishop ray): 角 / 馬 のいずれか
///
/// Path D: 旧 API (= `InfluencedSliderSet` への `push_unique`) は O(n) 線形検索が
/// hot path で 累積 cost になっていたため、 bitboard 直接 OR (= 重複除去は bit OR の
/// 冪等性で自動成立) に変更。
///
/// `occupied`: 現在の occupied bitboard。
/// `pos`: 現局面 (= 駒種判定用)。
#[inline]
pub fn collect_sliders_from_sq_bb(sq: Square, occupied: Bitboard, pos: &Position) -> Bitboard {
    // 直線 (= rook / 龍 / 香、 lance を含む) attacker:
    let rook_attackers = rook_effect(sq, occupied)
        & (pos.pieces_pt(PieceType::Rook)
            | pos.pieces_pt(PieceType::Dragon)
            | pos.pieces_pt(PieceType::Lance));

    // 斜方向 (= 角 / 馬) attacker:
    let bishop_attackers = bishop_effect(sq, occupied)
        & (pos.pieces_pt(PieceType::Bishop) | pos.pieces_pt(PieceType::Horse));

    rook_attackers | bishop_attackers
}

/// 現局面 `pos` (= do_move 後) から `MoveDelta` を構築する。
///
/// `pos.state().last_move` から最後の手を取得し、 そこから:
/// - moved_from / moved_to / moved_color / moved_piece_after / moved_piece_before
/// - captured (= pos.captured_piece() より)
/// - influenced_sliders (= from/to/captured_sq から ray-walk で集める)
/// - king_moved_any (= moved_piece_after が King なら true)
///
/// 戻り値 None: last_move が NULL / PASS / WIN / NONE などの特殊手で delta 計算不可。
/// caller 側で fallback (= full refresh) すること。
pub fn compute_move_delta(pos: &Position) -> Option<MoveDelta> {
    let m = pos.state().last_move;

    if m.is_none() || m.is_pass() || m.is_win() {
        return None;
    }
    if !m.has_piece_info() {
        // 駒情報が encode されていない手 (= 古い move format) では delta 不可
        return None;
    }

    let moved_piece_after_piece = m.moved_piece_after();
    let moved_color = moved_piece_after_piece.color();
    let moved_piece_after = moved_piece_after_piece.piece_type();

    let to_sq = m.to();

    let (moved_from, moved_piece_before) = if m.is_drop() {
        // drop: from なし、 piece_type は drop_piece_type (= promote しない)
        (None, m.drop_piece_type())
    } else {
        // 通常手: from は m.from()、 promote 前は piece_after の demote
        let from = m.from();
        let before = if m.is_promote() {
            // promote 前の piece_type を計算: PieceType ID の bit 3 (= 0x08) を落とす
            // PieceType: Pawn=1, Lance=2, Knight=3, Silver=4, Bishop=5, Rook=6, Gold=7, King=8,
            //            ProPawn=9, ProLance=10, ProKnight=11, ProSilver=12, Horse=13, Dragon=14
            // promote: Pawn→ProPawn (1→9)、 Lance→ProLance (2→10) ... +8 シフト
            let pt_id = moved_piece_after as u8;
            if (9..=14).contains(&pt_id) {
                // SAFETY: pt_id-8 is in 1..=6 (= Pawn..Rook、 valid PieceType)
                unsafe { std::mem::transmute::<u8, PieceType>(pt_id - 8) }
            } else {
                moved_piece_after
            }
        } else {
            moved_piece_after
        };
        (Some(from), before)
    };

    // captured: pos.captured_piece() は do_move 後 state の captured info
    let captured_piece = pos.captured_piece();
    let captured = if captured_piece.is_some() {
        Some((to_sq, captured_piece.color(), captured_piece.piece_type()))
    } else {
        None
    };

    let king_moved_any = moved_piece_after == PieceType::King;

    // Path D: bitboard 直 OR で sliders を集約 (= push_unique の O(n) 線形検索を排除)。
    // 将棋では captured.cap_sq == to_sq が常に成立するため (= en passant なし)、
    // captured 経路の ray walk は to_sq 経路と完全冗長 → skip 可能。
    let occupied = pos.occupied();
    let mut influenced_sliders_bb = Bitboard::EMPTY;
    if let Some(from) = moved_from {
        influenced_sliders_bb |= collect_sliders_from_sq_bb(from, occupied, pos);
    }
    influenced_sliders_bb |= collect_sliders_from_sq_bb(to_sq, occupied, pos);
    if let Some((cap_sq, _, _)) = captured
        && Some(cap_sq) != moved_from
        && cap_sq != to_sq
    {
        // 通常 capture は to_sq と同じで上で処理済み。 想定外の特殊 move のみ追加 walk。
        influenced_sliders_bb |= collect_sliders_from_sq_bb(cap_sq, occupied, pos);
    }

    Some(MoveDelta {
        moved_from,
        moved_to: to_sq,
        moved_color,
        moved_piece_after,
        moved_piece_before,
        captured,
        influenced_sliders_bb,
        king_moved_any,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::position::SFEN_HIRATE;

    fn make_pos() -> Position {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        pos
    }

    /// 開始局面で do_move しないと last_move = NONE → compute_move_delta は None
    #[test]
    fn compute_move_delta_no_move_returns_none() {
        let pos = make_pos();
        let result = compute_move_delta(&pos);
        assert!(result.is_none(), "no move yet, should return None");
    }

    /// 開始局面で 7g7f (= 7七歩→7六) を do_move し、 delta を計算 + 基本 field を check
    #[test]
    fn compute_move_delta_basic_pawn_move() {
        let mut pos = make_pos();
        use crate::movegen::{MoveList, generate_legal};
        let mut list = MoveList::new();
        generate_legal(&pos, &mut list);
        // 7g7f を探す: from = 7g (= File6, Rank6 = idx 60)、 to = 7f (= File6, Rank5 = idx 59)
        let sq_7g = Square::new(crate::types::File::File7, crate::types::Rank::Rank7);
        let sq_7f = Square::new(crate::types::File::File7, crate::types::Rank::Rank6);
        let mut target_mv = None;
        for i in 0..list.len() {
            let mv = list.at(i);
            if !mv.is_drop() && mv.from() == sq_7g && mv.to() == sq_7f {
                target_mv = Some(mv);
                break;
            }
        }
        let mv = target_mv.expect("7g7f should be a legal move");
        let gc = pos.gives_check(mv);
        let _dirty = pos.do_move(mv, gc);

        let delta = compute_move_delta(&pos).expect("should produce delta");
        assert_eq!(delta.moved_from, Some(sq_7g));
        assert_eq!(delta.moved_to, sq_7f);
        assert_eq!(delta.moved_color, Color::Black);
        assert_eq!(delta.moved_piece_after, PieceType::Pawn);
        assert_eq!(delta.moved_piece_before, PieceType::Pawn);
        assert!(delta.captured.is_none());
        assert!(!delta.king_moved_any);
    }
}
