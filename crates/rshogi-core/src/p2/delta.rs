//! Move + context → P2FeatureDelta 差分計算

use crate::position::Position;
use crate::types::{Color, Move, Piece, PieceType};

use super::extract::extract_tail_features;
use super::feature_index::{
    board_piece_index, compute_game_phase, compute_king_bucket, hand_bucket_index,
    king_square_index,
};
use super::feature_types::{P2_SIDE_TO_MOVE_INDEX, P2FeatureDelta, P2NnueInput, P2PieceType};

/// 差分を計算する
///
/// - `pos`: do_move 後の局面
/// - `mv`: 実行された指し手
/// - `captured`: 駒取りがあった場合の駒 (なければ Piece::NONE)
/// - `stm_before`: do_move 前の手番
/// - `prev_input`: do_move 前の P2NnueInput
pub fn compute_delta(
    pos: &Position,
    mv: Move,
    captured: Piece,
    stm_before: Color,
    prev_input: &P2NnueInput,
) -> P2FeatureDelta {
    let mut delta = P2FeatureDelta {
        changes: [(0, 0, 0); 12],
        num_changes: 0,
        recompute_tail: true,
    };

    if mv.is_drop() {
        compute_drop_delta(&mut delta, pos, mv, stm_before, prev_input);
    } else {
        compute_move_delta(&mut delta, pos, mv, captured, stm_before, prev_input);
    }

    // side_to_move は毎手 flip
    let old_stm_val = prev_input.piece_features[P2_SIDE_TO_MOVE_INDEX];
    let new_stm_val = if stm_before == Color::Black { 0 } else { 1 };
    push_change(&mut delta, P2_SIDE_TO_MOVE_INDEX as u16, old_stm_val, new_stm_val);

    delta
}

/// 打ち手の差分
fn compute_drop_delta(
    delta: &mut P2FeatureDelta,
    pos: &Position,
    mv: Move,
    stm_before: Color,
    _prev_input: &P2NnueInput,
) {
    let pt = mv.drop_piece_type();
    let to = mv.to();
    let p2type = P2PieceType::from_piece_type(pt);

    // 盤上に駒が現れる
    let idx = board_piece_index(p2type, stm_before, to);
    push_change(delta, idx as u16, 0, 1);

    // 持ち駒が減る: old bucket → new bucket
    let hand_before = pos.hand(stm_before).count(pt) + 1; // do_move 後なので +1 で復元
    let hand_after = hand_before - 1;
    let old_idx = hand_bucket_index(stm_before, pt, hand_before);
    let new_idx = hand_bucket_index(stm_before, pt, hand_after);
    if old_idx != new_idx {
        push_change(delta, old_idx as u16, 1, 0);
        push_change(delta, new_idx as u16, 0, 1);
    }
}

/// 盤上の手の差分
fn compute_move_delta(
    delta: &mut P2FeatureDelta,
    pos: &Position,
    mv: Move,
    captured: Piece,
    stm_before: Color,
    _prev_input: &P2NnueInput,
) {
    let from = mv.from();
    let to = mv.to();
    let piece_after = mv.moved_piece_after();
    let pt_after = piece_after.piece_type();

    // 移動元の駒種: 成りなら成り前、それ以外はそのまま
    let pt_before = if mv.is_promote() {
        pt_after.unpromote()
    } else {
        pt_after
    };

    let is_king_move = pt_before == PieceType::King;

    if is_king_move {
        // 玉移動: king_square セクション更新
        let old_king_idx = king_square_index(stm_before, from);
        let new_king_idx = king_square_index(stm_before, to);
        push_change(delta, old_king_idx as u16, 1, 0);
        push_change(delta, new_king_idx as u16, 0, 1);
    } else {
        // 通常駒移動: board_piece セクション更新
        let p2_before = P2PieceType::from_piece_type(pt_before);
        let old_idx = board_piece_index(p2_before, stm_before, from);
        push_change(delta, old_idx as u16, 1, 0);

        let p2_after = P2PieceType::from_piece_type(pt_after);
        let new_idx = board_piece_index(p2_after, stm_before, to);
        push_change(delta, new_idx as u16, 0, 1);
    }

    // 駒取り
    if captured.is_some() {
        let cap_pt = captured.piece_type();
        let cap_color = captured.color();

        if cap_pt == PieceType::King {
            // 玉取りは通常起こらないが念のため
            return;
        }

        let p2_cap = P2PieceType::from_piece_type(cap_pt);
        let cap_idx = board_piece_index(p2_cap, cap_color, to);
        push_change(delta, cap_idx as u16, 1, 0);

        // 持ち駒が増える (成り駒は生駒に戻る)
        let hand_pt = cap_pt.unpromote();
        let hand_after_count = pos.hand(stm_before).count(hand_pt);
        let hand_before_count = hand_after_count - 1; // do_move 前は 1 枚少なかった
        let old_hand_idx = hand_bucket_index(stm_before, hand_pt, hand_before_count);
        let new_hand_idx = hand_bucket_index(stm_before, hand_pt, hand_after_count);
        if old_hand_idx != new_hand_idx {
            push_change(delta, old_hand_idx as u16, 1, 0);
            push_change(delta, new_hand_idx as u16, 0, 1);
        }
    }
}

/// 差分を P2NnueInput に適用する
pub fn apply_delta(input: &mut P2NnueInput, delta: &P2FeatureDelta, pos: &Position) {
    // 個別変更を適用
    for i in 0..delta.num_changes as usize {
        let (idx, _old, new_val) = delta.changes[i];
        input.piece_features[idx as usize] = new_val;
    }

    // tail セクションは常に full recompute
    if delta.recompute_tail {
        extract_tail_features(pos, &mut input.piece_features);
    }

    // メタデータ更新
    input.side_to_move = if pos.side_to_move() == Color::Black {
        1
    } else {
        0
    };
    input.game_phase = compute_game_phase(pos);
    input.king_bucket = compute_king_bucket(pos);
}

#[inline]
fn push_change(delta: &mut P2FeatureDelta, idx: u16, old_val: i32, new_val: i32) {
    let n = delta.num_changes as usize;
    debug_assert!(n < 12, "P2FeatureDelta overflow");
    delta.changes[n] = (idx, old_val, new_val);
    delta.num_changes += 1;
}
