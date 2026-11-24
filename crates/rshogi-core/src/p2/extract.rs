//! Position → P2NnueInput 特徴量抽出（純粋関数）

use crate::position::Position;
use crate::types::{Color, PieceType, Square};

use super::feature_index::{
    board_piece_index, compute_game_phase, compute_king_bucket, hand_bucket_index,
    king_square_index,
};
use super::feature_types::{
    P2_FEATURE_DIM, P2_PHASE_ATTACK_START, P2_SIDE_TO_MOVE_INDEX, P2NnueInput, P2PieceType,
};

/// Position から 2048 次元の特徴量を抽出する
///
/// 純粋関数。副作用なし、ヒープ割り当てなし。
pub fn extract_features(pos: &Position) -> P2NnueInput {
    let mut features = [0i32; P2_FEATURE_DIM];

    // ─── board_piece [0..1781] ───
    for sq_idx in 0..Square::NUM {
        // SAFETY: 0..81 は有効な Square 範囲
        let sq = unsafe { Square::from_u8_unchecked(sq_idx as u8) };
        let piece = pos.piece_on(sq);
        if piece.is_none() {
            continue;
        }
        let pt = piece.piece_type();
        if pt == PieceType::King {
            continue;
        }
        let p2type = P2PieceType::from_piece_type(pt);
        let color = piece.color();
        let idx = board_piece_index(p2type, color, sq);
        features[idx] = 1;
    }

    // ─── king_square [1782..1943] ───
    for &color in &[Color::Black, Color::White] {
        let king_sq = pos.king_square(color);
        let idx = king_square_index(color, king_sq);
        features[idx] = 1;
    }

    // ─── side_to_move [1944] ───
    let stm = pos.side_to_move();
    features[P2_SIDE_TO_MOVE_INDEX] = if stm == Color::Black { 1 } else { 0 };

    // ─── hand_bucket [1945..2024] ───
    for &color in &[Color::Black, Color::White] {
        let hand = pos.hand(color);
        for &pt in &PieceType::HAND_PIECES {
            let count = hand.count(pt);
            let idx = hand_bucket_index(color, pt, count);
            features[idx] = 1;
        }
    }

    // ─── phase/attack/reserved [2025..2047] ───
    extract_tail_features(pos, &mut features);

    P2NnueInput {
        piece_features: features,
        side_to_move: if stm == Color::Black { 1 } else { 0 },
        game_phase: compute_game_phase(pos),
        king_bucket: compute_king_bucket(pos),
    }
}

/// [2025..2047] セクションを書き込む（差分更新時にも呼ばれる）
pub(crate) fn extract_tail_features(pos: &Position, features: &mut [i32; P2_FEATURE_DIM]) {
    let base = P2_PHASE_ATTACK_START;

    // [2025] in_check
    features[base] = if pos.in_check() { 1 } else { 0 };

    // [2026] game_ply (clamped 0..512)
    features[base + 1] = pos.game_ply().clamp(0, 512);

    // [2027] board_pieces (non-king pieces on board)
    let non_king = pos.occupied().count().saturating_sub(2);
    features[base + 2] = non_king as i32;

    // [2028] my_promo_zone: STM の駒が相手の promotion zone にある数
    let stm = pos.side_to_move();
    features[base + 3] = count_pieces_in_promo_zone(pos, stm) as i32;

    // [2029] opp_promo_zone: 相手の駒が STM の promotion zone にある数
    features[base + 4] = count_pieces_in_promo_zone(pos, stm.opponent()) as i32;

    // [2030..2047] reserved (18 slots, always 0)
    for i in 5..23 {
        features[base + i] = 0;
    }
}

/// 指定 color の駒が相手の promotion zone にある枚数（玉除く）
fn count_pieces_in_promo_zone(pos: &Position, color: Color) -> u32 {
    let my_pieces = pos.pieces_c(color);
    let king_bb = pos.pieces(color, PieceType::King);
    let non_king = my_pieces & !king_bb;

    // 相手の promotion zone = 自分から見た敵陣
    // Black の敵陣 = Rank1-3, White の敵陣 = Rank7-9
    let mut count = 0u32;
    for sq in non_king.iter() {
        let rank = sq.rank();
        if rank.can_promote(color) {
            count += 1;
        }
    }
    count
}
