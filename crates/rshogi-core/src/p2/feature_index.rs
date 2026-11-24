//! P2 特徴量の index 計算関数

use crate::position::Position;
use crate::types::{Color, PieceType, Square};

use super::feature_types::{
    P2_BLACK_KING_START, P2_BOARD_PIECE_START, P2_HAND_BUCKET_START, P2_WHITE_KING_START,
    P2PieceType,
};

/// 盤上駒 index: p2type * 2 * 81 + color.index() * 81 + sq.index()
#[inline]
pub const fn board_piece_index(p2type: P2PieceType, color: Color, sq: Square) -> usize {
    P2_BOARD_PIECE_START + p2type.index() * 2 * 81 + color.index() * 81 + sq.index()
}

/// 玉位置 index (one-hot 81)
#[inline]
pub const fn king_square_index(color: Color, sq: Square) -> usize {
    match color.index() {
        0 => P2_BLACK_KING_START + sq.index(),
        _ => P2_WHITE_KING_START + sq.index(),
    }
}

// ─── 持ち駒バケット ───
// 各駒種のバケット数
const HAND_PAWN_BUCKETS: u32 = 14; // 0..12 exact, 13+ grouped
const HAND_LANCE_BUCKETS: u32 = 5;
const HAND_KNIGHT_BUCKETS: u32 = 5;
const HAND_SILVER_BUCKETS: u32 = 5;
const HAND_GOLD_BUCKETS: u32 = 5;
const HAND_BISHOP_BUCKETS: u32 = 3;
const HAND_ROOK_BUCKETS: u32 = 3;

/// 片側あたりの持ち駒バケット数
const HAND_BUCKETS_PER_SIDE: u32 = HAND_PAWN_BUCKETS
    + HAND_LANCE_BUCKETS
    + HAND_KNIGHT_BUCKETS
    + HAND_SILVER_BUCKETS
    + HAND_GOLD_BUCKETS
    + HAND_BISHOP_BUCKETS
    + HAND_ROOK_BUCKETS;

const _: () = assert!(HAND_BUCKETS_PER_SIDE == 40);

/// 駒種ごとのバケット内オフセットと最大 bucket index を返す
#[inline]
const fn hand_piece_offset_and_max(pt: PieceType) -> (u32, u32) {
    match pt.index() {
        1 => (0, HAND_PAWN_BUCKETS - 1),    // Pawn: offset 0, max 13
        2 => (14, HAND_LANCE_BUCKETS - 1),  // Lance: offset 14, max 4
        3 => (19, HAND_KNIGHT_BUCKETS - 1), // Knight: offset 19, max 4
        4 => (24, HAND_SILVER_BUCKETS - 1), // Silver: offset 24, max 4
        7 => (29, HAND_GOLD_BUCKETS - 1),   // Gold: offset 29, max 4
        5 => (34, HAND_BISHOP_BUCKETS - 1), // Bishop: offset 34, max 2
        6 => (37, HAND_ROOK_BUCKETS - 1),   // Rook: offset 37, max 2
        _ => unreachable!(),
    }
}

/// 持ち駒バケット index
///
/// count は実際の枚数。bucket に clamp して one-hot 化する。
#[inline]
pub fn hand_bucket_index(color: Color, pt: PieceType, count: u32) -> usize {
    let (offset, max_bucket) = hand_piece_offset_and_max(pt);
    let bucket = if count > max_bucket {
        max_bucket
    } else {
        count
    };
    let side_offset = color.index() as u32 * HAND_BUCKETS_PER_SIDE;
    P2_HAND_BUCKET_START + (side_offset + offset + bucket) as usize
}

/// 玉バケット (0..14): STM 視点の玉位置を 15 バケットに分類
///
/// 1. STM が White なら玉位置を inverse (180° 回転)
/// 2. mirror_file = min(file, 8 - file)  → 0..4
/// 3. rank_group = rank / 3              → 0..2
/// 4. bucket = mirror_file * 3 + rank_group
#[inline]
pub fn compute_king_bucket(pos: &Position) -> u8 {
    let stm = pos.side_to_move();
    let king_sq = pos.king_square(stm);
    let sq = match stm {
        Color::White => king_sq.inverse(),
        _ => king_sq,
    };
    let file = sq.file().index();
    let rank = sq.rank().index();
    let mirror_file = file.min(8 - file);
    let rank_group = rank / 3;
    (mirror_file * 3 + rank_group) as u8
}

/// ゲームフェーズ (0=endgame, 255=opening)
///
/// 盤上の非王駒数ベース。
#[inline]
pub fn compute_game_phase(pos: &Position) -> u8 {
    let total = pos.occupied().count();
    // 玉 2 枚を除く
    let non_king = total.saturating_sub(2);
    // 38 = 開始局面の非王駒数 (max)
    ((non_king * 255 / 38).min(255)) as u8
}
