//! 玉相対座標 → 289 bucket 変換 (branch-less)。
//!
//! ```text
//! dx = file(piece_sq) - file(king_sq)
//! dy = rank(piece_sq) - rank(king_sq)
//! clamp(dx, -8, 8)
//! clamp(dy, -8, 8)
//! index = (dx + 8) * 17 + (dy + 8)   // 0..289
//! ```
//!
//! file / rank はそれぞれ 0..9 なので、 dx と dy は -8..=+8 の範囲に必ず収まる。
//! ただし将来的に他用途 (e.g. 持ち駒の擬似 sq) で範囲外値が渡される可能性を考慮し、
//! clamp は明示的に行う (branch-free min/max)。
//!
//! 玉の自身位置 (dx=dy=0) は index = 8*17 + 8 = 144。

use rshogi_core::types::Square;

use super::vocab::KING_REL_BUCKET_COUNT;

/// 玉相対 index を返す (0..289)。
///
/// branch-less (条件分岐は debug_assert のみ、 release では消える)。
#[inline]
pub fn king_relative_index_clamped(piece_sq: Square, king_sq: Square) -> u16 {
    // Square::index() = file * 9 + rank なので、 file = idx / 9、 rank = idx % 9。
    let p = piece_sq.index() as i32;
    let k = king_sq.index() as i32;
    let p_file = p / 9;
    let p_rank = p % 9;
    let k_file = k / 9;
    let k_rank = k % 9;
    let dx = p_file - k_file; // -8..=+8
    let dy = p_rank - k_rank; // -8..=+8

    // -8..=8 → 0..=16 (branch-free clamp + shift)
    let dx_c = clamp_to_17(dx);
    let dy_c = clamp_to_17(dy);
    let idx = dx_c * 17 + dy_c;
    debug_assert!((idx as u16) < KING_REL_BUCKET_COUNT);
    idx as u16
}

/// -8..=8 → 0..=16 に branch-less で写像する (範囲外は clamp)。
#[inline]
const fn clamp_to_17(v: i32) -> i32 {
    let lo = if v < -8 { -8 } else { v };
    let hi = if lo > 8 { 8 } else { lo };
    hi + 8
}

/// 自陣 (Black 視点で rank 7-9 = idx 6-8、 White 視点で rank 1-3 = idx 0-2) を 0、
/// 中段 (rank 4-6 = idx 3-5) を 1、 敵陣を 2 とする。
///
/// `color_idx` は 0=Black、 1=White。 White の場合は rank を反転する。
#[inline]
pub fn promotion_zone(sq: Square, color_idx: u8) -> u8 {
    let rank = (sq.index() % 9) as i32; // 0..9 (0 = 1段目、 8 = 9段目)
    // Black 視点で「自陣」は rank 6-8 (idx 7-9段)、 「敵陣」は rank 0-2 (1-3段)。
    // White 視点はその逆。
    let view_rank = if color_idx == 0 { rank } else { 8 - rank };
    if view_rank >= 6 {
        0 // 自陣
    } else if view_rank >= 3 {
        1 // 中段
    } else {
        2 // 敵陣
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::types::{File, Rank};

    #[test]
    fn king_relative_self_position() {
        let king = Square::new(File::File5, Rank::Rank9);
        let idx = king_relative_index_clamped(king, king);
        assert_eq!(idx, 8 * 17 + 8);
    }

    #[test]
    fn king_relative_north_one() {
        // king = 5九、 piece = 5八 (1 段上)
        let king = Square::new(File::File5, Rank::Rank9);
        let piece = Square::new(File::File5, Rank::Rank8);
        let idx = king_relative_index_clamped(piece, king);
        // dx = 0、 dy = rank8 - rank9 = -1
        // idx = (0+8)*17 + (-1+8) = 8*17 + 7 = 143
        assert_eq!(idx, 8 * 17 + 7);
    }

    #[test]
    fn king_relative_max_corner() {
        // king = 1一、 piece = 9九
        let king = Square::new(File::File1, Rank::Rank1);
        let piece = Square::new(File::File9, Rank::Rank9);
        let idx = king_relative_index_clamped(piece, king);
        // dx = 8、 dy = 8
        // idx = (8+8)*17 + (8+8) = 16*17 + 16 = 288
        assert_eq!(idx, 16 * 17 + 16);
        assert!(idx < KING_REL_BUCKET_COUNT);
    }

    #[test]
    fn promotion_zone_black_view() {
        // 1段目 (rank index 0) は black 視点で敵陣
        let sq = Square::new(File::File5, Rank::Rank1);
        assert_eq!(promotion_zone(sq, 0), 2); // 敵陣
        // 5段目 (rank index 4) は中段
        let sq = Square::new(File::File5, Rank::Rank5);
        assert_eq!(promotion_zone(sq, 0), 1);
        // 9段目 (rank index 8) は自陣
        let sq = Square::new(File::File5, Rank::Rank9);
        assert_eq!(promotion_zone(sq, 0), 0);
    }

    #[test]
    fn promotion_zone_white_view_is_mirrored() {
        let sq = Square::new(File::File5, Rank::Rank1);
        assert_eq!(promotion_zone(sq, 1), 0); // white 視点では自陣
        let sq = Square::new(File::File5, Rank::Rank9);
        assert_eq!(promotion_zone(sq, 1), 2); // white 視点では敵陣
    }
}
