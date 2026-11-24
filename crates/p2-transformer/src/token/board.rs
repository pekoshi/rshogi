//! 盤上駒 → vocab id 変換。
//!
//! `Piece::raw()` (0..=30) を index にした lookup table を使う。 Piece の
//! 内部表現は `bit 0-3 = PieceType (1..=14)` / `bit 4 = Color (0=B, 1=W)` で、
//! 15 / 16 は未使用。 `Piece::NONE` (= 0) には `EMPTY` を割り当てる。
//!
//! 巨大 match nest 禁止のため、 全 31 entry の lookup を const で構築する。

use rshogi_core::types::Piece;

use super::vocab::{BLACK_PIECE_BASE, EMPTY, PAD, WHITE_PIECE_BASE};

/// `Piece::raw()` (0..31) → vocab id の lookup table。
///
/// 配列の各 slot に対応するエンコード:
/// - 0:        `Piece::NONE` → `EMPTY`
/// - 1..=14:   Black 駒 (Pawn..Dragon)
/// - 15, 16:   未使用 (Piece の bit layout 上有効値なし、 PAD で埋める)
/// - 17..=30:  White 駒 (Pawn..Dragon)
const PIECE_VOCAB_TABLE: [u16; 31] = {
    let mut t = [PAD; 31];
    t[0] = EMPTY;
    // Black 駒 (raw = 1..=14、 PieceType の生 u8)
    let mut i = 1u16;
    while i <= 14 {
        t[i as usize] = BLACK_PIECE_BASE + (i - 1);
        i += 1;
    }
    // White 駒 (raw = 17..=30)
    let mut i = 17u16;
    while i <= 30 {
        t[i as usize] = WHITE_PIECE_BASE + (i - 17);
        i += 1;
    }
    t
};

/// 盤上の `Piece` を vocab id に変換する (branch-less)。
///
/// `Piece::NONE` → `EMPTY`、 有効な駒 → 対応する Black/White piece id。
#[inline]
pub fn piece_vocab_id(piece: Piece) -> u16 {
    // SAFETY: Piece::raw() の値域は {0, 1..=14, 17..=30}。 PIECE_VOCAB_TABLE の長さは 31 で、
    // 全 valid raw が index 範囲内であることが Piece::from_raw のドキュメントで保証される。
    // 未使用の 15 / 16 も table 内で PAD として初期化済みのため fall-through 安全。
    debug_assert!((piece.raw() as usize) < PIECE_VOCAB_TABLE.len());
    PIECE_VOCAB_TABLE[piece.raw() as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::types::{Color, PieceType};

    #[test]
    fn empty_maps_to_empty() {
        assert_eq!(piece_vocab_id(Piece::NONE), EMPTY);
    }

    #[test]
    fn black_pieces_map_to_black_range() {
        assert_eq!(piece_vocab_id(Piece::B_PAWN), BLACK_PIECE_BASE);
        assert_eq!(piece_vocab_id(Piece::B_LANCE), BLACK_PIECE_BASE + 1);
        assert_eq!(piece_vocab_id(Piece::B_KNIGHT), BLACK_PIECE_BASE + 2);
        assert_eq!(piece_vocab_id(Piece::B_SILVER), BLACK_PIECE_BASE + 3);
        assert_eq!(piece_vocab_id(Piece::B_BISHOP), BLACK_PIECE_BASE + 4);
        assert_eq!(piece_vocab_id(Piece::B_ROOK), BLACK_PIECE_BASE + 5);
        assert_eq!(piece_vocab_id(Piece::B_GOLD), BLACK_PIECE_BASE + 6);
        assert_eq!(piece_vocab_id(Piece::B_KING), BLACK_PIECE_BASE + 7);
        assert_eq!(piece_vocab_id(Piece::B_PRO_PAWN), BLACK_PIECE_BASE + 8);
        assert_eq!(piece_vocab_id(Piece::B_DRAGON), BLACK_PIECE_BASE + 13);
    }

    #[test]
    fn white_pieces_map_to_white_range() {
        assert_eq!(piece_vocab_id(Piece::W_PAWN), WHITE_PIECE_BASE);
        assert_eq!(piece_vocab_id(Piece::W_KING), WHITE_PIECE_BASE + 7);
        assert_eq!(piece_vocab_id(Piece::W_HORSE), WHITE_PIECE_BASE + 12);
        assert_eq!(piece_vocab_id(Piece::W_DRAGON), WHITE_PIECE_BASE + 13);
    }

    #[test]
    fn promoted_pieces_map_correctly() {
        for color in [Color::Black, Color::White] {
            for pt in [
                PieceType::ProPawn,
                PieceType::ProLance,
                PieceType::ProKnight,
                PieceType::ProSilver,
                PieceType::Horse,
                PieceType::Dragon,
            ] {
                let pc = Piece::make(color, pt);
                let expected = super::super::vocab::piece_vocab(color, pt);
                assert_eq!(piece_vocab_id(pc), expected, "piece={:?}", pc);
            }
        }
    }
}
