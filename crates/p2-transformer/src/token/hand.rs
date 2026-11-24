//! 手駒の count-aware vocab 構築。
//!
//! Hand token は 14 slot (Black 7 + White 7) を順に埋める。 駒種の順序は
//! `PieceType::HAND_PIECES` (Pawn, Lance, Knight, Silver, Gold, Bishop, Rook)。
//!
//! 各 slot の vocab id は [`super::vocab::hand_vocab`] で得られる。
//! 例: Black Pawn=3 枚 → `HAND_B_BASE + 0 + 3`、 White Rook=0 → `HAND_W_BASE + 42`。

use rshogi_core::position::Position;
use rshogi_core::types::{Color, PieceType};

use super::vocab::hand_vocab;

/// HAND_PIECES の順序 (vocab module と完全に揃える必要がある)。
pub const HAND_PIECES_ORDER: [PieceType; 7] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Gold,
    PieceType::Bishop,
    PieceType::Rook,
];

/// `Position` の手駒 (両色) を hand vocab id 列に書き出す。
///
/// `out` の先頭 14 entry に `[Black 7 駒種、 White 7 駒種]` の順に書き込む。
/// 残りの slot は呼び出し側で PAD に初期化されていることを想定。
///
/// branch-less / allocation 0。
#[inline]
pub fn write_hand_tokens(pos: &Position, out: &mut [u16; 14]) {
    for color_idx in 0..2 {
        let color = if color_idx == 0 {
            Color::Black
        } else {
            Color::White
        };
        let hand = pos.hand(color);
        let base = color_idx * 7;
        // unroll 全 7 駒種。 各 iter は Hand::count 1 回 + add 1 回 + table lookup 1 回。
        let mut i = 0usize;
        while i < HAND_PIECES_ORDER.len() {
            let pt = HAND_PIECES_ORDER[i];
            let count = hand.count(pt) as u8;
            out[base + i] = hand_vocab(color, pt, count);
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::position::Position;

    #[test]
    fn startpos_has_empty_hands() {
        let mut pos = Position::new();
        pos.set_hirate();
        let mut buf = [u16::MAX; 14];
        write_hand_tokens(&pos, &mut buf);
        // 全 14 entry が "count=0" の vocab id (= base + offset[i] + 0) のはず。
        for (i, &id) in buf.iter().enumerate() {
            let color = if i < 7 { Color::Black } else { Color::White };
            let pt = HAND_PIECES_ORDER[i % 7];
            assert_eq!(id, hand_vocab(color, pt, 0), "slot {} mismatch", i);
        }
    }

    #[test]
    fn max_hand_clamps_correctly() {
        // 歩 18 枚を Black に持たせた局面を構築するのは大変なので、 vocab 単体で確認。
        use crate::token::vocab::{HAND_B_BASE, HAND_OFFSET, HAND_W_BASE};
        // Black Pawn 18 = HAND_B_BASE + 0 + 18
        assert_eq!(hand_vocab(Color::Black, PieceType::Pawn, 18), HAND_B_BASE + 18);
        // Black Lance 4 = HAND_B_BASE + 19 + 4 = HAND_B_BASE + 23
        assert_eq!(hand_vocab(Color::Black, PieceType::Lance, 4), HAND_B_BASE + HAND_OFFSET[1] + 4);
        // White Rook 2 = HAND_W_BASE + 42 + 2 = HAND_W_BASE + 44
        assert_eq!(hand_vocab(Color::White, PieceType::Rook, 2), HAND_W_BASE + HAND_OFFSET[6] + 2);
        // overflow は clamp される
        assert_eq!(hand_vocab(Color::Black, PieceType::Pawn, 99), HAND_B_BASE + 18);
        assert_eq!(
            hand_vocab(Color::White, PieceType::Bishop, 99),
            HAND_W_BASE + HAND_OFFSET[5] + 2
        );
    }
}
