//! Position → Transformer 入力 tokens 変換。
//!
//! 出力: `[T]` i64 token 列 (T = [`NUM_TOKENS`])。
//!
//! ## token layout
//!
//! ```text
//! [0]                                       CLS
//! [1 .. 1+81)         = [1 .. 82)           board square (sq=0..80)
//! [82 .. 82+7)        = [82 .. 89)          Black 持ち駒 count token (7 種)
//! [89 .. 89+7)        = [89 .. 96)          White 持ち駒 count token (7 種)
//! [96]                                       side_to_move token
//! [97]                                       ply bucket token (0..15)
//! [98]                                       Black king square (96 + sq)
//! [99]                                       White king square (192 + sq)
//! ```
//!
//! 全 token は同一 vocab を共有する (`vocab_size` = [`VOCAB_SIZE`])。
//! - 0:           PAD
//! - 1:           CLS
//! - 2:           EMPTY
//! - 3..17  (14): Black piece (PieceType raw 1..14 を +2)
//! - 17..31 (14): White piece (PieceType raw 1..14 を +16)
//! - 31..81 (50): 手駒 bucket (Black 7 種 × 5 + White 7 種 × 5 を圧縮、 実際は count 0..max を bucket 化)
//! - 81..83 (2):  SIDE_BLACK / SIDE_WHITE
//! - 83..99 (16): PLY_BUCKET (16 段、 game_ply / 16)
//! - 99..180 (81): Black king square
//! - 180..261 (81): White king square
//! - 261..512: 予約 (将来 phase / material 等)
//!
//! Vocab 余裕を持って 512 と設定。

use rshogi_core::position::Position;
use rshogi_core::types::{Color, PieceType, Square};

/// 1 token 列に含まれる token 数。
pub const NUM_TOKENS: usize = 100;
/// 盤上の square token 数。
pub const NUM_BOARD_TOKENS: usize = 81;
/// 手駒 token 数 (Black 7 + White 7)。
pub const NUM_HAND_TOKENS: usize = 14;
/// メタ token 数 (side + ply + king_b + king_w)。
pub const NUM_META_TOKENS: usize = 4;

/// 同一 vocab で扱える最大 id (Transformer の embedding table 行数)。
pub const VOCAB_SIZE: usize = 512;

// ===========================================================================
// vocab id 規約
// ===========================================================================
/// Padding token id (将来 token 数を可変にした際の埋め草、 現状は使用しないが
/// vocab 規約上 reserve)。
pub const ID_PAD: u16 = 0;
const ID_CLS: u16 = 1;
const ID_EMPTY: u16 = 2;
/// Black piece token base: `BLACK_PIECE_BASE + (pt as u16 - 1)` (range 3..17)
const BLACK_PIECE_BASE: u16 = 3;
/// White piece token base: `WHITE_PIECE_BASE + (pt as u16 - 1)` (range 17..31)
const WHITE_PIECE_BASE: u16 = 17;
/// Hand bucket base。 14 種類 (B 7 + W 7) × 5 count bucket = 70 → 80..150 確保。
///
/// hand bucket は count を 0..4 (>=4 を飽和) で 5 段に丸める。
const HAND_BUCKET_BASE: u16 = 31;
const HAND_BUCKETS_PER_PIECE: u16 = 5;
const HAND_BUCKET_RANGE: u16 = (NUM_HAND_TOKENS as u16) * HAND_BUCKETS_PER_PIECE; // 70
/// side_to_move base: SIDE_BLACK = base、 SIDE_WHITE = base + 1
const SIDE_BASE: u16 = HAND_BUCKET_BASE + HAND_BUCKET_RANGE; // 31+70 = 101
/// ply bucket base (16 段)
const PLY_BASE: u16 = SIDE_BASE + 2; // 103
const PLY_BUCKETS: u16 = 16;
/// king square base: Black 119..200、 White 200..281
const KING_B_BASE: u16 = PLY_BASE + PLY_BUCKETS; // 119
const KING_W_BASE: u16 = KING_B_BASE + 81; // 200

const _: () = assert!(KING_W_BASE as usize + 81 <= VOCAB_SIZE);

/// Position → tokens 変換。 allocation 0、 fixed-size `[i64; NUM_TOKENS]` を返す。
///
/// 推論側でも訓練側でも同じ encoding を使うこと。
#[inline]
pub fn encode_position_to_tokens(pos: &Position) -> [i64; NUM_TOKENS] {
    let mut tokens = [0i64; NUM_TOKENS];

    // [0] CLS
    tokens[0] = ID_CLS as i64;

    // [1..82) board: square 0..80
    for sq_raw in 0..81u8 {
        // SAFETY: 0..=80 は Square の値域
        let sq = unsafe { Square::from_u8_unchecked(sq_raw) };
        let piece = pos.piece_on(sq);
        let id = if piece.is_none() {
            ID_EMPTY
        } else {
            let pt = piece.piece_type() as u16; // 1..=14
            let base = match piece.color() {
                Color::Black => BLACK_PIECE_BASE,
                Color::White => WHITE_PIECE_BASE,
            };
            base + (pt - 1)
        };
        tokens[1 + sq_raw as usize] = id as i64;
    }

    // [82..89) Black hand、 [89..96) White hand
    let hand_pieces: [PieceType; 7] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
    ];
    let bh = pos.hand(Color::Black);
    let wh = pos.hand(Color::White);
    for (i, &pt) in hand_pieces.iter().enumerate() {
        let cnt_b = bh.count(pt).min(4) as u16; // 0..=4 bucket
        let cnt_w = wh.count(pt).min(4) as u16;
        // black: offset = i (0..7)、 white: offset = 7 + i
        let b_offset = i as u16;
        let w_offset = 7u16 + i as u16;
        tokens[82 + i] = (HAND_BUCKET_BASE + b_offset * HAND_BUCKETS_PER_PIECE + cnt_b) as i64;
        tokens[89 + i] = (HAND_BUCKET_BASE + w_offset * HAND_BUCKETS_PER_PIECE + cnt_w) as i64;
    }

    // [96] side_to_move
    let side_id = match pos.side_to_move() {
        Color::Black => SIDE_BASE,
        Color::White => SIDE_BASE + 1,
    };
    tokens[96] = side_id as i64;

    // [97] ply bucket (game_ply / 16、 clamp 0..15)
    let ply = pos.game_ply().max(0) as u16;
    let ply_bucket = (ply / 16).min(PLY_BUCKETS - 1);
    tokens[97] = (PLY_BASE + ply_bucket) as i64;

    // [98] Black king sq、 [99] White king sq
    tokens[98] = (KING_B_BASE + pos.king_square(Color::Black).raw() as u16) as i64;
    tokens[99] = (KING_W_BASE + pos.king_square(Color::White).raw() as u16) as i64;

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::position::Position;

    #[test]
    fn startpos_tokens_have_expected_shape() {
        let mut pos = Position::new();
        pos.set_hirate();
        let tokens = encode_position_to_tokens(&pos);
        assert_eq!(tokens.len(), NUM_TOKENS);
        assert_eq!(tokens[0], ID_CLS as i64);
        // 全 token は VOCAB_SIZE 未満
        for &t in tokens.iter() {
            assert!((0..VOCAB_SIZE as i64).contains(&t), "token id out of range: {t}");
        }
        // side_to_move = Black
        assert_eq!(tokens[96], SIDE_BASE as i64);
        // Black king at SQ_59 (raw 49) かどうかは set_startpos 仕様に依存するため、 範囲チェックのみ
        assert!(tokens[98] >= KING_B_BASE as i64 && tokens[98] < KING_W_BASE as i64);
        assert!(tokens[99] >= KING_W_BASE as i64 && tokens[99] < VOCAB_SIZE as i64);
    }

    #[test]
    fn empty_board_yields_pad_safe_tokens() {
        let pos = Position::new();
        let tokens = encode_position_to_tokens(&pos);
        // 盤面は全て EMPTY
        for (i, &t) in tokens.iter().enumerate().take(82).skip(1) {
            assert_eq!(t, ID_EMPTY as i64, "board slot {i} expected EMPTY");
        }
    }
}
