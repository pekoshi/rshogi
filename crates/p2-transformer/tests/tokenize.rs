//! `p2-transformer` の統合テスト。
//!
//! - startpos の token layout 全体検証
//! - 各種 promotion 局面
//! - 最大手駒 (歩 18 等)
//! - 王手局面
//! - king-relative の正しさ
//! - token bounds (全 token vocab 範囲内)
//! - u16 / u8 overflow なし
//! - 同一 Position に対する deterministic 性

use p2_transformer::token::board::piece_vocab_id;
use p2_transformer::token::convert::position_to_tokens;
use p2_transformer::token::king_relative::king_relative_index_clamped;
use p2_transformer::token::vocab::{
    BLACK_PIECE_BASE, CHK_NO, CHK_YES, CLS, EMPTY, HAND_B_BASE, HAND_OFFSET, KING_B_BASE,
    KING_REL_BUCKET_COUNT, KING_REL_NONE, KING_W_BASE, MAX_TOKENS, PAD, SIDE_BLACK, SIDE_WHITE,
    SLOT_BOARD_BEGIN, SLOT_BOARD_END, SLOT_CLS, SLOT_HAND_BEGIN, SLOT_HAND_END, SLOT_KING_BLACK,
    SLOT_KING_WHITE, SLOT_PADDING_BEGIN, SLOT_SUMMARY_BEGIN, SQUARE_NONE, SUMMARY_RESERVED,
    SUMMARY_SLOT_CHECK, SUMMARY_SLOT_PLY, SUMMARY_SLOT_SIDE, VOCAB_SIZE, WHITE_PIECE_BASE,
    ZONE_COUNT, ZONE_NONE, king_vocab, piece_vocab,
};
use rshogi_core::position::Position;
use rshogi_core::types::{Color, File, Piece, PieceType, Rank, Square};

fn pos_from_sfen(sfen: &str) -> Position {
    let mut p = Position::new();
    p.set_sfen(sfen).expect("valid sfen");
    p
}

fn startpos() -> Position {
    let mut p = Position::new();
    p.set_hirate();
    p
}

#[test]
fn startpos_token_layout() {
    let pos = startpos();
    let t = position_to_tokens(&pos);

    // CLS slot
    assert_eq!(t.tokens[SLOT_CLS], CLS);
    assert_eq!(t.squares[SLOT_CLS], SQUARE_NONE);
    assert_eq!(t.own_king_rel[SLOT_CLS], KING_REL_NONE);
    assert_eq!(t.zones[SLOT_CLS], ZONE_NONE);

    // board: 40 駒
    let mut filled = 0;
    for i in SLOT_BOARD_BEGIN..SLOT_BOARD_END {
        if t.tokens[i] != EMPTY {
            filled += 1;
        }
        assert_eq!(t.squares[i] as usize, i - SLOT_BOARD_BEGIN);
    }
    assert_eq!(filled, 40, "startpos must have 40 pieces");

    // hand: 全て count=0
    for i in SLOT_HAND_BEGIN..SLOT_HAND_END {
        let id = t.tokens[i];
        assert!(id >= HAND_B_BASE);
        assert!(id < SIDE_BLACK);
    }

    // king tokens
    assert_eq!(
        t.tokens[SLOT_KING_BLACK],
        king_vocab(Color::Black, Square::new(File::File5, Rank::Rank9).raw())
    );
    assert_eq!(
        t.tokens[SLOT_KING_WHITE],
        king_vocab(Color::White, Square::new(File::File5, Rank::Rank1).raw())
    );

    // summary: side=Black, check=No
    assert_eq!(t.tokens[SLOT_SUMMARY_BEGIN + SUMMARY_SLOT_SIDE], SIDE_BLACK);
    assert_eq!(t.tokens[SLOT_SUMMARY_BEGIN + SUMMARY_SLOT_CHECK], CHK_NO);

    // padding
    for i in SLOT_PADDING_BEGIN..MAX_TOKENS {
        assert_eq!(t.tokens[i], PAD);
        assert_eq!(t.squares[i], SQUARE_NONE);
        assert_eq!(t.own_king_rel[i], KING_REL_NONE);
        assert_eq!(t.enemy_king_rel[i], KING_REL_NONE);
        assert_eq!(t.zones[i], ZONE_NONE);
    }
}

#[test]
fn startpos_white_to_move() {
    // 手番を白に変更
    let pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1");
    let t = position_to_tokens(&pos);
    assert_eq!(t.tokens[SLOT_SUMMARY_BEGIN + SUMMARY_SLOT_SIDE], SIDE_WHITE);
}

#[test]
fn promoted_pieces_each_kind() {
    // 各成駒を含む局面を SFEN で構築。
    // 5五に成駒を 1 つだけ置く + 玉 2 枚。
    let cases: &[(&str, Piece)] = &[
        // と金 (Black ProPawn)
        ("9/9/9/9/4+P4/9/9/9/k3K4 b - 1", Piece::B_PRO_PAWN),
        // 成香 (Black ProLance)
        ("9/9/9/9/4+L4/9/9/9/k3K4 b - 1", Piece::B_PRO_LANCE),
        // 成桂
        ("9/9/9/9/4+N4/9/9/9/k3K4 b - 1", Piece::B_PRO_KNIGHT),
        // 成銀
        ("9/9/9/9/4+S4/9/9/9/k3K4 b - 1", Piece::B_PRO_SILVER),
        // 馬
        ("9/9/9/9/4+B4/9/9/9/k3K4 b - 1", Piece::B_HORSE),
        // 龍
        ("9/9/9/9/4+R4/9/9/9/k3K4 b - 1", Piece::B_DRAGON),
        // 白の成駒
        ("9/9/9/9/4+p4/9/9/9/k3K4 b - 1", Piece::W_PRO_PAWN),
        ("9/9/9/9/4+r4/9/9/9/k3K4 b - 1", Piece::W_DRAGON),
    ];
    for (sfen, expected) in cases {
        let pos = pos_from_sfen(sfen);
        let t = position_to_tokens(&pos);
        // 5五 = File5(idx 4) * 9 + Rank5(idx 4) = 40
        let slot = SLOT_BOARD_BEGIN + 40;
        assert_eq!(
            t.tokens[slot],
            piece_vocab_id(*expected),
            "sfen={} expected piece={:?}",
            sfen,
            expected
        );
    }
}

#[test]
fn max_hand_counts() {
    // 全駒種を最大近くまで持たせる局面。 SFEN の手駒部分で記述。
    // 注意: 平手より少なめにする (合計駒数の整合性確認のため)。
    // 歩 18 + 香 4 + 桂 4 + 銀 4 + 金 4 + 角 2 + 飛 2 = 38 駒、 ただし元の駒種で
    // 既に盤上にもあるため SFEN 全体の整合性が必要。 ここでは盤上を最小化する。
    //
    // 玉 2 枚と歩 18 枚 (黒手駒) + 香 4 桂 4 銀 4 金 4 角 2 飛 2 (白手駒) = SFEN 化可能。
    // 盤上に何もない 9/9/...は SFEN として有効。 手番 b で手駒記述 "18P4l4n4s4g2b2r"
    let sfen = "k7K/9/9/9/9/9/9/9/9 b 18P4l4n4s4g2b2r 1";
    let pos = pos_from_sfen(sfen);
    let t = position_to_tokens(&pos);

    // Black Pawn 18
    let bp_slot = SLOT_HAND_BEGIN;
    assert_eq!(t.tokens[bp_slot], HAND_B_BASE + HAND_OFFSET[0] + 18);

    // White の各駒の vocab id を確認
    // hand layout: [Black Pawn, Lance, Knight, Silver, Gold, Bishop, Rook,
    //               White Pawn, Lance, Knight, Silver, Gold, Bishop, Rook]
    let wp_base = SLOT_HAND_BEGIN + 7;
    use p2_transformer::token::vocab::HAND_W_BASE;
    // White Pawn = 0
    assert_eq!(t.tokens[wp_base], (HAND_W_BASE + HAND_OFFSET[0]));
    // White Lance = 4
    assert_eq!(t.tokens[wp_base + 1], HAND_W_BASE + HAND_OFFSET[1] + 4);
    // White Knight = 4
    assert_eq!(t.tokens[wp_base + 2], HAND_W_BASE + HAND_OFFSET[2] + 4);
    // White Silver = 4
    assert_eq!(t.tokens[wp_base + 3], HAND_W_BASE + HAND_OFFSET[3] + 4);
    // White Gold = 4
    assert_eq!(t.tokens[wp_base + 4], HAND_W_BASE + HAND_OFFSET[4] + 4);
    // White Bishop = 2
    assert_eq!(t.tokens[wp_base + 5], HAND_W_BASE + HAND_OFFSET[5] + 2);
    // White Rook = 2
    assert_eq!(t.tokens[wp_base + 6], HAND_W_BASE + HAND_OFFSET[6] + 2);
}

#[test]
fn in_check_summary() {
    // Black 玉が 5九、 白龍が 5八 で王手
    let sfen = "9/9/9/9/9/9/9/4+r4/4K3k b - 1";
    let pos = pos_from_sfen(sfen);
    assert!(pos.in_check());
    let t = position_to_tokens(&pos);
    assert_eq!(t.tokens[SLOT_SUMMARY_BEGIN + SUMMARY_SLOT_CHECK], CHK_YES);
}

#[test]
fn king_relative_at_startpos() {
    let pos = startpos();
    let t = position_to_tokens(&pos);
    // Black 玉 = 5九 (file=4, rank=8)、 stm=Black なので own=BK、 enemy=WK
    let bk = Square::new(File::File5, Rank::Rank9);
    let wk = Square::new(File::File5, Rank::Rank1);
    // King[96] slot の own_king_rel = (BK rel BK) = (0,0) → idx 144
    assert_eq!(t.own_king_rel[SLOT_KING_BLACK], 8 * 17 + 8);
    // King[96] slot の enemy_king_rel = (BK rel WK) = (0, +8) → idx (0+8)*17 + (8+8) = 152
    let expected = king_relative_index_clamped(bk, wk);
    assert_eq!(t.enemy_king_rel[SLOT_KING_BLACK], expected);
}

#[test]
fn all_tokens_within_vocab() {
    let positions = [
        startpos(),
        pos_from_sfen("9/9/9/9/4+R4/9/9/9/k3K4 b - 1"),
        pos_from_sfen("k7K/9/9/9/9/9/9/9/9 b 18P4l4n4s4g2b2r 1"),
        pos_from_sfen("9/9/9/9/9/9/9/4+r4/4K3k b - 1"),
    ];
    for pos in &positions {
        let t = position_to_tokens(pos);
        for &id in t.tokens.iter() {
            assert!((id as usize) < VOCAB_SIZE, "token vocab overflow: {}", id);
        }
        for &sq in t.squares.iter() {
            // sentinel 255 か 0..81
            assert!(sq == SQUARE_NONE || (sq as usize) < 81, "square overflow: {}", sq);
        }
        for &kr in t.own_king_rel.iter() {
            assert!(kr == KING_REL_NONE || kr < KING_REL_BUCKET_COUNT);
        }
        for &kr in t.enemy_king_rel.iter() {
            assert!(kr == KING_REL_NONE || kr < KING_REL_BUCKET_COUNT);
        }
        for &z in t.zones.iter() {
            assert!(z == ZONE_NONE || z < ZONE_COUNT);
        }
    }
}

#[test]
fn deterministic_same_position() {
    let pos = startpos();
    let t1 = position_to_tokens(&pos);
    let t2 = position_to_tokens(&pos);
    assert_eq!(t1.tokens, t2.tokens);
    assert_eq!(t1.squares, t2.squares);
    assert_eq!(t1.own_king_rel, t2.own_king_rel);
    assert_eq!(t1.enemy_king_rel, t2.enemy_king_rel);
    assert_eq!(t1.zones, t2.zones);
}

#[test]
fn deterministic_complex_position() {
    let pos = pos_from_sfen("k7K/9/9/9/9/9/9/9/9 b 18P4l4n4s4g2b2r 1");
    let t1 = position_to_tokens(&pos);
    let t2 = position_to_tokens(&pos);
    assert_eq!(t1.tokens, t2.tokens);
}

#[test]
fn vocab_ranges_disjoint() {
    // Black piece と White piece、 hand B と hand W、 等が disjoint。
    assert!(
        piece_vocab(Color::Black, PieceType::Dragon) < piece_vocab(Color::White, PieceType::Pawn)
    );
    assert!(piece_vocab(Color::White, PieceType::Dragon) < KING_B_BASE);
    assert!(king_vocab(Color::Black, 80) < KING_W_BASE);
    assert!(king_vocab(Color::White, 80) < HAND_B_BASE);
}

#[test]
fn padding_slots_use_pad_token() {
    let pos = startpos();
    let t = position_to_tokens(&pos);
    for i in SLOT_PADDING_BEGIN..MAX_TOKENS {
        assert_eq!(t.tokens[i], PAD);
    }
}

#[test]
fn summary_reserved_slots() {
    let pos = startpos();
    let t = position_to_tokens(&pos);
    // [98..103) は有効 summary、 [103..116) は予約
    for i in (SLOT_SUMMARY_BEGIN + 5)..(SLOT_SUMMARY_BEGIN + 18) {
        assert_eq!(t.tokens[i], SUMMARY_RESERVED, "summary reserved at i={}", i);
    }
}

#[test]
fn black_white_piece_base_distinct() {
    assert_ne!(BLACK_PIECE_BASE, WHITE_PIECE_BASE);
    assert_eq!(WHITE_PIECE_BASE - BLACK_PIECE_BASE, 14);
}

#[test]
fn ply_bucket_advances_with_moves() {
    // Note: 厳密な ply 進行は make_move を呼ばずに SFEN で擬似的に変える。
    let mid = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 30");
    let t = position_to_tokens(&mid);
    let early = position_to_tokens(&startpos());
    let early_ply = early.tokens[SLOT_SUMMARY_BEGIN + SUMMARY_SLOT_PLY];
    let mid_ply = t.tokens[SLOT_SUMMARY_BEGIN + SUMMARY_SLOT_PLY];
    assert!(mid_ply >= early_ply, "ply bucket must be non-decreasing");
}
