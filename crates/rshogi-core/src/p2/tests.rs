//! P2 特徴量のテスト

use crate::movegen::MoveList;
use crate::movegen::generate_legal;
use crate::position::Position;
use crate::types::{Color, File, PieceType, Rank, Square};

use super::delta::{apply_delta, compute_delta};
use super::extract::extract_features;
use super::feature_index::{board_piece_index, hand_bucket_index, king_square_index};
use super::feature_types::{
    P2_BOARD_PIECE_LEN, P2_BOARD_PIECE_START, P2_FEATURE_DIM, P2_HAND_BUCKET_LEN,
    P2_HAND_BUCKET_START, P2_PHASE_ATTACK_LEN, P2_PHASE_ATTACK_START, P2_SIDE_TO_MOVE_INDEX,
    P2NnueInput, P2PieceType,
};

/// startpos を作成するヘルパー
fn make_startpos() -> Position {
    let mut pos = Position::new();
    pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
        .unwrap();
    pos
}

/// 特徴量ベクトルが一致するかを比較
fn features_equal(a: &P2NnueInput, b: &P2NnueInput) -> bool {
    a.piece_features == b.piece_features
        && a.side_to_move == b.side_to_move
        && a.game_phase == b.game_phase
        && a.king_bucket == b.king_bucket
}

// ─── 初期局面テスト ───

#[test]
fn test_startpos_board_pieces() {
    let pos = make_startpos();
    let input = extract_features(&pos);

    // 先手歩: 7七〜1七（7筋〜1筋の7段目）= 9 枚
    let mut pawn_count = 0;
    for sq_idx in 0..81 {
        let idx = board_piece_index(P2PieceType::Pawn, Color::Black, unsafe {
            Square::from_u8_unchecked(sq_idx)
        });
        if input.piece_features[idx] == 1 {
            pawn_count += 1;
        }
    }
    assert_eq!(pawn_count, 9, "先手歩は9枚");

    // 後手歩: 9 枚
    pawn_count = 0;
    for sq_idx in 0..81 {
        let idx = board_piece_index(P2PieceType::Pawn, Color::White, unsafe {
            Square::from_u8_unchecked(sq_idx)
        });
        if input.piece_features[idx] == 1 {
            pawn_count += 1;
        }
    }
    assert_eq!(pawn_count, 9, "後手歩は9枚");
}

#[test]
fn test_startpos_king_squares() {
    let pos = make_startpos();
    let input = extract_features(&pos);

    // 先手玉: 5九 = Square::new(File5, Rank9)
    let black_king_sq = Square::new(File::File5, Rank::Rank9);
    let idx = king_square_index(Color::Black, black_king_sq);
    assert_eq!(input.piece_features[idx], 1, "先手玉の位置");

    // 後手玉: 5一 = Square::new(File5, Rank1)
    let white_king_sq = Square::new(File::File5, Rank::Rank1);
    let idx = king_square_index(Color::White, white_king_sq);
    assert_eq!(input.piece_features[idx], 1, "後手玉の位置");

    // 玉位置は各色1つだけ
    let mut black_king_count = 0;
    let mut white_king_count = 0;
    for i in 0..81 {
        let sq = unsafe { Square::from_u8_unchecked(i) };
        if input.piece_features[king_square_index(Color::Black, sq)] == 1 {
            black_king_count += 1;
        }
        if input.piece_features[king_square_index(Color::White, sq)] == 1 {
            white_king_count += 1;
        }
    }
    assert_eq!(black_king_count, 1);
    assert_eq!(white_king_count, 1);
}

#[test]
fn test_startpos_side_to_move() {
    let pos = make_startpos();
    let input = extract_features(&pos);
    assert_eq!(input.piece_features[P2_SIDE_TO_MOVE_INDEX], 1, "先手番");
    assert_eq!(input.side_to_move, 1);
}

#[test]
fn test_startpos_hand_empty() {
    let pos = make_startpos();
    let input = extract_features(&pos);

    // 初期局面では持ち駒なし → 全駒種 count=0 バケットが 1
    for &color in &[Color::Black, Color::White] {
        for &pt in &PieceType::HAND_PIECES {
            let idx = hand_bucket_index(color, pt, 0);
            assert_eq!(
                input.piece_features[idx], 1,
                "持ち駒なしバケットが 1 であること ({:?}, {:?})",
                color, pt,
            );
        }
    }
}

#[test]
fn test_startpos_phase() {
    let pos = make_startpos();
    let input = extract_features(&pos);

    // 盤上 40 駒 - 2 玉 = 38 非王駒
    assert_eq!(input.piece_features[P2_PHASE_ATTACK_START + 2], 38, "非王駒数");
    assert_eq!(input.game_phase, 255, "opening phase");
}

#[test]
fn test_startpos_not_in_check() {
    let pos = make_startpos();
    let input = extract_features(&pos);
    assert_eq!(input.piece_features[P2_PHASE_ATTACK_START], 0, "王手なし");
}

// ─── Determinism テスト ───

#[test]
fn test_determinism() {
    let pos = make_startpos();
    let input1 = extract_features(&pos);
    for _ in 0..100 {
        let input2 = extract_features(&pos);
        assert!(features_equal(&input1, &input2), "決定性テスト失敗");
    }
}

// ─── Bounds チェック ───

#[test]
fn test_board_piece_index_bounds() {
    for p2type_idx in 0..P2PieceType::NUM {
        let p2type = unsafe { std::mem::transmute::<u8, P2PieceType>(p2type_idx as u8) };
        for color_idx in 0..2 {
            let color = if color_idx == 0 {
                Color::Black
            } else {
                Color::White
            };
            for sq_idx in 0..81 {
                let sq = unsafe { Square::from_u8_unchecked(sq_idx) };
                let idx = board_piece_index(p2type, color, sq);
                assert!(idx < P2_FEATURE_DIM, "board_piece_index out of bounds: {}", idx);
                assert!(
                    idx < P2_BOARD_PIECE_START + P2_BOARD_PIECE_LEN,
                    "board_piece_index exceeds section: {}",
                    idx
                );
            }
        }
    }
}

#[test]
fn test_king_square_index_bounds() {
    for color_idx in 0..2 {
        let color = if color_idx == 0 {
            Color::Black
        } else {
            Color::White
        };
        for sq_idx in 0..81 {
            let sq = unsafe { Square::from_u8_unchecked(sq_idx) };
            let idx = king_square_index(color, sq);
            assert!(idx < P2_FEATURE_DIM, "king_square_index out of bounds");
        }
    }
}

#[test]
fn test_hand_bucket_index_bounds() {
    let max_counts = [18, 4, 4, 4, 4, 2, 2]; // Pawn, Lance, Knight, Silver, Gold, Bishop, Rook
    for color_idx in 0..2 {
        let color = if color_idx == 0 {
            Color::Black
        } else {
            Color::White
        };
        for (i, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
            for count in 0..=max_counts[i] {
                let idx = hand_bucket_index(color, pt, count);
                assert!(
                    (P2_HAND_BUCKET_START..P2_HAND_BUCKET_START + P2_HAND_BUCKET_LEN)
                        .contains(&idx),
                    "hand_bucket_index out of section: {} (color={:?}, pt={:?}, count={})",
                    idx,
                    color,
                    pt,
                    count,
                );
            }
        }
    }
}

#[test]
fn test_startpos_binary_features_01() {
    let pos = make_startpos();
    let input = extract_features(&pos);

    // board_piece セクション: 全て 0 or 1
    for i in P2_BOARD_PIECE_START..P2_BOARD_PIECE_START + P2_BOARD_PIECE_LEN {
        assert!(
            input.piece_features[i] == 0 || input.piece_features[i] == 1,
            "board_piece[{}] = {} (expected 0 or 1)",
            i,
            input.piece_features[i],
        );
    }

    // hand_bucket セクション: 全て 0 or 1
    for i in P2_HAND_BUCKET_START..P2_HAND_BUCKET_START + P2_HAND_BUCKET_LEN {
        assert!(
            input.piece_features[i] == 0 || input.piece_features[i] == 1,
            "hand_bucket[{}] = {} (expected 0 or 1)",
            i,
            input.piece_features[i],
        );
    }
}

// ─── 差分 vs 全計算 一致テスト ───

#[test]
fn test_delta_vs_full_startpos_all_moves() {
    let mut pos = make_startpos();
    let input_before = extract_features(&pos);

    let mut list = MoveList::new();
    generate_legal(&pos, &mut list);

    for mv in &list {
        let stm_before = pos.side_to_move();
        let gives_check = pos.gives_check(*mv);
        let captured = pos.piece_on(mv.to());

        pos.do_move(*mv, gives_check);

        // 差分計算
        let mut delta_input = P2NnueInput {
            piece_features: input_before.piece_features,
            side_to_move: input_before.side_to_move,
            game_phase: input_before.game_phase,
            king_bucket: input_before.king_bucket,
        };
        let delta = compute_delta(&pos, *mv, captured, stm_before, &input_before);
        apply_delta(&mut delta_input, &delta, &pos);

        // 全計算
        let full_input = extract_features(&pos);

        // 比較
        for i in 0..P2_FEATURE_DIM {
            assert_eq!(
                delta_input.piece_features[i],
                full_input.piece_features[i],
                "差分 vs 全計算 不一致: feature[{}], move={} (delta={}, full={})",
                i,
                mv.to_usi(),
                delta_input.piece_features[i],
                full_input.piece_features[i],
            );
        }
        assert_eq!(delta_input.side_to_move, full_input.side_to_move);
        assert_eq!(delta_input.game_phase, full_input.game_phase);
        assert_eq!(delta_input.king_bucket, full_input.king_bucket);

        pos.undo_move(*mv);
    }
}

/// 駒取り・成り・打ちを含む複雑な局面での差分テスト
#[test]
fn test_delta_vs_full_complex_positions() {
    let sfens = [
        // 中盤局面（持ち駒あり）
        "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RsijO9telefonocelular 1",
        // 駒取りが多い局面
        "lnsg3nl/1r2k1gb1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 1",
        // 後手番
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2",
    ];

    for sfen in &sfens {
        let mut pos = Position::new();
        if pos.set_sfen(sfen).is_err() {
            continue;
        }

        let input_before = extract_features(&pos);
        let mut list = MoveList::new();
        generate_legal(&pos, &mut list);

        for mv in &list {
            let stm_before = pos.side_to_move();
            let gives_check = pos.gives_check(*mv);
            let captured = pos.piece_on(mv.to());

            pos.do_move(*mv, gives_check);

            let mut delta_input = P2NnueInput {
                piece_features: input_before.piece_features,
                side_to_move: input_before.side_to_move,
                game_phase: input_before.game_phase,
                king_bucket: input_before.king_bucket,
            };
            let delta = compute_delta(&pos, *mv, captured, stm_before, &input_before);
            apply_delta(&mut delta_input, &delta, &pos);

            let full_input = extract_features(&pos);

            for i in 0..P2_FEATURE_DIM {
                assert_eq!(
                    delta_input.piece_features[i],
                    full_input.piece_features[i],
                    "差分 vs 全計算 不一致: feature[{}], move={}, sfen={} (delta={}, full={})",
                    i,
                    mv.to_usi(),
                    sfen,
                    delta_input.piece_features[i],
                    full_input.piece_features[i],
                );
            }

            pos.undo_move(*mv);
        }
    }
}

/// 2手連続で差分を適用して一致を確認
#[test]
fn test_delta_chain_two_moves() {
    let mut pos = make_startpos();

    let mut list1 = MoveList::new();
    generate_legal(&pos, &mut list1);

    // 最初の合法手を実行
    let mv1 = list1[0];
    let stm1 = pos.side_to_move();
    let gc1 = pos.gives_check(mv1);
    let cap1 = pos.piece_on(mv1.to());
    let input0 = extract_features(&pos);

    pos.do_move(mv1, gc1);

    let delta1 = compute_delta(&pos, mv1, cap1, stm1, &input0);
    let mut input1 = P2NnueInput {
        piece_features: input0.piece_features,
        side_to_move: input0.side_to_move,
        game_phase: input0.game_phase,
        king_bucket: input0.king_bucket,
    };
    apply_delta(&mut input1, &delta1, &pos);

    // 2手目
    let mut list2 = MoveList::new();
    generate_legal(&pos, &mut list2);
    let mv2 = list2[0];
    let stm2 = pos.side_to_move();
    let gc2 = pos.gives_check(mv2);
    let cap2 = pos.piece_on(mv2.to());

    pos.do_move(mv2, gc2);

    let delta2 = compute_delta(&pos, mv2, cap2, stm2, &input1);
    let mut input2 = P2NnueInput {
        piece_features: input1.piece_features,
        side_to_move: input1.side_to_move,
        game_phase: input1.game_phase,
        king_bucket: input1.king_bucket,
    };
    apply_delta(&mut input2, &delta2, &pos);

    let full_input2 = extract_features(&pos);

    for i in 0..P2_FEATURE_DIM {
        assert_eq!(
            input2.piece_features[i], full_input2.piece_features[i],
            "2手連鎖差分 vs 全計算 不一致: feature[{}] (delta={}, full={})",
            i, input2.piece_features[i], full_input2.piece_features[i],
        );
    }
}

// ─── reserved スロットテスト ───

#[test]
fn test_reserved_slots_zero() {
    let pos = make_startpos();
    let input = extract_features(&pos);
    for i in (P2_PHASE_ATTACK_START + 5)..(P2_PHASE_ATTACK_START + P2_PHASE_ATTACK_LEN) {
        assert_eq!(input.piece_features[i], 0, "reserved slot [{}] should be 0", i);
    }
}

// ─── king_bucket テスト ───

#[test]
fn test_king_bucket_range() {
    let pos = make_startpos();
    let input = extract_features(&pos);
    assert!(input.king_bucket < 15, "king_bucket は 0..14");
}

// ─── feature_dim コンパイル時保証 ───

#[test]
fn test_feature_dim_2048() {
    assert_eq!(P2_FEATURE_DIM, 2048);
}

// ─── 盤上駒数の整合性 ───

#[test]
fn test_startpos_total_board_pieces() {
    let pos = make_startpos();
    let input = extract_features(&pos);

    // board_piece セクションで 1 の数をカウント
    let board_ones: i32 = input.piece_features
        [P2_BOARD_PIECE_START..P2_BOARD_PIECE_START + P2_BOARD_PIECE_LEN]
        .iter()
        .sum();
    // 初期局面: 40 駒 - 2 玉 = 38
    assert_eq!(board_ones, 38, "盤上非王駒数");
}
