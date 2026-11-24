//! Position → P2NnueInput 特徴量抽出 (V3: dlshogi 互換)
//!
//! 出力は i32 binary (0/1) の 9728 dim ベクトル。
//! 内訳: features1 (5022) + features2 (4617) + padding (89)。

use rshogi_core::bitboard::{
    Bitboard, bishop_effect, dragon_effect, gold_effect, horse_effect, king_effect, knight_effect,
    lance_effect, pawn_effect, rook_effect, silver_effect,
};
use rshogi_core::position::Position;
use rshogi_core::types::{Color, PieceType, Square};

use super::feature_index::{compute_game_phase, compute_king_bucket};
use super::feature_types::{
    FEATURES1_CHANNELS_PER_COLOR, FEATURES1_START, FEATURES2_START, HAND_MAXES, HAND_OFFSETS,
    MAX_ATTACK_NUM, MAX_FEATURES2_HAND_NUM, MAX_PIECES_IN_HAND_SUM, P2_FEATURE_DIM, P2NnueInput,
    PIECETYPE_NUM, SQUARE_NUM,
};

/// 駒種 + 自玉視点の符号で attacks bitboard を取得
fn piece_attacks(pt: PieceType, color: Color, sq: Square, occupied: Bitboard) -> Bitboard {
    match pt {
        PieceType::Pawn => pawn_effect(color, sq),
        PieceType::Lance => lance_effect(color, sq, occupied),
        PieceType::Knight => knight_effect(color, sq),
        PieceType::Silver => silver_effect(color, sq),
        PieceType::Gold
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver => gold_effect(color, sq),
        PieceType::Bishop => bishop_effect(sq, occupied),
        PieceType::Rook => rook_effect(sq, occupied),
        PieceType::Horse => horse_effect(sq, occupied),
        PieceType::Dragon => dragon_effect(sq, occupied),
        PieceType::King => king_effect(sq),
    }
}

#[inline]
fn set_f1(features: &mut [i32; P2_FEATURE_DIM], color_idx: usize, ch: usize, sq_idx: usize) {
    let off = FEATURES1_START
        + color_idx * (FEATURES1_CHANNELS_PER_COLOR * SQUARE_NUM)
        + ch * SQUARE_NUM
        + sq_idx;
    features[off] = 1;
}

#[inline]
fn set_f2_hand(
    features: &mut [i32; P2_FEATURE_DIM],
    color_idx: usize,
    hand_offset: usize,
    count: usize,
) {
    // dlshogi の thermometer 表現: count 枚あれば 0..count の plane を全部 1 に
    let base = color_idx * MAX_PIECES_IN_HAND_SUM + hand_offset;
    for i in 0..count {
        let plane = base + i;
        let off = FEATURES2_START + plane * SQUARE_NUM;
        // 全マス 1
        for sq in 0..SQUARE_NUM {
            features[off + sq] = 1;
        }
    }
}

#[inline]
fn set_f2_in_check(features: &mut [i32; P2_FEATURE_DIM]) {
    let plane = MAX_FEATURES2_HAND_NUM; // 56
    let off = FEATURES2_START + plane * SQUARE_NUM;
    for sq in 0..SQUARE_NUM {
        features[off + sq] = 1;
    }
}

/// Position から P2NnueInput を抽出する (heap 上に Box で確保)
pub fn extract_features(pos: &Position) -> P2NnueInput {
    let mut features = Box::new([0i32; P2_FEATURE_DIM]);

    let turn = pos.side_to_move();
    let is_white_turn = turn == Color::White;
    let occupied = pos.occupied();

    // 利き数集計用 [color][square]
    let mut attack_num = [[0u8; SQUARE_NUM]; 2];

    // ─── 歩以外: 配置 + 利き + 利き数 ───
    let pawns_bb = pos.pieces_pt(PieceType::Pawn);
    let without_pawns = occupied & !pawns_bb;

    for sq in without_pawns.iter() {
        let pc = pos.piece_on(sq);
        let pt = pc.piece_type();
        let orig_color = pc.color();
        let attacks = piece_attacks(pt, orig_color, sq, occupied);

        // 出力用 (後手番なら盤面反転)
        let (c, out_sq) = if is_white_turn {
            (orig_color.opponent() as usize, sq.inverse().index())
        } else {
            (orig_color as usize, sq.index())
        };

        let pt_idx = pt as usize - 1;
        set_f1(&mut features, c, pt_idx, out_sq);

        for to in attacks.iter() {
            let out_to = if is_white_turn {
                to.inverse().index()
            } else {
                to.index()
            };
            set_f1(&mut features, c, PIECETYPE_NUM + pt_idx, out_to);

            let num = &mut attack_num[c][out_to];
            if (*num as usize) < MAX_ATTACK_NUM {
                set_f1(&mut features, c, 2 * PIECETYPE_NUM + *num as usize, out_to);
                *num += 1;
            }
        }
    }

    // ─── 歩 + 手駒 ───
    let colors = [Color::Black, Color::White];
    for logical_c in 0..2usize {
        let board_color = if is_white_turn {
            colors[1 - logical_c]
        } else {
            colors[logical_c]
        };

        // 歩の配置と利き
        let my_pawns = pawns_bb & pos.pieces_c(board_color);
        for sq in my_pawns.iter() {
            let out_sq = if is_white_turn {
                sq.inverse().index()
            } else {
                sq.index()
            };

            set_f1(&mut features, logical_c, 0, out_sq);

            let pawn_color = if logical_c == 0 {
                Color::Black
            } else {
                Color::White
            };
            if let Some(out_sq_sq) = Square::from_u8(out_sq as u8) {
                let effect = pawn_effect(pawn_color, out_sq_sq);
                for to in effect.iter() {
                    let out_to = to.index();
                    set_f1(&mut features, logical_c, PIECETYPE_NUM, out_to);

                    let num = &mut attack_num[logical_c][out_to];
                    if (*num as usize) < MAX_ATTACK_NUM {
                        set_f1(&mut features, logical_c, 2 * PIECETYPE_NUM + *num as usize, out_to);
                        *num += 1;
                    }
                }
            }
        }

        // 持ち駒
        let hand = pos.hand(colors[logical_c]);
        let board_c_idx = board_color as usize;
        let hand_piece_types = [
            PieceType::Pawn,
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::Bishop,
            PieceType::Rook,
        ];
        for (i, &hpt) in hand_piece_types.iter().enumerate() {
            let num = hand.count(hpt) as usize;
            let num = num.min(HAND_MAXES[i]);
            if num > 0 {
                set_f2_hand(&mut features, board_c_idx, HAND_OFFSETS[i], num);
            }
        }
    }

    // ─── 王手 ───
    if pos.in_check() {
        set_f2_in_check(&mut features);
    }

    P2NnueInput {
        piece_features: features,
        side_to_move: if turn == Color::Black { 1 } else { 0 },
        game_phase: compute_game_phase(pos),
        king_bucket: compute_king_bucket(pos),
    }
}
