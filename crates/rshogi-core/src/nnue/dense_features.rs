//! Compact NNUE 特徴量生成モジュール
//!
//! Policy/MCTS用NNUEの入力として2048次元の密特徴量ベクトルを生成する。
//! 既存のHalfKP/HalfKAスパースインデックス方式とは独立したフォーマット。
//!
//! ## Feature Layout (2048次元)
//!
//! | 区間 | 内容 | 次元数 |
//! |------|------|--------|
//! | `[0..1782)` | 盤上駒 11型×2色×81マス (binary) | 1782 |
//! | `[1782..1863)` | 先手玉位置 one-hot | 81 |
//! | `[1863..1944)` | 後手玉位置 one-hot | 81 |
//! | `[1944]` | 手番 (Black=0, White=1) | 1 |
//! | `[1945..2025)` | 手駒 thermometer | 80 |
//! | `[2025..2048)` | phase/attack/reserved | 23 |

use crate::position::Position;
use crate::types::{Color, PieceType, Square};

use super::bona_piece_halfka_hm::king_bucket;

// =============================================================================
// 定数
// =============================================================================

/// フォーマットバージョン
pub const FORMAT_VERSION: u32 = 1;

/// 特徴量の総次元数
pub const FEATURE_DIM: usize = 2048;

/// 盤上駒セグメント: 11型 × 2色 × 81マス = 1782
const BOARD_PIECE_OFFSET: usize = 0;
const DENSE_PIECE_TYPES: usize = 11;
const BOARD_PIECE_DIM: usize = DENSE_PIECE_TYPES * 2 * Square::NUM; // 1782

/// 先手玉位置 one-hot
const BLACK_KING_OFFSET: usize = BOARD_PIECE_DIM; // 1782
/// 後手玉位置 one-hot
const WHITE_KING_OFFSET: usize = BLACK_KING_OFFSET + Square::NUM; // 1863

/// 手番
const SIDE_TO_MOVE_OFFSET: usize = WHITE_KING_OFFSET + Square::NUM; // 1944

/// 手駒 thermometer
const HAND_OFFSET: usize = SIDE_TO_MOVE_OFFSET + 1; // 1945
/// 1色あたりの手駒次元: P(18)+L(4)+N(4)+S(4)+G(4)+B(2)+R(2) = 38
const HAND_DIM_PER_COLOR: usize = 38;
/// 手駒reserved
const HAND_RESERVED: usize = 4;
/// 手駒セグメント合計: 38*2 + 4 = 80
const HAND_DIM: usize = HAND_DIM_PER_COLOR * 2 + HAND_RESERVED; // 80

/// phase/attack/reserved
const PHASE_OFFSET: usize = HAND_OFFSET + HAND_DIM; // 2025

// レイアウト整合性の静的アサート
const _: () = assert!(PHASE_OFFSET + 23 == FEATURE_DIM);

// =============================================================================
// DensePieceType - 盤上11駒種
// =============================================================================

/// 盤上駒の密インデックス用駒種（11種、玉を除く）
///
/// 成香・成桂・成銀は「成小」(idx=8) に統合される。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DensePieceType {
    /// 歩
    Pawn = 0,
    /// 香
    Lance = 1,
    /// 桂
    Knight = 2,
    /// 銀
    Silver = 3,
    /// 金
    Gold = 4,
    /// 角
    Bishop = 5,
    /// 飛
    Rook = 6,
    /// と
    ProPawn = 7,
    /// 成小 (成香/成桂/成銀統合)
    ProMinor = 8,
    /// 馬
    Horse = 9,
    /// 龍
    Dragon = 10,
}

/// `PieceType` → `DensePieceType` 変換
///
/// 玉 (`King`) に対しては `None` を返す。
#[inline]
pub fn to_dense_piece_type(pt: PieceType) -> Option<DensePieceType> {
    match pt {
        PieceType::Pawn => Some(DensePieceType::Pawn),
        PieceType::Lance => Some(DensePieceType::Lance),
        PieceType::Knight => Some(DensePieceType::Knight),
        PieceType::Silver => Some(DensePieceType::Silver),
        PieceType::Gold => Some(DensePieceType::Gold),
        PieceType::Bishop => Some(DensePieceType::Bishop),
        PieceType::Rook => Some(DensePieceType::Rook),
        PieceType::ProPawn => Some(DensePieceType::ProPawn),
        PieceType::ProLance | PieceType::ProKnight | PieceType::ProSilver => {
            Some(DensePieceType::ProMinor)
        }
        PieceType::Horse => Some(DensePieceType::Horse),
        PieceType::Dragon => Some(DensePieceType::Dragon),
        PieceType::King => None,
    }
}

// =============================================================================
// NNUEInput
// =============================================================================

/// Compact NNUE 入力特徴量
pub struct NNUEInput {
    /// 2048次元の密特徴量ベクトル (binary 0/1 が主)
    pub piece_features: [i32; FEATURE_DIM],
    /// 手番 (Black=0, White=1)
    pub side_to_move: u8,
    /// ゲーム進行度 (= min(game_ply, 255))
    pub game_phase: u8,
    /// 手番側玉の45バケットインデックス (Half-Mirror方式)
    pub king_bucket: u8,
}

// =============================================================================
// extract - フル抽出
// =============================================================================

/// 局面から特徴量をフル抽出する（純粋関数）
///
/// 81マスのスキャンと手駒のthermometerエンコードにより、
/// 2048次元のi32ベクトルを生成する。ヒープ割り当ては行わない。
pub fn extract(pos: &Position) -> NNUEInput {
    let mut features = [0i32; FEATURE_DIM];

    // --- 盤上駒 + 玉 one-hot ---
    for sq in Square::all() {
        let pc = pos.piece_on(sq);
        if pc.is_none() {
            continue;
        }

        let pt = pc.piece_type();
        let color = pc.color();

        if pt == PieceType::King {
            // 玉は専用セクションへ
            let offset = if color == Color::Black {
                BLACK_KING_OFFSET
            } else {
                WHITE_KING_OFFSET
            };
            features[offset + sq.index()] = 1;
        } else if let Some(dense_type) = to_dense_piece_type(pt) {
            // 盤上駒: dense_type * 162 + color_index * 81 + square_index
            let color_index = color.index(); // Black=0, White=1
            let idx = BOARD_PIECE_OFFSET
                + (dense_type as usize) * (2 * Square::NUM)
                + color_index * Square::NUM
                + sq.index();
            features[idx] = 1;
        }
    }

    // --- 手番 ---
    let stm = pos.side_to_move();
    let stm_value = if stm == Color::Black { 0 } else { 1 };
    features[SIDE_TO_MOVE_OFFSET] = stm_value;

    // --- 手駒 thermometer ---
    encode_hand_thermometer(pos.hand(Color::Black), HAND_OFFSET, &mut features);
    encode_hand_thermometer(
        pos.hand(Color::White),
        HAND_OFFSET + HAND_DIM_PER_COLOR,
        &mut features,
    );

    // --- game_phase ---
    let ply = pos.game_ply();
    let phase = if ply < 0 {
        0u8
    } else {
        (ply as u32).min(255) as u8
    };
    features[PHASE_OFFSET] = phase as i32;

    // --- king_bucket ---
    let kb = king_bucket(pos.king_square(stm), stm) as u8;

    NNUEInput {
        piece_features: features,
        side_to_move: stm_value as u8,
        game_phase: phase,
        king_bucket: kb,
    }
}

// =============================================================================
// 手駒 thermometer エンコード
// =============================================================================

/// 手駒をthermometer符号化して `features[offset..]` に書き込む
///
/// 1色あたり38次元: P(18)+L(4)+N(4)+S(4)+G(4)+B(2)+R(2)
/// 例: 歩3枚 → `[1,1,1,0,...,0]` (18個中3個が1)
fn encode_hand_thermometer(
    hand: crate::types::Hand,
    offset: usize,
    features: &mut [i32; FEATURE_DIM],
) {
    /// 駒種ごとの最大枚数（thermometer次元数）
    const HAND_PIECES: [(PieceType, usize); 7] = [
        (PieceType::Pawn, 18),
        (PieceType::Lance, 4),
        (PieceType::Knight, 4),
        (PieceType::Silver, 4),
        (PieceType::Gold, 4),
        (PieceType::Bishop, 2),
        (PieceType::Rook, 2),
    ];

    let mut cursor = offset;
    for &(pt, max_count) in &HAND_PIECES {
        let count = hand.count(pt) as usize;
        for i in 0..max_count {
            if i < count {
                features[cursor] = 1;
            }
            cursor += 1;
        }
    }
    debug_assert_eq!(cursor - offset, HAND_DIM_PER_COLOR);
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Position, SFEN_HIRATE};
    use crate::types::Hand;

    /// ヘルパー: SFENから局面を作成
    fn pos_from_sfen(sfen: &str) -> Position {
        let mut pos = Position::new();
        pos.set_sfen(sfen).unwrap();
        pos
    }

    // --------------------------------------------------
    // 1. 初期局面テスト
    // --------------------------------------------------
    #[test]
    fn test_hirate_basic() {
        let pos = pos_from_sfen(SFEN_HIRATE);
        let input = extract(&pos);

        // 手番は先手
        assert_eq!(input.side_to_move, 0);
        // SFEN_HIRATEの手数フィールドは1 → game_ply()=1 → phase=1
        assert_eq!(input.game_phase, 1);

        // 特徴量の非ゼロ要素数を確認
        // 盤上38駒 + 玉2 one-hot + game_phase(=1) = 41
        // (手番=0はカウントされない)
        let nonzero_count = input.piece_features.iter().filter(|&&v| v != 0).count();
        assert_eq!(nonzero_count, 41);
    }

    #[test]
    fn test_hirate_piece_count_by_type() {
        let pos = pos_from_sfen(SFEN_HIRATE);
        let input = extract(&pos);

        // 歩: 先手9 + 後手9 = 18
        let pawn_start = BOARD_PIECE_OFFSET + (DensePieceType::Pawn as usize) * 162;
        let pawn_count: i32 = input.piece_features[pawn_start..pawn_start + 162].iter().sum();
        assert_eq!(pawn_count, 18);

        // 香: 先手2 + 後手2 = 4
        let lance_start = BOARD_PIECE_OFFSET + (DensePieceType::Lance as usize) * 162;
        let lance_count: i32 = input.piece_features[lance_start..lance_start + 162].iter().sum();
        assert_eq!(lance_count, 4);

        // 金: 先手2 + 後手2 = 4
        let gold_start = BOARD_PIECE_OFFSET + (DensePieceType::Gold as usize) * 162;
        let gold_count: i32 = input.piece_features[gold_start..gold_start + 162].iter().sum();
        assert_eq!(gold_count, 4);

        // 角: 先手1 + 後手1 = 2
        let bishop_start = BOARD_PIECE_OFFSET + (DensePieceType::Bishop as usize) * 162;
        let bishop_count: i32 = input.piece_features[bishop_start..bishop_start + 162].iter().sum();
        assert_eq!(bishop_count, 2);

        // 飛: 先手1 + 後手1 = 2
        let rook_start = BOARD_PIECE_OFFSET + (DensePieceType::Rook as usize) * 162;
        let rook_count: i32 = input.piece_features[rook_start..rook_start + 162].iter().sum();
        assert_eq!(rook_count, 2);
    }

    #[test]
    fn test_hirate_king_onehot() {
        let pos = pos_from_sfen(SFEN_HIRATE);
        let input = extract(&pos);

        // 先手玉 one-hot: 1マスだけ1
        let bk_sum: i32 = input.piece_features[BLACK_KING_OFFSET..WHITE_KING_OFFSET].iter().sum();
        assert_eq!(bk_sum, 1);

        // 後手玉 one-hot: 1マスだけ1
        let wk_sum: i32 = input.piece_features[WHITE_KING_OFFSET..WHITE_KING_OFFSET + Square::NUM]
            .iter()
            .sum();
        assert_eq!(wk_sum, 1);
    }

    // --------------------------------------------------
    // 2. 空盤テスト（玉2枚のみ）
    // --------------------------------------------------
    #[test]
    fn test_kings_only() {
        // 先手玉5九、後手玉5一
        let pos = pos_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        let input = extract(&pos);

        // 盤上駒セグメントは全て0
        let board_sum: i32 = input.piece_features[BOARD_PIECE_OFFSET..BOARD_PIECE_DIM].iter().sum();
        assert_eq!(board_sum, 0);

        // 玉 one-hot はそれぞれ1
        let bk_sum: i32 = input.piece_features[BLACK_KING_OFFSET..WHITE_KING_OFFSET].iter().sum();
        assert_eq!(bk_sum, 1);
        let wk_sum: i32 = input.piece_features[WHITE_KING_OFFSET..WHITE_KING_OFFSET + Square::NUM]
            .iter()
            .sum();
        assert_eq!(wk_sum, 1);

        // 手駒は全て0
        let hand_sum: i32 = input.piece_features[HAND_OFFSET..HAND_OFFSET + HAND_DIM].iter().sum();
        assert_eq!(hand_sum, 0);
    }

    // --------------------------------------------------
    // 3. 対称性テスト
    // --------------------------------------------------
    #[test]
    fn test_color_symmetry() {
        // 先手歩77 vs 後手歩33（鏡像的配置）
        let pos_b = pos_from_sfen("4k4/9/9/9/9/9/2P6/9/4K4 b - 1");
        let pos_w = pos_from_sfen("4k4/9/6p2/9/9/9/9/9/4K4 b - 1");

        let input_b = extract(&pos_b);
        let input_w = extract(&pos_w);

        // 先手歩の合計は pos_b で 1、pos_w で 0
        let pawn_b_start = BOARD_PIECE_OFFSET + (DensePieceType::Pawn as usize) * 162;
        let pawn_b_black: i32 =
            input_b.piece_features[pawn_b_start..pawn_b_start + 81].iter().sum();
        assert_eq!(pawn_b_black, 1);

        let pawn_w_white_start = pawn_b_start + 81;
        let pawn_w_white: i32 =
            input_w.piece_features[pawn_w_white_start..pawn_w_white_start + 81].iter().sum();
        assert_eq!(pawn_w_white, 1);
    }

    // --------------------------------------------------
    // 4. 手駒テスト
    // --------------------------------------------------
    #[test]
    fn test_hand_thermometer() {
        // 先手: 歩3 金1, 後手: 角1
        let pos = pos_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b 3PGb 1");
        let input = extract(&pos);

        // 先手手駒: P(18)のうち先頭3つが1、残り15は0
        let black_hand_start = HAND_OFFSET;
        for i in 0..3 {
            assert_eq!(input.piece_features[black_hand_start + i], 1, "先手歩thermometer[{i}]");
        }
        for i in 3..18 {
            assert_eq!(input.piece_features[black_hand_start + i], 0, "先手歩thermometer[{i}]");
        }

        // 先手金: offset = 18+4+4+4 = 30 from hand start, 1枚なので先頭1が1
        let gold_offset = black_hand_start + 18 + 4 + 4 + 4;
        assert_eq!(input.piece_features[gold_offset], 1);
        assert_eq!(input.piece_features[gold_offset + 1], 0);

        // 後手手駒: 角1枚 → offset = HAND_OFFSET + 38 + 18 + 4 + 4 + 4 + 4 = +72
        let white_hand_start = HAND_OFFSET + HAND_DIM_PER_COLOR;
        let white_bishop_offset = white_hand_start + 18 + 4 + 4 + 4 + 4;
        assert_eq!(input.piece_features[white_bishop_offset], 1);
        assert_eq!(input.piece_features[white_bishop_offset + 1], 0);
    }

    #[test]
    fn test_hand_thermometer_max_pawn() {
        // 歩18枚持ち（最大）
        let pos = pos_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b 18P 1");
        let input = extract(&pos);

        let black_hand_start = HAND_OFFSET;
        for i in 0..18 {
            assert_eq!(input.piece_features[black_hand_start + i], 1, "歩18枚: thermometer[{i}]");
        }
    }

    #[test]
    fn test_hand_thermometer_encoding() {
        // 直接 encode_hand_thermometer をテスト
        let hand = Hand::EMPTY
            .set(PieceType::Pawn, 5)
            .set(PieceType::Lance, 2)
            .set(PieceType::Rook, 1);
        let mut features = [0i32; FEATURE_DIM];
        encode_hand_thermometer(hand, HAND_OFFSET, &mut features);

        // P(18): 先頭5が1
        for i in 0..5 {
            assert_eq!(features[HAND_OFFSET + i], 1);
        }
        for i in 5..18 {
            assert_eq!(features[HAND_OFFSET + i], 0);
        }

        // L(4): 先頭2が1
        let lance_off = HAND_OFFSET + 18;
        assert_eq!(features[lance_off], 1);
        assert_eq!(features[lance_off + 1], 1);
        assert_eq!(features[lance_off + 2], 0);
        assert_eq!(features[lance_off + 3], 0);

        // R(2): 先頭1が1
        let rook_off = HAND_OFFSET + 18 + 4 + 4 + 4 + 4 + 2;
        assert_eq!(features[rook_off], 1);
        assert_eq!(features[rook_off + 1], 0);
    }

    // --------------------------------------------------
    // 5. 成小統合テスト
    // --------------------------------------------------
    #[test]
    fn test_pro_minor_unified() {
        // 成香・成桂・成銀がすべて ProMinor (idx=8) にマッピングされること
        assert_eq!(to_dense_piece_type(PieceType::ProLance), Some(DensePieceType::ProMinor));
        assert_eq!(to_dense_piece_type(PieceType::ProKnight), Some(DensePieceType::ProMinor));
        assert_eq!(to_dense_piece_type(PieceType::ProSilver), Some(DensePieceType::ProMinor));

        // 盤上に成香と成桂を配置して同じ ProMinor セグメントに入ることを確認
        let pos = pos_from_sfen("4k4/9/9/9/9/9/1+L+N6/9/4K4 b - 1");
        let input = extract(&pos);

        let pro_minor_start = BOARD_PIECE_OFFSET + (DensePieceType::ProMinor as usize) * 162;
        let pro_minor_count: i32 =
            input.piece_features[pro_minor_start..pro_minor_start + 162].iter().sum();
        assert_eq!(pro_minor_count, 2);
    }

    // --------------------------------------------------
    // 6. game_phase テスト
    // --------------------------------------------------
    #[test]
    fn test_game_phase_clamping() {
        // ply=0 → phase=0
        let pos = pos_from_sfen(SFEN_HIRATE);
        let input = extract(&pos);
        // SFEN_HIRATE の game_ply() は 0
        assert!(input.game_phase <= 1);

        // game_phase が features にも反映されていること
        assert_eq!(input.piece_features[PHASE_OFFSET], input.game_phase as i32);
    }

    // --------------------------------------------------
    // 7. king_bucket テスト
    // --------------------------------------------------
    #[test]
    fn test_king_bucket_matches() {
        let pos = pos_from_sfen(SFEN_HIRATE);
        let stm = pos.side_to_move();
        let expected = king_bucket(pos.king_square(stm), stm) as u8;
        let input = extract(&pos);
        assert_eq!(input.king_bucket, expected);
    }

    #[test]
    fn test_king_bucket_range() {
        // king_bucket は 0..44
        let pos = pos_from_sfen(SFEN_HIRATE);
        let input = extract(&pos);
        assert!(input.king_bucket < 45);
    }

    // --------------------------------------------------
    // 8. 全次元が2048であること
    // --------------------------------------------------
    #[test]
    fn test_feature_dim() {
        let pos = pos_from_sfen(SFEN_HIRATE);
        let input = extract(&pos);
        assert_eq!(input.piece_features.len(), 2048);
    }

    // --------------------------------------------------
    // 9. 決定性テスト（同一入力 → 同一出力）
    // --------------------------------------------------
    #[test]
    fn test_deterministic() {
        let pos = pos_from_sfen(SFEN_HIRATE);
        let a = extract(&pos);
        let b = extract(&pos);
        assert_eq!(a.piece_features, b.piece_features);
        assert_eq!(a.side_to_move, b.side_to_move);
        assert_eq!(a.game_phase, b.game_phase);
        assert_eq!(a.king_bucket, b.king_bucket);
    }

    // --------------------------------------------------
    // 10. 複雑な中盤局面テスト
    // --------------------------------------------------
    #[test]
    fn test_complex_position() {
        // YaneuraOu のテスト局面
        let sfen = "8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w Sbgn3p 124";
        let pos = pos_from_sfen(sfen);
        let input = extract(&pos);

        // 後手番
        assert_eq!(input.side_to_move, 1);

        // 盤上に成駒があること (ProPawn, ProMinor (=成+r は Dragon))
        // +R = Dragon, +r = Dragon, +p = ProPawn
        // 局面の駒を数える
        let total_board: i32 =
            input.piece_features[BOARD_PIECE_OFFSET..BOARD_PIECE_DIM].iter().sum();
        // 盤上の玉以外の駒をカウント
        assert!(total_board > 0);

        // 手駒があること
        let hand_sum: i32 = input.piece_features[HAND_OFFSET..HAND_OFFSET + HAND_DIM].iter().sum();
        // S(1) + b(1) + g(1) + n(1) + 3p = 7 相当の thermometer
        assert!(hand_sum > 0);

        // king_bucket が有効範囲
        assert!(input.king_bucket < 45);
    }

    // --------------------------------------------------
    // 11. DensePieceType 全種テスト
    // --------------------------------------------------
    #[test]
    fn test_to_dense_piece_type_all() {
        assert_eq!(to_dense_piece_type(PieceType::Pawn), Some(DensePieceType::Pawn));
        assert_eq!(to_dense_piece_type(PieceType::Lance), Some(DensePieceType::Lance));
        assert_eq!(to_dense_piece_type(PieceType::Knight), Some(DensePieceType::Knight));
        assert_eq!(to_dense_piece_type(PieceType::Silver), Some(DensePieceType::Silver));
        assert_eq!(to_dense_piece_type(PieceType::Gold), Some(DensePieceType::Gold));
        assert_eq!(to_dense_piece_type(PieceType::Bishop), Some(DensePieceType::Bishop));
        assert_eq!(to_dense_piece_type(PieceType::Rook), Some(DensePieceType::Rook));
        assert_eq!(to_dense_piece_type(PieceType::ProPawn), Some(DensePieceType::ProPawn));
        assert_eq!(to_dense_piece_type(PieceType::Horse), Some(DensePieceType::Horse));
        assert_eq!(to_dense_piece_type(PieceType::Dragon), Some(DensePieceType::Dragon));
        assert_eq!(to_dense_piece_type(PieceType::King), None);
    }

    // --------------------------------------------------
    // 12. レイアウト定数の整合性テスト
    // --------------------------------------------------
    #[test]
    fn test_layout_constants() {
        assert_eq!(BOARD_PIECE_OFFSET, 0);
        assert_eq!(BOARD_PIECE_DIM, 1782);
        assert_eq!(BLACK_KING_OFFSET, 1782);
        assert_eq!(WHITE_KING_OFFSET, 1863);
        assert_eq!(SIDE_TO_MOVE_OFFSET, 1944);
        assert_eq!(HAND_OFFSET, 1945);
        assert_eq!(HAND_DIM_PER_COLOR, 38);
        assert_eq!(HAND_DIM, 80);
        assert_eq!(PHASE_OFFSET, 2025);
        assert_eq!(PHASE_OFFSET + 23, FEATURE_DIM);
    }
}
