//! P2 特徴量の型定義

use crate::types::PieceType;

/// 特徴量の総次元数
pub const P2_FEATURE_DIM: usize = 2048;

/// フォーマットバージョン
pub const P2_FORMAT_VERSION: u32 = 1;

// ─── セクション境界 ───
/// 盤上駒セクション開始 index
pub const P2_BOARD_PIECE_START: usize = 0;
/// 盤上駒セクション長: 11 types × 2 colors × 81 squares
pub const P2_BOARD_PIECE_LEN: usize = 11 * 2 * 81; // 1782

/// 先手玉位置セクション開始
pub const P2_BLACK_KING_START: usize = 1782;
/// 後手玉位置セクション開始
pub const P2_WHITE_KING_START: usize = 1863;
/// 玉位置セクション長 (片側)
pub const P2_KING_SQUARE_LEN: usize = 81;

/// 手番セクション index
pub const P2_SIDE_TO_MOVE_INDEX: usize = 1944;

/// 持ち駒バケットセクション開始
pub const P2_HAND_BUCKET_START: usize = 1945;
/// 持ち駒バケットセクション長: 40/side × 2
pub const P2_HAND_BUCKET_LEN: usize = 80;

/// フェーズ/攻撃/予約セクション開始
pub const P2_PHASE_ATTACK_START: usize = 2025;
/// フェーズ/攻撃/予約セクション長
pub const P2_PHASE_ATTACK_LEN: usize = 23;

// コンパイル時検証
const _: () = assert!(
    P2_BOARD_PIECE_LEN
        + P2_KING_SQUARE_LEN * 2
        + 1 // side_to_move
        + P2_HAND_BUCKET_LEN
        + P2_PHASE_ATTACK_LEN
        == P2_FEATURE_DIM
);

/// P2 用の 11 駒種
///
/// 成小（成香・成桂・成銀）を統合して 11 種に圧縮。
/// 玉は含まない（king_square セクションで別管理）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum P2PieceType {
    Pawn = 0,
    Lance = 1,
    Knight = 2,
    Silver = 3,
    Gold = 4,
    Bishop = 5,
    Rook = 6,
    ProPawn = 7,
    ProMinor = 8,
    Horse = 9,
    Dragon = 10,
}

impl P2PieceType {
    /// 駒種数
    pub const NUM: usize = 11;

    /// PieceType → P2PieceType 変換
    ///
    /// # Panics
    /// King を渡した場合 panic する（玉は board_piece に含めない）。
    #[inline]
    pub fn from_piece_type(pt: PieceType) -> P2PieceType {
        match pt {
            PieceType::Pawn => P2PieceType::Pawn,
            PieceType::Lance => P2PieceType::Lance,
            PieceType::Knight => P2PieceType::Knight,
            PieceType::Silver => P2PieceType::Silver,
            PieceType::Gold => P2PieceType::Gold,
            PieceType::Bishop => P2PieceType::Bishop,
            PieceType::Rook => P2PieceType::Rook,
            PieceType::ProPawn => P2PieceType::ProPawn,
            PieceType::ProLance | PieceType::ProKnight | PieceType::ProSilver => {
                P2PieceType::ProMinor
            }
            PieceType::Horse => P2PieceType::Horse,
            PieceType::Dragon => P2PieceType::Dragon,
            PieceType::King => unreachable!("King は board_piece に含めない"),
        }
    }

    /// index (0..10)
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// P2 NNUE 入力構造体
#[repr(C)]
pub struct P2NnueInput {
    /// 2048 次元の特徴量ベクトル
    pub piece_features: [i32; P2_FEATURE_DIM],
    /// 手番 (0=White, 1=Black)
    pub side_to_move: u8,
    /// ゲームフェーズ (0=endgame, 255=opening)
    pub game_phase: u8,
    /// 玉バケット (0..14)
    pub king_bucket: u8,
}

/// 差分更新情報
pub struct P2FeatureDelta {
    /// (index, old_val, new_val) の変更リスト
    pub changes: [(u16, i32, i32); 12],
    /// 有効な変更数
    pub num_changes: u8,
    /// [2025..2047] を full recompute するか
    pub recompute_tail: bool,
}
