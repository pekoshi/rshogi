//! P2 NNUE 入力特徴量の型定義 (Stage 2 V3: dlshogi-style features)
//!
//! V3 では dlshogi 互換の入力構造 (features1=5022 + features2=4617) を採用し、
//! 末尾を padding して 32 byte alignment が取れる 9728 次元にする。

use rshogi_core::types::PieceType;

// ─────────────────────────────────────────────────────────────
// dlshogi 互換 features ブロック定数
// ─────────────────────────────────────────────────────────────

pub const SQUARE_NUM: usize = 81;

/// features1 の駒種数 (歩〜竜の 14 種)
pub const PIECETYPE_NUM: usize = 14;

/// 利き数の最大カウント (3 以上は飽和)
pub const MAX_ATTACK_NUM: usize = 3;

/// features1 の 1 色あたり ch 数: 14 (配置) + 14 (利き) + 3 (利き数) = 31
pub const FEATURES1_CHANNELS_PER_COLOR: usize = PIECETYPE_NUM + PIECETYPE_NUM + MAX_ATTACK_NUM;

/// features1 全体: 2 色 × 31 ch × 81 sq = 5022
pub const FEATURES1_SIZE: usize = 2 * FEATURES1_CHANNELS_PER_COLOR * SQUARE_NUM;

/// 持ち駒の最大枚数 (各種)
pub const MAX_HPAWN_NUM: usize = 8;
pub const MAX_HLANCE_NUM: usize = 4;
pub const MAX_HKNIGHT_NUM: usize = 4;
pub const MAX_HSILVER_NUM: usize = 4;
pub const MAX_HGOLD_NUM: usize = 4;
pub const MAX_HBISHOP_NUM: usize = 2;
pub const MAX_HROOK_NUM: usize = 2;

/// 1 色の持ち駒合計 ch: 8+4+4+4+4+2+2 = 28
pub const MAX_PIECES_IN_HAND_SUM: usize = MAX_HPAWN_NUM
    + MAX_HLANCE_NUM
    + MAX_HKNIGHT_NUM
    + MAX_HSILVER_NUM
    + MAX_HGOLD_NUM
    + MAX_HBISHOP_NUM
    + MAX_HROOK_NUM;

/// features2 の持ち駒部分: 2 色 × 28 = 56 ch
pub const MAX_FEATURES2_HAND_NUM: usize = 2 * MAX_PIECES_IN_HAND_SUM;

/// features2 全体 ch: 持ち駒 56 + 王手 1 = 57
pub const FEATURES2_CHANNELS: usize = MAX_FEATURES2_HAND_NUM + 1;

/// features2 全体: 57 × 81 = 4617
pub const FEATURES2_SIZE: usize = FEATURES2_CHANNELS * SQUARE_NUM;

/// dlshogi 互換 features 合計: 9639
pub const DLSHOGI_FEATURES_SIZE: usize = FEATURES1_SIZE + FEATURES2_SIZE;

// ─────────────────────────────────────────────────────────────
// P2-NNUE 入力次元 (32 byte alignment)
// ─────────────────────────────────────────────────────────────

/// 学習・推論で使う入力次元
///
/// 9639 + 89 padding = 9728 (= 32 × 304、AVX2 32 byte alignment 完全)。
pub const P2_FEATURE_DIM: usize = 9728;

/// padding ch 数 (= P2_FEATURE_DIM - DLSHOGI_FEATURES_SIZE)
pub const P2_FEATURE_PADDING: usize = P2_FEATURE_DIM - DLSHOGI_FEATURES_SIZE;

/// フォーマットバージョン (V3)
pub const P2_FORMAT_VERSION: u32 = 3;

/// features1 開始 index (0)
pub const FEATURES1_START: usize = 0;

/// features2 開始 index (= FEATURES1_SIZE)
pub const FEATURES2_START: usize = FEATURES1_SIZE;

/// 持ち駒種別ごとのオフセット (features2 内、片色)
pub const HAND_OFFSETS: [usize; 7] = [
    0,
    MAX_HPAWN_NUM,
    MAX_HPAWN_NUM + MAX_HLANCE_NUM,
    MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM,
    MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM,
    MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM + MAX_HGOLD_NUM,
    MAX_HPAWN_NUM
        + MAX_HLANCE_NUM
        + MAX_HKNIGHT_NUM
        + MAX_HSILVER_NUM
        + MAX_HGOLD_NUM
        + MAX_HBISHOP_NUM,
];

pub const HAND_MAXES: [usize; 7] = [
    MAX_HPAWN_NUM,
    MAX_HLANCE_NUM,
    MAX_HKNIGHT_NUM,
    MAX_HSILVER_NUM,
    MAX_HGOLD_NUM,
    MAX_HBISHOP_NUM,
    MAX_HROOK_NUM,
];

// コンパイル時 alignment 検証
const _: () = assert!(P2_FEATURE_DIM.is_multiple_of(32), "P2_FEATURE_DIM must be 32-aligned");
const _: () = assert!(P2_FEATURE_DIM >= DLSHOGI_FEATURES_SIZE);

// ─────────────────────────────────────────────────────────────
// P2NnueInput
// ─────────────────────────────────────────────────────────────

/// P2 NNUE 入力構造体
///
/// `piece_features` は 9728 次元の i32 binary (0/1)。 38KB なので Box で heap alloc。
pub struct P2NnueInput {
    pub piece_features: Box<[i32; P2_FEATURE_DIM]>,
    /// 手番 (0=White, 1=Black)
    pub side_to_move: u8,
    /// ゲームフェーズ (0=endgame, 255=opening)
    pub game_phase: u8,
    /// 玉バケット (0..14)
    pub king_bucket: u8,
}

impl P2NnueInput {
    /// 全ゼロ初期化
    pub fn new_zeroed() -> Self {
        Self {
            piece_features: Box::new([0i32; P2_FEATURE_DIM]),
            side_to_move: 1,
            game_phase: 128,
            king_bucket: 0,
        }
    }
}

/// 差分更新情報 (V3 では disabled)
///
/// dlshogi-style features は王の移動・駒の利きで広範囲に影響するため、
/// incremental 更新コストが大きい。 V3 では毎手 fresh 抽出する。
/// 互換性のため型定義は残すが、 関連関数は no-op になる。
pub struct P2FeatureDelta {
    pub recompute_full: bool,
}

impl Default for P2FeatureDelta {
    fn default() -> Self {
        Self {
            recompute_full: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// PieceType ヘルパ (dlshogi 抽出用)
// ─────────────────────────────────────────────────────────────

/// PieceType → 0-indexed 駒種 index (歩=0..竜=13)
#[inline]
pub fn piece_type_to_dlshogi_index(pt: PieceType) -> usize {
    pt as usize - 1
}
