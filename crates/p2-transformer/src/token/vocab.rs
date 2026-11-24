//! vocab id 定数と range 定義。
//!
//! vocab size は 512 で固定 (`u16` で表現、 Transformer の embedding table 行数)。
//! 後方互換性を保つため、 既存の id 割当ては変更しない。 新規 token は予約領域
//! ([`SUMMARY_RESERVED_BASE`] 以降の未使用 id) に追加する。
//!
//! # vocab table (id 範囲、 用途)
//!
//! | id range      | 用途                                            | エントリ数 |
//! |---------------|-------------------------------------------------|-----------|
//! | 0             | `PAD` (no-emb sentinel に近いが embedding は持つ) | 1         |
//! | 1             | `CLS` (token[0])                                | 1         |
//! | 2             | `EMPTY` (board 上の空マス)                       | 1         |
//! | 3             | `SUMMARY_RESERVED` (summary 未使用 slot 埋め)    | 1         |
//! | 4..8          | special 予約                                     | 4         |
//! | 8..22         | Black 駒 (Pawn..Dragon、 PieceType 順)          | 14        |
//! | 22..36        | White 駒 (Pawn..Dragon、 PieceType 順)          | 14        |
//! | 36..117       | `KING_B_<sq>` (sq=0..80)                        | 81        |
//! | 117..198      | `KING_W_<sq>` (sq=0..80)                        | 81        |
//! | 198..243      | Black hand count-aware (歩 19 / 香桂銀金 5×4 / 角飛 3×2) | 45 |
//! | 243..288      | White hand count-aware (同上)                   | 45        |
//! | 288..290      | `SIDE_BLACK`、 `SIDE_WHITE`                     | 2         |
//! | 290..293      | `PHASE_OPEN / MIDDLE / END`                     | 3         |
//! | 293..310      | `MATERIAL_BUCKET[-8..+8]` (17 段)                | 17        |
//! | 310..326      | `PLY_BUCKET[0..15]` (16 段)                     | 16        |
//! | 326..328      | `CHK_NO`、 `CHK_YES`                            | 2         |
//! | 328..512      | 将来拡張用 (king safety / attack pressure 等)    | 184       |
//!
//! 合計 使用済み: 328、 余裕: 184。

use rshogi_core::types::{Color, PieceType};

/// Transformer 入力の固定 token 長。
pub const MAX_TOKENS: usize = 128;

/// vocab size (embedding table 行数の上限)。
pub const VOCAB_SIZE: usize = 512;

/// dlshogi `make_move_label` 互換 policy 出力次元。
pub const POLICY_SIZE: usize = 2187;

// ============================================================
// Special tokens
// ============================================================

/// Padding token。 token[116..128) や 未使用 hand slot を埋める。
pub const PAD: u16 = 0;
/// CLS token。 token[0] に固定配置。
pub const CLS: u16 = 1;
/// 空マス token (board の空升)。
pub const EMPTY: u16 = 2;
/// Summary slot 予約値 (token[103..116) 等の未使用 summary slot)。
pub const SUMMARY_RESERVED: u16 = 3;

// ============================================================
// Board piece tokens (Black 8..22、 White 22..36)
// ============================================================

/// Black 駒 token base。 PieceType の生 u8 (1..=14) を加算した id を持つ。
/// すなわち `BLACK_PIECE_BASE + (pt as u16 - 1)`。
pub const BLACK_PIECE_BASE: u16 = 8;
/// White 駒 token base。
pub const WHITE_PIECE_BASE: u16 = 22;

/// 駒種ごとの vocab id を返す。
///
/// `PieceType::Pawn` (=1) が `_BASE + 0` に対応する。
#[inline]
pub const fn piece_vocab(color: Color, pt: PieceType) -> u16 {
    let base = match color {
        Color::Black => BLACK_PIECE_BASE,
        Color::White => WHITE_PIECE_BASE,
    };
    base + (pt as u16 - 1)
}

// ============================================================
// King tokens (KING_B_<sq>: 36..117、 KING_W_<sq>: 117..198)
// ============================================================

/// `KING_B_<sq>` の base。 `KING_B_BASE + sq` で id が得られる (sq=0..81)。
pub const KING_B_BASE: u16 = 36;
/// `KING_W_<sq>` の base。
pub const KING_W_BASE: u16 = 117;

#[inline]
pub const fn king_vocab(color: Color, sq_idx: u8) -> u16 {
    debug_assert!(sq_idx < 81);
    let base = match color {
        Color::Black => KING_B_BASE,
        Color::White => KING_W_BASE,
    };
    base + sq_idx as u16
}

// ============================================================
// Hand count-aware tokens (Black 198..243、 White 243..288)
// ============================================================

/// 駒種ごとの最大手駒枚数 (HAND_PIECES の順番に対応)。
///
/// 順番: Pawn, Lance, Knight, Silver, Gold, Bishop, Rook
pub const HAND_MAX_COUNT: [u8; 7] = [18, 4, 4, 4, 4, 2, 2];

/// 各駒種のエントリ数 (`max + 1`、 0 枚も含む)。
pub const HAND_ENTRY_COUNT: [u8; 7] = [19, 5, 5, 5, 5, 3, 3];

/// 1 色あたりの hand vocab entry 数 (19+5+5+5+5+3+3 = 45)。
pub const HAND_ENTRIES_PER_COLOR: u16 = 45;

/// 駒種ごとの累積 offset (HAND_PIECES の順)。
///
/// `HAND_OFFSET[i] = sum(HAND_ENTRY_COUNT[0..i])`。
pub const HAND_OFFSET: [u16; 7] = [0, 19, 24, 29, 34, 39, 42];

/// Black hand vocab base。
pub const HAND_B_BASE: u16 = 198;
/// White hand vocab base。
pub const HAND_W_BASE: u16 = HAND_B_BASE + HAND_ENTRIES_PER_COLOR; // 243

/// 駒種を HAND_PIECES の index に変換する。
///
/// 手駒になりうる駒種以外 (King / 成駒) が渡された場合は `None` を返す。
#[inline]
pub const fn hand_piece_index(pt: PieceType) -> Option<usize> {
    match pt {
        PieceType::Pawn => Some(0),
        PieceType::Lance => Some(1),
        PieceType::Knight => Some(2),
        PieceType::Silver => Some(3),
        PieceType::Gold => Some(4),
        PieceType::Bishop => Some(5),
        PieceType::Rook => Some(6),
        _ => None,
    }
}

/// Hand count-aware vocab id を返す。
///
/// `count` は対応する駒種の最大値以下を想定する (内部で clamp)。
#[inline]
pub fn hand_vocab(color: Color, pt: PieceType, count: u8) -> u16 {
    let Some(idx) = hand_piece_index(pt) else {
        return PAD;
    };
    let max = HAND_MAX_COUNT[idx];
    let clamped = if count > max { max } else { count };
    let base = match color {
        Color::Black => HAND_B_BASE,
        Color::White => HAND_W_BASE,
    };
    base + HAND_OFFSET[idx] + clamped as u16
}

// ============================================================
// Summary tokens (288..328)
// ============================================================

/// 手番 token base (`SIDE_BLACK` = base、 `SIDE_WHITE` = base+1)。
pub const SIDE_BASE: u16 = 288;
pub const SIDE_BLACK: u16 = SIDE_BASE;
pub const SIDE_WHITE: u16 = SIDE_BASE + 1;

/// 局面進行度 phase token base。
pub const PHASE_BASE: u16 = 290;
pub const PHASE_OPEN: u16 = PHASE_BASE;
pub const PHASE_MIDDLE: u16 = PHASE_BASE + 1;
pub const PHASE_END: u16 = PHASE_BASE + 2;

/// material balance bucket (17 段、 index 0 = -8、 index 16 = +8)。
pub const MATERIAL_BASE: u16 = 293;
pub const MATERIAL_BUCKET_COUNT: u16 = 17;

/// ply bucket (16 段)。
pub const PLY_BASE: u16 = 310;
pub const PLY_BUCKET_COUNT: u16 = 16;

/// 王手有無 token。
pub const CHECK_BASE: u16 = 326;
pub const CHK_NO: u16 = CHECK_BASE;
pub const CHK_YES: u16 = CHECK_BASE + 1;

/// 将来拡張用予約領域の base id。 新規 token は 328..512 から割り当てる。
pub const SUMMARY_RESERVED_BASE: u16 = 328;

// ============================================================
// Sentinel 値 (lookup id 用)
// ============================================================

/// square index の sentinel (no square emb)。
pub const SQUARE_NONE: u8 = u8::MAX;
/// promotion zone index の sentinel (no zone emb)。
pub const ZONE_NONE: u8 = u8::MAX;
/// king-relative index の sentinel (no king-relative emb)。
pub const KING_REL_NONE: u16 = u16::MAX;

/// promotion zone の bucket 数 (0=自陣 / 1=中段 / 2=敵陣)。
/// Black 視点で rank 1-3 = 敵陣 (2)、 rank 4-6 = 中段 (1)、 rank 7-9 = 自陣 (0)。
pub const ZONE_COUNT: u8 = 3;

/// king-relative bucket 数 (17 × 17 = 289)。
pub const KING_REL_BUCKET_COUNT: u16 = 17 * 17;

// ============================================================
// token slot 配置
// ============================================================

/// CLS slot index。
pub const SLOT_CLS: usize = 0;
/// board slot の開始 index。 length は 81。
pub const SLOT_BOARD_BEGIN: usize = 1;
pub const SLOT_BOARD_END: usize = SLOT_BOARD_BEGIN + 81; // 82
/// hand slot の開始 index。 length は 14 (Black 7 + White 7)。
pub const SLOT_HAND_BEGIN: usize = SLOT_BOARD_END; // 82
pub const SLOT_HAND_END: usize = SLOT_HAND_BEGIN + 14; // 96
/// king token slot index (Black, White)。
pub const SLOT_KING_BLACK: usize = SLOT_HAND_END; // 96
pub const SLOT_KING_WHITE: usize = SLOT_HAND_END + 1; // 97
/// summary slot の開始 index。 length は 18。
pub const SLOT_SUMMARY_BEGIN: usize = SLOT_HAND_END + 2; // 98
pub const SLOT_SUMMARY_END: usize = SLOT_SUMMARY_BEGIN + 18; // 116
/// reserved/padding slot 開始 index。 length は 12。
pub const SLOT_PADDING_BEGIN: usize = SLOT_SUMMARY_END; // 116

// summary slot 内の意味づけ
pub const SUMMARY_SLOT_SIDE: usize = 0;
pub const SUMMARY_SLOT_PHASE: usize = 1;
pub const SUMMARY_SLOT_MATERIAL: usize = 2;
pub const SUMMARY_SLOT_PLY: usize = 3;
pub const SUMMARY_SLOT_CHECK: usize = 4;
// SLOT_SUMMARY_BEGIN + 5..18 は SUMMARY_RESERVED で埋める (13 entries)。

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_contiguous_and_within_vocab() {
        // Black/White piece tokens
        assert_eq!(piece_vocab(Color::Black, PieceType::Pawn), BLACK_PIECE_BASE);
        assert_eq!(piece_vocab(Color::Black, PieceType::Dragon), BLACK_PIECE_BASE + 13);
        assert_eq!(piece_vocab(Color::White, PieceType::Pawn), WHITE_PIECE_BASE);
        assert_eq!(piece_vocab(Color::White, PieceType::Dragon), WHITE_PIECE_BASE + 13);
        assert!(piece_vocab(Color::White, PieceType::Dragon) < KING_B_BASE);

        // King tokens
        assert_eq!(king_vocab(Color::Black, 0), KING_B_BASE);
        assert_eq!(king_vocab(Color::Black, 80), KING_B_BASE + 80);
        assert_eq!(king_vocab(Color::White, 0), KING_W_BASE);
        assert_eq!(king_vocab(Color::White, 80), KING_W_BASE + 80);
        assert!(king_vocab(Color::White, 80) < HAND_B_BASE);

        // Hand tokens: 最終 entry (White Rook = 2)
        let last_hand = hand_vocab(Color::White, PieceType::Rook, 2);
        assert_eq!(last_hand, HAND_W_BASE + 44);
        assert!(last_hand < SIDE_BASE);

        // Summary tokens
        assert_eq!(SIDE_BLACK, 288);
        assert_eq!(PHASE_END, 292);
        assert_eq!(MATERIAL_BASE + MATERIAL_BUCKET_COUNT, 310);
        assert_eq!(PLY_BASE + PLY_BUCKET_COUNT, 326);
        assert_eq!(CHK_YES, 327);

        // Reserved
        assert_eq!(SUMMARY_RESERVED_BASE, 328);
        assert!((VOCAB_SIZE as u16) >= SUMMARY_RESERVED_BASE);
    }

    #[test]
    fn slot_layout_matches_spec() {
        assert_eq!(SLOT_CLS, 0);
        assert_eq!(SLOT_BOARD_BEGIN, 1);
        assert_eq!(SLOT_BOARD_END, 82);
        assert_eq!(SLOT_HAND_BEGIN, 82);
        assert_eq!(SLOT_HAND_END, 96);
        assert_eq!(SLOT_KING_BLACK, 96);
        assert_eq!(SLOT_KING_WHITE, 97);
        assert_eq!(SLOT_SUMMARY_BEGIN, 98);
        assert_eq!(SLOT_SUMMARY_END, 116);
        assert_eq!(SLOT_PADDING_BEGIN, 116);
        assert_eq!(SLOT_PADDING_BEGIN + 12, MAX_TOKENS);
    }
}
