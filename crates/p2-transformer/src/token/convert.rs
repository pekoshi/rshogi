//! `Position` → [`TransformerTokens`] 変換 (allocation 0、 branch-minimised)。
//!
//! 本モジュールは Transformer 学習・推論時の hot path。 計測なしの最適化は
//! 行わないが、 設計上以下の保証を満たす:
//!
//! - **ホットパスで heap allocation を一切行わない** (stack 上で構築)。
//! - **dynamic dispatch なし** (trait object / Box<dyn Trait> 不使用)。
//! - **branch は閉じた範囲のみ** (81 squares のループ内で 1 段)。
//! - **連続 memory write** (5 つの SoA 配列を順に埋める)。
//! - **SIMD friendly** (各配列は単一型の固定長 array で、 cache line を踏まない)。
//!
//! ## Embedding 加算の前提
//!
//! Transformer 側では 1 layer 目の embedding を以下のように構築する:
//!
//! ```text
//! x[i] = piece_emb[tokens[i]]
//!      + square_emb[squares[i]]            (squares[i] == SQUARE_NONE なら zero)
//!      + own_king_rel_emb[own_king_rel[i]] (sentinel なら zero)
//!      + enemy_king_rel_emb[enemy_king_rel[i]] (sentinel なら zero)
//!      + zone_emb[zones[i]]                (sentinel なら zero)
//! ```
//!
//! sentinel 行 (各 emb table の最終 row) を学習時に zero vector に固定すれば
//! masking 不要で「embedding なし」を表現できる。

use rshogi_core::position::Position;
use rshogi_core::types::{Color, PieceType, Square};

use super::board::piece_vocab_id;
use super::hand::write_hand_tokens;
use super::king_relative::{king_relative_index_clamped, promotion_zone};
use super::summary::{check_token, material_bucket, phase_token, ply_bucket, side_token};
use super::vocab::{
    CLS, KING_REL_NONE, MAX_TOKENS, PAD, SLOT_BOARD_BEGIN, SLOT_BOARD_END, SLOT_CLS,
    SLOT_HAND_BEGIN, SLOT_HAND_END, SLOT_KING_BLACK, SLOT_KING_WHITE, SLOT_PADDING_BEGIN,
    SLOT_SUMMARY_BEGIN, SQUARE_NONE, SUMMARY_RESERVED, SUMMARY_SLOT_CHECK, SUMMARY_SLOT_MATERIAL,
    SUMMARY_SLOT_PHASE, SUMMARY_SLOT_PLY, SUMMARY_SLOT_SIDE, ZONE_NONE, king_vocab,
};

/// Transformer 入力 token (SoA + repr(C) で SIMD / CUDA 転送 friendly)。
///
/// 全配列の長さは [`MAX_TOKENS`] (= 128) で固定。 配列はそれぞれ:
///
/// - [`tokens`]: piece embedding 用 vocab id (0..[`VOCAB_SIZE`])。
/// - [`squares`]: board square index (0..81)、 sentinel = [`SQUARE_NONE`] (255)。
/// - [`own_king_rel`]: own king-relative bucket (0..289)、 sentinel = [`KING_REL_NONE`] (u16::MAX)。
/// - [`enemy_king_rel`]: enemy king-relative bucket、 sentinel 同上。
/// - [`zones`]: promotion zone (0..3)、 sentinel = [`ZONE_NONE`] (255)。
///
/// [`tokens`]: TransformerTokens::tokens
/// [`squares`]: TransformerTokens::squares
/// [`own_king_rel`]: TransformerTokens::own_king_rel
/// [`enemy_king_rel`]: TransformerTokens::enemy_king_rel
/// [`zones`]: TransformerTokens::zones
/// [`VOCAB_SIZE`]: super::vocab::VOCAB_SIZE
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TransformerTokens {
    /// piece embedding 用 vocab id (0..VOCAB_SIZE)。
    pub tokens: [u16; MAX_TOKENS],
    /// square embedding 用 (0..81、 sentinel = `SQUARE_NONE`)。
    pub squares: [u8; MAX_TOKENS],
    /// own king-relative index (0..289、 sentinel = `KING_REL_NONE`)。
    pub own_king_rel: [u16; MAX_TOKENS],
    /// enemy king-relative index (0..289、 sentinel = `KING_REL_NONE`)。
    pub enemy_king_rel: [u16; MAX_TOKENS],
    /// promotion zone (0..3、 sentinel = `ZONE_NONE`)。
    pub zones: [u8; MAX_TOKENS],
}

impl TransformerTokens {
    /// 全 token を PAD / sentinel で初期化した値を返す。
    #[inline]
    pub const fn empty() -> Self {
        TransformerTokens {
            tokens: [PAD; MAX_TOKENS],
            squares: [SQUARE_NONE; MAX_TOKENS],
            own_king_rel: [KING_REL_NONE; MAX_TOKENS],
            enemy_king_rel: [KING_REL_NONE; MAX_TOKENS],
            zones: [ZONE_NONE; MAX_TOKENS],
        }
    }
}

impl Default for TransformerTokens {
    fn default() -> Self {
        Self::empty()
    }
}

/// `Position` を Transformer 入力 token 列に変換する。
///
/// allocation 0 (stack のみ)、 branch-minimised。
///
/// ## token layout (再掲)
///
/// - `tokens[0]` = CLS
/// - `tokens[1..82)` = board (sq 0..81 を順に走査、 駒があれば piece vocab、 空マスは EMPTY)
/// - `tokens[82..96)` = hand (Black 7 駒種 + White 7 駒種)
/// - `tokens[96]` = `KING_B_<sq>`
/// - `tokens[97]` = `KING_W_<sq>`
/// - `tokens[98..116)` = summary (side / phase / material / ply / check / 13 reserved)
/// - `tokens[116..128)` = PAD
#[inline]
pub fn position_to_tokens(pos: &Position) -> TransformerTokens {
    let mut t = TransformerTokens::empty();

    // ============================================================
    // CLS slot
    // ============================================================
    t.tokens[SLOT_CLS] = CLS;
    // squares / king_rel / zones は sentinel のまま (Self::empty で初期化済み)。

    // ============================================================
    // king square を最初に確定 (board ループの相対座標で参照する)
    // ============================================================
    let bk_sq = pos.king_square(Color::Black);
    let wk_sq = pos.king_square(Color::White);
    let stm = pos.side_to_move();
    let (own_king_sq, enemy_king_sq) = match stm {
        Color::Black => (bk_sq, wk_sq),
        Color::White => (wk_sq, bk_sq),
    };
    let stm_idx: u8 = stm.index() as u8;

    // ============================================================
    // board slot (sq 0..81)
    // ============================================================
    // Square は file*9 + rank で encode されており、 (0..81) で 81 個。
    // PieceList を介さず、 board をそのまま走査する (cache locality 重視)。
    let mut sq_idx = 0u8;
    while sq_idx < 81 {
        // SAFETY: sq_idx < 81 で from_u8 を呼ぶため、 必ず Some を返す。
        // パニック回避のため Option::unwrap ではなく from_u8_unchecked を使う。
        // ただし debug_assert 経由で範囲を再確認。
        let sq = unsafe { Square::from_u8_unchecked(sq_idx) };
        let piece = pos.piece_on(sq);
        let slot = SLOT_BOARD_BEGIN + sq_idx as usize;

        t.tokens[slot] = piece_vocab_id(piece);
        t.squares[slot] = sq_idx;
        t.own_king_rel[slot] = king_relative_index_clamped(sq, own_king_sq);
        t.enemy_king_rel[slot] = king_relative_index_clamped(sq, enemy_king_sq);
        t.zones[slot] = promotion_zone(sq, stm_idx);

        sq_idx += 1;
    }
    debug_assert_eq!(SLOT_BOARD_BEGIN + 81, SLOT_BOARD_END);

    // ============================================================
    // hand slot (Black 7 + White 7 = 14)
    // ============================================================
    let mut hand_buf = [PAD; 14];
    write_hand_tokens(pos, &mut hand_buf);
    // squares / king_rel / zones は sentinel のまま (hand は盤上 sq を持たない)。
    let mut i = 0usize;
    while i < 14 {
        t.tokens[SLOT_HAND_BEGIN + i] = hand_buf[i];
        i += 1;
    }
    debug_assert_eq!(SLOT_HAND_BEGIN + 14, SLOT_HAND_END);

    // ============================================================
    // king tokens (token[96] = Black king、 token[97] = White king)
    // ============================================================
    let bk_idx = bk_sq.raw();
    let wk_idx = wk_sq.raw();
    t.tokens[SLOT_KING_BLACK] = king_vocab(Color::Black, bk_idx);
    t.squares[SLOT_KING_BLACK] = bk_idx;
    // king-relative は手番視点 (own/enemy) で計算。 stm=Black の場合 black_king が own、
    // stm=White の場合 black_king は enemy。 own_king_sq / enemy_king_sq に既に反映済み
    // のため、 dx/dy は両 slot で同一式で求まる。
    t.own_king_rel[SLOT_KING_BLACK] = king_relative_index_clamped(bk_sq, own_king_sq);
    t.enemy_king_rel[SLOT_KING_BLACK] = king_relative_index_clamped(bk_sq, enemy_king_sq);
    t.zones[SLOT_KING_BLACK] = promotion_zone(bk_sq, stm_idx);

    t.tokens[SLOT_KING_WHITE] = king_vocab(Color::White, wk_idx);
    t.squares[SLOT_KING_WHITE] = wk_idx;
    t.own_king_rel[SLOT_KING_WHITE] = king_relative_index_clamped(wk_sq, own_king_sq);
    t.enemy_king_rel[SLOT_KING_WHITE] = king_relative_index_clamped(wk_sq, enemy_king_sq);
    t.zones[SLOT_KING_WHITE] = promotion_zone(wk_sq, stm_idx);

    // ============================================================
    // summary slot (98..116)
    // ============================================================
    let ply = pos.game_ply();
    let total_hand_count = total_hand_pieces(pos);

    let summary_base = SLOT_SUMMARY_BEGIN;
    t.tokens[summary_base + SUMMARY_SLOT_SIDE] = side_token(stm);
    t.tokens[summary_base + SUMMARY_SLOT_PHASE] = phase_token(ply, total_hand_count);
    t.tokens[summary_base + SUMMARY_SLOT_MATERIAL] =
        material_bucket(simple_material_score_black_pov(pos));
    t.tokens[summary_base + SUMMARY_SLOT_PLY] = ply_bucket(ply);
    t.tokens[summary_base + SUMMARY_SLOT_CHECK] = check_token(pos.in_check());
    // 残り 13 slot は SUMMARY_RESERVED で埋める (将来拡張用)。
    let mut j = SUMMARY_SLOT_CHECK + 1;
    while j < 18 {
        t.tokens[summary_base + j] = SUMMARY_RESERVED;
        j += 1;
    }

    // ============================================================
    // padding (116..128) はそのまま PAD (empty 初期化)。
    // ============================================================
    debug_assert_eq!(SLOT_PADDING_BEGIN + 12, MAX_TOKENS);
    // 念のため明示的に再設定 (compiler に最適化させる)
    let mut k = SLOT_PADDING_BEGIN;
    while k < MAX_TOKENS {
        t.tokens[k] = PAD;
        k += 1;
    }

    t
}

/// 手駒枚数の合計 (Black + White)。 phase 判定で使用。
#[inline]
fn total_hand_pieces(pos: &Position) -> u32 {
    let mut total = 0u32;
    for color in [Color::Black, Color::White] {
        let hand = pos.hand(color);
        for pt in [
            PieceType::Pawn,
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::Bishop,
            PieceType::Rook,
        ] {
            total += hand.count(pt);
        }
    }
    total
}

/// 簡易 material score (Black 視点、 cp)。 summary token 用の粗い指標。
///
/// 駒価値は概算 (Pawn=100, Lance=300, Knight=300, Silver=500, Gold=600,
/// Bishop=800, Rook=1000、 成駒は +200)。 King は除外。
#[inline]
fn simple_material_score_black_pov(pos: &Position) -> i32 {
    let mut score = 0i32;
    // 盤上駒
    let mut sq_idx = 0u8;
    while sq_idx < 81 {
        // SAFETY: sq_idx < 81。
        let sq = unsafe { Square::from_u8_unchecked(sq_idx) };
        let pc = pos.piece_on(sq);
        if pc.is_some() {
            let val = piece_value_cp(pc.piece_type());
            score += if pc.color() == Color::Black {
                val
            } else {
                -val
            };
        }
        sq_idx += 1;
    }
    // 手駒 (King は手駒にならないので除外不要)
    for color in [Color::Black, Color::White] {
        let hand = pos.hand(color);
        for pt in [
            PieceType::Pawn,
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::Bishop,
            PieceType::Rook,
        ] {
            let count = hand.count(pt) as i32;
            let val = piece_value_cp(pt) * count;
            score += if color == Color::Black { val } else { -val };
        }
    }
    score
}

#[inline]
const fn piece_value_cp(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => 100,
        PieceType::Lance => 300,
        PieceType::Knight => 300,
        PieceType::Silver => 500,
        PieceType::Gold => 600,
        PieceType::Bishop => 800,
        PieceType::Rook => 1000,
        PieceType::King => 0,
        PieceType::ProPawn => 300,
        PieceType::ProLance => 500,
        PieceType::ProKnight => 500,
        PieceType::ProSilver => 600,
        PieceType::Horse => 1000,
        PieceType::Dragon => 1200,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::position::Position;

    fn startpos() -> Position {
        let mut p = Position::new();
        p.set_hirate();
        p
    }

    #[test]
    fn startpos_layout_smoke() {
        let pos = startpos();
        let t = position_to_tokens(&pos);
        // CLS
        assert_eq!(t.tokens[SLOT_CLS], CLS);
        // board: 81 個分のうち、 startpos では 40 駒が存在 (各色 20 駒)。
        let board_filled = (SLOT_BOARD_BEGIN..SLOT_BOARD_END)
            .filter(|&i| t.tokens[i] != super::super::vocab::EMPTY)
            .count();
        assert_eq!(board_filled, 40);
        // hand: 全 14 slot は count=0 の vocab id
        for i in SLOT_HAND_BEGIN..SLOT_HAND_END {
            assert!(t.tokens[i] >= super::super::vocab::HAND_B_BASE);
            assert!(t.tokens[i] < super::super::vocab::SIDE_BASE);
        }
        // king
        assert!(t.tokens[SLOT_KING_BLACK] >= super::super::vocab::KING_B_BASE);
        assert!(t.tokens[SLOT_KING_BLACK] < super::super::vocab::KING_W_BASE);
        assert!(t.tokens[SLOT_KING_WHITE] >= super::super::vocab::KING_W_BASE);
        assert!(t.tokens[SLOT_KING_WHITE] < super::super::vocab::HAND_B_BASE);
        // padding
        for i in SLOT_PADDING_BEGIN..MAX_TOKENS {
            assert_eq!(t.tokens[i], PAD);
        }
    }
}
