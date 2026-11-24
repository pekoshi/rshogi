//! KingRing Threat PSQT scalar head (arm 11、 2026-05-28、 revert 後)。
//!
//! 玉周辺 24 sq (= 5×5 zone から玉自身を除く) × **13 piece** × **4 state** ×
//! 8 bucket × 2 persp = **19,968 i16 weight** (~40 KB)。 PSQT scalar head として加算する。
//!
//! # 設計 (= a37d arm 9 仕様、 bullet 側 `P2KRING\0` + 19,968 dim と整合)
//!
//! ## feature 集合
//!
//! - ring 24 sq (= 5x5 zone − king 自身、 perspective swap 対応)
//! - 各 sq の 4 state:
//!   - state 0 = empty sq + **own (= perspective) 側 smallest attacker** あり
//!   - state 1 = own piece on sq (= 自軍駒で守られている、 = 守り)
//!   - state 2 = enemy piece on sq (= 敵駒に占拠されている、 = 圧力)
//!   - state 3 = empty sq + **enemy 側 smallest attacker** あり (= king-zone pressure)
//! - 各 active entry の piece_idx = 0..12 (= Pawn..Dragon 13 種、 King 除外)
//!
//! ## hot path 最適化 (= 2026-05-29 追加: ring table precompute + ring_idx 固定 slot + runtime bucket-inner i32)
//!
//! - `classify_sq` 内で `attackers_to_occ` を **1 回** だけ呼ぶ (= occupied bb 引数経由)、
//!   own/enemy 分割は bitboard mask で派生 (= sliding effect 2 回呼びを 1 回に圧縮)。
//! - smallest attacker は `pieces_pt(pt)` intersection の `.is_empty()` で piece type 順
//!   早期 return (= 13 piece × 1 cmp、 hot path で 81 sq loop は走らせない)。
//! - `Square::from_u8_unchecked` で hot loop の bounds check / panic を除去。
//! - **ring table precompute (= 候補 4)**: `KINGRING_TABLE[persp][king_sq][24] -> u8` を
//!   `LazyLock` で構築 (= 81 × 2 × 24 = 3,888 byte)。 hot path で `Option<Square>` 構築を
//!   完全に排除、 単 byte load + sentinel cmp で ring sq を取得する。
//! - **ring_idx 固定 slot 化 (= 候補 1)**: active list を「compact + len」 ではなく、
//!   `[u16; 24]` の **ring_idx 固定 slot** で保持 (= SENTINEL = u16::MAX で空 marker)。
//!   diff は 24 slot を ring_idx で順次 cmp するだけ = removed/added buffer 不要、
//!   sort 前提も不要、 active_len も不要。 sq が動かないので old vs new で 1 対 1 比較。
//! - **runtime weight bucket-inner 展開 (= 候補 3)**: disk format は P2KRING + 19,968 dim を
//!   維持しつつ、 読み込み後に runtime 用に **転置** して `[[[i32; 8]; 1248]; 2]` の
//!   bucket-inner i32 配置を構築 (= ~80 KB / persp、 計 ~160 KB)。 hot path で
//!   `row[bucket]` の **8 連続 read** が可能、 raw_idx 分解の除算/剰余と i16→i32 cast を
//!   削除。 disk shape (= `KingRingWeights`) は ckpt 互換性維持のため残し、 既存 test も
//!   そのまま動かす (= disk shape 経路は legacy / test 用)。
//!
//! ## ckpt format (= disk layout、 変更なし)
//!
//! sub_magic = `"P2KRING\0"` (= 8 byte、 bullet `P2KRING\0` と一致)。
//!
//! ```text
//! sub_magic = "P2KRING\0" (8 byte)
//! version   = u32 LE      (= 1)
//! count     = u32 LE      (= 19,968、 全 i16 entry 数)
//! weights   = i16[bucket 8][persp 2][sq 24][piece 13][state 4] LE  (= bucket-outer)
//! ```
//!
//! 合計 byte 数: `8 + 4 + 4 + 2 × 19,968 = 39,952 byte` (~40 KB)。

use crate::bitboard::{Bitboard, bishop_effect, king_effect, knight_effect, rook_effect};
use crate::nnue::dual_delta::MoveDelta;
use crate::position::Position;
use crate::types::{Color, PieceType, Square};
#[cfg(test)]
use crate::types::{File, Rank};
use std::sync::LazyLock;

// =============================================================================
// 定数
// =============================================================================

/// bucket 数 (= LayerStacks の progress8kpabs と共有)。
pub const KINGRING_NUM_BUCKETS: usize = 8;

/// ring 内 sq 数 (= 5×5 zone から king 自身を除く = 24)。
pub const KINGRING_NUM_SQ: usize = 24;

/// piece 数 (= King 除く Pawn..Dragon の 13 種)。
///
/// idx mapping (= PieceType raw - 1 for raw<King、 raw - 2 for raw>King):
/// - 0 = Pawn / 1 = Lance / 2 = Knight / 3 = Silver / 4 = Bishop / 5 = Rook
/// - 6 = Gold / 7 = ProPawn / 8 = ProLance / 9 = ProKnight / 10 = ProSilver
/// - 11 = Horse / 12 = Dragon
pub const KINGRING_NUM_PIECE: usize = 13;

/// state 数 (= 4):
///
/// - 0 = empty sq + own smallest attacker (= own が king zone empty sq に攻撃)
/// - 1 = own piece on sq (= own defended)
/// - 2 = enemy piece on sq (= enemy occupies)
/// - 3 = empty sq + enemy smallest attacker (= enemy が king zone empty sq に圧力)
pub const KINGRING_NUM_STATE: usize = 4;

/// 1 perspective × 1 bucket あたりの flat dim (= 24 × 13 × 4 = 1,248)。
pub const KINGRING_PER_BUCKET_DIM: usize =
    KINGRING_NUM_SQ * KINGRING_NUM_PIECE * KINGRING_NUM_STATE;

const _: () = {
    assert!(KINGRING_PER_BUCKET_DIM == 1_248);
};

/// 1 perspective の (sq, piece, state) flat 空間サイズ (= 1,248)。
/// `encode_active_idx` の値域。
pub const KINGRING_PER_PERSP_DIM: usize = KINGRING_PER_BUCKET_DIM;

/// weight 総 dim (= 8 bucket × 2 persp × 24 sq × 13 piece × 4 state = 19,968)。
pub const KINGRING_WEIGHT_DIM: usize = KINGRING_NUM_BUCKETS * 2 * KINGRING_PER_BUCKET_DIM;

const _: () = {
    assert!(KINGRING_WEIGHT_DIM == 19_968);
};

/// ckpt section の sub_magic ("P2KRING\0" = 8 byte)。
pub const KINGRING_SUB_MAGIC: [u8; 8] = *b"P2KRING\0";

/// ckpt section format version (現在 1)。
pub const KINGRING_VERSION: u32 = 1;

/// ckpt section の i16 entry 総数 (= KINGRING_WEIGHT_DIM)。
pub const KINGRING_COUNT: u32 = KINGRING_WEIGHT_DIM as u32;

/// output scale (= eval 加算前の正規化定数、 1)。
pub const KINGRING_OUTPUT_SCALE: i32 = 1;

/// 1 perspective × 1 sq 最大 entry 数 (= 24)。
pub const KINGRING_MAX_ACTIVE: usize = KINGRING_NUM_SQ;

/// ring_idx 固定 slot の SENTINEL 値 (= 「該当 ring sq に active entry なし」 marker)。
///
/// active flat idx は 0..1248 なので、 u16::MAX は値域外で sentinel として安全に使える。
pub const KINGRING_SLOT_SENTINEL: u16 = u16::MAX;

/// ring table 内 sentinel (= 「盤外 ring sq」 marker、 0..81 の値域外)。
const RING_SQ_SENTINEL: u8 = 81;

// =============================================================================
// 型定義
// =============================================================================

/// flat weight tensor (= **bucket-outer** memory layout、 disk format と完全一致)。
///
/// `weights[bucket][persp][sq][piece][state] -> i16`。
///
/// disk format も同じ順 (`[bucket 8][persp 2][sq 24][piece 13][state 4]`) なので、
/// `read_kingring_section` は転置なしで sequential read できる。
///
/// 19,968 i16 = 39,936 byte ≈ 40 KB (= heap Box 推奨)。
///
/// **runtime hot path では使わない** (= [`KingRingRuntimeWeights`] が bucket-inner i32 で
/// 高速 read 可能)。 disk read & test 用に残る形式。
pub type KingRingWeights =
    [[[[[i16; KINGRING_NUM_STATE]; KINGRING_NUM_PIECE]; KINGRING_NUM_SQ]; 2]; KINGRING_NUM_BUCKETS];

/// hot path 用 **bucket-inner i32 row** layout (= 候補 3、 2026-05-29 追加)。
///
/// `runtime[persp][raw_idx][bucket] -> i32`、 raw_idx は `encode_active_idx` 戻り値
/// (= 0..1248)。 1 entry に対する 8 bucket は **連続 memory** に並ぶため、 hot path で
/// `row[bucket]` を順次 read するだけで 8 個の i32 を一気に sum できる。
///
/// メモリ消費: 2 persp × 1,248 raw_idx × 8 bucket × 4 byte = **79,872 byte (~80 KB)**。
/// disk shape の `KingRingWeights` (40 KB) からの +40 KB 増だが、 hot path 削減効果
/// (= 5 次元 index decode と i16→i32 cast を除去) で NPS 退行を緩和する。
pub type KingRingRuntimeWeights = [[[i32; KINGRING_NUM_BUCKETS]; KINGRING_PER_BUCKET_DIM]; 2];

/// 1 perspective の accumulator (= per-bucket i32 scalar)。
pub type KingRingAccumulation = [i32; KINGRING_NUM_BUCKETS];

/// 1 entry の sparse encode 値:
/// `idx = sq * 13 * 4 + piece * 4 + state` (= 0..1248、 u16 で表現)。
pub type KingRingActiveIdx = u16;

/// **ring_idx 固定 slot** の active 集合表現 (= 候補 1、 2026-05-29 追加)。
///
/// `slots[ring_idx] = encoded_idx` (= 0..1248) または `KINGRING_SLOT_SENTINEL` (= 空)。
/// ring 24 sq の slot 位置は固定なので、 diff update は old/new slot を ring_idx 順に
/// **1 対 1 で cmp** するだけで済む (= removed/added buffer + sort 不要)。
pub type KingRingActiveSlots = [u16; KINGRING_NUM_SQ];

// =============================================================================
// disk shape → runtime shape 転置
// =============================================================================

/// `KingRingWeights` (= bucket-outer i16) を `KingRingRuntimeWeights` (= bucket-inner i32) に
/// 転置構築する。 ckpt read 後に **1 回だけ** 呼ぶ (= hot path ではない)。
///
/// disk shape は ckpt 互換性のため変更不可、 runtime hot path は bucket-inner で 8 連続
/// read したいので、 read 後にこの転置を 1 回行う。 メモリ +40 KB 程度 = 許容範囲。
pub fn build_runtime_weights(disk: &KingRingWeights) -> Box<KingRingRuntimeWeights> {
    let mut rt: Box<KingRingRuntimeWeights> =
        Box::new([[[0i32; KINGRING_NUM_BUCKETS]; KINGRING_PER_BUCKET_DIM]; 2]);
    for (bucket, bucket_arr) in disk.iter().enumerate() {
        for (persp, persp_arr) in bucket_arr.iter().enumerate() {
            for (sq, sq_arr) in persp_arr.iter().enumerate() {
                for (piece, piece_arr) in sq_arr.iter().enumerate() {
                    for (state, &w) in piece_arr.iter().enumerate() {
                        let raw_idx = sq * (KINGRING_NUM_PIECE * KINGRING_NUM_STATE)
                            + piece * KINGRING_NUM_STATE
                            + state;
                        rt[persp][raw_idx][bucket] = w as i32;
                    }
                }
            }
        }
    }
    rt
}

// =============================================================================
// helper: piece_type → piece idx 変換
// =============================================================================

/// `PieceType` (1..=14) を kingring の piece idx (0..=12) に変換する。
///
/// 戻り値 `Some(idx)`: idx ∈ 0..13。 King の場合は `None` (= kingring では含めない)。
///
/// mapping (= bullet 側 `piece_type_to_kingring_idx` と一致):
/// raw < King なら `raw - 1`、 raw > King なら `raw - 2`。
#[inline]
const fn piece_type_to_kingring_idx(pt: PieceType) -> Option<usize> {
    let raw = pt as u8;
    let king_raw = PieceType::King as u8;
    if raw == king_raw {
        return None;
    }
    let idx = if raw < king_raw {
        (raw - 1) as usize
    } else {
        (raw - 2) as usize
    };
    debug_assert!(idx < KINGRING_NUM_PIECE);
    Some(idx)
}

/// piece_type を 0..=12 の順に並べた逆引き table (= smallest attacker 検索用)。
///
/// hot path で `pieces_pt(pt)` を順次 mask 取得し、 active な最初の pt を返す。
const PIECE_ORDER: [PieceType; KINGRING_NUM_PIECE] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Gold,
    PieceType::ProPawn,
    PieceType::ProLance,
    PieceType::ProKnight,
    PieceType::ProSilver,
    PieceType::Horse,
    PieceType::Dragon,
];

// =============================================================================
// ring 24 sq table (= 候補 4、 precompute)
// =============================================================================

/// `KINGRING_TABLE[persp][king_sq.raw()][ring_idx] = sq_raw_or_sentinel`。
///
/// - `sq_raw` ∈ 0..81 = 盤内 ring sq
/// - `RING_SQ_SENTINEL (= 81)` = 盤外 (= 該当 ring sq なし)
///
/// 起動時に 1 回だけ build される (= LazyLock)。 サイズ = 2 × 81 × 24 = **3,888 byte**
/// (= stack ではなく static、 L1 cache に常駐想定)。
static KINGRING_TABLE: LazyLock<[[[u8; KINGRING_NUM_SQ]; 81]; 2]> = LazyLock::new(|| {
    let mut table = [[[RING_SQ_SENTINEL; KINGRING_NUM_SQ]; 81]; 2];
    for (p, persp_table) in table.iter_mut().enumerate() {
        let sign: i8 = if p == 0 { 1 } else { -1 };
        for (king_raw, king_row) in persp_table.iter_mut().enumerate() {
            let king_file = (king_raw / 9) as i8;
            let king_rank = (king_raw % 9) as i8;
            let mut idx: usize = 0;
            for df in -2..=2i8 {
                for dr in -2..=2i8 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    let nf = king_file + sign * df;
                    let nr = king_rank + sign * dr;
                    if (0..9).contains(&nf) && (0..9).contains(&nr) {
                        let sq_raw = (nf as u8) * 9 + (nr as u8);
                        king_row[idx] = sq_raw;
                    }
                    idx += 1;
                }
            }
            debug_assert!(idx == KINGRING_NUM_SQ);
        }
    }
    table
});

/// king 位置の **perspective** に対する ring 24 sq を返す (= 候補 4、 precompute table lookup)。
///
/// `ring[ring_idx]` = `Some(Square)` (= 盤内) または `None` (= 盤外)。
///
/// 旧 `build_ring_for_king` の `Option<Square>` 24 個構築を **byte table の 24 直 read** に
/// 置き換えた。 ただし API 互換性のため `Option<Square>` を返す版も残す (= test 用)。
#[inline]
fn ring_table_row(king_sq: Square, perspective: Color) -> &'static [u8; KINGRING_NUM_SQ] {
    let p = perspective as usize;
    let king_raw = king_sq.raw() as usize;
    debug_assert!(p < 2);
    debug_assert!(king_raw < 81);
    // SAFETY: p < 2、 king_raw < 81 を上で確認済。
    unsafe { KINGRING_TABLE.get_unchecked(p).get_unchecked(king_raw) }
}

/// test 互換用の `Option<Square>` 配列版 (= test でしか使われない、 hot path は
/// `ring_table_row` で byte 直 read)。
#[cfg(test)]
#[inline]
fn build_ring_for_king(king_sq: Square, perspective: Color) -> [Option<Square>; KINGRING_NUM_SQ] {
    let row = ring_table_row(king_sq, perspective);
    let mut out = [None; KINGRING_NUM_SQ];
    for (idx, &sq_raw) in row.iter().enumerate() {
        if sq_raw != RING_SQ_SENTINEL {
            // SAFETY: KINGRING_TABLE build 時に sq_raw ∈ 0..81 を保証済。
            out[idx] = Some(unsafe { Square::from_u8_unchecked(sq_raw) });
        }
    }
    out
}

// =============================================================================
// classify_sq: 1 sq の状態判定 (4 state)
// =============================================================================

/// `attackers` (= 全色 attackers bitboard、 sq に対する利き) から、 `pieces_c(side) & attackers`
/// 内の **最小 piece_type** を返す (= bullet `kingring_smallest_attacker` と等価)。
///
/// `own_side_attackers` は呼出側で `attackers_to_occ(sq, occ) & pieces_c(side)` 済み。
///
/// # 戻り値
///
/// `Some(PieceType)` = 最小駒種 / `None` = attackers なし。
#[inline]
fn smallest_attacker_in(pos: &Position, side_attackers: Bitboard) -> Option<PieceType> {
    if side_attackers.is_empty() {
        return None;
    }
    for pt in PIECE_ORDER {
        let pt_bb = pos.pieces_pt(pt);
        if !(side_attackers & pt_bb).is_empty() {
            return Some(pt);
        }
    }
    None
}

/// `perspective` 視点で `ring_sq` の (piece_idx, state_idx) を判定する。
///
/// 優先順位:
/// 1. sq に own 駒あり → state 1、 piece = sq の駒
/// 2. sq に enemy 駒あり → state 2、 piece = sq の駒
/// 3. empty + own smallest attacker → state 0、 piece = attacker の最小駒
/// 4. empty + enemy smallest attacker → state 3、 piece = attacker の最小駒
/// 5. 該当なし → None
///
/// `attackers_to_occ` は **1 回** だけ呼ぶ (= empty sq の場合のみ)。 occupied sq は早期 return
/// で利き計算を skip。
#[inline]
fn classify_sq(pos: &Position, perspective: Color, ring_sq: Square) -> Option<(usize, usize)> {
    let piece = pos.piece_on(ring_sq);
    if piece.is_some() {
        let pt = piece.piece_type();
        let piece_color = piece.color();
        let piece_idx = piece_type_to_kingring_idx(pt)?;
        let state = if piece_color == perspective { 1 } else { 2 };
        return Some((piece_idx, state));
    }
    // empty: own → enemy の順で smallest attacker を確認。
    // attackers_to_occ を 1 回呼んで own/enemy で bitop 分割 (= 利き計算 1 回化)。
    let occ = pos.occupied();
    let all_attackers = pos.attackers_to_occ(ring_sq, occ);
    let own_attackers = all_attackers & pos.pieces_c(perspective);
    if let Some(pt) = smallest_attacker_in(pos, own_attackers) {
        let piece_idx = piece_type_to_kingring_idx(pt)?;
        return Some((piece_idx, 0));
    }
    let enemy_attackers = all_attackers & pos.pieces_c(perspective.opponent());
    if let Some(pt) = smallest_attacker_in(pos, enemy_attackers) {
        let piece_idx = piece_type_to_kingring_idx(pt)?;
        return Some((piece_idx, 3));
    }
    None
}

// =============================================================================
// active index 構築
// =============================================================================

/// (ring_idx, piece_idx, state_idx) を flat sparse index に encode。
///
/// `idx = ring_idx * 13 * 4 + piece_idx * 4 + state_idx`、 値域 0..1248。
#[inline]
pub const fn encode_active_idx(ring_idx: usize, piece_idx: usize, state_idx: usize) -> u16 {
    debug_assert!(ring_idx < KINGRING_NUM_SQ);
    debug_assert!(piece_idx < KINGRING_NUM_PIECE);
    debug_assert!(state_idx < KINGRING_NUM_STATE);
    (ring_idx * (KINGRING_NUM_PIECE * KINGRING_NUM_STATE)
        + piece_idx * KINGRING_NUM_STATE
        + state_idx) as u16
}

/// flat sparse index を (ring_idx, piece_idx, state_idx) に decode。
///
/// hot path では使わない (= test の roundtrip 検証 + runtime layout で raw_idx 直 read
/// するため)。
#[cfg(test)]
#[inline]
const fn decode_active_idx(idx: u16) -> (usize, usize, usize) {
    let raw = idx as usize;
    let state = raw % KINGRING_NUM_STATE;
    let rest = raw / KINGRING_NUM_STATE;
    let piece = rest % KINGRING_NUM_PIECE;
    let ring_idx = rest / KINGRING_NUM_PIECE;
    debug_assert!(ring_idx < KINGRING_NUM_SQ);
    (ring_idx, piece, state)
}

/// **ring_idx 固定 slot** 版 active 集合構築 (= 候補 1、 hot path 主経路、 2026-05-29)。
///
/// `out[ring_idx]` に encoded_idx (= 0..1248) を書き込む。 該当 ring sq が空 / 盤外 /
/// classify 失敗の場合は `KINGRING_SLOT_SENTINEL` を書き込む。 ring_idx 固定なので
/// active_len は不要。
#[inline]
pub fn collect_active_slots(pos: &Position, perspective: Color, out: &mut KingRingActiveSlots) {
    let king_sq = pos.king_square(perspective);
    let row = ring_table_row(king_sq, perspective);
    for ring_idx in 0..KINGRING_NUM_SQ {
        // SAFETY: ring_idx < KINGRING_NUM_SQ = 24、 row len = 24。
        let sq_raw = unsafe { *row.get_unchecked(ring_idx) };
        if sq_raw == RING_SQ_SENTINEL {
            // SAFETY: ring_idx < 24 = out.len。
            unsafe {
                *out.get_unchecked_mut(ring_idx) = KINGRING_SLOT_SENTINEL;
            }
            continue;
        }
        // SAFETY: KINGRING_TABLE build 時に sq_raw ∈ 0..81 を保証。
        let sq = unsafe { Square::from_u8_unchecked(sq_raw) };
        let slot = match classify_sq(pos, perspective, sq) {
            Some((piece_idx, state_idx)) => encode_active_idx(ring_idx, piece_idx, state_idx),
            None => KINGRING_SLOT_SENTINEL,
        };
        // SAFETY: ring_idx < 24 = out.len。
        unsafe {
            *out.get_unchecked_mut(ring_idx) = slot;
        }
    }
}

/// 旧 API (= compact list + len) 互換用、 test 専用。
#[cfg(test)]
#[inline]
pub fn collect_active_indices(
    pos: &Position,
    perspective: Color,
    out: &mut [u16; KINGRING_MAX_ACTIVE],
) -> usize {
    let mut slots: KingRingActiveSlots = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
    collect_active_slots(pos, perspective, &mut slots);
    let mut len: usize = 0;
    for &s in slots.iter() {
        if s != KINGRING_SLOT_SENTINEL {
            out[len] = s;
            len += 1;
        }
    }
    len
}

// =============================================================================
// accumulator 計算
// =============================================================================

/// **runtime layout** (= bucket-inner i32 row) から 1 entry を加減算する。
///
/// `row[bucket]` の **8 連続 read** = メモリ局所性 max、 raw_idx 分解不要、
/// i16 → i32 cast 不要。 hot path 主経路。
///
/// # 安全性
///
/// `raw_idx < KINGRING_PER_BUCKET_DIM`、 `p < 2` を debug_assert で確認。
#[inline(always)]
fn accumulate_entry_runtime(
    rt: &KingRingRuntimeWeights,
    p: usize,
    raw_idx: u16,
    sign: i32,
    sums: &mut KingRingAccumulation,
) {
    debug_assert!(p < 2);
    debug_assert!((raw_idx as usize) < KINGRING_PER_BUCKET_DIM);
    // SAFETY: p < 2、 raw_idx < KINGRING_PER_BUCKET_DIM を上で debug_assert。
    let row = unsafe { rt.get_unchecked(p).get_unchecked(raw_idx as usize) };
    if sign >= 0 {
        for (sum, &w) in sums.iter_mut().zip(row.iter()) {
            *sum = sum.wrapping_add(w);
        }
    } else {
        for (sum, &w) in sums.iter_mut().zip(row.iter()) {
            *sum = sum.wrapping_sub(w);
        }
    }
}

/// 旧 disk shape (= bucket-outer i16) から entry 加減算 (= test / legacy 用)。
#[cfg(test)]
#[inline(always)]
fn accumulate_entry(
    weights: &KingRingWeights,
    p: usize,
    raw_idx: u16,
    sign: i32,
    sums: &mut KingRingAccumulation,
) {
    debug_assert!(p < 2);
    debug_assert!((raw_idx as usize) < KINGRING_PER_BUCKET_DIM);
    let raw = raw_idx as usize;
    let state = raw % KINGRING_NUM_STATE;
    let rest = raw / KINGRING_NUM_STATE;
    let piece = rest % KINGRING_NUM_PIECE;
    let sq = rest / KINGRING_NUM_PIECE;
    debug_assert!(sq < KINGRING_NUM_SQ);

    for (bucket, sum) in sums.iter_mut().enumerate() {
        // SAFETY: bucket < 8、 p < 2、 sq < 24、 piece < 13、 state < 4。
        let w = unsafe {
            *weights
                .get_unchecked(bucket)
                .get_unchecked(p)
                .get_unchecked(sq)
                .get_unchecked(piece)
                .get_unchecked(state)
        };
        if sign >= 0 {
            *sum = sum.wrapping_add(w as i32);
        } else {
            *sum = sum.wrapping_sub(w as i32);
        }
    }
}

// =============================================================================
// hot path 主経路: ring_idx 固定 slot ベース refresh / update
// =============================================================================

/// **ring_idx 固定 slot** 集合から 8 bucket の sum を計算 (= runtime layout、 hot path)。
#[inline]
fn sum_slots_runtime(
    rt: &KingRingRuntimeWeights,
    perspective: Color,
    slots: &KingRingActiveSlots,
) -> KingRingAccumulation {
    let mut sums: KingRingAccumulation = [0i32; KINGRING_NUM_BUCKETS];
    let p = perspective as usize;
    for &slot in slots.iter() {
        if slot != KINGRING_SLOT_SENTINEL {
            accumulate_entry_runtime(rt, p, slot, 1, &mut sums);
        }
    }
    sums
}

/// `perspective` の accumulator を full refresh する (= runtime layout、 hot path 主経路)。
///
/// 戻り値: 8 bucket × i32 sum、 active slots (= ring_idx 固定 24 slot) を caller 側
/// buffer に書き出す。
pub fn refresh_kingring_perspective_runtime(
    pos: &Position,
    rt: &KingRingRuntimeWeights,
    perspective: Color,
    slots_out: &mut KingRingActiveSlots,
) -> KingRingAccumulation {
    collect_active_slots(pos, perspective, slots_out);
    sum_slots_runtime(rt, perspective, slots_out)
}

/// prev / current の **ring_idx 固定 slot** を 1 対 1 で比較して diff を accumulator に適用。
///
/// removed/added buffer は不要、 sort 前提も不要 (= ring_idx 位置が固定なので old vs new で
/// 直接 cmp)。 hot path 主経路。
pub fn apply_slot_diff_runtime(
    rt: &KingRingRuntimeWeights,
    perspective: Color,
    prev_slots: &KingRingActiveSlots,
    cur_slots: &KingRingActiveSlots,
    acc: &mut KingRingAccumulation,
) {
    let p = perspective as usize;
    for ring_idx in 0..KINGRING_NUM_SQ {
        // SAFETY: ring_idx < 24 = slots.len。
        let old = unsafe { *prev_slots.get_unchecked(ring_idx) };
        let new = unsafe { *cur_slots.get_unchecked(ring_idx) };
        if old == new {
            continue;
        }
        if old != KINGRING_SLOT_SENTINEL {
            accumulate_entry_runtime(rt, p, old, -1, acc);
        }
        if new != KINGRING_SLOT_SENTINEL {
            accumulate_entry_runtime(rt, p, new, 1, acc);
        }
    }
}

// =============================================================================
// incremental reuse (= MoveDelta affected sq set ベース、 hot path 真因削減)
// =============================================================================
//
// `classify_sq` の `attackers_to_occ` は ring 24 sq 全部呼ぶと **48 回/ply** に達して
// hot path bottleneck になる。 1 手で本当に変化する ring sq は drop/quiet/capture で
// 1-3 個、 slider 移動でも ~8 個程度なので、 **affected sq 集合に含まれない ring sq は
// 前 ply の slot をそのまま再利用** して classify_sq 呼び出しを削減する。
//
// `MoveDelta` (= `nnue::dual_delta::compute_move_delta`) から:
//   - moved_from / moved_to / captured_sq (= 直接 affected)
//   - 各 sq の 8 近傍 + 桂 jumps (= 短距離 attacker から見た own_sq の利き変化)
//   - influenced_sliders_bb (= ray 上の slider 駒 自身の利き変化)
// の和集合を「affected sq 集合」 と定義する。 ring sq が この集合に含まれていれば
// state / piece / smallest_attacker が変化しうるので再 classify、 そうでなければ
// 前 ply の slot をそのまま使う。 king が動いた perspective は ring 自体が shift する
// ので caller 側で full refresh に fallback (= 本関数を呼ばない契約)。

/// 1 手分の影響を受ける ring sq 候補 集合 (= bitboard、 kingring 用)。
///
/// kingring の classify_sq は ring sq r の attacker 集合に依存するため、 r の attacker
/// 集合が変化しうる ring sq のみ再 classify すれば bit-exact に full refresh と一致する。
/// r の attacker 集合は以下の場合に変化する:
///
/// 1. r ∈ {from, to, cap_sq}: r 自身の occupant 駒が変わる (= state 1/2 自体が変化)
/// 2. 短距離 attacker (= 歩/銀/金/桂/玉) が from/to/cap_sq に出入り:
///    r ∈ king_effect(from/to/cap_sq) ∪ knight_effect(_, from/to/cap_sq)
/// 3. slider attacker (= 飛/角/香/馬/竜) の利き が変化:
///    r が from/to/cap_sq から見える slider line 上にいる (= ray-extending /
///    blocking が r の最寄り attacker を変える可能性)
///
/// 3 を取りこぼすと slot diff が誤って old を流用してしまうので、 影響範囲は
/// やや conservative に **rook_effect | bishop_effect (= 全 ray-walk、 occupancy 考慮)** を
/// 含める。
#[inline]
fn build_affected_sq_set(pos: &Position, delta: &MoveDelta) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    let to_sq = delta.moved_to;
    let occ = pos.occupied();

    if let Some(from) = delta.moved_from {
        bb |= Bitboard::from_square(from);
        bb |= king_effect(from);
        bb |= knight_effect(Color::Black, from);
        bb |= knight_effect(Color::White, from);
        // slider line: from を経由する ray 上の ring sq は attacker 集合が変化しうる。
        bb |= rook_effect(from, occ);
        bb |= bishop_effect(from, occ);
    }

    bb |= Bitboard::from_square(to_sq);
    bb |= king_effect(to_sq);
    bb |= knight_effect(Color::Black, to_sq);
    bb |= knight_effect(Color::White, to_sq);
    bb |= rook_effect(to_sq, occ);
    bb |= bishop_effect(to_sq, occ);

    // 将棋では captured_sq == moved_to が常に成立 (= en passant なし) のため、
    // 通常 capture は to 経路で既に処理済。 cap_sq != to_sq の特殊 move のみ追加。
    if let Some((cap_sq, _, _)) = delta.captured
        && cap_sq != to_sq
    {
        bb |= Bitboard::from_square(cap_sq);
        bb |= king_effect(cap_sq);
        bb |= knight_effect(Color::Black, cap_sq);
        bb |= knight_effect(Color::White, cap_sq);
        bb |= rook_effect(cap_sq, occ);
        bb |= bishop_effect(cap_sq, occ);
    }

    bb
}

/// `MoveDelta` から affected sq 集合を build する (= 2 perspective で共有可能、 hot path)。
///
/// 集合 = `{from, to, captured_sq} ∪ {各 sq の 8 近傍 + 桂 jumps + rook/bishop rays}`。
/// perspective に依存しないため、 caller 側で 1 度だけ計算して両 perspective の
/// `update_kingring_perspective_incremental_with_affected` に渡せる。
#[inline]
pub fn build_kingring_affected_sq_set(pos: &Position, delta: &MoveDelta) -> Bitboard {
    build_affected_sq_set(pos, delta)
}

/// `perspective` の ring 24 slot を「affected ring sq のみ classify、 残りは prev 流用」
/// で更新する (= incremental reuse、 hot path 主経路)。
///
/// king は **動いていない前提** (= caller 契約)。 king moved 時は ring 自体が shift する
/// ので full refresh に fallback すること。
///
/// 戻り値: 更新後 8 bucket × i32 sum。 `slots` には new slots が書き込まれる
/// (= prev slot 流用 + affected 再 classify の混合)。
pub fn update_kingring_perspective_incremental(
    pos: &Position,
    rt: &KingRingRuntimeWeights,
    perspective: Color,
    delta: &MoveDelta,
    prev_slots: &KingRingActiveSlots,
    prev_acc: &KingRingAccumulation,
    slots: &mut KingRingActiveSlots,
) -> KingRingAccumulation {
    let affected = build_affected_sq_set(pos, delta);
    update_kingring_perspective_incremental_with_affected(
        pos,
        rt,
        perspective,
        delta,
        affected,
        prev_slots,
        prev_acc,
        slots,
    )
}

/// `update_kingring_perspective_incremental` の lean variant (= affected を caller が
/// pre-compute して 2 perspective で共有)。 動作は本体と bit-exact 同等。
#[allow(clippy::too_many_arguments)]
pub fn update_kingring_perspective_incremental_with_affected(
    pos: &Position,
    rt: &KingRingRuntimeWeights,
    perspective: Color,
    delta: &MoveDelta,
    affected: Bitboard,
    prev_slots: &KingRingActiveSlots,
    prev_acc: &KingRingAccumulation,
    slots: &mut KingRingActiveSlots,
) -> KingRingAccumulation {
    debug_assert!(!delta.king_moved_any, "king moved should use full refresh");

    let king_sq = pos.king_square(perspective);
    let row = ring_table_row(king_sq, perspective);
    let mut acc = *prev_acc;
    let p = perspective as usize;

    for ring_idx in 0..KINGRING_NUM_SQ {
        // SAFETY: ring_idx < KINGRING_NUM_SQ = 24、 row len = 24、 slots len = 24。
        let sq_raw = unsafe { *row.get_unchecked(ring_idx) };
        if sq_raw == RING_SQ_SENTINEL {
            // 盤外 ring sq: 必ず SENTINEL、 prev も SENTINEL のはず。
            // SAFETY: ring_idx < 24。
            unsafe {
                *slots.get_unchecked_mut(ring_idx) = KINGRING_SLOT_SENTINEL;
            }
            continue;
        }
        // SAFETY: KINGRING_TABLE build 時に sq_raw ∈ 0..81 を保証。
        let sq = unsafe { Square::from_u8_unchecked(sq_raw) };

        // ring sq が affected に含まれなければ prev slot をそのまま流用 (= classify 不要)。
        if !affected.contains(sq) {
            // SAFETY: ring_idx < 24。
            let prev_slot = unsafe { *prev_slots.get_unchecked(ring_idx) };
            unsafe {
                *slots.get_unchecked_mut(ring_idx) = prev_slot;
            }
            continue;
        }

        // affected ring sq: 再 classify + slot diff 適用。
        let new_slot = match classify_sq(pos, perspective, sq) {
            Some((piece_idx, state_idx)) => encode_active_idx(ring_idx, piece_idx, state_idx),
            None => KINGRING_SLOT_SENTINEL,
        };
        // SAFETY: ring_idx < 24。
        let old_slot = unsafe { *prev_slots.get_unchecked(ring_idx) };
        if old_slot != new_slot {
            if old_slot != KINGRING_SLOT_SENTINEL {
                accumulate_entry_runtime(rt, p, old_slot, -1, &mut acc);
            }
            if new_slot != KINGRING_SLOT_SENTINEL {
                accumulate_entry_runtime(rt, p, new_slot, 1, &mut acc);
            }
        }
        // SAFETY: ring_idx < 24。
        unsafe {
            *slots.get_unchecked_mut(ring_idx) = new_slot;
        }
    }

    acc
}

// =============================================================================
// legacy 経路 (= disk shape ベース、 test 専用)
// =============================================================================

/// 与えられた active 集合から 8 bucket の sum を計算 (= disk shape、 test 用)。
#[cfg(test)]
#[inline]
fn sum_active(
    weights: &KingRingWeights,
    perspective: Color,
    active: &[u16],
) -> KingRingAccumulation {
    let mut sums: KingRingAccumulation = [0i32; KINGRING_NUM_BUCKETS];
    let p = perspective as usize;
    for &idx in active {
        accumulate_entry(weights, p, idx, 1, &mut sums);
    }
    sums
}

/// `perspective` の accumulator を full refresh する (= disk shape、 test 用)。
///
/// hot path では `refresh_kingring_perspective_runtime` を使うこと。
#[cfg(test)]
pub fn refresh_kingring_perspective(
    pos: &Position,
    weights: &KingRingWeights,
    perspective: Color,
    active_out: &mut [u16; KINGRING_MAX_ACTIVE],
    active_len_out: &mut u8,
) -> KingRingAccumulation {
    let len = collect_active_indices(pos, perspective, active_out);
    *active_len_out = len as u8;
    sum_active(weights, perspective, &active_out[..len])
}

/// removed / added の sparse diff を accumulator に適用する (= disk shape、 test 用)。
#[cfg(test)]
pub fn apply_diff(
    weights: &KingRingWeights,
    perspective: Color,
    removed: &[u16],
    added: &[u16],
    acc: &mut KingRingAccumulation,
) {
    let p = perspective as usize;
    for &idx in removed {
        accumulate_entry(weights, p, idx, -1, acc);
    }
    for &idx in added {
        accumulate_entry(weights, p, idx, 1, acc);
    }
}

/// parent / current の sparse active index 集合の symmetric difference を抽出 (= test 用)。
///
/// 戻り値: (removed_len, added_len)。 stack alloc buffer のサイズは `KINGRING_MAX_ACTIVE`
/// で十分 (= 入力長 ≤ 24)。
///
/// hot path では `apply_slot_diff_runtime` を使うこと (= ring_idx 固定 slot で直接 diff)。
#[cfg(test)]
pub fn diff_active_indices(
    parent: &[u16],
    current: &[u16],
    removed_out: &mut [u16; KINGRING_MAX_ACTIVE],
    added_out: &mut [u16; KINGRING_MAX_ACTIVE],
) -> (usize, usize) {
    debug_assert!(parent.len() <= KINGRING_MAX_ACTIVE);
    debug_assert!(current.len() <= KINGRING_MAX_ACTIVE);
    debug_assert!(parent.windows(2).all(|w| w[0] < w[1]), "parent must be sorted");
    debug_assert!(current.windows(2).all(|w| w[0] < w[1]), "current must be sorted");

    let mut removed_len = 0usize;
    let mut added_len = 0usize;
    let mut i = 0usize;
    let mut j = 0usize;
    let plen = parent.len();
    let clen = current.len();
    while i < plen && j < clen {
        let p = parent[i];
        let c = current[j];
        match p.cmp(&c) {
            std::cmp::Ordering::Less => {
                removed_out[removed_len] = p;
                removed_len += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                added_out[added_len] = c;
                added_len += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    while i < plen {
        removed_out[removed_len] = parent[i];
        removed_len += 1;
        i += 1;
    }
    while j < clen {
        added_out[added_len] = current[j];
        added_len += 1;
        j += 1;
    }
    (removed_len, added_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::SFEN_HIRATE;

    fn make_pos() -> Position {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        pos
    }

    #[test]
    fn dims_match_spec() {
        assert_eq!(KINGRING_NUM_BUCKETS, 8);
        assert_eq!(KINGRING_NUM_SQ, 24);
        assert_eq!(KINGRING_NUM_PIECE, 13);
        assert_eq!(KINGRING_NUM_STATE, 4);
        assert_eq!(KINGRING_PER_BUCKET_DIM, 1_248);
        assert_eq!(KINGRING_WEIGHT_DIM, 19_968);
        assert_eq!(KINGRING_COUNT, 19_968);
        assert_eq!(KINGRING_SUB_MAGIC, *b"P2KRING\0");
    }

    #[test]
    fn piece_type_to_idx_groups_correctly() {
        assert_eq!(piece_type_to_kingring_idx(PieceType::Pawn), Some(0));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Lance), Some(1));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Knight), Some(2));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Silver), Some(3));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Bishop), Some(4));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Rook), Some(5));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Gold), Some(6));
        assert_eq!(piece_type_to_kingring_idx(PieceType::King), None);
        assert_eq!(piece_type_to_kingring_idx(PieceType::ProPawn), Some(7));
        assert_eq!(piece_type_to_kingring_idx(PieceType::ProLance), Some(8));
        assert_eq!(piece_type_to_kingring_idx(PieceType::ProKnight), Some(9));
        assert_eq!(piece_type_to_kingring_idx(PieceType::ProSilver), Some(10));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Horse), Some(11));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Dragon), Some(12));
    }

    #[test]
    fn encode_decode_roundtrip() {
        for ring_idx in 0..KINGRING_NUM_SQ {
            for piece_idx in 0..KINGRING_NUM_PIECE {
                for state_idx in 0..KINGRING_NUM_STATE {
                    let enc = encode_active_idx(ring_idx, piece_idx, state_idx);
                    assert!((enc as usize) < KINGRING_PER_BUCKET_DIM);
                    let dec = decode_active_idx(enc);
                    assert_eq!(dec, (ring_idx, piece_idx, state_idx));
                }
            }
        }
    }

    #[test]
    fn ring_at_center_has_24_inside_board() {
        let center = Square::SQ_55;
        let ring = build_ring_for_king(center, Color::Black);
        let inside: usize = ring.iter().filter(|s| s.is_some()).count();
        assert_eq!(inside, 24, "center 5x5 should have all 24 ring sq inside");
    }

    #[test]
    fn ring_at_corner_has_some_outside() {
        let corner = Square::new(File::File1, Rank::Rank1);
        let ring = build_ring_for_king(corner, Color::Black);
        let inside: usize = ring.iter().filter(|s| s.is_some()).count();
        assert!(inside < 24, "corner ring should have some outside-board sq");
        assert!(inside >= 6, "corner ring should still have at least king + 3x3 nearby");
    }

    #[test]
    fn collect_active_indices_startpos_smoke() {
        let pos = make_pos();
        let mut out = [0u16; KINGRING_MAX_ACTIVE];
        let len_b = collect_active_indices(&pos, Color::Black, &mut out);
        let len_w = collect_active_indices(&pos, Color::White, &mut out);
        // 開始局面: ring 内には own/enemy 駒 + own/enemy 攻撃 sq の混合あり、 少なくとも数件は active。
        assert!(len_b > 0, "black ring active should not be empty");
        assert!(len_w > 0, "white ring active should not be empty");
        assert!(len_b <= KINGRING_MAX_ACTIVE);
        assert!(len_w <= KINGRING_MAX_ACTIVE);
    }

    #[test]
    fn refresh_zero_weights_sums_zero() {
        let pos = make_pos();
        let zero_weights: Box<KingRingWeights> = Box::new(
            [[[[[0i16; KINGRING_NUM_STATE]; KINGRING_NUM_PIECE]; KINGRING_NUM_SQ]; 2];
                KINGRING_NUM_BUCKETS],
        );
        let mut active = [0u16; KINGRING_MAX_ACTIVE];
        let mut len = 0u8;
        let sums =
            refresh_kingring_perspective(&pos, &zero_weights, Color::Black, &mut active, &mut len);
        for (b, &v) in sums.iter().enumerate() {
            assert_eq!(v, 0, "zero weight should yield zero sum at bucket {b}");
        }
    }

    #[test]
    fn refresh_vs_diff_consistent_random_walk() {
        // 開始局面 → 適当な合法手列を 50 ply 進めて、
        // 各 ply で full refresh と prev からの diff update が完全一致することを確認。
        use crate::movegen::{MoveList, generate_legal};

        let mut pos = make_pos();
        // 1 weight = 一意な値 (= bit-exact 比較用)。
        // layout = [bucket][persp][sq][piece][state] (= bucket-outer)
        let mut weights: Box<KingRingWeights> = Box::new(
            [[[[[0i16; KINGRING_NUM_STATE]; KINGRING_NUM_PIECE]; KINGRING_NUM_SQ]; 2];
                KINGRING_NUM_BUCKETS],
        );
        let mut seed: i16 = 1;
        for b in 0..KINGRING_NUM_BUCKETS {
            for p in 0..2 {
                for s in 0..KINGRING_NUM_SQ {
                    for pi in 0..KINGRING_NUM_PIECE {
                        for st in 0..KINGRING_NUM_STATE {
                            weights[b][p][s][pi][st] = seed;
                            seed = seed.wrapping_add(3);
                        }
                    }
                }
            }
        }

        // 初期 active + acc
        let mut prev_active_b = [0u16; KINGRING_MAX_ACTIVE];
        let mut prev_len_b = 0u8;
        let mut prev_acc_b = refresh_kingring_perspective(
            &pos,
            &weights,
            Color::Black,
            &mut prev_active_b,
            &mut prev_len_b,
        );
        let mut prev_active_w = [0u16; KINGRING_MAX_ACTIVE];
        let mut prev_len_w = 0u8;
        let mut prev_acc_w = refresh_kingring_perspective(
            &pos,
            &weights,
            Color::White,
            &mut prev_active_w,
            &mut prev_len_w,
        );

        let mut rng_state: u64 = 0xC0FFEE;
        for _step in 0..50 {
            let mut list = MoveList::new();
            generate_legal(&pos, &mut list);
            if list.is_empty() {
                break;
            }
            rng_state =
                rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pick = (rng_state >> 33) as usize % list.len();
            let mv = list.at(pick);
            let gc = pos.gives_check(mv);
            let _dirty = pos.do_move(mv, gc);

            // full refresh
            let mut cur_active_b = [0u16; KINGRING_MAX_ACTIVE];
            let mut cur_len_b = 0u8;
            let refreshed_b = refresh_kingring_perspective(
                &pos,
                &weights,
                Color::Black,
                &mut cur_active_b,
                &mut cur_len_b,
            );
            let mut cur_active_w = [0u16; KINGRING_MAX_ACTIVE];
            let mut cur_len_w = 0u8;
            let refreshed_w = refresh_kingring_perspective(
                &pos,
                &weights,
                Color::White,
                &mut cur_active_w,
                &mut cur_len_w,
            );

            // diff update from prev
            let mut diff_acc_b = prev_acc_b;
            let mut removed = [0u16; KINGRING_MAX_ACTIVE];
            let mut added = [0u16; KINGRING_MAX_ACTIVE];
            let (rlen, alen) = diff_active_indices(
                &prev_active_b[..prev_len_b as usize],
                &cur_active_b[..cur_len_b as usize],
                &mut removed,
                &mut added,
            );
            apply_diff(&weights, Color::Black, &removed[..rlen], &added[..alen], &mut diff_acc_b);

            let mut diff_acc_w = prev_acc_w;
            let (rlen, alen) = diff_active_indices(
                &prev_active_w[..prev_len_w as usize],
                &cur_active_w[..cur_len_w as usize],
                &mut removed,
                &mut added,
            );
            apply_diff(&weights, Color::White, &removed[..rlen], &added[..alen], &mut diff_acc_w);

            assert_eq!(refreshed_b, diff_acc_b, "Black: refresh vs diff mismatch at step");
            assert_eq!(refreshed_w, diff_acc_w, "White: refresh vs diff mismatch at step");

            // runtime layout (= bucket-inner i32 + ring_idx 固定 slot) も同 acc を出すことを確認。
            let rt = build_runtime_weights(&weights);
            let mut slot_b = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
            let rt_acc_b =
                refresh_kingring_perspective_runtime(&pos, &rt, Color::Black, &mut slot_b);
            let mut slot_w = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
            let rt_acc_w =
                refresh_kingring_perspective_runtime(&pos, &rt, Color::White, &mut slot_w);
            assert_eq!(refreshed_b, rt_acc_b, "Black: disk vs runtime refresh mismatch");
            assert_eq!(refreshed_w, rt_acc_w, "White: disk vs runtime refresh mismatch");

            prev_active_b = cur_active_b;
            prev_len_b = cur_len_b;
            prev_acc_b = refreshed_b;
            prev_active_w = cur_active_w;
            prev_len_w = cur_len_w;
            prev_acc_w = refreshed_w;
        }
    }

    #[test]
    fn ring_slot_diff_matches_full_refresh_when_king_stationary() {
        // ring_idx 固定 slot diff は king 不動 ply で必ず full refresh と一致することを確認。
        // king 移動 ply は ring 自体が shift するため slot diff の意味が変わる → skip。
        use crate::movegen::{MoveList, generate_legal};

        let mut pos = make_pos();
        let mut disk_weights: Box<KingRingWeights> = Box::new(
            [[[[[0i16; KINGRING_NUM_STATE]; KINGRING_NUM_PIECE]; KINGRING_NUM_SQ]; 2];
                KINGRING_NUM_BUCKETS],
        );
        let mut seed: i16 = 7;
        for b in 0..KINGRING_NUM_BUCKETS {
            for p in 0..2 {
                for s in 0..KINGRING_NUM_SQ {
                    for pi in 0..KINGRING_NUM_PIECE {
                        for st in 0..KINGRING_NUM_STATE {
                            disk_weights[b][p][s][pi][st] = seed;
                            seed = seed.wrapping_add(5);
                        }
                    }
                }
            }
        }
        let rt = build_runtime_weights(&disk_weights);

        let mut prev_slot_b = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
        let mut prev_slot_w = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
        let mut prev_acc_b =
            refresh_kingring_perspective_runtime(&pos, &rt, Color::Black, &mut prev_slot_b);
        let mut prev_acc_w =
            refresh_kingring_perspective_runtime(&pos, &rt, Color::White, &mut prev_slot_w);
        let mut prev_king_b = pos.king_square(Color::Black);
        let mut prev_king_w = pos.king_square(Color::White);

        let mut rng_state: u64 = 0xBEEF_CAFE;
        let mut checked_steps = 0usize;
        for _step in 0..80 {
            let mut list = MoveList::new();
            generate_legal(&pos, &mut list);
            if list.is_empty() {
                break;
            }
            rng_state =
                rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pick = (rng_state >> 33) as usize % list.len();
            let mv = list.at(pick);
            let gc = pos.gives_check(mv);
            let _dirty = pos.do_move(mv, gc);

            let cur_king_b = pos.king_square(Color::Black);
            let cur_king_w = pos.king_square(Color::White);
            let king_moved_b = cur_king_b != prev_king_b;
            let king_moved_w = cur_king_w != prev_king_w;

            let mut cur_slot_b = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
            let cur_acc_b =
                refresh_kingring_perspective_runtime(&pos, &rt, Color::Black, &mut cur_slot_b);
            let mut cur_slot_w = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
            let cur_acc_w =
                refresh_kingring_perspective_runtime(&pos, &rt, Color::White, &mut cur_slot_w);

            if !king_moved_b {
                let mut diff_acc_b = prev_acc_b;
                apply_slot_diff_runtime(
                    &rt,
                    Color::Black,
                    &prev_slot_b,
                    &cur_slot_b,
                    &mut diff_acc_b,
                );
                assert_eq!(cur_acc_b, diff_acc_b, "Black: slot diff mismatch (king stationary)");
                checked_steps += 1;
            }
            if !king_moved_w {
                let mut diff_acc_w = prev_acc_w;
                apply_slot_diff_runtime(
                    &rt,
                    Color::White,
                    &prev_slot_w,
                    &cur_slot_w,
                    &mut diff_acc_w,
                );
                assert_eq!(cur_acc_w, diff_acc_w, "White: slot diff mismatch (king stationary)");
                checked_steps += 1;
            }

            prev_slot_b = cur_slot_b;
            prev_slot_w = cur_slot_w;
            prev_acc_b = cur_acc_b;
            prev_acc_w = cur_acc_w;
            prev_king_b = cur_king_b;
            prev_king_w = cur_king_w;
        }
        // 50 step / 2 persp ≈ 100 機会、 king 移動 ply は少数なので大半が check 対象になるはず。
        assert!(checked_steps > 20, "expected many stationary-king checks");
    }

    /// incremental reuse (= affected sq mask + classify_sq selective) が full refresh と
    /// **完全 bit-exact** であることを 100-step random walk で確認 (= 2026-05-29 追加)。
    /// king 移動 ply は incremental の契約外なので skip。
    #[test]
    fn incremental_reuse_matches_full_refresh_random_walk() {
        use crate::movegen::{MoveList, generate_legal};
        use crate::nnue::dual_delta::compute_move_delta;

        let mut pos = make_pos();
        let mut disk_weights: Box<KingRingWeights> = Box::new(
            [[[[[0i16; KINGRING_NUM_STATE]; KINGRING_NUM_PIECE]; KINGRING_NUM_SQ]; 2];
                KINGRING_NUM_BUCKETS],
        );
        let mut seed: i16 = 11;
        for b in 0..KINGRING_NUM_BUCKETS {
            for p in 0..2 {
                for s in 0..KINGRING_NUM_SQ {
                    for pi in 0..KINGRING_NUM_PIECE {
                        for st in 0..KINGRING_NUM_STATE {
                            disk_weights[b][p][s][pi][st] = seed;
                            seed = seed.wrapping_add(7);
                        }
                    }
                }
            }
        }
        let rt = build_runtime_weights(&disk_weights);

        let mut prev_slot_b = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
        let mut prev_slot_w = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
        let mut prev_acc_b =
            refresh_kingring_perspective_runtime(&pos, &rt, Color::Black, &mut prev_slot_b);
        let mut prev_acc_w =
            refresh_kingring_perspective_runtime(&pos, &rt, Color::White, &mut prev_slot_w);
        let mut prev_king_b = pos.king_square(Color::Black);
        let mut prev_king_w = pos.king_square(Color::White);

        let mut rng_state: u64 = 0xFEED_BABE;
        let mut checked_steps = 0usize;
        for _step in 0..100 {
            let mut list = MoveList::new();
            generate_legal(&pos, &mut list);
            if list.is_empty() {
                break;
            }
            rng_state =
                rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pick = (rng_state >> 33) as usize % list.len();
            let mv = list.at(pick);
            let gc = pos.gives_check(mv);
            let _dirty = pos.do_move(mv, gc);

            let cur_king_b = pos.king_square(Color::Black);
            let cur_king_w = pos.king_square(Color::White);
            let king_moved_b = cur_king_b != prev_king_b;
            let king_moved_w = cur_king_w != prev_king_w;

            let mut full_slot_b = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
            let full_acc_b =
                refresh_kingring_perspective_runtime(&pos, &rt, Color::Black, &mut full_slot_b);
            let mut full_slot_w = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
            let full_acc_w =
                refresh_kingring_perspective_runtime(&pos, &rt, Color::White, &mut full_slot_w);

            let delta = compute_move_delta(&pos);

            if let Some(ref d) = delta {
                if !king_moved_b && !d.king_moved_any {
                    let mut inc_slot_b = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
                    let inc_acc_b = update_kingring_perspective_incremental(
                        &pos,
                        &rt,
                        Color::Black,
                        d,
                        &prev_slot_b,
                        &prev_acc_b,
                        &mut inc_slot_b,
                    );
                    assert_eq!(
                        full_acc_b, inc_acc_b,
                        "Black: incremental acc mismatch (king stationary)"
                    );
                    assert_eq!(
                        full_slot_b, inc_slot_b,
                        "Black: incremental slots mismatch (king stationary)"
                    );
                    checked_steps += 1;
                }
                if !king_moved_w && !d.king_moved_any {
                    let mut inc_slot_w = [KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ];
                    let inc_acc_w = update_kingring_perspective_incremental(
                        &pos,
                        &rt,
                        Color::White,
                        d,
                        &prev_slot_w,
                        &prev_acc_w,
                        &mut inc_slot_w,
                    );
                    assert_eq!(
                        full_acc_w, inc_acc_w,
                        "White: incremental acc mismatch (king stationary)"
                    );
                    assert_eq!(
                        full_slot_w, inc_slot_w,
                        "White: incremental slots mismatch (king stationary)"
                    );
                    checked_steps += 1;
                }
            }

            prev_slot_b = full_slot_b;
            prev_slot_w = full_slot_w;
            prev_acc_b = full_acc_b;
            prev_acc_w = full_acc_w;
            prev_king_b = cur_king_b;
            prev_king_w = cur_king_w;
        }
        // 100 step / 2 persp ≈ 200 機会、 king 移動 ply は少数なので大半 check 対象になるはず。
        assert!(checked_steps > 40, "expected many incremental checks");
    }
}
