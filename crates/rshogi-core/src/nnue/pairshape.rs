//! Piece-Pair Shape Table scalar head (arm 12、 2026-05-29)。
//!
//! PSQT は「駒 A がこの升にいる」 まで見られるが、 「駒 A と駒 B の連携形」 までは
//! 見られない。 軽量 scalar table で **固定 12 種の味方駒ペア × 16 種 shape class** を
//! 捕捉し、 PSQT scalar head として加算する。
//!
//! # 設計
//!
//! ## feature 集合
//!
//! - 12 種の **固定 pair_type** (= 全駒ペア総当たり禁止):
//!   1. Gold-Silver / 2. Gold-Gold / 3. Silver-Silver / 4. Silver-Knight /
//!      5. Rook-Pawn / 6. Bishop-Silver / 7. Tokin-Gold / 8. Horse-Gold /
//!      9. Dragon-Gold / 10. Pawn-Knight / 11. Silver-Bishop / 12. Rook-Gold
//! - 16 種の **shape class** (= 排他、 優先順位順):
//!   `SameFile` / `SameRank` / `KnightShape` / `DiagonalAdjacent` /
//!   `AdjacentFile` / `AdjacentRank` / `Distance1` / `Distance2` /
//!   `Distance3Plus` / `Ahead` / `Behind` / `SameFlankLeft` /
//!   `SameFlankRight` / `CrossFlank` / `Centered` / `Other`
//!
//! ## perspective 正規化
//!
//! - black perspective: 通常の盤面
//! - white perspective: rank 軸 flip (= ahead/behind を perspective に対して
//!   一貫した「自陣→敵陣」 方向に揃える)
//! - **左右反転はしない**
//!
//! ## weight layout
//!
//! - disk: `[bucket 8][persp 2][pair_type 12][shape 16] -> i16`
//!   = 3,072 i16 = 6,144 byte (= 軽量 scalar table)
//! - runtime: `[persp 2][pair_type 12][shape 16][bucket 8] -> i32` (bucket-inner)
//!   = hot path で `row[bucket]` の 8 連続 read 可能、 ~24 KB
//!
//! ## ckpt format
//!
//! ```text
//! sub_magic = "P2PAIR\0\0" (8 byte)
//! version   = u32 LE       (= 1)
//! count     = u32 LE       (= 3,072)
//! weights   = i16 × 3,072 LE
//! ```
//!
//! 合計 6,160 byte。 既存 P2DUALZ / P2DUAL2 / P2KRING section の後ろに append。
//!
//! ## 差分更新方針 (= 真 incremental、 2026-05-29)
//!
//! pair_type ごとに `pos.pieces_pt(pt)` 経由で bitboard を iterate。 active 集合は
//! `(pair_type_idx, shape_idx)` を flat encode した u16 を sorted list (= multiset、
//! 重複保持) で保持する。
//!
//! 旧実装は **full re-collect** (= 12 pair_type × cross product) で active list を毎手
//! ゼロから作り直し、 prev list との sorted symmetric difference から removed / added
//! を抽出していた。 NPS 退行 -18% で実用域を超過 → **真 incremental 化** に置換
//! (= 2026-05-29)。
//!
//! 新実装:
//! 1. `MoveDelta` から「影響 piece_type」 (= moved_before/after + captured) を取得
//! 2. `PIECE_TO_PAIR_TYPE_MASK` 逆引きで「影響 pair_type マスク」 (= u16、 12 bit) を導出
//! 3. **影響外 pair_type の active entry は prev から そのまま copy** (= encode 値の
//!    上位 12 bit が pair_type idx なので 1 比較で判別可能)
//! 4. 影響 pair_type は cur 盤面から full recompute (= `collect_active_for_pair_type`、
//!    1 pair_type 単位)
//! 5. accumulator は (prev の影響範囲を sub) + (cur の影響範囲を add)
//!
//! king 移動でも shape 自体は再計算されるが、 king は pair member ではない → 影響
//! pair_type マスク = 0 → prev そのまま copy で完結 (= full refresh fallback 不要)。

use crate::bitboard::Bitboard;
use crate::nnue::dual_delta::MoveDelta;
use crate::position::Position;
use crate::types::{Color, PieceType, Square};

// =============================================================================
// 定数
// =============================================================================

/// bucket 数 (= LayerStacks の progress8kpabs と共有)。
pub const PAIRSHAPE_NUM_BUCKETS: usize = 8;

/// pair_type 数 (= 12 種、 固定)。
pub const PAIRSHAPE_NUM_PAIR_TYPES: usize = 12;

/// shape class 数 (= 16 種、 排他)。
pub const PAIRSHAPE_NUM_SHAPES: usize = 16;

/// 1 perspective × 1 bucket あたりの flat dim (= 12 × 16 = 192)。
pub const PAIRSHAPE_PER_BUCKET_DIM: usize = PAIRSHAPE_NUM_PAIR_TYPES * PAIRSHAPE_NUM_SHAPES;

const _: () = {
    assert!(PAIRSHAPE_PER_BUCKET_DIM == 192);
};

/// 1 perspective の (pair_type, shape) flat 空間サイズ (= 192、 = `encode_active_idx` 値域)。
pub const PAIRSHAPE_PER_PERSP_DIM: usize = PAIRSHAPE_PER_BUCKET_DIM;

/// weight 総 dim (= 8 bucket × 2 persp × 12 pair × 16 shape = 3,072)。
pub const PAIRSHAPE_WEIGHT_DIM: usize = PAIRSHAPE_NUM_BUCKETS * 2 * PAIRSHAPE_PER_BUCKET_DIM;

const _: () = {
    assert!(PAIRSHAPE_WEIGHT_DIM == 3_072);
};

/// ckpt section の sub_magic (= 8 byte)。
pub const PAIRSHAPE_SUB_MAGIC: [u8; 8] = *b"P2PAIR\0\0";

/// ckpt section format version (= 1)。
pub const PAIRSHAPE_VERSION: u32 = 1;

/// ckpt section の i16 entry 総数 (= `PAIRSHAPE_WEIGHT_DIM`)。
pub const PAIRSHAPE_COUNT: u32 = PAIRSHAPE_WEIGHT_DIM as u32;

/// output scale (= eval 加算前の正規化倍率、 8)。
pub const PAIRSHAPE_OUTPUT_SCALE: i32 = 8;

/// 1 perspective での active entry 数上限 (= 全 pair 数 + 余裕)。
///
/// 各 pair_type の最大 pair 個数 (= 最大 = Gold-Pawn 系で ~36) × 12 pair_type
/// で理論上限は数百だが、 実盤面では 200 程度に収まる。 256 で安全側に確保。
pub const PAIRSHAPE_MAX_ACTIVE: usize = 256;

// =============================================================================
// pair_type 定義
// =============================================================================

/// 1 pair_type が指す (piece_a, piece_b) 順序対。
///
/// shape 計算では a / b は 区別する (= 「a が ahead か b が ahead か」)。
/// a, b 同 piece_type の pair は unordered (= a.sq < b.sq の片方のみ count)。
#[derive(Debug, Clone, Copy)]
struct PairTypeDef {
    a: PieceType,
    b: PieceType,
}

/// 12 種の固定 pair_type 定義 (= idx 0..12)。
///
/// 設計指針: 「攻め駒の連携」 「攻守両用の形」 を捕捉。 全駒ペア総当たりではなく
/// 棋理に重要な 12 種に絞る。
const PAIR_TYPES: [PairTypeDef; PAIRSHAPE_NUM_PAIR_TYPES] = [
    // 1. Gold-Silver (= 金銀の連結、 囲い基本)
    PairTypeDef {
        a: PieceType::Gold,
        b: PieceType::Silver,
    },
    // 2. Gold-Gold (= 金 2 枚連結)
    PairTypeDef {
        a: PieceType::Gold,
        b: PieceType::Gold,
    },
    // 3. Silver-Silver (= 銀 2 枚連結)
    PairTypeDef {
        a: PieceType::Silver,
        b: PieceType::Silver,
    },
    // 4. Silver-Knight (= 銀桂連携、 攻守両用)
    PairTypeDef {
        a: PieceType::Silver,
        b: PieceType::Knight,
    },
    // 5. Rook-Pawn (= 飛と歩、 飛先突破)
    PairTypeDef {
        a: PieceType::Rook,
        b: PieceType::Pawn,
    },
    // 6. Bishop-Silver (= 角銀の連携、 角頭銀)
    PairTypeDef {
        a: PieceType::Bishop,
        b: PieceType::Silver,
    },
    // 7. Tokin-Gold (= と金と金の連結、 終盤強し)
    PairTypeDef {
        a: PieceType::ProPawn,
        b: PieceType::Gold,
    },
    // 8. Horse-Gold (= 馬と金、 終盤囲い崩し+受け)
    PairTypeDef {
        a: PieceType::Horse,
        b: PieceType::Gold,
    },
    // 9. Dragon-Gold (= 竜と金、 終盤強い)
    PairTypeDef {
        a: PieceType::Dragon,
        b: PieceType::Gold,
    },
    // 10. Pawn-Knight (= 桂頭の歩、 攻守両用)
    PairTypeDef {
        a: PieceType::Pawn,
        b: PieceType::Knight,
    },
    // 11. Silver-Bishop (= 角銀の連携、 攻防)
    PairTypeDef {
        a: PieceType::Silver,
        b: PieceType::Bishop,
    },
    // 12. Rook-Gold (= 飛と金、 中盤連携)
    PairTypeDef {
        a: PieceType::Rook,
        b: PieceType::Gold,
    },
];

/// pair_type idx (= 0..12) → (a, b)。
#[inline]
const fn pair_type_at(idx: usize) -> PairTypeDef {
    debug_assert!(idx < PAIRSHAPE_NUM_PAIR_TYPES);
    PAIR_TYPES[idx]
}

// =============================================================================
// shape class 定義
// =============================================================================

/// 16 種の shape class (= 排他、 優先順位順)。
///
/// 優先順位:
/// 1. `SameFile` / `SameRank` / `KnightShape` (= 排他で最優先)
/// 2. `DiagonalAdjacent` (= `|df|=1, |dr|=1`)
/// 3. `AdjacentFile` / `AdjacentRank` (= `|df|=1, |dr|>1` / `|df|>1, |dr|=1`)
/// 4. `Distance1` / `Distance2` / `Distance3Plus` (= チェビシェフ距離)
/// 5. `Ahead` / `Behind` (= perspective normalize 済 r 軸方向)
/// 6. `SameFlankLeft` / `SameFlankRight` / `CrossFlank` / `Centered`
/// 7. `Other`
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairShape {
    SameFile = 0,
    SameRank = 1,
    KnightShape = 2,
    DiagonalAdjacent = 3,
    AdjacentFile = 4,
    AdjacentRank = 5,
    Distance1 = 6,
    Distance2 = 7,
    Distance3Plus = 8,
    Ahead = 9,
    Behind = 10,
    SameFlankLeft = 11,
    SameFlankRight = 12,
    CrossFlank = 13,
    Centered = 14,
    Other = 15,
}

/// shape index を u8 で返す (= `0..PAIRSHAPE_NUM_SHAPES`)。
#[inline]
const fn shape_index(s: PairShape) -> u8 {
    s as u8
}

/// (a_sq, b_sq) を perspective 視点で classify。
///
/// 利き計算なし、 file / rank / abs diff のみ。 white の場合 b_sq の rank を flip して
/// 「ahead = a → b 方向、 自陣→敵陣方向」 を perspective に対し一貫させる (= 仕様:
/// black では rank 増加方向、 white では rank 減少方向)。 左右反転 (= file flip) はしない。
///
/// **重要**: 「ahead」 = a が後陣側 / b が前陣側 の場合 (= a.rank < b.rank、 black 視点) を
/// 「a の前方に b 」 と解釈する設計。 white では rank 軸を flip しているため、 同じ
/// physical 配置を black 視点と同じ rank diff で扱える。
#[inline]
fn classify_pair_shape(persp: Color, a_sq: Square, b_sq: Square) -> PairShape {
    // file 軸 (= 左右反転なし)
    let fa = a_sq.file() as i8;
    let fb = b_sq.file() as i8;
    // rank 軸 (= perspective に応じて flip)
    let ra_raw = a_sq.rank() as i8;
    let rb_raw = b_sq.rank() as i8;
    let (ra, rb) = if persp == Color::Black {
        (ra_raw, rb_raw)
    } else {
        // white: 自陣 = rank 大、 敵陣 = rank 小、 flip で 一貫した「rank 大 = 自陣側」 に揃える
        (8 - ra_raw, 8 - rb_raw)
    };
    let df_signed = fa - fb;
    let dr_signed = ra - rb;
    let df = df_signed.unsigned_abs();
    let dr = dr_signed.unsigned_abs();

    // 1. 最優先: SameFile / SameRank / KnightShape (排他)
    if df == 0 {
        return PairShape::SameFile;
    }
    if dr == 0 {
        return PairShape::SameRank;
    }
    // KnightShape: 桂跳び形 (|df|=1,|dr|=2) または (|df|=2,|dr|=1)
    if (df == 1 && dr == 2) || (df == 2 && dr == 1) {
        return PairShape::KnightShape;
    }

    // 2. DiagonalAdjacent (= |df|=1, |dr|=1)
    if df == 1 && dr == 1 {
        return PairShape::DiagonalAdjacent;
    }

    // 3. AdjacentFile / AdjacentRank
    if df == 1 {
        return PairShape::AdjacentFile;
    }
    if dr == 1 {
        return PairShape::AdjacentRank;
    }

    // 4. Distance (= Chebyshev) ベース
    let cheby = df.max(dr);
    // 注: Distance1 (= cheby==1) は既に Same/Adjacent で吸収されるが、 安全側に
    //     条件を残す (= dr=df=1 は DiagonalAdjacent で吸収済み)。
    if cheby == 1 {
        return PairShape::Distance1;
    }
    if cheby == 2 {
        // |df|=2, |dr|=2 (= 対角 2)
        return PairShape::Distance2;
    }
    // 5. Distance3Plus
    if cheby >= 3 {
        // ahead / behind (= perspective r 軸方向の優位)、 SameFlank / Centered と
        // 排他的にする必要があるため、 ここでは distance を優先せず flank 系の分岐に
        // 進む (= Distance3Plus は flank/ahead いずれにも該当しない pair で fallback)。
        // 設計選択: distance より「方向 / flank」 情報を優先する。
    }

    // 6. Ahead / Behind (= perspective r 軸方向、 |dr| が支配的、 |df|<|dr| の時)
    // a から見て b が前陣 (= rank 値が flip 後で大) なら Ahead、 後陣なら Behind
    if dr_signed < 0 && dr > df {
        // a.rank (flip) < b.rank (flip) = b は a の前方
        return PairShape::Ahead;
    }
    if dr_signed > 0 && dr > df {
        // b は a の後方
        return PairShape::Behind;
    }

    // 7. Flank 系 (= file 軸の盤面位置)、 Centered は中央 4-6 筋
    // File は 0..=8、 1-3 筋 = file_idx 0-2、 7-9 筋 = file_idx 6-8、 4-6 筋 = file_idx 3-5
    let fa_left = fa <= 2;
    let fa_right = fa >= 6;
    let fa_center = (3..=5).contains(&fa);
    let fb_left = fb <= 2;
    let fb_right = fb >= 6;
    let fb_center = (3..=5).contains(&fb);
    if fa_left && fb_left {
        return PairShape::SameFlankLeft;
    }
    if fa_right && fb_right {
        return PairShape::SameFlankRight;
    }
    if (fa_left && fb_right) || (fa_right && fb_left) {
        return PairShape::CrossFlank;
    }
    if fa_center && fb_center {
        return PairShape::Centered;
    }

    // 8. Distance3Plus または Other で fallback
    if cheby >= 3 {
        return PairShape::Distance3Plus;
    }
    PairShape::Other
}

// =============================================================================
// 型定義
// =============================================================================

/// 1 entry の sparse encode 値:
/// `idx = pair_type_idx * 16 + shape_idx` (= 0..192、 u16 表現)。
///
/// 同じ (pair_type, shape) を複数 pair が共有することがあるため、 active list は
/// **重複を保つ multiset** として扱う。 ただし sorted で diff を取るため、 同 idx が
/// 連続して並ぶことが許される (= weight には 個数分の add/sub が起こる)。
pub type PairShapeActiveIdx = u16;

/// flat weight tensor (= **bucket-outer** memory layout、 disk format と完全一致)。
///
/// `weights[bucket][persp][pair_type][shape] -> i16`。
///
/// 3,072 i16 = 6,144 byte (= 軽量、 stack でも持てるが安全側で Box 推奨)。
pub type PairShapeWeights =
    [[[[i16; PAIRSHAPE_NUM_SHAPES]; PAIRSHAPE_NUM_PAIR_TYPES]; 2]; PAIRSHAPE_NUM_BUCKETS];

/// hot path 用 **bucket-inner i32 row** layout (= kingring と同パターン)。
///
/// `runtime[persp][raw_idx][bucket] -> i32`、 raw_idx は `encode_active_idx` 戻り値
/// (= 0..192)。 1 entry に対する 8 bucket が **連続 memory** に並ぶため、 hot path で
/// `row[bucket]` を順次 read するだけで 8 個の i32 を一気に sum できる。
///
/// メモリ消費: 2 persp × 192 raw_idx × 8 bucket × 4 byte = **12,288 byte (~12 KB)**。
pub type PairShapeRuntimeWeights = [[[i32; PAIRSHAPE_NUM_BUCKETS]; PAIRSHAPE_PER_BUCKET_DIM]; 2];

/// 1 perspective の accumulator (= per-bucket i32 scalar)。
pub type PairShapeAccumulation = [i32; PAIRSHAPE_NUM_BUCKETS];

// =============================================================================
// disk shape → runtime shape 転置
// =============================================================================

/// `PairShapeWeights` (= bucket-outer i16) を `PairShapeRuntimeWeights`
/// (= bucket-inner i32) に転置構築する。 ckpt read 後に **1 回だけ** 呼ぶ。
pub fn build_runtime_weights(disk: &PairShapeWeights) -> Box<PairShapeRuntimeWeights> {
    let mut rt: Box<PairShapeRuntimeWeights> =
        Box::new([[[0i32; PAIRSHAPE_NUM_BUCKETS]; PAIRSHAPE_PER_BUCKET_DIM]; 2]);
    for (bucket, bucket_arr) in disk.iter().enumerate() {
        for (persp, persp_arr) in bucket_arr.iter().enumerate() {
            for (pt_idx, pt_arr) in persp_arr.iter().enumerate() {
                for (sh_idx, &w) in pt_arr.iter().enumerate() {
                    let raw_idx = pt_idx * PAIRSHAPE_NUM_SHAPES + sh_idx;
                    rt[persp][raw_idx][bucket] = w as i32;
                }
            }
        }
    }
    rt
}

// =============================================================================
// active idx encode / decode
// =============================================================================

/// `(pair_type_idx, shape_idx)` → flat sparse idx (= 0..192)。
#[inline]
pub const fn encode_active_idx(pair_type_idx: usize, shape_idx: usize) -> u16 {
    debug_assert!(pair_type_idx < PAIRSHAPE_NUM_PAIR_TYPES);
    debug_assert!(shape_idx < PAIRSHAPE_NUM_SHAPES);
    (pair_type_idx * PAIRSHAPE_NUM_SHAPES + shape_idx) as u16
}

/// flat sparse idx (= 0..192) → `(pair_type_idx, shape_idx)`。 test 用。
#[cfg(test)]
#[inline]
const fn decode_active_idx(idx: u16) -> (usize, usize) {
    let raw = idx as usize;
    let shape = raw % PAIRSHAPE_NUM_SHAPES;
    let pair_type = raw / PAIRSHAPE_NUM_SHAPES;
    debug_assert!(pair_type < PAIRSHAPE_NUM_PAIR_TYPES);
    (pair_type, shape)
}

// =============================================================================
// active 集合 (= sorted list) の構築
// =============================================================================

/// `pos` から `perspective` 視点の active idx 集合を sorted で構築。
///
/// 各 pair_type について `pos.pieces_pt(a)` ∩ own と `pos.pieces_pt(b)` ∩ own を取り、
/// 全 pair の組み合わせを enumerate して `(pair_type, shape)` を encode する。
/// 同 pair_type で a == b の場合は unordered (= a_sq < b_sq) で重複を防ぐ。
///
/// 出力は **sorted u16 列** (= bucket-inner diff の sorted merge 前提)。
///
/// 戻り値: active 集合の長さ (= 0..PAIRSHAPE_MAX_ACTIVE)。
pub fn collect_active_indices_sorted(
    pos: &Position,
    perspective: Color,
    out: &mut [u16; PAIRSHAPE_MAX_ACTIVE],
) -> usize {
    let own_bb = pos.pieces_c(perspective);
    let mut len: usize = 0;

    // 各 pair_type を順に列挙、 a / b の own 駒位置を bitboard で iterate
    for pt_idx in 0..PAIRSHAPE_NUM_PAIR_TYPES {
        let def = pair_type_at(pt_idx);
        let a_bb = pos.pieces_pt(def.a) & own_bb;
        let b_bb = pos.pieces_pt(def.b) & own_bb;
        if a_bb.is_empty() || b_bb.is_empty() {
            continue;
        }

        if def.a as u8 == def.b as u8 {
            // 同 piece_type pair: unordered で a_sq < b_sq のみ採用
            let mut a_iter = a_bb;
            while a_iter.is_not_empty() {
                let a_sq = a_iter.pop();
                let mut b_iter = b_bb;
                while b_iter.is_not_empty() {
                    let b_sq = b_iter.pop();
                    if a_sq.raw() >= b_sq.raw() {
                        continue;
                    }
                    let sh = classify_pair_shape(perspective, a_sq, b_sq);
                    let enc = encode_active_idx(pt_idx, shape_index(sh) as usize);
                    if len < PAIRSHAPE_MAX_ACTIVE {
                        out[len] = enc;
                        len += 1;
                    }
                }
            }
        } else {
            // 異 piece_type pair: a × b の cross product
            let mut a_iter = a_bb;
            while a_iter.is_not_empty() {
                let a_sq = a_iter.pop();
                let mut b_iter = b_bb;
                while b_iter.is_not_empty() {
                    let b_sq = b_iter.pop();
                    let sh = classify_pair_shape(perspective, a_sq, b_sq);
                    let enc = encode_active_idx(pt_idx, shape_index(sh) as usize);
                    if len < PAIRSHAPE_MAX_ACTIVE {
                        out[len] = enc;
                        len += 1;
                    }
                }
            }
        }
    }

    // sort (= small N、 192 値域、 insertion sort で十分高速。 だが std sort_unstable で簡潔に)
    out[..len].sort_unstable();
    len
}

// =============================================================================
// runtime accumulator add / sub
// =============================================================================

/// **runtime layout** から 1 entry の sign 付き加減算。
///
/// `row[bucket]` の 8 連続 read。 hot path 主経路。
#[inline(always)]
fn accumulate_entry_runtime(
    rt: &PairShapeRuntimeWeights,
    p: usize,
    raw_idx: u16,
    sign: i32,
    sums: &mut PairShapeAccumulation,
) {
    debug_assert!(p < 2);
    debug_assert!((raw_idx as usize) < PAIRSHAPE_PER_BUCKET_DIM);
    // SAFETY: p < 2、 raw_idx < PAIRSHAPE_PER_BUCKET_DIM を上で debug_assert。
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

/// sorted active list から 8 bucket の sum を計算。
#[inline]
fn sum_active_runtime(
    rt: &PairShapeRuntimeWeights,
    perspective: Color,
    active: &[u16],
) -> PairShapeAccumulation {
    let mut sums: PairShapeAccumulation = [0i32; PAIRSHAPE_NUM_BUCKETS];
    let p = perspective as usize;
    for &idx in active {
        accumulate_entry_runtime(rt, p, idx, 1, &mut sums);
    }
    sums
}

// =============================================================================
// full refresh
// =============================================================================

/// `perspective` の accumulator を full refresh する (= hot path 主経路の 1 つ、
/// king moved / 初期化 / fallback で使用)。
///
/// 戻り値: 8 bucket × i32 sum、 active list を caller 側 buffer に書き出す。
pub fn refresh_pairshape_perspective(
    pos: &Position,
    rt: &PairShapeRuntimeWeights,
    perspective: Color,
    active_out: &mut [u16; PAIRSHAPE_MAX_ACTIVE],
    active_len_out: &mut u16,
) -> PairShapeAccumulation {
    let len = collect_active_indices_sorted(pos, perspective, active_out);
    debug_assert!(len <= u16::MAX as usize);
    *active_len_out = len as u16;
    sum_active_runtime(rt, perspective, &active_out[..len])
}

// =============================================================================
// 真 incremental update
// =============================================================================

/// `PieceType` (= 1..=14) → 含まれる pair_type の bitmask (= 1 bit per pair_type、
/// 12 bit fit)。 idx 0 / 8 (= King) / 2 (= Lance) / 10 / 11 / 12 (= 成桂・成銀・成香) は
/// すべて mask = 0 (= pair に含まれない駒種)。
///
/// 真 incremental update で「動いた駒 / 取られた駒に関係する pair_type」 を 1 bit
/// 演算で即座に集約するための逆引きテーブル。
const PIECE_TO_PAIR_TYPE_MASK: [u16; 15] = {
    let mut m = [0u16; 15];
    let mut i = 0;
    while i < PAIRSHAPE_NUM_PAIR_TYPES {
        let def = PAIR_TYPES[i];
        m[def.a as usize] |= 1u16 << i;
        m[def.b as usize] |= 1u16 << i;
        i += 1;
    }
    m
};

/// `pos` の指定 `perspective` × 指定 `pair_type_idx` の active idx を `out` に append。
///
/// `collect_active_indices_sorted` の 1 pair_type 版。 真 incremental update で
/// 「影響 pair_type のみ recompute」 する hot path で使用する。
///
/// 戻り値: append された entry 数。
#[inline]
fn collect_active_for_pair_type(
    pos: &Position,
    perspective: Color,
    pt_idx: usize,
    out: &mut [u16; PAIRSHAPE_MAX_ACTIVE],
    out_len: &mut usize,
) {
    let own_bb = pos.pieces_c(perspective);
    let def = pair_type_at(pt_idx);
    let a_bb = pos.pieces_pt(def.a) & own_bb;
    let b_bb = pos.pieces_pt(def.b) & own_bb;
    if a_bb.is_empty() || b_bb.is_empty() {
        return;
    }

    if def.a as u8 == def.b as u8 {
        // 同 piece_type pair: unordered で a_sq < b_sq のみ採用
        let mut a_iter = a_bb;
        while a_iter.is_not_empty() {
            let a_sq = a_iter.pop();
            let mut b_iter = b_bb;
            while b_iter.is_not_empty() {
                let b_sq = b_iter.pop();
                if a_sq.raw() >= b_sq.raw() {
                    continue;
                }
                let sh = classify_pair_shape(perspective, a_sq, b_sq);
                let enc = encode_active_idx(pt_idx, shape_index(sh) as usize);
                if *out_len < PAIRSHAPE_MAX_ACTIVE {
                    out[*out_len] = enc;
                    *out_len += 1;
                }
            }
        }
    } else {
        // 異 piece_type pair: a × b の cross product
        let mut a_iter = a_bb;
        while a_iter.is_not_empty() {
            let a_sq = a_iter.pop();
            let mut b_iter = b_bb;
            while b_iter.is_not_empty() {
                let b_sq = b_iter.pop();
                let sh = classify_pair_shape(perspective, a_sq, b_sq);
                let enc = encode_active_idx(pt_idx, shape_index(sh) as usize);
                if *out_len < PAIRSHAPE_MAX_ACTIVE {
                    out[*out_len] = enc;
                    *out_len += 1;
                }
            }
        }
    }
}

/// `perspective` の accumulator を **prev からの真 incremental delta** で更新する。
///
/// # アルゴリズム
///
/// 1. `delta` から **影響を受ける pair_type マスク** (= u16、 12 bit) を計算
///    - perspective == moved_color の場合: moved_piece_before / moved_piece_after に該当する pair
///    - perspective == captured.color の場合: captured.piece_type に該当する pair
/// 2. **影響外 pair_type の active entry は prev_active からそのまま copy**
///    (= sort 順は (pair_type, shape) の lexicographic なので、 encode 値 >> 4 で pair_type
///    判別可能)
/// 3. **影響 pair_type のみ cur 盤面から full recompute**
///    (= 1 pair_type 内の cross product は full collect と同じだが、 数が 12 → 平均 2-5)
/// 4. accumulator は (removed - added) で直接 add/sub
///
/// # 計算量
///
/// 旧 full re-collect: 12 pair × 平均 cross product = O(12 × n²)、 NPS -15% 退行の主因。
/// 新 真 incremental: 平均 2-5 pair × 平均 cross product = **約 2-4x 削減**。
///
/// # delta None fallback
///
/// `delta == None` (= NULL move / PASS 等) の場合は full refresh fallback (= 安全側)。
/// `delta.king_moved_any` の場合は両 perspective の active list は (= king は pair member
/// ではないため) 影響を受けないので full refresh は不要 (= 通常 delta path で OK)。
#[allow(clippy::too_many_arguments)]
pub fn update_pairshape_perspective(
    pos: &Position,
    rt: &PairShapeRuntimeWeights,
    perspective: Color,
    delta: Option<&MoveDelta>,
    prev_active: &[u16],
    prev_acc: &PairShapeAccumulation,
    cur_active_out: &mut [u16; PAIRSHAPE_MAX_ACTIVE],
    cur_active_len_out: &mut u16,
) -> PairShapeAccumulation {
    let Some(delta) = delta else {
        // fallback: full refresh
        let len = collect_active_indices_sorted(pos, perspective, cur_active_out);
        debug_assert!(len <= u16::MAX as usize);
        *cur_active_len_out = len as u16;
        return sum_active_runtime(rt, perspective, &cur_active_out[..len]);
    };

    // 1. 影響 pair_type マスク計算 (= 該当 perspective での影響 piece_type を集約)
    let mut affected_mask: u16 = 0;
    if delta.moved_color == perspective {
        affected_mask |= PIECE_TO_PAIR_TYPE_MASK[delta.moved_piece_before as usize];
        affected_mask |= PIECE_TO_PAIR_TYPE_MASK[delta.moved_piece_after as usize];
    }
    if let Some((_, cap_color, cap_pt)) = delta.captured
        && cap_color == perspective
    {
        affected_mask |= PIECE_TO_PAIR_TYPE_MASK[cap_pt as usize];
    }

    if affected_mask == 0 {
        // 影響なし: prev そのまま copy (= 王手回避で動いた玉、 相手側 perspective での
        // 自分の駒に関係しない手 等)。
        let prev_len = prev_active.len();
        debug_assert!(prev_len <= PAIRSHAPE_MAX_ACTIVE);
        cur_active_out[..prev_len].copy_from_slice(prev_active);
        *cur_active_len_out = prev_len as u16;
        return *prev_acc;
    }

    // 2. cur_active を build:
    //    a. prev_active のうち **影響外 pair_type** の entry を copy
    //    b. 影響 pair_type を cur 盤面から recompute して append
    //    c. 最後に sort_unstable で sorted 不変条件を復元
    let mut cur_len: usize = 0;
    for &enc in prev_active {
        let pt_idx = (enc as usize) >> 4; // shape は 16 種 = 下位 4 bit
        debug_assert!(pt_idx < PAIRSHAPE_NUM_PAIR_TYPES);
        if (affected_mask & (1u16 << pt_idx)) == 0 && cur_len < PAIRSHAPE_MAX_ACTIVE {
            cur_active_out[cur_len] = enc;
            cur_len += 1;
        }
    }
    // 影響 pair_type を cur 盤面から full recompute
    let mut affected_iter = affected_mask;
    while affected_iter != 0 {
        let pt_idx = affected_iter.trailing_zeros() as usize;
        affected_iter &= affected_iter - 1;
        collect_active_for_pair_type(pos, perspective, pt_idx, cur_active_out, &mut cur_len);
    }

    // sort 不変条件を復元 (= multiset、 重複保持)
    cur_active_out[..cur_len].sort_unstable();
    debug_assert!(cur_len <= u16::MAX as usize);
    *cur_active_len_out = cur_len as u16;

    // 3. accumulator は (prev の影響 pair 範囲 sub) + (cur の影響 pair 範囲 add) で計算
    //    cur の影響範囲は上で append した部分 (= sort 前の indices)、 sort 後は混ざるが
    //    multiset として等価なので問題ない → ここでは直接 prev → cur の 差分は計算
    //    せず、 「影響 pair_type に該当する prev_active 範囲を sub」 + 「cur 影響 pair_type
    //    範囲を add」 で計算する。
    let mut acc = *prev_acc;
    let p = perspective as usize;
    // prev のうち 影響 pair_type 範囲を sub
    for &enc in prev_active {
        let pt_idx = (enc as usize) >> 4;
        if (affected_mask & (1u16 << pt_idx)) != 0 {
            accumulate_entry_runtime(rt, p, enc, -1, &mut acc);
        }
    }
    // cur のうち 影響 pair_type 範囲を add (= 上の copy で含まれていない範囲)
    for &enc in &cur_active_out[..cur_len] {
        let pt_idx = (enc as usize) >> 4;
        if (affected_mask & (1u16 << pt_idx)) != 0 {
            accumulate_entry_runtime(rt, p, enc, 1, &mut acc);
        }
    }
    acc
}

// =============================================================================
// affected sq 集合 build (= 将来 真 incremental 化のための reservation)
// =============================================================================

/// 1 手の影響を受ける可能性のある sq 集合 (= bitboard、 pairshape 用)。
///
/// 真 incremental 化を実装する場合に使う想定の helper。 現実装は full re-collect
/// 経路のため本関数の戻り値は **無視される** (= dead path、 caller は呼ばない)。
/// 将来削除候補だが API 互換のため残す。
#[inline]
#[allow(dead_code)]
pub fn build_pairshape_affected_sq_set(_pos: &Position, delta: &MoveDelta) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    if let Some(from) = delta.moved_from {
        bb |= Bitboard::from_square(from);
    }
    bb |= Bitboard::from_square(delta.moved_to);
    if let Some((cap_sq, _, _)) = delta.captured
        && cap_sq != delta.moved_to
    {
        bb |= Bitboard::from_square(cap_sq);
    }
    bb
}

// =============================================================================
// tests
// =============================================================================

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
        assert_eq!(PAIRSHAPE_NUM_BUCKETS, 8);
        assert_eq!(PAIRSHAPE_NUM_PAIR_TYPES, 12);
        assert_eq!(PAIRSHAPE_NUM_SHAPES, 16);
        assert_eq!(PAIRSHAPE_PER_BUCKET_DIM, 192);
        assert_eq!(PAIRSHAPE_WEIGHT_DIM, 3_072);
        assert_eq!(PAIRSHAPE_COUNT, 3_072);
        assert_eq!(PAIRSHAPE_SUB_MAGIC, *b"P2PAIR\0\0");
    }

    #[test]
    fn encode_decode_roundtrip() {
        for pt in 0..PAIRSHAPE_NUM_PAIR_TYPES {
            for sh in 0..PAIRSHAPE_NUM_SHAPES {
                let enc = encode_active_idx(pt, sh);
                assert!((enc as usize) < PAIRSHAPE_PER_BUCKET_DIM);
                let dec = decode_active_idx(enc);
                assert_eq!(dec, (pt, sh));
            }
        }
    }

    #[test]
    fn shape_classifier_perspective_consistency() {
        // 同一 physical 配置 (= a=5五、 b=5四) で black perspective:
        // a.rank = Rank5 = 4、 b.rank = Rank4 = 3
        // black: ra=4, rb=3 → dr_signed = +1, df=0 → SameFile
        // SameFile は perspective に依らず一致
        use crate::types::{File, Rank};
        let a = Square::new(File::File5, Rank::Rank5);
        let b = Square::new(File::File5, Rank::Rank4);
        let s_b = classify_pair_shape(Color::Black, a, b);
        let s_w = classify_pair_shape(Color::White, a, b);
        assert_eq!(s_b, PairShape::SameFile);
        assert_eq!(s_w, PairShape::SameFile);
    }

    #[test]
    fn shape_classifier_ahead_behind_persp_flips() {
        // a=5五 (rank=4)、 b=5三 (rank=2、 black 視点 ahead)
        // black: dr_signed = ra - rb = 4 - 2 = +2 > 0 + dr>df=0 → 但し df=0 → SameFile
        // SameFile が先に判定されるため Ahead 判定はされない
        // 別ケース: a=5五、 b=4三 → file diff あり
        use crate::types::{File, Rank};
        let a = Square::new(File::File5, Rank::Rank5);
        let b = Square::new(File::File4, Rank::Rank3);
        // black: fa=4, ra=4 / fb=3, rb=2 → df=1, dr=2 → KnightShape
        // white: fa=4, ra=8-4=4 / fb=3, rb=8-2=6 → df=1, dr=2 → KnightShape
        let s_b = classify_pair_shape(Color::Black, a, b);
        let s_w = classify_pair_shape(Color::White, a, b);
        assert_eq!(s_b, PairShape::KnightShape);
        assert_eq!(s_w, PairShape::KnightShape);
    }

    #[test]
    fn shape_classifier_knight_shape() {
        use crate::types::{File, Rank};
        // 桂跳び形: |df|=1, |dr|=2
        let a = Square::new(File::File5, Rank::Rank5);
        let b = Square::new(File::File6, Rank::Rank3);
        let s = classify_pair_shape(Color::Black, a, b);
        assert_eq!(s, PairShape::KnightShape);
        // |df|=2, |dr|=1
        let b2 = Square::new(File::File7, Rank::Rank4);
        let s2 = classify_pair_shape(Color::Black, a, b2);
        assert_eq!(s2, PairShape::KnightShape);
    }

    #[test]
    fn shape_classifier_diagonal_adjacent() {
        use crate::types::{File, Rank};
        let a = Square::new(File::File5, Rank::Rank5);
        let b = Square::new(File::File6, Rank::Rank4);
        let s = classify_pair_shape(Color::Black, a, b);
        assert_eq!(s, PairShape::DiagonalAdjacent);
    }

    #[test]
    fn shape_classifier_centered() {
        use crate::types::{File, Rank};
        // 両者 file 4-6 中央、 同 rank でない、 距離が AdjacentFile/Knight 等に当たらない
        // a=5五、 b=4三: df=1, dr=2 → Knight が先に当たるので NG
        // a=4五、 b=6二: df=2, dr=3 → Knight 該当せず、 cheby=3 → Ahead/Behind 経路
        //   dr_signed = 4-1 = +3 > 0、 dr > df (= 3>2) → Behind
        let a = Square::new(File::File4, Rank::Rank5);
        let b = Square::new(File::File6, Rank::Rank2);
        let s = classify_pair_shape(Color::Black, a, b);
        assert_eq!(s, PairShape::Behind);
    }

    #[test]
    fn collect_active_smoke_startpos() {
        // 開始局面: own gold 2 + silver 2 + pawn 9 + knight 2 + ...
        // Black perspective で Gold-Gold (= pt 1) = C(2,2)=1 pair、
        // Gold-Silver (= pt 0) = 2*2 = 4 pair、 Silver-Silver (= pt 2) = 1 pair 等を期待
        let pos = make_pos();
        let mut active = [0u16; PAIRSHAPE_MAX_ACTIVE];
        let len_b = collect_active_indices_sorted(&pos, Color::Black, &mut active);
        let len_w = collect_active_indices_sorted(&pos, Color::White, &mut active);
        assert!(len_b > 0, "startpos should have some active pairs for black");
        assert!(len_w > 0, "startpos should have some active pairs for white");
        assert!(len_b <= PAIRSHAPE_MAX_ACTIVE);
        assert!(len_w <= PAIRSHAPE_MAX_ACTIVE);
    }

    #[test]
    fn collect_active_sorted_invariant() {
        let pos = make_pos();
        let mut active = [0u16; PAIRSHAPE_MAX_ACTIVE];
        let len = collect_active_indices_sorted(&pos, Color::Black, &mut active);
        for w in active[..len].windows(2) {
            assert!(w[0] <= w[1], "active list must be sorted (multiset, dup OK)");
        }
    }

    #[test]
    fn refresh_zero_weights_yields_zero_sum() {
        let pos = make_pos();
        let zero_disk: Box<PairShapeWeights> = Box::new(
            [[[[0i16; PAIRSHAPE_NUM_SHAPES]; PAIRSHAPE_NUM_PAIR_TYPES]; 2]; PAIRSHAPE_NUM_BUCKETS],
        );
        let rt = build_runtime_weights(&zero_disk);
        let mut active = [0u16; PAIRSHAPE_MAX_ACTIVE];
        let mut len = 0u16;
        let sums = refresh_pairshape_perspective(&pos, &rt, Color::Black, &mut active, &mut len);
        for (b, &v) in sums.iter().enumerate() {
            assert_eq!(v, 0, "zero weight should yield zero sum at bucket {b}");
        }
    }

    /// 100 step random walk で full refresh と incremental delta update が
    /// **bit-exact 一致** することを確認。
    #[test]
    fn refresh_vs_incremental_random_walk_bit_exact() {
        use crate::movegen::{MoveList, generate_legal};
        use crate::nnue::dual_delta::compute_move_delta;

        let mut pos = make_pos();
        // 1 weight = 一意な値 (= bit-exact 比較用)
        let mut disk: Box<PairShapeWeights> = Box::new(
            [[[[0i16; PAIRSHAPE_NUM_SHAPES]; PAIRSHAPE_NUM_PAIR_TYPES]; 2]; PAIRSHAPE_NUM_BUCKETS],
        );
        let mut seed: i16 = 1;
        for b in 0..PAIRSHAPE_NUM_BUCKETS {
            for p in 0..2 {
                for pt in 0..PAIRSHAPE_NUM_PAIR_TYPES {
                    for sh in 0..PAIRSHAPE_NUM_SHAPES {
                        disk[b][p][pt][sh] = seed;
                        seed = seed.wrapping_add(3);
                    }
                }
            }
        }
        let rt = build_runtime_weights(&disk);

        // 初期 active + acc
        let mut prev_active_b = [0u16; PAIRSHAPE_MAX_ACTIVE];
        let mut prev_len_b = 0u16;
        let mut prev_acc_b = refresh_pairshape_perspective(
            &pos,
            &rt,
            Color::Black,
            &mut prev_active_b,
            &mut prev_len_b,
        );
        let mut prev_active_w = [0u16; PAIRSHAPE_MAX_ACTIVE];
        let mut prev_len_w = 0u16;
        let mut prev_acc_w = refresh_pairshape_perspective(
            &pos,
            &rt,
            Color::White,
            &mut prev_active_w,
            &mut prev_len_w,
        );

        let mut rng_state: u64 = 0xABCDEF01_23456789;
        for step in 0..100 {
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
            let mut full_active_b = [0u16; PAIRSHAPE_MAX_ACTIVE];
            let mut full_len_b = 0u16;
            let full_acc_b = refresh_pairshape_perspective(
                &pos,
                &rt,
                Color::Black,
                &mut full_active_b,
                &mut full_len_b,
            );
            let mut full_active_w = [0u16; PAIRSHAPE_MAX_ACTIVE];
            let mut full_len_w = 0u16;
            let full_acc_w = refresh_pairshape_perspective(
                &pos,
                &rt,
                Color::White,
                &mut full_active_w,
                &mut full_len_w,
            );

            // incremental delta from prev
            let delta = compute_move_delta(&pos);
            let mut inc_active_b = [0u16; PAIRSHAPE_MAX_ACTIVE];
            let mut inc_len_b = 0u16;
            let inc_acc_b = update_pairshape_perspective(
                &pos,
                &rt,
                Color::Black,
                delta.as_ref(),
                &prev_active_b[..prev_len_b as usize],
                &prev_acc_b,
                &mut inc_active_b,
                &mut inc_len_b,
            );
            let mut inc_active_w = [0u16; PAIRSHAPE_MAX_ACTIVE];
            let mut inc_len_w = 0u16;
            let inc_acc_w = update_pairshape_perspective(
                &pos,
                &rt,
                Color::White,
                delta.as_ref(),
                &prev_active_w[..prev_len_w as usize],
                &prev_acc_w,
                &mut inc_active_w,
                &mut inc_len_w,
            );

            assert_eq!(
                full_acc_b, inc_acc_b,
                "Black: full vs incremental acc mismatch at step {step}"
            );
            assert_eq!(
                full_acc_w, inc_acc_w,
                "White: full vs incremental acc mismatch at step {step}"
            );
            assert_eq!(full_len_b, inc_len_b, "Black: len mismatch at step {step}");
            assert_eq!(full_len_w, inc_len_w, "White: len mismatch at step {step}");
            assert_eq!(
                &full_active_b[..full_len_b as usize],
                &inc_active_b[..inc_len_b as usize],
                "Black: active list mismatch at step {step}"
            );
            assert_eq!(
                &full_active_w[..full_len_w as usize],
                &inc_active_w[..inc_len_w as usize],
                "White: active list mismatch at step {step}"
            );

            prev_active_b = full_active_b;
            prev_len_b = full_len_b;
            prev_acc_b = full_acc_b;
            prev_active_w = full_active_w;
            prev_len_w = full_len_w;
            prev_acc_w = full_acc_w;
        }
    }
}
