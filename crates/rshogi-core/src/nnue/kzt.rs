//! King Zone Threat (KZT) sparse PSQT-like head (2026-06-30、 **骨組み v0**)。
//!
//! HalfKA_hm + FT1024 が拾いにくい 「**敵玉周辺の条件付き脅威**」 を補完する
//! ための **大重み数 / 少 active feature** な sparse head。 既存の小型 scalar
//! class head (= KDGR / ADEV / SXRY、 各 128 weight) と異なり、 weight table は
//! 大きい (= ~70 万) が、 1 局面で立つ active feature は少数 (= sparse) に保つ。
//!
//! 本ファイルは **v0 骨組み** であり、 以下のみを実装する:
//! - section format (= magic / version / count) と index 計算式
//! - `OUTPUT_SCALE` 機構 (= KDGR / ADEV / SXRY と同方式、 USI option で runtime 上書き)
//! - **0-weight bit-exact** な `evaluate` 骨組み (= feature gather は未実装の stub、
//!   現状は常に 0 を返すため、 既存 eval に対し完全 bit-exact)
//!
//! **feature 計算本体 (= active feature の収集) は次段階** とし、 ユーザ判断を仰ぐ。
//! `collect_hand_features` は no-heap (= stack 固定 buffer) の sparse gather API
//! 形状だけ提示し、 中身は未実装 (= 空集合) で返す。
//!
//! # v0 family: KZT_HAND (= 推奨初手、 [`KZT_HAND_*`] 定数)
//!
//! 敵玉 5x5 zone の target square に、 **持駒を打てるか / 王手か / 逃げ道封鎖か /
//! 攻防状態** を見る。 feature key:
//!
//! ```text
//! KZT_HAND(persp, bucket, enemy_king_region, hand_piece, target_rel, drop_state)
//! ```
//!
//! - `persp`             : 2  (= 自分 / 相手 視点。 評価は (us - them) 反対称合成)
//! - `bucket`            : 8  (= LayerStack bucket、 既存 head と共通)
//! - `enemy_king_region` : 16 (= 敵玉の盤面領域。 例: file 3band × rank 5band を 16 に量子化)
//! - `hand_piece`        : 7  (= 歩 / 香 / 桂 / 銀 / 金 / 角 / 飛、 [`PieceType::HAND_PIECES`])
//! - `target_rel`        : 25 (= 敵玉中心 5x5 zone 内の相対 square。 dx,dy∈[-2,2])
//! - `drop_state`        : 16 (= 「打てる/王手/逃げ道封鎖/利き有」 等の 4 bit 状態)
//!
//! 合計 = 2 × 8 × 16 × 7 × 25 × 16 = **716,800 weight** (= i16 で ~1.4 MB)。
//! 「打てない」 target は原則 active にしない (= sparse)。
//!
//! # Family B: KZT_XRAY (= v0 では未実装、 設計のみ。 [`king_danger`] / [`slider_xray`] 参照)
//!
//! 敵玉から 8 方向 ray で 飛角香 / 馬竜 の線圧・遮蔽物・距離を粗く評価する別 family。
//! v0 骨組みでは KZT_HAND のみを対象とする。
//!
//! # ckpt format
//!
//! ```text
//! sub_magic = "P2KZT\0\0\0" (8 byte)
//! version   = u32 LE        (= 1)
//! count     = u32 LE        (= KZT_HAND_WEIGHT_COUNT = 716_800)
//! weights   = i16 × count LE
//! ```
//!
//! 既存 section (= ... / P2SXRY) の **物理末尾** に append。 旧 ckpt (= KZT 不在) は
//! section 欠落 → weight `None` → eval 寄与 0 (= bit-exact)。
//!
//! # evaluate (= 骨組み)
//!
//! stm 視点で `(Σ us_active_w - Σ them_active_w) × OUTPUT_SCALE` を返す設計。
//! v0 骨組みでは active feature gather が空のため常に 0。 `OUTPUT_SCALE == 0` でも
//! 即 0 (= scan 省略) で、 いずれの場合も既存 eval と bit-exact。

use crate::bitboard::{
    Bitboard, FILE_BB, bishop_effect, gold_effect, king_5x5_mask, king_effect, knight_effect,
    lance_effect, pawn_effect, rook_effect, silver_effect,
};
use crate::position::Position;
use crate::types::{Color, PieceType, Square};
use std::sync::atomic::{AtomicI32, Ordering};

// KZT-FT (= FT 入力版) と KZT-PSQT (= output 加算版) は同じ KZT 寄与を別経路で
// eval に足すため、 同時有効化すると二重計上になる。 相互排他を強制する。
#[cfg(all(feature = "nnue-kzt-ft", feature = "nnue-kzt-psqt"))]
compile_error!(
    "feature `nnue-kzt-ft` と `nnue-kzt-psqt` は同時に有効化できません \
     (KZT 寄与の二重計上になるため。 FT 入力版 = nnue-kzt-ft、 output 加算版 = nnue-kzt-psqt)"
);

// ---------------------------------------------------------------------------
// section / 次元 定数 (= KZT_HAND)
// ---------------------------------------------------------------------------

/// section magic `P2KZT\0\0\0` (= 8 byte)
pub const KZT_MAGIC: &[u8; 8] = b"P2KZT\0\0\0";

/// section version (= 2、 factored v1 = 5 テーブル分解)。
///
/// version 1 は dense 716,800 i16 (= 廃止)。 version 2 は factored 10,336 i16
/// (= core + region + bucket + persp + bias の 5 テーブルを固定順で連結)。 version 不一致は
/// load 時に明示 reject する (= 旧 dense net は互換なし)。
pub const KZT_VERSION: u32 = 2;

/// perspective 数 (= 自分 / 相手)
pub const KZT_HAND_NUM_PERSP: usize = 2;
/// LayerStack bucket 数 (= 既存 head と共通)
pub const KZT_HAND_NUM_BUCKET: usize = 8;
/// 敵玉領域の量子化数 (= 盤面を 16 領域に分割)
pub const KZT_HAND_NUM_KING_REGION: usize = 16;
/// 持駒の種類数 (= 歩香桂銀金角飛、 [`PieceType::HAND_PIECES`])
pub const KZT_HAND_NUM_PIECE: usize = 7;
/// 敵玉中心 5x5 zone の相対 square 数 (= dx,dy ∈ [-2,2])
pub const KZT_HAND_NUM_TARGET_REL: usize = 25;
/// drop 状態 bit 集合の数 (= 4 bit: 打てる / 王手 / 逃げ道封鎖 / 利き有 など)
pub const KZT_HAND_NUM_DROP_STATE: usize = 16;

/// KZT_HAND の総 weight 数 = 2 × 8 × 16 × 7 × 25 × 16 = 716,800。
pub const KZT_HAND_WEIGHT_COUNT: usize = KZT_HAND_NUM_PERSP
    * KZT_HAND_NUM_BUCKET
    * KZT_HAND_NUM_KING_REGION
    * KZT_HAND_NUM_PIECE
    * KZT_HAND_NUM_TARGET_REL
    * KZT_HAND_NUM_DROP_STATE;

/// 1 局面 / 1 perspective あたりに立ちうる active feature の上限 (= sparse 想定)。
///
/// 最悪でも 「敵玉 5x5 の 25 target × 自分が持つ駒種 (最大 7)」 が上限だが、
/// 「打てる候補のみ active」 制約で実効はこれより遥かに小さい。 hot path での
/// stack buffer サイズに使う (= no-heap、 [`collect_hand_features`])。
pub const KZT_HAND_MAX_ACTIVE: usize = KZT_HAND_NUM_TARGET_REL * KZT_HAND_NUM_PIECE;

// ---------------------------------------------------------------------------
// factored v1 テーブル (= dense 716,800 を 5 低ランクテーブル 10,336 に分解)
// ---------------------------------------------------------------------------
//
// dense 版 (= 6 軸直積 716,800 weight) は過学習で eval 極値のブレが大きいため、 同じ
// active feature を 5 つの低ランクテーブルの和で近似する (= 平滑化)。 gather
// (`collect_hand_features`) は不変で、 evaluate 側で dense flat index を 6 軸に分解し
// (`kzt_hand_decode`)、 各軸を下記 5 テーブルに射影して合算する。
//
// ```text
// score(persp,bucket,king_region,hand_piece,target_rel,drop_state)
//   = W_core  [hand_piece][target_rel][drop_state]
//   + W_region[king_region][target_rel][drop_state]
//   + W_bucket[bucket][hand_piece][drop_state]
//   + W_persp [persp][hand_piece][drop_state]
//   + W_bias  [drop_state]
// ```

/// W_core テーブル要素数 = hand_piece(7) × target_rel(25) × drop_state(16) = 2,800。
pub const KZT_CORE_COUNT: usize =
    KZT_HAND_NUM_PIECE * KZT_HAND_NUM_TARGET_REL * KZT_HAND_NUM_DROP_STATE;
/// W_region テーブル要素数 = king_region(16) × target_rel(25) × drop_state(16) = 6,400。
pub const KZT_REGION_COUNT: usize =
    KZT_HAND_NUM_KING_REGION * KZT_HAND_NUM_TARGET_REL * KZT_HAND_NUM_DROP_STATE;
/// W_bucket テーブル要素数 = bucket(8) × hand_piece(7) × drop_state(16) = 896。
pub const KZT_BUCKET_COUNT: usize =
    KZT_HAND_NUM_BUCKET * KZT_HAND_NUM_PIECE * KZT_HAND_NUM_DROP_STATE;
/// W_persp テーブル要素数 = persp(2) × hand_piece(7) × drop_state(16) = 224。
pub const KZT_PERSP_COUNT: usize =
    KZT_HAND_NUM_PERSP * KZT_HAND_NUM_PIECE * KZT_HAND_NUM_DROP_STATE;
/// W_bias テーブル要素数 = drop_state(16) = 16。
pub const KZT_BIAS_COUNT: usize = KZT_HAND_NUM_DROP_STATE;

/// factored 5 テーブル合計要素数 = 2,800 + 6,400 + 896 + 224 + 16 = 10,336。
///
/// disk section (= version 2) は core → region → bucket → persp → bias の固定順で
/// i16 連結する (= [`KztFactoredWeights::from_flat`] / [`KztFactoredWeights::write_flat`])。
pub const KZT_FACTORED_COUNT: usize =
    KZT_CORE_COUNT + KZT_REGION_COUNT + KZT_BUCKET_COUNT + KZT_PERSP_COUNT + KZT_BIAS_COUNT;

/// W_core の flat index (= `(hand_piece, target_rel, drop_state)`、 drop_state stride 1)。
#[inline]
pub const fn kzt_core_index(hand_piece: usize, target_rel: usize, drop_state: usize) -> usize {
    debug_assert!(hand_piece < KZT_HAND_NUM_PIECE);
    debug_assert!(target_rel < KZT_HAND_NUM_TARGET_REL);
    debug_assert!(drop_state < KZT_HAND_NUM_DROP_STATE);
    (hand_piece * KZT_HAND_NUM_TARGET_REL + target_rel) * KZT_HAND_NUM_DROP_STATE + drop_state
}

/// W_region の flat index (= `(king_region, target_rel, drop_state)`)。
#[inline]
pub const fn kzt_region_index(king_region: usize, target_rel: usize, drop_state: usize) -> usize {
    debug_assert!(king_region < KZT_HAND_NUM_KING_REGION);
    debug_assert!(target_rel < KZT_HAND_NUM_TARGET_REL);
    debug_assert!(drop_state < KZT_HAND_NUM_DROP_STATE);
    (king_region * KZT_HAND_NUM_TARGET_REL + target_rel) * KZT_HAND_NUM_DROP_STATE + drop_state
}

/// W_bucket の flat index (= `(bucket, hand_piece, drop_state)`)。
#[inline]
pub const fn kzt_bucket_index(bucket: usize, hand_piece: usize, drop_state: usize) -> usize {
    debug_assert!(bucket < KZT_HAND_NUM_BUCKET);
    debug_assert!(hand_piece < KZT_HAND_NUM_PIECE);
    debug_assert!(drop_state < KZT_HAND_NUM_DROP_STATE);
    (bucket * KZT_HAND_NUM_PIECE + hand_piece) * KZT_HAND_NUM_DROP_STATE + drop_state
}

/// W_persp の flat index (= `(persp, hand_piece, drop_state)`)。
#[inline]
pub const fn kzt_persp_index(persp: usize, hand_piece: usize, drop_state: usize) -> usize {
    debug_assert!(persp < KZT_HAND_NUM_PERSP);
    debug_assert!(hand_piece < KZT_HAND_NUM_PIECE);
    debug_assert!(drop_state < KZT_HAND_NUM_DROP_STATE);
    (persp * KZT_HAND_NUM_PIECE + hand_piece) * KZT_HAND_NUM_DROP_STATE + drop_state
}

/// W_bias の flat index (= `drop_state`)。
#[inline]
pub const fn kzt_bias_index(drop_state: usize) -> usize {
    debug_assert!(drop_state < KZT_HAND_NUM_DROP_STATE);
    drop_state
}

// ---------------------------------------------------------------------------
// KZT-FT (= Feature Transformer 入力版、 feature `nnue-kzt-ft`)
// ---------------------------------------------------------------------------
//
// KZT_HAND の sparse feature を eval に scalar 加算する (= kzt-psqt) のではなく、
// **FT 入力として 1024 accumulator へ流す** 版。 threat_accumulation と同じ機構
// (= per-perspective [[i16; L1]; 2]、 SCReLU 前に piece accumulation と加算) で、
// SCReLU → L1 → L2 を通り piece features と非線形に co-adapt させる。
//
// # 次元空間 (= factored 5 テーブルから W_bucket を除外した 4 テーブルの連結)
//
// FT accumulator は **bucket 非依存** が設計 (= 一度計算して全 bucket L1 stack で共有、
// bucket 条件付けは LayerStack の 8 bucket L1→L2 が担う)。 よって factored の
// `W_bucket` (= bucket 依存 896 weight) は FT 入力から除外する。 残り 4 テーブルを
// **core → region → persp → bias** の固定順で連結した 9,440 次元空間を使う:
//
// ```text
//   [0, 2800)      core   : kzt_core_index(hand_piece, target_rel, drop_state)
//   [2800, 9200)   region : kzt_region_index(king_region, target_rel, drop_state)
//   [9200, 9424)   persp  : kzt_persp_index(persp, hand_piece, drop_state)
//   [9424, 9440)   bias   : kzt_bias_index(drop_state)
// ```
//
// 1 つの active hand feature (= `collect_hand_features` が返す 1 dense index) は、
// 上記 4 テーブルへ射影され **4 個の FT active index** を emit する (= bucket 成分は捨てる)。
// trainer もこの 9,440 × L1 の i8 weight を同順で生成すること (= 下記 section 仕様参照)。
//
// # ckpt section 仕様 (= trainer が生成すべき weight 形状)
//
// ```text
// sub_magic = "P2KZTF\0\0" (8 byte、 [`KZT_FT_MAGIC`])
// version   = u32 LE       (= 1、 [`KZT_FT_VERSION`])
// count     = u32 LE       (= KZT_FT_DIMENSIONS * L1)
// weights   = i8 × count LE (= row-major [dim 9,440][L1]、 逆量子化なし)
// ```
//
// 既存 section (= ... / P2SXRY / P2KZT) の **物理末尾** に append。 旧 ckpt (= section 欠落)
// は weight `None` → has_kzt_ft=false → eval 寄与 0 (= bit-exact)。

/// KZT-FT section magic `P2KZTF\0\0` (= 8 byte、 scalar 版 `P2KZT\0\0\0` と非衝突)。
pub const KZT_FT_MAGIC: &[u8; 8] = b"P2KZTF\0\0";

/// KZT-FT section version (= 1)。
pub const KZT_FT_VERSION: u32 = 1;

/// KZT-FT 入力次元数 = core(2,800) + region(6,400) + persp(224) + bias(16) = 9,440
/// (= factored 5 テーブルから W_bucket(896) を除外)。
pub const KZT_FT_DIMENSIONS: usize =
    KZT_CORE_COUNT + KZT_REGION_COUNT + KZT_PERSP_COUNT + KZT_BIAS_COUNT;

/// FT 連結空間での core テーブル先頭 offset (= 0)。
pub const KZT_FT_CORE_OFFSET: usize = 0;
/// FT 連結空間での region テーブル先頭 offset (= 2,800)。
pub const KZT_FT_REGION_OFFSET: usize = KZT_CORE_COUNT;
/// FT 連結空間での persp テーブル先頭 offset (= 9,200)。
pub const KZT_FT_PERSP_OFFSET: usize = KZT_CORE_COUNT + KZT_REGION_COUNT;
/// FT 連結空間での bias テーブル先頭 offset (= 9,424)。
pub const KZT_FT_BIAS_OFFSET: usize = KZT_CORE_COUNT + KZT_REGION_COUNT + KZT_PERSP_COUNT;

/// 1 つの active hand feature が emit する FT active index 数 (= core/region/persp/bias)。
pub const KZT_FT_COMPONENTS_PER_FEATURE: usize = 4;

/// 1 perspective あたりの FT active index 上限 (= 175 hand feature × 4 = 700)。
pub const KZT_FT_MAX_ACTIVE_PER_PERSP: usize = KZT_HAND_MAX_ACTIVE * KZT_FT_COMPONENTS_PER_FEATURE;

/// `attacker` の持駒で敵玉 5x5 zone を攻める KZT active feature を gather し、 各 feature を
/// **FT 入力の 4 テーブル成分 (= core/region/persp/bias)** に射影して `f` へ index を渡す
/// (= no-heap、 threat の `for_each_active_threat_index` に相当)。
///
/// FT accumulator は bucket 非依存のため W_bucket 成分は emit しない (= bucket 引数には
/// 0 を渡し、 dense index の bucket 成分は [`kzt_hand_decode`] で捨てる)。 各 index は
/// `[0, KZT_FT_DIMENSIONS)` に収まる (= 連結空間)。
///
/// `kzt_accumulation[p]` は **attacker = p の単一側** feature のみを集める (= us/them 分離は
/// network の `[acc[stm], acc[nstm]]` 連結で学習)。 呼び出し側は perspective p の色を
/// `attacker` に渡すこと。
#[cfg(feature = "nnue-kzt-ft")]
#[inline]
pub fn for_each_active_kzt_ft_index<F: FnMut(usize)>(pos: &Position, attacker: Color, mut f: F) {
    let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
    // bucket は FT では次元から除外済み。 collect_hand_features は bucket 引数を取るが、
    // 返す dense index の bucket 成分は下で捨てるため任意 (= 0 を渡す)。
    let n = collect_hand_features(pos, attacker, 0, &mut buf);
    for feat in buf.iter().take(n) {
        let (persp, _bucket, king_region, hand_piece, target_rel, drop_state) =
            kzt_hand_decode(feat.index);
        f(KZT_FT_CORE_OFFSET + kzt_core_index(hand_piece, target_rel, drop_state));
        f(KZT_FT_REGION_OFFSET + kzt_region_index(king_region, target_rel, drop_state));
        f(KZT_FT_PERSP_OFFSET + kzt_persp_index(persp, hand_piece, drop_state));
        f(KZT_FT_BIAS_OFFSET + kzt_bias_index(drop_state));
    }
}

// ---------------------------------------------------------------------------
// OUTPUT_SCALE (= KDGR / ADEV / SXRY と同方式)
// ---------------------------------------------------------------------------

/// `OUTPUT_SCALE` 初期値 (= 4、 既存 scalar head と揃える)。
pub const KZT_OUTPUT_SCALE_DEFAULT: i32 = 4;

/// USI option `KZT_OUTPUT_SCALE` で runtime 上書き可能な eval scale。
pub static KZT_OUTPUT_SCALE: AtomicI32 = AtomicI32::new(KZT_OUTPUT_SCALE_DEFAULT);

/// USI option 経由で eval scale を更新する。
pub fn set_output_scale(scale: i32) {
    KZT_OUTPUT_SCALE.store(scale, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// head 合計 clamp (= factored v1 の平滑化 / 過学習極値抑制、 runtime option)
// ---------------------------------------------------------------------------

/// `KZT_CLAMP_RAW` 初期値 (= 0 = clamp 無効 = ∞、 従来 dense 挙動)。
///
/// 0 のとき head 合計を clamp しない (= 旧挙動 / bit-exact 検証時に使用)。 正の K を
/// 指定すると各 perspective の head 合計 (= fv_scale 適用前 raw cp) を `[-K, +K]` に clamp
/// する (= factored の平滑化実験用)。
pub const KZT_CLAMP_RAW_DEFAULT: i32 = 0;

/// USI option `KZT_CLAMP_RAW` で runtime 上書き可能な head 合計 clamp 幅 (= 0 で無効)。
pub static KZT_CLAMP_RAW: AtomicI32 = AtomicI32::new(KZT_CLAMP_RAW_DEFAULT);

/// USI option 経由で head 合計 clamp 幅 K を更新する (= 0 で clamp 無効)。
pub fn set_clamp_raw(k: i32) {
    KZT_CLAMP_RAW.store(k, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// weight type
// ---------------------------------------------------------------------------

/// factored v1 の 5 テーブル weight (= 合計 [`KZT_FACTORED_COUNT`] = 10,336 i16 ≈ 20 KB)。
///
/// dense 版 (`[i16; 716_800]` ≈ 1.4 MB) を廃し、 同じ active feature を 5 低ランクテーブルの
/// 和で近似する (= 平滑化)。 disk section (= version 2) は core → region → bucket → persp →
/// bias の固定順で i16 連結する。
#[derive(Clone)]
pub struct KztFactoredWeights {
    /// W_core[hand_piece][target_rel][drop_state] (= [`kzt_core_index`])。
    pub core: [i16; KZT_CORE_COUNT],
    /// W_region[king_region][target_rel][drop_state] (= [`kzt_region_index`])。
    pub region: [i16; KZT_REGION_COUNT],
    /// W_bucket[bucket][hand_piece][drop_state] (= [`kzt_bucket_index`])。
    pub bucket: [i16; KZT_BUCKET_COUNT],
    /// W_persp[persp][hand_piece][drop_state] (= [`kzt_persp_index`])。
    pub persp: [i16; KZT_PERSP_COUNT],
    /// W_bias[drop_state] (= [`kzt_bias_index`])。
    pub bias: [i16; KZT_BIAS_COUNT],
}

impl KztFactoredWeights {
    /// 全 0 の weight を heap に確保する。
    pub fn zeroed() -> Box<Self> {
        Box::new(Self {
            core: [0i16; KZT_CORE_COUNT],
            region: [0i16; KZT_REGION_COUNT],
            bucket: [0i16; KZT_BUCKET_COUNT],
            persp: [0i16; KZT_PERSP_COUNT],
            bias: [0i16; KZT_BIAS_COUNT],
        })
    }

    /// disk 固定順 (= core → region → bucket → persp → bias) の flat i16 から構築する。
    ///
    /// # Panics
    /// `flat.len() != KZT_FACTORED_COUNT` のとき panic。
    pub fn from_flat(flat: &[i16]) -> Box<Self> {
        assert_eq!(
            flat.len(),
            KZT_FACTORED_COUNT,
            "kzt factored flat len mismatch: expected {KZT_FACTORED_COUNT}, got {}",
            flat.len()
        );
        let mut w = Self::zeroed();
        let mut off = 0usize;
        w.core.copy_from_slice(&flat[off..off + KZT_CORE_COUNT]);
        off += KZT_CORE_COUNT;
        w.region.copy_from_slice(&flat[off..off + KZT_REGION_COUNT]);
        off += KZT_REGION_COUNT;
        w.bucket.copy_from_slice(&flat[off..off + KZT_BUCKET_COUNT]);
        off += KZT_BUCKET_COUNT;
        w.persp.copy_from_slice(&flat[off..off + KZT_PERSP_COUNT]);
        off += KZT_PERSP_COUNT;
        w.bias.copy_from_slice(&flat[off..off + KZT_BIAS_COUNT]);
        w
    }

    /// disk 固定順 (= core → region → bucket → persp → bias) の flat i16 に書き出す。
    ///
    /// # Panics
    /// `out.len() != KZT_FACTORED_COUNT` のとき panic。
    pub fn write_flat(&self, out: &mut [i16]) {
        assert_eq!(out.len(), KZT_FACTORED_COUNT);
        let mut off = 0usize;
        out[off..off + KZT_CORE_COUNT].copy_from_slice(&self.core);
        off += KZT_CORE_COUNT;
        out[off..off + KZT_REGION_COUNT].copy_from_slice(&self.region);
        off += KZT_REGION_COUNT;
        out[off..off + KZT_BUCKET_COUNT].copy_from_slice(&self.bucket);
        off += KZT_BUCKET_COUNT;
        out[off..off + KZT_PERSP_COUNT].copy_from_slice(&self.persp);
        off += KZT_PERSP_COUNT;
        out[off..off + KZT_BIAS_COUNT].copy_from_slice(&self.bias);
    }

    /// 1 つの active feature (= 6 軸) の factored score (= 5 テーブル和) を返す。
    #[inline]
    pub fn feature_score(
        &self,
        persp: usize,
        bucket: usize,
        king_region: usize,
        hand_piece: usize,
        target_rel: usize,
        drop_state: usize,
    ) -> i32 {
        self.core[kzt_core_index(hand_piece, target_rel, drop_state)] as i32
            + self.region[kzt_region_index(king_region, target_rel, drop_state)] as i32
            + self.bucket[kzt_bucket_index(bucket, hand_piece, drop_state)] as i32
            + self.persp[kzt_persp_index(persp, hand_piece, drop_state)] as i32
            + self.bias[kzt_bias_index(drop_state)] as i32
    }
}

// ---------------------------------------------------------------------------
// index 計算
// ---------------------------------------------------------------------------

/// KZT_HAND の flat index を返す (= disk / runtime 共通 layout)。
///
/// flat order (= most → least significant):
/// `persp, bucket, king_region, hand_piece, target_rel, drop_state`。
///
/// # Panics (debug)
/// 各引数が次元上限を超える場合 debug build で panic (= index 安全性の検証用)。
#[inline]
pub const fn kzt_hand_index(
    persp: usize,
    bucket: usize,
    king_region: usize,
    hand_piece: usize,
    target_rel: usize,
    drop_state: usize,
) -> usize {
    debug_assert!(persp < KZT_HAND_NUM_PERSP);
    debug_assert!(bucket < KZT_HAND_NUM_BUCKET);
    debug_assert!(king_region < KZT_HAND_NUM_KING_REGION);
    debug_assert!(hand_piece < KZT_HAND_NUM_PIECE);
    debug_assert!(target_rel < KZT_HAND_NUM_TARGET_REL);
    debug_assert!(drop_state < KZT_HAND_NUM_DROP_STATE);

    ((((persp * KZT_HAND_NUM_BUCKET + bucket) * KZT_HAND_NUM_KING_REGION + king_region)
        * KZT_HAND_NUM_PIECE
        + hand_piece)
        * KZT_HAND_NUM_TARGET_REL
        + target_rel)
        * KZT_HAND_NUM_DROP_STATE
        + drop_state
}

/// [`PieceType::HAND_PIECES`] 内での持駒 index (= 0..7) を返す。
///
/// 歩=0 / 香=1 / 桂=2 / 銀=3 / 金=4 / 角=5 / 飛=6。 hand piece でなければ `None`。
#[inline]
pub fn hand_piece_index(pt: PieceType) -> Option<usize> {
    PieceType::HAND_PIECES.iter().position(|&p| p == pt)
}

// ---------------------------------------------------------------------------
// sparse feature gather (= 骨組み stub、 no-heap)
// ---------------------------------------------------------------------------

/// 1 つの active KZT_HAND feature (= flat index + 立っている事実)。
///
/// 値は 1 (= active) 固定の binary feature を想定 (= PSQT 同様、 重みは weight 側)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KztHandFeature {
    /// [`kzt_hand_index`] が返す flat index。
    pub index: usize,
}

/// 敵玉中心 5x5 zone の相対升 index 計算で扱う 1 軸あたりの幅 (= 5、 dx,dy ∈ [-2,2])。
const TARGET_REL_SIDE: i32 = 5;

/// perspective 正規化 rank (= attacker の **自陣側 = 0 / 敵陣側 = 8**) を返す。
///
/// `Rank::relative(attacker.opponent())` と同値。 attacker = Black なら `8 - rank`、
/// White なら `rank` (= 180° 回転の rank 成分)。 attacker 自玉前進方向で単調増加する。
#[inline]
fn norm_rank(attacker: Color, sq: Square) -> usize {
    sq.rank().relative(attacker.opponent()).index()
}

/// perspective 正規化 file (= 180° 回転の file 成分)。
///
/// attacker = Black なら `8 - file`、 White なら `file`。 norm_rank と合わせて 「Black は
/// 180° 回転 (= file/rank 両反転)、 White は恒等」 の attacker 標準 frame を作る。
/// これにより同一形状が先後で同一 (drow, dcol) を返す (= 色非依存)。
#[inline]
fn norm_file(attacker: Color, sq: Square) -> usize {
    match attacker {
        Color::Black => 8 - sq.file().index(),
        Color::White => sq.file().index(),
    }
}

/// 敵玉 `king_sq` の `king_region` (= 0..16) を attacker 標準 frame で算出する。
///
/// `rank_band(4) * 4 + file_band(4)`:
/// - rank_band: 正規化 rank (0=自陣..8=敵陣) を `0-2 / 3-4 / 5-6 / 7-8` の 4 band に量子化。
/// - file_band: `file_norm = min(file, 8-file)` (= 左右ミラー共有) を `0 / 1 / 2 / 3-4` の
///   4 band に量子化。 file の左右はここで吸収する (= target_rel は折り畳まない)。
#[inline]
fn compute_king_region(attacker: Color, king_sq: Square) -> usize {
    let nr = norm_rank(attacker, king_sq);
    let rank_band = match nr {
        0..=2 => 0,
        3..=4 => 1,
        5..=6 => 2,
        _ => 3, // 7..=8
    };
    let file_idx = king_sq.file().index();
    let file_norm = file_idx.min(8 - file_idx); // 0..=4
    let file_band = match file_norm {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => 3, // 3 | 4
    };
    rank_band * 4 + file_band
}

/// 投下した駒種 `pt` が升 `sq` から利かせる升集合 (= drop 利き)。
///
/// `dropper` = 投下する側の色 (= 歩 / 香 / 桂 は色依存前方利き、 角 / 飛 / 香 は `occupied`
/// 考慮の遠隔利き)。 玉以外の持駒種 (= 歩香桂銀金角飛) のみを想定する。
#[inline]
fn drop_attack(pt: PieceType, dropper: Color, sq: Square, occupied: Bitboard) -> Bitboard {
    match pt {
        PieceType::Pawn => pawn_effect(dropper, sq),
        PieceType::Lance => lance_effect(dropper, sq, occupied),
        PieceType::Knight => knight_effect(dropper, sq),
        PieceType::Silver => silver_effect(dropper, sq),
        PieceType::Gold => gold_effect(dropper, sq),
        PieceType::Bishop => bishop_effect(sq, occupied),
        PieceType::Rook => rook_effect(sq, occupied),
        _ => Bitboard::EMPTY,
    }
}

/// `attacker` が升 `sq` に駒種 `pt` を **合法に打てるか** (= v0 の粗い合法性)。
///
/// 判定するのは「行き所のない駒」 (= 歩香の最終段 / 桂の最終 2 段) と **二歩** のみ。
/// 打ち歩詰め・詰めろ・両取り・SEE は判定しない (= v0 許容、 stage2 で精緻化)。
/// `nifu` = 同 file に attacker の不成歩が既に在るか (= 呼び出し側で算出済)。
#[inline]
fn is_legal_drop_square(pt: PieceType, attacker: Color, sq: Square, nifu: bool) -> bool {
    // attacker 視点の段 (= relative(attacker)、 0 = 最終段 = 行き所なし)。
    let rrel = sq.rank().relative(attacker).index();
    match pt {
        PieceType::Pawn => rrel != 0 && !nifu,
        PieceType::Lance => rrel != 0,
        PieceType::Knight => rrel > 1,
        _ => true,
    }
}

/// `attacker` の持駒で **敵玉 5x5 zone を攻める** active feature を `out` (= stack 固定
/// buffer) に収集し、 個数を返す (= no-heap sparse gather)。
///
/// 立てる feature は「敵玉中心 5x5 の空升 × attacker が持つ駒種」 のうち **合法に打てる**
/// もの (= 二歩 / 行き所のない駒を除外) のみ (= sparse)。 各 feature の `persp` 次元は
/// **attacker の絶対色** (= `attacker.index()`) を焼き込む (= KDGR / DROP 等 既存 scalar
/// head と同じ絶対色 indexing。 (us - them) 反対称合成は [`evaluate_kzt_hand`] 側で行う)。
///
/// drop_state (= 4 bit) は `check | escape_block<<1 | supported<<2 | enemy_attacked<<3`:
/// - check (bit0): 打った駒が敵玉に王手
/// - escape_block (bit1): 打ち先が敵玉の利き升 (= 玉の逃げ升候補) を塞ぐ
/// - supported (bit2): 打ち先に attacker 自軍の利きが在る
/// - enemy_attacked (bit3): 打ち先に敵の利きが在る
///
/// # Arguments
/// * `pos`      - 局面
/// * `attacker` - 攻め手番 (= この色の持駒で **相手玉** を攻める脅威を見る)
/// * `bucket`   - LayerStack bucket (= 0..8)
/// * `out`      - 収集先 stack buffer (= 長さ [`KZT_HAND_MAX_ACTIVE`])
///
/// # Returns
/// 収集した active feature 数 (= `out[..n]` が有効)。 持駒なし / 打ち先なしでは 0。
#[inline]
pub fn collect_hand_features(
    pos: &Position,
    attacker: Color,
    bucket: usize,
    out: &mut [KztHandFeature; KZT_HAND_MAX_ACTIVE],
) -> usize {
    let hand = pos.hand(attacker);
    if hand.is_empty() {
        return 0;
    }
    let opp = attacker.opponent();
    let enemy_king = pos.king_square(opp);
    let occupied = pos.occupied();

    // 敵玉中心 5x5 zone のうち投下可能な空升 (= zone & ~occupied)。
    let drop = king_5x5_mask(enemy_king) & !occupied;
    if drop.is_empty() {
        return 0;
    }

    let persp = attacker.index();
    let king_region = compute_king_region(attacker, enemy_king);
    let kr_nrank = norm_rank(attacker, enemy_king) as i32;
    let kr_nfile = norm_file(attacker, enemy_king) as i32;
    let escape = king_effect(enemy_king); // 敵玉の逃げ升候補 (= 8 近傍)。
    let my_pawns = pos.pieces(attacker, PieceType::Pawn); // 不成歩のみ (二歩判定用)。

    let mut n = 0usize;
    for sq in drop.iter() {
        // target_rel (= attacker 標準 frame の敵玉相対 5x5 index、 0..25)。
        let drow = norm_rank(attacker, sq) as i32 - kr_nrank;
        let dcol = norm_file(attacker, sq) as i32 - kr_nfile;
        // 5x5 zone 内なので |drow|,|dcol| <= 2 が保証される。
        let target_rel = ((drow + 2) * TARGET_REL_SIDE + (dcol + 2)) as usize;

        // 升単位の drop_state bit (= 駒種非依存): escape_block / supported / enemy_attacked。
        // `attackers_to` を 1 回だけ計算し、 色フィルタで両 bit を導出する
        // (= attackers_to_c を 2 回呼ぶと同一 scan が二重になるため 1 回に統合)。
        let escape_block = escape.contains(sq);
        let attackers = pos.attackers_to(sq);
        let supported = !(attackers & pos.pieces_c(attacker)).is_empty();
        let enemy_attacked = !(attackers & pos.pieces_c(opp)).is_empty();
        let square_bits = ((escape_block as usize) << 1)
            | ((supported as usize) << 2)
            | ((enemy_attacked as usize) << 3);
        // 二歩判定 (= この file に attacker の不成歩が在るか) は歩のみで使う。
        let nifu = !(my_pawns & FILE_BB[sq.file().index()]).is_empty();

        for (hp_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
            if hand.count(pt) == 0 {
                continue;
            }
            if !is_legal_drop_square(pt, attacker, sq, nifu) {
                continue;
            }
            // check bit (= 駒種依存): 打った駒が敵玉に利くか。
            let check = drop_attack(pt, attacker, sq, occupied).contains(enemy_king);
            let drop_state = (check as usize) | square_bits;
            let index = kzt_hand_index(persp, bucket, king_region, hp_idx, target_rel, drop_state);
            if n < KZT_HAND_MAX_ACTIVE {
                out[n] = KztHandFeature { index };
                n += 1;
            }
        }
    }
    n
}

/// flat index を 6 軸 (= persp, bucket, king_region, hand_piece, target_rel, drop_state) に
/// 分解する (= [`kzt_hand_index`] の逆写像、 debug / 集計用)。
#[inline]
pub fn kzt_hand_decode(index: usize) -> (usize, usize, usize, usize, usize, usize) {
    let drop_state = index % KZT_HAND_NUM_DROP_STATE;
    let r = index / KZT_HAND_NUM_DROP_STATE;
    let target_rel = r % KZT_HAND_NUM_TARGET_REL;
    let r = r / KZT_HAND_NUM_TARGET_REL;
    let hand_piece = r % KZT_HAND_NUM_PIECE;
    let r = r / KZT_HAND_NUM_PIECE;
    let king_region = r % KZT_HAND_NUM_KING_REGION;
    let r = r / KZT_HAND_NUM_KING_REGION;
    let bucket = r % KZT_HAND_NUM_BUCKET;
    let persp = r / KZT_HAND_NUM_BUCKET;
    (persp, bucket, king_region, hand_piece, target_rel, drop_state)
}

// ---------------------------------------------------------------------------
// debug / 集計 (= hot path 外、 eval コマンドや test 用)
// ---------------------------------------------------------------------------

/// 1 局面 1 perspective の KZT_HAND active feature 内訳 (= debug 集計、 hot path 外)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KztHandDebugSummary {
    /// active feature 総数。
    pub total: usize,
    /// drop_state 別 active 数 (= 4 bit、 16 区分)。
    pub by_drop_state: [usize; KZT_HAND_NUM_DROP_STATE],
    /// king_region 別 active 数 (= 16 区分)。
    pub by_king_region: [usize; KZT_HAND_NUM_KING_REGION],
    /// hand_piece 別 active 数 (= 歩香桂銀金角飛、 7 区分)。
    pub by_hand_piece: [usize; KZT_HAND_NUM_PIECE],
}

/// `attacker` の KZT_HAND active feature を収集し、 軸別に集計する (= debug / 集計用)。
///
/// hot path では呼ばない。 `eval` コマンドや test から 1 局面の内訳を出すための API。
/// 内部で [`collect_hand_features`] を呼び、 各 index を [`kzt_hand_decode`] で分解集計する。
pub fn summarize_hand_features(
    pos: &Position,
    attacker: Color,
    bucket: usize,
) -> KztHandDebugSummary {
    let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
    let n = collect_hand_features(pos, attacker, bucket, &mut buf);
    let mut summary = KztHandDebugSummary {
        total: n,
        ..Default::default()
    };
    for f in buf.iter().take(n) {
        let (_persp, _bucket, king_region, hand_piece, _target_rel, drop_state) =
            kzt_hand_decode(f.index);
        summary.by_drop_state[drop_state] += 1;
        summary.by_king_region[king_region] += 1;
        summary.by_hand_piece[hand_piece] += 1;
    }
    summary
}

// ---------------------------------------------------------------------------
// factored weight 統計 (= 訓練後 per-table / per-axis L2 レポート、 hot path 外)
// ---------------------------------------------------------------------------

/// factored 5 テーブル weight の per-table / per-axis 統計 (= 訓練後の育成度診断)。
///
/// L2 は `sqrt(Σ w^2)` (= i16 weight を f64 で二乗和)。 per-axis L2 は、 その軸を含む
/// 全テーブルの weight を軸値ごとに集約した二乗和の平方根 (= drop_state は全 5 テーブル、
/// hand_piece は core/bucket/persp、 target_rel は core/region、 bucket は bucket、
/// king_region は region に由来)。
#[derive(Clone, Debug, PartialEq)]
pub struct KztWeightStats {
    /// テーブル別 L2 (= core, region, bucket, persp, bias の順)。
    pub table_l2: [f64; 5],
    /// テーブル別 max |w|。
    pub table_max_abs: [i16; 5],
    /// drop_state 別 L2 (= 全テーブル集約、 16 区分)。
    pub by_drop_state_l2: [f64; KZT_HAND_NUM_DROP_STATE],
    /// hand_piece 別 L2 (= core/bucket/persp 集約、 7 区分)。
    pub by_hand_piece_l2: [f64; KZT_HAND_NUM_PIECE],
    /// target_rel 別 L2 (= core/region 集約、 25 区分)。
    pub by_target_rel_l2: [f64; KZT_HAND_NUM_TARGET_REL],
    /// bucket 別 L2 (= bucket テーブル、 8 区分)。
    pub by_bucket_l2: [f64; KZT_HAND_NUM_BUCKET],
    /// king_region 別 L2 (= region テーブル、 16 区分)。
    pub by_king_region_l2: [f64; KZT_HAND_NUM_KING_REGION],
}

/// factored weight の per-table / per-axis L2 / max_abs を集計する (= 訓練後レポート用、
/// hot path 外)。
pub fn kzt_weight_stats(w: &KztFactoredWeights) -> KztWeightStats {
    let mut table_sq = [0f64; 5];
    let mut table_max = [0i16; 5];
    let mut ds_sq = [0f64; KZT_HAND_NUM_DROP_STATE];
    let mut hp_sq = [0f64; KZT_HAND_NUM_PIECE];
    let mut tr_sq = [0f64; KZT_HAND_NUM_TARGET_REL];
    let mut bk_sq = [0f64; KZT_HAND_NUM_BUCKET];
    let mut kr_sq = [0f64; KZT_HAND_NUM_KING_REGION];

    let mut acc = |val: i16, table: usize| {
        let sq = (val as f64) * (val as f64);
        table_sq[table] += sq;
        if val.unsigned_abs() > table_max[table].unsigned_abs() {
            table_max[table] = val;
        }
        sq
    };

    // 各テーブル配列を直接走査し、 flat index を軸へ逆分解して集約する
    // (= index 式 `(a*B+b)*C+c` の逆写像、 kzt_*_index と対応)。
    // core[hand_piece][target_rel][drop_state]
    for (i, &val) in w.core.iter().enumerate() {
        let ds = i % KZT_HAND_NUM_DROP_STATE;
        let tr = (i / KZT_HAND_NUM_DROP_STATE) % KZT_HAND_NUM_TARGET_REL;
        let hp = i / KZT_HAND_NUM_DROP_STATE / KZT_HAND_NUM_TARGET_REL;
        let sq = acc(val, 0);
        ds_sq[ds] += sq;
        hp_sq[hp] += sq;
        tr_sq[tr] += sq;
    }
    // region[king_region][target_rel][drop_state]
    for (i, &val) in w.region.iter().enumerate() {
        let ds = i % KZT_HAND_NUM_DROP_STATE;
        let tr = (i / KZT_HAND_NUM_DROP_STATE) % KZT_HAND_NUM_TARGET_REL;
        let kr = i / KZT_HAND_NUM_DROP_STATE / KZT_HAND_NUM_TARGET_REL;
        let sq = acc(val, 1);
        ds_sq[ds] += sq;
        tr_sq[tr] += sq;
        kr_sq[kr] += sq;
    }
    // bucket[bucket][hand_piece][drop_state]
    for (i, &val) in w.bucket.iter().enumerate() {
        let ds = i % KZT_HAND_NUM_DROP_STATE;
        let hp = (i / KZT_HAND_NUM_DROP_STATE) % KZT_HAND_NUM_PIECE;
        let bk = i / KZT_HAND_NUM_DROP_STATE / KZT_HAND_NUM_PIECE;
        let sq = acc(val, 2);
        ds_sq[ds] += sq;
        hp_sq[hp] += sq;
        bk_sq[bk] += sq;
    }
    // persp[persp][hand_piece][drop_state]
    for (i, &val) in w.persp.iter().enumerate() {
        let ds = i % KZT_HAND_NUM_DROP_STATE;
        let hp = (i / KZT_HAND_NUM_DROP_STATE) % KZT_HAND_NUM_PIECE;
        let sq = acc(val, 3);
        ds_sq[ds] += sq;
        hp_sq[hp] += sq;
    }
    // bias[drop_state]
    for (ds, &val) in w.bias.iter().enumerate() {
        let sq = acc(val, 4);
        ds_sq[ds] += sq;
    }

    let sqrt = |a: [f64; 5]| a.map(f64::sqrt);
    KztWeightStats {
        table_l2: sqrt(table_sq),
        table_max_abs: table_max,
        by_drop_state_l2: ds_sq.map(f64::sqrt),
        by_hand_piece_l2: hp_sq.map(f64::sqrt),
        by_target_rel_l2: tr_sq.map(f64::sqrt),
        by_bucket_l2: bk_sq.map(f64::sqrt),
        by_king_region_l2: kr_sq.map(f64::sqrt),
    }
}

// ---------------------------------------------------------------------------
// evaluate (= gather 本体配線済み、 0-weight / scale 0 / 空 active で bit-exact)
// ---------------------------------------------------------------------------

/// KZT_HAND sparse head を評価する (= stm 視点 `(us - them) × OUTPUT_SCALE`)。
///
/// - `us`   = stm の持駒で **敵玉** を攻める脅威 feature の weight 総和 (= 攻め)。
/// - `them` = 相手の持駒で **自玉** を攻める脅威 feature の weight 総和 (= 受け)。
///
/// 各 perspective で [`collect_hand_features`] を呼び、 sparse active feature の weight を
/// 総和する。 weight が全 0 (= 訓練前 / 旧 ckpt) なら当然 0、 `OUTPUT_SCALE == 0` の
/// ときは scan を省略して即 0 を返す (= head 無効化時の NPS 確保、 KDGR / ADEV / SXRY と
/// 同方針)。 いずれも既存 eval と bit-exact。
///
/// # Arguments
/// * `weights` - factored KZT_HAND weight (= [`KztFactoredWeights`]、 i16)
/// * `pos`     - 局面
/// * `bucket`  - LayerStack bucket (= 0..8)
/// * `stm`     - 手番 (= side to move)
pub fn evaluate_kzt_hand(
    weights: &KztFactoredWeights,
    pos: &Position,
    bucket: usize,
    stm: Color,
) -> i32 {
    let scale = KZT_OUTPUT_SCALE.load(Ordering::Relaxed);
    if scale == 0 {
        return 0;
    }
    let nstm = stm.opponent();

    let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];

    // us (= stm の持駒で敵玉を攻める脅威)
    let mut us = kzt_sum_side(weights, pos, stm, bucket, &mut buf);
    // them (= nstm の持駒で自玉を攻める脅威)
    let mut them = kzt_sum_side(weights, pos, nstm, bucket, &mut buf);

    // head 合計 clamp (= factored v1 の平滑化、 K==0 で無効 = 従来 dense 挙動)。
    // 各 perspective の raw 合計を個別に clamp し、 その後 (us - them) × scale。
    let clamp_k = KZT_CLAMP_RAW.load(Ordering::Relaxed);
    if clamp_k > 0 {
        us = us.clamp(-clamp_k, clamp_k);
        them = them.clamp(-clamp_k, clamp_k);
    }

    us.saturating_sub(them).saturating_mul(scale)
}

/// `attacker` の active feature を gather し、 各 feature の factored score を合算する
/// (= 1 perspective 分の head 合計、 clamp 前)。
///
/// gather (`collect_hand_features`) は dense flat index を返すため、 [`kzt_hand_decode`] で
/// 6 軸に分解し [`KztFactoredWeights::feature_score`] で 5 テーブル和を積算する。
#[inline]
fn kzt_sum_side(
    weights: &KztFactoredWeights,
    pos: &Position,
    attacker: Color,
    bucket: usize,
    buf: &mut [KztHandFeature; KZT_HAND_MAX_ACTIVE],
) -> i32 {
    let n = collect_hand_features(pos, attacker, bucket, buf);
    let mut acc: i32 = 0;
    for f in buf.iter().take(n) {
        let (persp, bk, king_region, hand_piece, target_rel, drop_state) = kzt_hand_decode(f.index);
        acc = acc.saturating_add(weights.feature_score(
            persp,
            bk,
            king_region,
            hand_piece,
            target_rel,
            drop_state,
        ));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Position, SFEN_HIRATE};
    use std::sync::Mutex;

    /// KZT_OUTPUT_SCALE / KZT_CLAMP_RAW は process 共有 atomic なので、 それらを書き換える
    /// test は並列実行で干渉する。 直列化用の共有 lock。
    static SCALE_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn make_pos() -> Position {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        pos
    }

    #[test]
    fn test_magic_version_count() {
        assert_eq!(KZT_MAGIC, b"P2KZT\0\0\0");
        assert_eq!(KZT_MAGIC.len(), 8);
        assert_eq!(KZT_VERSION, 2);
        assert_eq!(KZT_HAND_WEIGHT_COUNT, 716_800);
        assert_eq!(KZT_HAND_WEIGHT_COUNT, 2 * 8 * 16 * 7 * 25 * 16);
        // factored v1 テーブル要素数。
        assert_eq!(KZT_CORE_COUNT, 7 * 25 * 16);
        assert_eq!(KZT_REGION_COUNT, 16 * 25 * 16);
        assert_eq!(KZT_BUCKET_COUNT, 8 * 7 * 16);
        assert_eq!(KZT_PERSP_COUNT, 2 * 7 * 16);
        assert_eq!(KZT_BIAS_COUNT, 16);
        assert_eq!(KZT_FACTORED_COUNT, 2_800 + 6_400 + 896 + 224 + 16);
        assert_eq!(KZT_FACTORED_COUNT, 10_336);
    }

    #[test]
    fn test_factored_index_bijective_per_table() {
        // 各 factored テーブルの index が 0..count の全単射 (= 重複 / 抜け無し) であることを確認。
        let mut core = vec![false; KZT_CORE_COUNT];
        for hp in 0..KZT_HAND_NUM_PIECE {
            for tr in 0..KZT_HAND_NUM_TARGET_REL {
                for ds in 0..KZT_HAND_NUM_DROP_STATE {
                    let i = kzt_core_index(hp, tr, ds);
                    assert!(!core[i], "core dup {i}");
                    core[i] = true;
                }
            }
        }
        assert!(core.iter().all(|&b| b), "core not full");

        let mut region = vec![false; KZT_REGION_COUNT];
        for kr in 0..KZT_HAND_NUM_KING_REGION {
            for tr in 0..KZT_HAND_NUM_TARGET_REL {
                for ds in 0..KZT_HAND_NUM_DROP_STATE {
                    let i = kzt_region_index(kr, tr, ds);
                    assert!(!region[i], "region dup {i}");
                    region[i] = true;
                }
            }
        }
        assert!(region.iter().all(|&b| b), "region not full");

        let mut bucket = vec![false; KZT_BUCKET_COUNT];
        for bk in 0..KZT_HAND_NUM_BUCKET {
            for hp in 0..KZT_HAND_NUM_PIECE {
                for ds in 0..KZT_HAND_NUM_DROP_STATE {
                    let i = kzt_bucket_index(bk, hp, ds);
                    assert!(!bucket[i], "bucket dup {i}");
                    bucket[i] = true;
                }
            }
        }
        assert!(bucket.iter().all(|&b| b), "bucket not full");

        let mut persp = vec![false; KZT_PERSP_COUNT];
        for p in 0..KZT_HAND_NUM_PERSP {
            for hp in 0..KZT_HAND_NUM_PIECE {
                for ds in 0..KZT_HAND_NUM_DROP_STATE {
                    let i = kzt_persp_index(p, hp, ds);
                    assert!(!persp[i], "persp dup {i}");
                    persp[i] = true;
                }
            }
        }
        assert!(persp.iter().all(|&b| b), "persp not full");

        let mut bias = [false; KZT_BIAS_COUNT];
        for ds in 0..KZT_HAND_NUM_DROP_STATE {
            let i = kzt_bias_index(ds);
            assert!(!bias[i], "bias dup {i}");
            bias[i] = true;
        }
        assert!(bias.iter().all(|&b| b), "bias not full");
    }

    #[test]
    fn test_factored_flat_roundtrip() {
        // from_flat → write_flat の往復一致 (= disk 固定順の検証)。
        let flat: Vec<i16> = (0..KZT_FACTORED_COUNT).map(|i| (i as i16).wrapping_mul(3)).collect();
        let w = KztFactoredWeights::from_flat(&flat);
        let mut out = vec![0i16; KZT_FACTORED_COUNT];
        w.write_flat(&mut out);
        assert_eq!(flat, out);
        // 固定順の先頭/境界の確認。
        assert_eq!(w.core[0], flat[0]);
        assert_eq!(w.region[0], flat[KZT_CORE_COUNT]);
        assert_eq!(w.bucket[0], flat[KZT_CORE_COUNT + KZT_REGION_COUNT]);
        assert_eq!(w.persp[0], flat[KZT_CORE_COUNT + KZT_REGION_COUNT + KZT_BUCKET_COUNT]);
        assert_eq!(
            w.bias[0],
            flat[KZT_CORE_COUNT + KZT_REGION_COUNT + KZT_BUCKET_COUNT + KZT_PERSP_COUNT]
        );
    }

    #[test]
    fn test_kzt_weight_stats_axes() {
        // 既知 weight で per-table / per-axis L2 が期待通り集計されることを確認 + 出力。
        let mut w = KztFactoredWeights::zeroed();
        // core[hp=2][tr=3][ds=5] = 6、 region[kr=1][tr=3][ds=5] = 8、 bias[ds=5]=1。
        w.core[kzt_core_index(2, 3, 5)] = 6;
        w.region[kzt_region_index(1, 3, 5)] = 8;
        w.bias[kzt_bias_index(5)] = 1;
        let s = kzt_weight_stats(&w);
        // table L2: core=6, region=8, bias=1。
        assert!((s.table_l2[0] - 6.0).abs() < 1e-9);
        assert!((s.table_l2[1] - 8.0).abs() < 1e-9);
        assert!((s.table_l2[4] - 1.0).abs() < 1e-9);
        assert_eq!(s.table_max_abs[0], 6);
        assert_eq!(s.table_max_abs[1], 8);
        // target_rel=3 は core(6)+region(8) 集約 → sqrt(36+64)=10。
        assert!((s.by_target_rel_l2[3] - 10.0).abs() < 1e-9);
        // drop_state=5 は core(6)+region(8)+bias(1) → sqrt(36+64+1)=sqrt(101)。
        assert!((s.by_drop_state_l2[5] - 101f64.sqrt()).abs() < 1e-9);
        // hand_piece=2 は core のみ(6)。
        assert!((s.by_hand_piece_l2[2] - 6.0).abs() < 1e-9);
        // king_region=1 は region のみ(8)。
        assert!((s.by_king_region_l2[1] - 8.0).abs() < 1e-9);
        println!(
            "KZT_STATS table_l2={:?} table_max={:?}\n  by_drop_state_l2={:?}\n  by_hand_piece_l2={:?}\n  by_bucket_l2={:?}\n  by_king_region_l2={:?}",
            s.table_l2,
            s.table_max_abs,
            s.by_drop_state_l2,
            s.by_hand_piece_l2,
            s.by_bucket_l2,
            s.by_king_region_l2
        );
    }

    #[test]
    fn test_feature_score_sums_five_tables() {
        let mut w = KztFactoredWeights::zeroed();
        let (p, bk, kr, hp, tr, ds) = (1usize, 3usize, 9usize, 4usize, 12usize, 5usize);
        w.core[kzt_core_index(hp, tr, ds)] = 10;
        w.region[kzt_region_index(kr, tr, ds)] = 20;
        w.bucket[kzt_bucket_index(bk, hp, ds)] = 30;
        w.persp[kzt_persp_index(p, hp, ds)] = 40;
        w.bias[kzt_bias_index(ds)] = 50;
        assert_eq!(w.feature_score(p, bk, kr, hp, tr, ds), 150);
        // 別 feature (= 全テーブル 0 のはず)。
        assert_eq!(w.feature_score(0, 0, 0, 0, 0, 0), 0);
    }

    #[test]
    fn test_index_first_last() {
        assert_eq!(kzt_hand_index(0, 0, 0, 0, 0, 0), 0);
        assert_eq!(
            kzt_hand_index(
                KZT_HAND_NUM_PERSP - 1,
                KZT_HAND_NUM_BUCKET - 1,
                KZT_HAND_NUM_KING_REGION - 1,
                KZT_HAND_NUM_PIECE - 1,
                KZT_HAND_NUM_TARGET_REL - 1,
                KZT_HAND_NUM_DROP_STATE - 1,
            ),
            KZT_HAND_WEIGHT_COUNT - 1
        );
    }

    #[test]
    fn test_index_is_bijective_dense() {
        // 全 index を走査して 0..count の全単射 (= 重複 / 抜け無し) を確認。
        // 各次元の stride が正しいことの強い保証。
        let mut seen = vec![false; KZT_HAND_WEIGHT_COUNT];
        for persp in 0..KZT_HAND_NUM_PERSP {
            for bucket in 0..KZT_HAND_NUM_BUCKET {
                for region in 0..KZT_HAND_NUM_KING_REGION {
                    for hp in 0..KZT_HAND_NUM_PIECE {
                        for trel in 0..KZT_HAND_NUM_TARGET_REL {
                            for dstate in 0..KZT_HAND_NUM_DROP_STATE {
                                let idx = kzt_hand_index(persp, bucket, region, hp, trel, dstate);
                                assert!(idx < KZT_HAND_WEIGHT_COUNT, "idx {idx} out of range");
                                assert!(!seen[idx], "duplicate index {idx}");
                                seen[idx] = true;
                            }
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&b| b), "index space not fully covered");
    }

    #[test]
    fn test_index_stride_drop_state_is_one() {
        // drop_state が最下位次元 (= stride 1)。
        let a = kzt_hand_index(1, 2, 3, 4, 5, 6);
        let b = kzt_hand_index(1, 2, 3, 4, 5, 7);
        assert_eq!(b - a, 1);
    }

    #[test]
    fn test_hand_piece_index() {
        assert_eq!(hand_piece_index(PieceType::Pawn), Some(0));
        assert_eq!(hand_piece_index(PieceType::Rook), Some(6));
        // 玉 / 成駒は hand piece でない
        assert_eq!(hand_piece_index(PieceType::King), None);
        assert_eq!(hand_piece_index(PieceType::ProPawn), None);
    }

    #[test]
    fn test_collect_features_startpos_empty_hand() {
        // 初期局面は両者持駒ゼロ → gather は空集合 (= 0 個)。
        let pos = make_pos();
        let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
        assert_eq!(collect_hand_features(&pos, Color::Black, 0, &mut buf), 0);
        assert_eq!(collect_hand_features(&pos, Color::White, 3, &mut buf), 0);
    }

    #[test]
    fn test_evaluate_zero_weight_is_zero() {
        // 全 weight 0 で評価 0 (= zero-load 時無害)。
        let weights = KztFactoredWeights::zeroed();
        let pos = make_pos();
        for bucket in 0..KZT_HAND_NUM_BUCKET {
            assert_eq!(evaluate_kzt_hand(&weights, &pos, bucket, Color::Black), 0);
        }
    }

    #[test]
    fn test_evaluate_empty_active_is_zero_even_with_nonzero_weight() {
        // startpos は両者持駒ゼロ → active gather が空なので、 weight が非零でも寄与 0。
        let mut weights = KztFactoredWeights::zeroed();
        weights.core[0] = 12345;
        weights.bias[KZT_BIAS_COUNT - 1] = -9999;
        let pos = make_pos();
        assert_eq!(evaluate_kzt_hand(&weights, &pos, 0, Color::Black), 0);
    }

    #[test]
    fn test_output_scale_zero_short_circuits() {
        let _guard = SCALE_TEST_LOCK.lock().unwrap();
        let weights = KztFactoredWeights::zeroed();
        let pos = make_pos();
        set_output_scale(0);
        assert_eq!(evaluate_kzt_hand(&weights, &pos, 0, Color::Black), 0);
        set_output_scale(KZT_OUTPUT_SCALE_DEFAULT);
    }

    // -----------------------------------------------------------------------
    // gather 本体 (= v0 実装) のテスト
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_roundtrip() {
        // index ↔ 6 軸 decode の往復一致 (= 全ストライド検証)。
        for &(p, b, kr, hp, tr, ds) in &[
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize),
            (1, 7, 15, 6, 24, 15),
            (0, 3, 9, 4, 12, 5),
        ] {
            let idx = kzt_hand_index(p, b, kr, hp, tr, ds);
            assert_eq!(kzt_hand_decode(idx), (p, b, kr, hp, tr, ds));
        }
    }

    #[test]
    fn test_collect_rook_drop_checks_exposed_king() {
        // 黒持駒に飛 1 枚、 白玉 5一 (= 露出)、 黒玉 5九。 黒 (= 攻め) は敵玉 5x5 zone に
        // 多数の飛打ちを持つ。 少なくとも 1 つは王手 (= check bit0) になるはず。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b R 1").unwrap();
        let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
        let n = collect_hand_features(&pos, Color::Black, 0, &mut buf);
        assert!(n > 0, "黒飛打ちの脅威が複数あるはず (got {n})");

        // 収集 feature を decode して check bit が立つものが存在することを確認。
        let mut has_check = false;
        for f in buf.iter().take(n) {
            let (persp, bucket, _kr, hp, _tr, ds) = kzt_hand_decode(f.index);
            assert_eq!(persp, Color::Black.index(), "persp = attacker 絶対色");
            assert_eq!(bucket, 0);
            assert_eq!(hp, 6, "飛 = hand_piece index 6");
            if ds & 1 != 0 {
                has_check = true;
            }
        }
        assert!(has_check, "飛を玉の筋/段に打てば王手になる feature があるはず");

        // 白 (= 攻め) は持駒ゼロ → 0 個。
        assert_eq!(collect_hand_features(&pos, Color::White, 0, &mut buf), 0);
    }

    #[test]
    fn test_collect_respects_nifu_and_hand() {
        // 黒持駒に歩 1 枚 + 同 file(5筋) に既に黒歩 → その file への歩打ちは二歩で除外。
        // 白玉 5一、 黒歩 5七 (= 5筋に不成歩) を置く。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/4P4/9/4K4 b P 1").unwrap();
        let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
        let n = collect_hand_features(&pos, Color::Black, 0, &mut buf);
        // 歩は hand_piece 0。 5筋(file index 4)は二歩で除外され、 zone 内の他 file のみ。
        for f in buf.iter().take(n) {
            let (_p, _b, _kr, hp, _tr, _ds) = kzt_hand_decode(f.index);
            assert_eq!(hp, 0, "持駒は歩のみ");
        }
        // 二歩除外の確認: 5筋空升への歩打ち feature が無いこと。 5一玉 zone の 5筋空升
        // (5二/5三/5四…のうち zone 内 = 5二/5三) への歩 feature が存在しないはず。
        // → 全 feature の file を復元するのは煩雑なので、 二歩無し版と件数比較で検証。
        let mut pos_no_pawn = Position::new();
        pos_no_pawn.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").unwrap();
        let n_free = collect_hand_features(&pos_no_pawn, Color::Black, 0, &mut buf);
        assert!(n_free > n, "二歩制約がある方が歩打ち候補は少ない (free {n_free} vs nifu {n})");
    }

    #[test]
    fn test_evaluate_nonzero_weight_fires_in_midgame() {
        let _guard = SCALE_TEST_LOCK.lock().unwrap();
        // gather が non-empty な局面で weight 非零なら eval が動く (= 配線確認)。
        let mut weights = KztFactoredWeights::zeroed();
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b R 1").unwrap();
        // 黒の active feature を 1 つ取り、 その core テーブル要素を立てる。
        let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
        let n = collect_hand_features(&pos, Color::Black, 0, &mut buf);
        assert!(n > 0);
        let (_p, _b, _kr, hp, tr, ds) = kzt_hand_decode(buf[0].index);
        weights.core[kzt_core_index(hp, tr, ds)] = 100;
        set_output_scale(4);
        let v = evaluate_kzt_hand(&weights, &pos, 0, Color::Black);
        set_output_scale(KZT_OUTPUT_SCALE_DEFAULT);
        // 黒 (stm) = us、 active feature が当たり weight 100 × scale 4 ぶん非零。
        assert_ne!(v, 0, "midgame で weight 非零なら eval が動くはず (got {v})");
    }

    #[test]
    fn test_clamp_raw_limits_head_sum() {
        let _guard = SCALE_TEST_LOCK.lock().unwrap();
        // 同一 core 値を多数 active feature が叩き、 head 合計が大きくなる局面で clamp が効く。
        let mut weights = KztFactoredWeights::zeroed();
        let mut pos = Position::new();
        // 露出玉 + 全持駒 → active feature 多数。
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b RBGSLNP 1").unwrap();
        let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
        let n = collect_hand_features(&pos, Color::Black, 0, &mut buf);
        assert!(n > 4, "多数 active が必要 (got {n})");
        // 全 drop_state に bias を入れて head 合計を必ず大きくする。
        for b in weights.bias.iter_mut() {
            *b = 100;
        }
        set_output_scale(1);

        // clamp 無効 (= 0): us = Σ = n × 100 で大きい。
        set_clamp_raw(0);
        let raw = evaluate_kzt_hand(&weights, &pos, 0, Color::Black);
        // 白 (them) は持駒ゼロ → them = 0。 us = n*100。
        assert_eq!(raw, (n as i32) * 100);

        // clamp K=50: us が 50 に clamp、 them=0 → 50。
        set_clamp_raw(50);
        let clamped = evaluate_kzt_hand(&weights, &pos, 0, Color::Black);
        assert_eq!(clamped, 50, "head 合計が K=50 に clamp されるはず (raw={raw})");

        set_clamp_raw(KZT_CLAMP_RAW_DEFAULT);
        set_output_scale(KZT_OUTPUT_SCALE_DEFAULT);
    }

    #[test]
    fn test_debug_summary_two_positions() {
        // Codex 要望の debug 集計を 2 局面で出力 (= active 内訳)。 `--nocapture` で確認。
        let cases = [
            ("4k4/9/9/9/9/9/9/9/4K4 b R 1", "黒飛 vs 露出白玉"),
            ("4k4/9/9/9/9/9/9/9/4K4 b RBGSLNP 1", "黒全駒種 vs 露出白玉"),
        ];
        for (sfen, label) in cases {
            let mut pos = Position::new();
            pos.set_sfen(sfen).unwrap();
            let s = summarize_hand_features(&pos, Color::Black, 0);
            println!(
                "KZT_DEBUG [{label}] sfen='{sfen}' total={} by_hand_piece={:?} by_drop_state={:?} by_king_region={:?}",
                s.total, s.by_hand_piece, s.by_drop_state, s.by_king_region
            );
            assert_eq!(
                s.total,
                s.by_hand_piece.iter().sum::<usize>(),
                "by_hand_piece の総和は total に一致"
            );
            assert_eq!(
                s.total,
                s.by_drop_state.iter().sum::<usize>(),
                "by_drop_state の総和は total に一致"
            );
            assert!(s.total > 0, "[{label}] active feature が存在するはず");
        }
    }

    /// KZT_HAND gather の純 scan コスト計測 (= scale 0 早期 return vs scale 4 scan)。
    ///
    /// `cargo test -p rshogi-core --release --features nnue-kzt-psqt -- --ignored \
    ///  --nocapture kzt_scan_nps` で実行。 weight 全 0 (= eval 結果は常に 0) なので
    /// 計測対象は collect_hand_features の scan コストのみ。
    #[test]
    #[ignore = "perf 計測用 (timing、 通常 test では skip)"]
    fn kzt_scan_nps_microbench() {
        use std::time::Instant;
        let weights = KztFactoredWeights::zeroed();
        const ITERS: u64 = 3_000_000;
        let cases = [
            // (a) 最悪ケース: 持駒全種 + 露出玉 (= 5x5 全空、 attackers_to scan 最大)。
            ("4k4/9/9/9/9/9/9/9/4K4 b RBGSLNP 1", "worst(exposed+all-hand)"),
            // (b) 現実的中盤: 矢倉囲い + 控えめ持駒 (= 玉周り密、 空升少)。
            (
                "ln1g3nl/1ks1g2b1/1ppp1p1pp/p3p1p2/5P3/2P1P1P2/PP1PS2PP/1BKGG2R1/LN5NL w Srs 1",
                "realistic(castle+small-hand)",
            ),
        ];
        for (sfen, label) in cases {
            let mut pos = Position::new();
            pos.set_sfen(sfen).unwrap();
            let stm = pos.side_to_move();

            set_output_scale(0);
            let t0 = Instant::now();
            let mut acc0: i64 = 0;
            for _ in 0..ITERS {
                acc0 += evaluate_kzt_hand(&weights, &pos, 0, stm) as i64;
            }
            let d0 = t0.elapsed();

            set_output_scale(4);
            let t1 = Instant::now();
            let mut acc4: i64 = 0;
            for _ in 0..ITERS {
                acc4 += evaluate_kzt_hand(&weights, &pos, 0, stm) as i64;
            }
            let d4 = t1.elapsed();
            set_output_scale(KZT_OUTPUT_SCALE_DEFAULT);

            let ns0 = d0.as_nanos() as f64 / ITERS as f64;
            let ns4 = d4.as_nanos() as f64 / ITERS as f64;
            println!(
                "KZT_SCAN_NPS [{label}] iters={ITERS} scale0={ns0:.2}ns/call scale4={ns4:.2}ns/call overhead={:.2}ns/call (acc0={acc0} acc4={acc4})",
                ns4 - ns0
            );
            // weight 全 0 なので eval は常に 0 (= scan コストに関わらず bit-exact)。
            assert_eq!(acc0, 0);
            assert_eq!(acc4, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// KZT-FT (= FT 入力版) の test
// ---------------------------------------------------------------------------
#[cfg(all(test, feature = "nnue-kzt-ft"))]
mod ft_tests {
    use super::*;
    use crate::position::{Position, SFEN_HIRATE};

    /// Black が飛角を持ち、 White 玉が上段で露出した局面 (= 打ち込み target 多数)。
    fn make_drop_pos() -> Position {
        let mut pos = Position::new();
        // 5a=White 玉、 5i=Black 玉、 Black 手駒 = 飛+角。 盤上ほぼ空 = 5x5 zone 全空升。
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b RB 1").unwrap();
        pos
    }

    /// `for_each_active_kzt_ft_index` の emit が `collect_hand_features` + 4 テーブル射影と
    /// 完全一致することを確認 (= 「KZT active index が collect_hand_features と一致」)。
    #[test]
    fn test_ft_index_matches_collect() {
        let pos = make_drop_pos();
        let attacker = Color::Black;

        // 期待値: collect_hand_features を decode し 4 テーブル (bucket 除外) へ手動射影。
        let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
        let n = collect_hand_features(&pos, attacker, 0, &mut buf);
        assert!(n > 0, "drop pos で active feature が 0 なのは想定外");
        let mut expected = Vec::new();
        for feat in buf.iter().take(n) {
            let (persp, _bucket, kr, hp, tr, ds) = kzt_hand_decode(feat.index);
            expected.push(KZT_FT_CORE_OFFSET + kzt_core_index(hp, tr, ds));
            expected.push(KZT_FT_REGION_OFFSET + kzt_region_index(kr, tr, ds));
            expected.push(KZT_FT_PERSP_OFFSET + kzt_persp_index(persp, hp, ds));
            expected.push(KZT_FT_BIAS_OFFSET + kzt_bias_index(ds));
        }

        let mut actual = Vec::new();
        for_each_active_kzt_ft_index(&pos, attacker, |idx| actual.push(idx));

        assert_eq!(actual, expected, "FT emit 順・値が collect + 射影と不一致");
        // 個数 = hand feature 数 × 4、 全 index が範囲内。
        assert_eq!(actual.len(), n * KZT_FT_COMPONENTS_PER_FEATURE);
        assert!(actual.iter().all(|&i| i < KZT_FT_DIMENSIONS));
        assert!(actual.len() <= KZT_FT_MAX_ACTIVE_PER_PERSP);
    }

    /// gather は pos の純関数 → 同一局面で複数回呼んでも同一 index 列 (= 到達経路非依存)。
    /// v0 は常に full refresh なので、 これが「refresh 経路の到達順不変性」を保証する。
    #[test]
    fn test_ft_gather_is_deterministic() {
        let pos = make_drop_pos();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for_each_active_kzt_ft_index(&pos, Color::Black, |i| a.push(i));
        for_each_active_kzt_ft_index(&pos, Color::Black, |i| b.push(i));
        assert_eq!(a, b);
    }

    /// 合成 weight を用いた accumulator 相当の総和が、 打てる持駒がある局面で動き、
    /// 無い局面 (= 持駒なし) で不動 (= 0) であることを確認 (= 「eval が効く/効かない」)。
    #[test]
    fn test_ft_accumulation_moves_only_with_droppable_hand() {
        const L: usize = 16; // accumulator 幅 (= test 用の小 L1)。
        // 全 weight = 1 の合成 [dim × L] i8。 各 active index が acc[j] を +1 する。
        let weights = vec![1i8; KZT_FT_DIMENSIONS * L];
        let accumulate = |pos: &Position, attacker: Color| -> Vec<i32> {
            let mut acc = vec![0i32; L];
            for_each_active_kzt_ft_index(pos, attacker, |idx| {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a += weights[idx * L + j] as i32;
                }
            });
            acc
        };

        // 打てる持駒あり → 各次元 = active index 数 (> 0)、 acc が動く。
        let drop_pos = make_drop_pos();
        let acc_drop = accumulate(&drop_pos, Color::Black);
        let active = {
            let mut c = 0usize;
            for_each_active_kzt_ft_index(&drop_pos, Color::Black, |_| c += 1);
            c
        };
        assert!(active > 0);
        assert!(acc_drop.iter().all(|&x| x == active as i32));

        // 持駒なし (= 平手初形) → active 0 → acc は全 0 (= weight が非ゼロでも不動)。
        let mut hirate = Position::new();
        hirate.set_sfen(SFEN_HIRATE).unwrap();
        let acc_hirate = accumulate(&hirate, Color::Black);
        assert!(acc_hirate.iter().all(|&x| x == 0));
    }
}
