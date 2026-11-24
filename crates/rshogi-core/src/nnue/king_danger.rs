//! King Danger (KDGR) PSQT scalar head (2026-06-15)。
//!
//! 自玉中心 3x3 (= Chebyshev 距離 ≤ 1 の 9 マス、 玉自身を含む) に居る
//! **敵駒の重み付き数** を計算し、 `min(weighted_sum, 7)` で `0..=7` の
//! 8 class に離散化、 8 class × 8 bucket × 2 persp = **128 i16 weight** の
//! 軽量 scalar table で eval 加算する。
//!
//! 残差解析で「自玉危険を約 -50cp 過小評価 (t=14.8、 2-shard 再現)」 が確認済。
//! 自玉が危険 (= class 高) なほど stm の eval を **下げる** 補正を学習で獲得する
//! ことが狙い (= 楽観バイアス補正)。
//!
//! # 設計
//!
//! ## feature 集合 (= class 計算)
//!
//! - perspective `persp` の自玉中心 3x3 zone (= `king_effect(king) | 玉自身`)
//! - zone に入る **敵駒** を重み付きで合計:
//!   - 重駒 (= 飛 / 角 / 金 / 銀 / 竜 / 馬 / と / 成香 / 成桂 / 成銀) = **3**
//!   - 軽駒 (= 歩 / 香 / 桂) = **1**
//!   - 玉は除外 (= weight 0)
//! - 合計を `min(7)` で `0..=7` の 8 class に clamp
//!
//! 実装は bitboard 直積 (= 敵重駒 mask / 敵軽駒 mask を 1 回構築 → zone と AND して
//! popcount) で計算し、 升単位の `piece_on` ループを避ける (= activity_risk の
//! `enemy_heavy_mask` 同方式、 NPS 重視)。
//!
//! ## weight layout (= KED と同形)
//!
//! - disk: `[bucket 8][persp 2][class 8] -> i16` = 128 i16 = 256 byte
//! - runtime: `[persp 2][class 8][bucket 8] -> i32` (bucket-inner)
//!   = hot path で `row[bucket]` の 8 連続 read 可能
//!
//! ## ckpt format
//!
//! ```text
//! sub_magic = "P2KDGR\0\0" (8 byte)
//! version   = u32 LE       (= 1)
//! count     = u32 LE       (= 128)
//! weights   = i16 × 128 LE (= 256 byte)
//! ```
//!
//! 合計 272 byte。 既存 section (= ... / P2KESC / P2AREL / P2ARS1) の後ろに append。
//!
//! ## evaluate
//!
//! `(us_acc[bucket] - them_acc[bucket]) × OUTPUT_SCALE`、 stm 視点で反対称合成。
//! PSQT / pairshape / activity_risk / king_escape と同 convention。
//! `us` = 自玉危険 class の weight (= 学習で負方向に育てば stm を下げる)、
//! `them` = 敵玉危険 class の weight。
//!
//! `OUTPUT_SCALE` は USI option `KING_DANGER_OUTPUT_SCALE` で runtime 上書き可能
//! (= ARH_V2_OUTPUT_SCALE 同方式、 AtomicI32、 default 4)。

use crate::bitboard::{
    Bitboard, bishop_effect, dragon_effect, horse_effect, king_effect, lance_effect, rook_effect,
};
use crate::position::Position;
use crate::types::{Color, PieceType};
use std::sync::atomic::{AtomicI32, Ordering};

pub const KING_DANGER_VERSION: u32 = 1;
pub const KING_DANGER_NUM_CLASS: usize = 8;
pub const KING_DANGER_NUM_BUCKET: usize = 8;
pub const KING_DANGER_NUM_PERSP: usize = 2;
pub const KING_DANGER_WEIGHT_COUNT: usize =
    KING_DANGER_NUM_CLASS * KING_DANGER_NUM_BUCKET * KING_DANGER_NUM_PERSP;

/// 重駒の weight (= 飛角金銀竜馬 + と/成香/成桂/成銀)
pub const KING_DANGER_HEAVY_WEIGHT: usize = 3;
/// 軽駒の weight (= 歩香桂)
pub const KING_DANGER_LIGHT_WEIGHT: usize = 1;

/// `OUTPUT_SCALE` 初期値 (= 4、 ARH_V2 と揃える)。
pub const KING_DANGER_OUTPUT_SCALE_DEFAULT: i32 = 4;

/// USI option `KING_DANGER_OUTPUT_SCALE` で runtime 上書き可能な eval scale。
pub static KING_DANGER_OUTPUT_SCALE: AtomicI32 = AtomicI32::new(KING_DANGER_OUTPUT_SCALE_DEFAULT);

/// USI option 経由で eval scale を更新する。
pub fn set_output_scale(scale: i32) {
    KING_DANGER_OUTPUT_SCALE.store(scale, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// 解析的 king_danger 補正 (= 訓練重み非依存、 2026-06-16)
// ---------------------------------------------------------------------------
//
// 中盤 gap = 自玉危険を約 -40cp 過小評価 (残差解析 king_danger3_diff coef -40,
// t14.8, 2-shard 再現) を、 **訓練重みでなく解析特徴 × 固定係数の直結項**で埋める。
//
// 訓練型 KDGR head (= 上記 class table) は残差軸と直交 (corr -0.01) しており
// OUTPUT_SCALE を上げても gap を埋められない。 そこで「自玉危険の生の重み付き合計」
// に固定係数を掛けて方向保証する。
//
// memory 法則「危険局面で eval を下げる悲観補正は勝つ」 に合致。

/// 解析的補正係数の default (= 0、 off)。 USI option `KING_DANGER_ANALYTIC_COEF` で
/// runtime 上書き。 `eval_cp -= COEF × king_danger3_diff` の単位 (= 最終 cp / unit)。
pub const KING_DANGER_ANALYTIC_COEF_DEFAULT: i32 = 0;

/// USI option `KING_DANGER_ANALYTIC_COEF` で runtime 上書き可能な解析補正係数。
///
/// `eval_cp -= COEF × king_danger3_diff` (= 最終 cp 単位)。 raw eval 加算時は
/// `fv_scale` を掛けて単位を揃える ([`analytic_king_danger_raw`] 参照)。
pub static KING_DANGER_ANALYTIC_COEF: AtomicI32 = AtomicI32::new(KING_DANGER_ANALYTIC_COEF_DEFAULT);

/// USI option 経由で解析補正係数を更新する。
pub fn set_analytic_coef(coef: i32) {
    KING_DANGER_ANALYTIC_COEF.store(coef, Ordering::Relaxed);
}

/// perspective 側の自玉中心 3x3 (= 玉自身含む、 盤外除外) に居る敵駒の
/// **生の重み付き合計** (= class clamp 前) を返す。
///
/// 解析的補正項で使用 (= class 離散化すると残差軸の線形性が崩れるため生値を使う)。
///
/// # 仕様
/// - zone = `king_effect(king) | 玉自身`
/// - 重駒 (= heavy mask) × 3 + 軽駒 (= light mask) × 1
/// - 玉は heavy / light どちらにも含めない
#[inline]
pub fn king_danger_weighted_sum(pos: &Position, persp: Color) -> i32 {
    let king_sq = pos.king_square(persp);
    let enemy = persp.opponent();
    let zone = king_effect(king_sq) | Bitboard::from_square(king_sq);

    let heavy = enemy_heavy_mask(pos, enemy) & zone;
    let light = enemy_light_mask(pos, enemy) & zone;

    (heavy.count() as i32) * (KING_DANGER_HEAVY_WEIGHT as i32)
        + (light.count() as i32) * (KING_DANGER_LIGHT_WEIGHT as i32)
}

/// `king_danger3_diff` (= 残差解析と同定義)。
///
/// `(自玉 3x3 内の敵駒重み付き数) − (敵玉 3x3 内の自駒重み付き数)`。
/// stm 視点。 正 = 自玉が相対的に危険。
#[inline]
pub fn king_danger3_diff(pos: &Position, stm: Color) -> i32 {
    let nstm = stm.opponent();
    king_danger_weighted_sum(pos, stm) - king_danger_weighted_sum(pos, nstm)
}

/// 解析的 king_danger 補正の **raw eval (= fv_scale 適用前)** 寄与を返す。
///
/// `eval_cp` への寄与は `-COEF × king_danger3_diff` (= 最終 cp 単位) であり、
/// raw_total (= fv_scale で割る前) に加算するため `× fv_scale` で単位を揃える:
///
/// ```text
/// raw 寄与 = -COEF × king_danger3_diff × fv_scale
/// total_cp = raw_total / fv_scale  ⇒  cp 寄与 = -COEF × king_danger3_diff
/// ```
///
/// `COEF == 0` (= default off) で常に 0 (= bit-exact 互換)。
///
/// # Arguments
/// * `pos`      - 局面
/// * `stm`      - 手番 (= side to move)
/// * `fv_scale` - 現在の fv_scale (= raw 単位換算用)
#[inline]
pub fn analytic_king_danger_raw(pos: &Position, stm: Color, fv_scale: i32) -> i32 {
    let coef = KING_DANGER_ANALYTIC_COEF.load(Ordering::Relaxed);
    if coef == 0 {
        return 0;
    }
    let diff = king_danger3_diff(pos, stm);
    // 自玉危険 (diff 正) → eval を下げる ⇒ 符号は負。
    (-coef).saturating_mul(diff).saturating_mul(fv_scale)
}

// ---------------------------------------------------------------------------
// slider 拡張: 玉 3x3 に届く敵 slider の **本物の利き** (block-aware) を捕捉
// (= 2026-06-29)
// ---------------------------------------------------------------------------
//
// 既存の `king_danger_weighted_sum` は自玉 3x3 に **物理的に居る** 敵駒のみを数える
// ため、 離れた位置から玉筋へ利いている飛角香 (= 遠隔の危険) を見落とす。
// ここでは occupied 考慮の本物の利き (= block-aware ray) が玉 3x3 zone に届く敵
// slider を捕捉し、 解析的補正の別軸として加算する。
//
// 物理在駒版 (= king_danger3_diff) と本軸 (= slider_diff) は別 USI option / 別係数で
// 独立に注入する。 既存項は不変。

/// 解析的 slider 補正係数の default (= 0、 off)。 USI option
/// `KING_DANGER_SLIDER_ANALYTIC_COEF` で runtime 上書き。
pub const KING_DANGER_SLIDER_ANALYTIC_COEF_DEFAULT: i32 = 0;

/// USI option `KING_DANGER_SLIDER_ANALYTIC_COEF` で runtime 上書き可能な
/// slider 解析補正係数。
///
/// `eval_cp -= COEF × king_danger_slider_diff` (= 最終 cp 単位)。 raw eval 加算時は
/// `fv_scale` を掛けて単位を揃える ([`analytic_king_danger_slider_raw`] 参照)。
pub static KING_DANGER_SLIDER_ANALYTIC_COEF: AtomicI32 =
    AtomicI32::new(KING_DANGER_SLIDER_ANALYTIC_COEF_DEFAULT);

/// USI option 経由で slider 解析補正係数を更新する。
pub fn set_slider_analytic_coef(coef: i32) {
    KING_DANGER_SLIDER_ANALYTIC_COEF.store(coef, Ordering::Relaxed);
}

/// perspective 側の自玉中心 3x3 zone に **本物の利き (= occupied 考慮の block-aware
/// ray)** が届く敵 slider の重み付き合計を返す。
///
/// `king_danger_weighted_sum` (= zone 内在駒のみ) と異なり、 zone から離れた位置に
/// 居ても利きが zone に通っていれば計上する (= 遠隔の飛角香の脅威を捕捉)。
///
/// # 仕様
/// - zone = `king_effect(king) | 玉自身` (= 3x3、 盤外自動除外)
/// - occupied = 全駒 (= block 考慮)
/// - 飛 / 竜 / 角 / 馬: 本物の利き `& zone` が非空なら × [`KING_DANGER_HEAVY_WEIGHT`]
/// - 香 (= enemy 色の前進方向): 本物の利き `& zone` が非空なら × [`KING_DANGER_LIGHT_WEIGHT`]
#[inline]
pub fn king_danger_slider_weighted_sum(pos: &Position, persp: Color) -> i32 {
    let king_sq = pos.king_square(persp);
    let enemy = persp.opponent();
    let zone = king_effect(king_sq) | Bitboard::from_square(king_sq);
    let occ = pos.occupied();

    let heavy_w = KING_DANGER_HEAVY_WEIGHT as i32;
    let light_w = KING_DANGER_LIGHT_WEIGHT as i32;
    let mut sum = 0i32;

    // 飛車 (= block 考慮の十字 ray)
    for sq in pos.pieces(enemy, PieceType::Rook).iter() {
        if !(rook_effect(sq, occ) & zone).is_empty() {
            sum += heavy_w;
        }
    }
    // 竜 (= 飛 + 玉利き)
    for sq in pos.pieces(enemy, PieceType::Dragon).iter() {
        if !(dragon_effect(sq, occ) & zone).is_empty() {
            sum += heavy_w;
        }
    }
    // 角 (= block 考慮の対角 ray)
    for sq in pos.pieces(enemy, PieceType::Bishop).iter() {
        if !(bishop_effect(sq, occ) & zone).is_empty() {
            sum += heavy_w;
        }
    }
    // 馬 (= 角 + 玉利き)
    for sq in pos.pieces(enemy, PieceType::Horse).iter() {
        if !(horse_effect(sq, occ) & zone).is_empty() {
            sum += heavy_w;
        }
    }
    // 香 (= enemy 色の前進方向 ray、 block 考慮)
    for sq in pos.pieces(enemy, PieceType::Lance).iter() {
        if !(lance_effect(enemy, sq, occ) & zone).is_empty() {
            sum += light_w;
        }
    }

    sum
}

/// `king_danger_slider_diff` (= slider 版の相対危険度)。
///
/// `(自玉 zone に届く敵 slider 重み) − (敵玉 zone に届く自 slider 重み)`。
/// stm 視点。 正 = 自玉が相対的に危険。
#[inline]
pub fn king_danger_slider_diff(pos: &Position, stm: Color) -> i32 {
    let nstm = stm.opponent();
    king_danger_slider_weighted_sum(pos, stm) - king_danger_slider_weighted_sum(pos, nstm)
}

/// 解析的 slider 補正の **raw eval (= fv_scale 適用前)** 寄与を返す。
///
/// `analytic_king_danger_raw` と完全同形 (= 物理在駒の代わりに block-aware slider 利き
/// を使う点のみ異なる)。
///
/// `COEF == 0` (= default off) で常に 0 (= bit-exact 互換)。
///
/// # Arguments
/// * `pos`      - 局面
/// * `stm`      - 手番 (= side to move)
/// * `fv_scale` - 現在の fv_scale (= raw 単位換算用)
#[inline]
pub fn analytic_king_danger_slider_raw(pos: &Position, stm: Color, fv_scale: i32) -> i32 {
    let coef = KING_DANGER_SLIDER_ANALYTIC_COEF.load(Ordering::Relaxed);
    if coef == 0 {
        return 0;
    }
    let diff = king_danger_slider_diff(pos, stm);
    // 自玉危険 (diff 正) → eval を下げる ⇒ 符号は負。
    (-coef).saturating_mul(diff).saturating_mul(fv_scale)
}

// ---------------------------------------------------------------------------
// slider 守備のみ (= 非対称) 版: 自玉危険の項だけを純悲観で注入 (= 2026-06-30)
// ---------------------------------------------------------------------------
//
// 上記 slider 版 (= `analytic_king_danger_slider_raw`) は対称
// (`slider_diff = self - enemy`) で、 攻めている局面 (= 敵玉危険、 diff 負) では
// eval を **上げて** しまい over-optimism を悪化させる。
//
// memory 法則「守備的盲点の補正は勝つ／攻めへの楽観は負ける」 に従い、 本軸は
// **自玉 (stm) が敵 slider に狙われている重み (= 常に >=0)** のみを使い、 敵玉項を
// 使わない純悲観とする (= 攻めでブーストしない非対称版)。 既存 slider / 物理在駒 /
// 3x3 版は一切変更せず、 独立 USI option / 別係数で共存させる。

/// 解析的 slider **守備のみ** 補正係数の default (= 0、 off)。 USI option
/// `KING_DANGER_SLIDER_DEF_ANALYTIC_COEF` で runtime 上書き。
pub const KING_DANGER_SLIDER_DEF_ANALYTIC_COEF_DEFAULT: i32 = 0;

/// USI option `KING_DANGER_SLIDER_DEF_ANALYTIC_COEF` で runtime 上書き可能な
/// slider **守備のみ** 解析補正係数。
///
/// `eval_cp -= COEF × king_danger_slider_weighted_sum(stm)` (= 最終 cp 単位、 自玉
/// 危険分のみ)。 raw eval 加算時は `fv_scale` を掛けて単位を揃える
/// ([`analytic_king_danger_slider_def_raw`] 参照)。
pub static KING_DANGER_SLIDER_DEF_ANALYTIC_COEF: AtomicI32 =
    AtomicI32::new(KING_DANGER_SLIDER_DEF_ANALYTIC_COEF_DEFAULT);

/// USI option 経由で slider **守備のみ** 解析補正係数を更新する。
pub fn set_slider_def_analytic_coef(coef: i32) {
    KING_DANGER_SLIDER_DEF_ANALYTIC_COEF.store(coef, Ordering::Relaxed);
}

/// 解析的 slider **守備のみ (= 非対称)** 補正の **raw eval (= fv_scale 適用前)**
/// 寄与を返す。
///
/// [`analytic_king_danger_slider_raw`] (= 対称 `self - enemy`) と異なり、 **自玉
/// (stm) が敵 slider に狙われている重み (= [`king_danger_slider_weighted_sum`]、
/// 常に >=0) のみ** を悲観方向に使う。 敵玉項は使わないため、 攻めている局面
/// (= 敵玉危険) で eval を上げることがない (= over-optimism を悪化させない)。
///
/// `COEF == 0` (= default off) で常に 0 (= bit-exact 互換)。
///
/// # Arguments
/// * `pos`      - 局面
/// * `stm`      - 手番 (= side to move)
/// * `fv_scale` - 現在の fv_scale (= raw 単位換算用)
#[inline]
pub fn analytic_king_danger_slider_def_raw(pos: &Position, stm: Color, fv_scale: i32) -> i32 {
    let coef = KING_DANGER_SLIDER_DEF_ANALYTIC_COEF.load(Ordering::Relaxed);
    if coef == 0 {
        return 0;
    }
    // 自玉 (stm) が敵 slider に狙われている重み (= 常に >=0)。 敵玉項は使わない。
    let danger = king_danger_slider_weighted_sum(pos, stm);
    // 自玉危険 → eval を下げる ⇒ 符号は負。 攻めでは danger=0 のため不変。
    (-coef).saturating_mul(danger).saturating_mul(fv_scale)
}

/// section magic `P2KDGR\0\0` (= 8 byte)
pub const KING_DANGER_MAGIC: &[u8; 8] = b"P2KDGR\0\0";

/// disk layout: `[bucket][persp][class]`
pub type KingDangerWeights =
    [[[i16; KING_DANGER_NUM_CLASS]; KING_DANGER_NUM_PERSP]; KING_DANGER_NUM_BUCKET];

/// runtime layout (= bucket-inner): `[persp][class][bucket]`
pub type KingDangerRuntimeWeights =
    [[[i32; KING_DANGER_NUM_BUCKET]; KING_DANGER_NUM_CLASS]; KING_DANGER_NUM_PERSP];

/// 敵 **重駒** mask (= 飛 / 角 / 金 / 銀 / 竜 / 馬 + と / 成香 / 成桂 / 成銀)。
///
/// activity_risk の `enemy_heavy_mask` と同 piece 集合。 玉は含まない。
/// 1 局面につき 1 回構築 → zone と AND して popcount する想定。
#[inline]
pub fn enemy_heavy_mask(pos: &Position, enemy: Color) -> Bitboard {
    pos.pieces(enemy, PieceType::Rook)
        | pos.pieces(enemy, PieceType::Bishop)
        | pos.pieces(enemy, PieceType::Gold)
        | pos.pieces(enemy, PieceType::Silver)
        | pos.pieces(enemy, PieceType::Dragon)
        | pos.pieces(enemy, PieceType::Horse)
        | pos.pieces(enemy, PieceType::ProPawn)
        | pos.pieces(enemy, PieceType::ProLance)
        | pos.pieces(enemy, PieceType::ProKnight)
        | pos.pieces(enemy, PieceType::ProSilver)
}

/// 敵 **軽駒** mask (= 歩 / 香 / 桂)。 玉は含まない。
#[inline]
pub fn enemy_light_mask(pos: &Position, enemy: Color) -> Bitboard {
    pos.pieces(enemy, PieceType::Pawn)
        | pos.pieces(enemy, PieceType::Lance)
        | pos.pieces(enemy, PieceType::Knight)
}

/// perspective 側の自玉中心 3x3 (= 玉自身含む、 盤外除外) に居る敵駒の重み付き数を
/// `0..=7` の class で返す。
///
/// # Arguments
/// * `pos`   - 局面
/// * `persp` - 視点 (= 自玉の色)
///
/// # 仕様
/// - zone = `king_effect(king) | 玉自身` (= 最大 9 升、 盤外自動除外)
/// - 重駒 (= heavy mask) × 3 + 軽駒 (= light mask) × 1 を zone 内で合計
/// - 玉は heavy / light どちらにも含めない (= weight 0)
/// - 最終 weighted_sum を `min(7)` で 8 class に clamp
pub fn compute_king_danger_class(pos: &Position, persp: Color) -> usize {
    let king_sq = pos.king_square(persp);
    let enemy = persp.opponent();
    let zone = king_effect(king_sq) | Bitboard::from_square(king_sq);

    let heavy = enemy_heavy_mask(pos, enemy) & zone;
    let light = enemy_light_mask(pos, enemy) & zone;

    let weighted = (heavy.count() as usize) * KING_DANGER_HEAVY_WEIGHT
        + (light.count() as usize) * KING_DANGER_LIGHT_WEIGHT;
    weighted.min(KING_DANGER_NUM_CLASS - 1)
}

/// flatten index (= disk layout 用): `[bucket][persp][class]` → flat
#[inline]
pub const fn flatten_disk_idx(bucket: usize, persp: usize, class: usize) -> usize {
    bucket * (KING_DANGER_NUM_PERSP * KING_DANGER_NUM_CLASS) + persp * KING_DANGER_NUM_CLASS + class
}

/// runtime weight build (= bucket-inner 転置)
pub fn build_runtime_weights(disk: &KingDangerWeights) -> KingDangerRuntimeWeights {
    let mut rt: KingDangerRuntimeWeights =
        [[[0i32; KING_DANGER_NUM_BUCKET]; KING_DANGER_NUM_CLASS]; KING_DANGER_NUM_PERSP];
    for (bucket, disk_bucket) in disk.iter().enumerate() {
        for (persp, disk_persp) in disk_bucket.iter().enumerate() {
            for (class, &w) in disk_persp.iter().enumerate() {
                rt[persp][class][bucket] = w as i32;
            }
        }
    }
    rt
}

/// evaluate KingDanger scalar head。
///
/// stm 視点で `(us - them) × OUTPUT_SCALE` を返す。
///
/// # Arguments
/// * `weights` - runtime layout の KingDanger weight (= bucket-inner i32)
/// * `pos`     - 局面
/// * `bucket`  - 現在の bucket index (= 0..=7)
/// * `stm`     - 手番 (= side to move)
pub fn evaluate_king_danger(
    weights: &KingDangerRuntimeWeights,
    pos: &Position,
    bucket: usize,
    stm: Color,
) -> i32 {
    // OUTPUT_SCALE == 0 のとき最終結果は必ず `× 0 == 0` になるため、
    // bitboard scan (compute_king_danger_class) を行わず即 0 を返す。
    // `× 0` と早期 return 0 は代数的に同値なので eval は全局面で bit-exact に不変であり、
    // head を無効化 (scale=0) した本番 config で per-node の利き計算を丸ごと省ける純粋な NPS 改善となる。
    let scale = KING_DANGER_OUTPUT_SCALE.load(Ordering::Relaxed);
    if scale == 0 {
        return 0;
    }
    let nstm = stm.opponent();
    let stm_class = compute_king_danger_class(pos, stm);
    let nstm_class = compute_king_danger_class(pos, nstm);
    let us = weights[stm as usize][stm_class][bucket];
    let them = weights[nstm as usize][nstm_class][bucket];
    us.saturating_sub(them).saturating_mul(scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Position, SFEN_HIRATE};

    fn make_pos() -> Position {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        pos
    }

    #[test]
    fn test_magic_and_version() {
        assert_eq!(KING_DANGER_MAGIC, b"P2KDGR\0\0");
        assert_eq!(KING_DANGER_VERSION, 1);
        assert_eq!(KING_DANGER_WEIGHT_COUNT, 128);
        assert_eq!(KING_DANGER_NUM_CLASS, 8);
    }

    #[test]
    fn test_startpos_danger_class_zero() {
        // 初期局面: 自玉周辺 3x3 に敵駒なし → class 0
        let pos = make_pos();
        assert_eq!(compute_king_danger_class(&pos, Color::Black), 0);
        assert_eq!(compute_king_danger_class(&pos, Color::White), 0);
    }

    #[test]
    fn test_danger_class_light_and_heavy() {
        // 自玉 5九 (= file 4, rank 8 / black 視点) 周辺に敵駒を配置して class 計算を検証。
        // SFEN で黒玉を 5九、 その周囲に白の歩 (= 軽 1) + 白の飛 (= 重 3) を置く。
        // 5九 = 5筋9段。 隣接升 5八 (= 真上)、 4八、 6八 など。
        // ここでは「白歩 1 枚 (= +1) + 白飛 1 枚 (= +3) = weighted 4 → class 4」 を狙う。
        let mut pos = Position::new();
        // 黒玉 5九、 白歩 5八、 白飛 4八、 残りは最小限。 白玉は 5一 に配置。
        // 段 = 1..9 (上から)、 筋 = 1..9 (右から)。 SFEN は段ごと記述。
        let sfen = "4k4/9/9/9/9/9/9/3rp4/4K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        // 黒玉 (= Black king) 周辺 3x3 = 5九 中心。 5八 (= 白歩 p)、 4八 (= 白飛 r) が隣接。
        let cls = compute_king_danger_class(&pos, Color::Black);
        // 軽 1 (= 歩) + 重 3 (= 飛) = 4
        assert_eq!(cls, 4, "black king danger class should be 1*1 + 1*3 = 4");
        // 白玉 5一 周辺には黒駒なし → class 0
        assert_eq!(compute_king_danger_class(&pos, Color::White), 0);
    }

    #[test]
    fn test_danger_class_saturates_at_7() {
        // 自玉周辺 3x3 に重駒を 3 枚以上 (= 3*3=9 → clamp 7)。
        let mut pos = Position::new();
        // 黒玉 5九、 周囲 4八/5八/6八 に白飛/白角/白金 (= 重 3 枚 = 9 → clamp 7)。
        let sfen = "4k4/9/9/9/9/9/9/3rbg3/4K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        let cls = compute_king_danger_class(&pos, Color::Black);
        assert_eq!(cls, 7, "3 heavy pieces (weighted 9) should saturate to class 7");
    }

    #[test]
    fn test_flatten_disk_idx() {
        // first slot (= bucket=0, persp=0, class=0) = 0
        assert_eq!(flatten_disk_idx(0, 0, 0), 0);
        // last slot (= bucket=7, persp=1, class=7) = 127
        assert_eq!(flatten_disk_idx(7, 1, 7), 127);
        assert_eq!(flatten_disk_idx(7, 1, 7) + 1, KING_DANGER_WEIGHT_COUNT);
    }

    #[test]
    fn test_build_runtime_weights_layout() {
        let mut disk: KingDangerWeights = [[[0; 8]; 2]; 8];
        // bucket 3, persp 0 (Black), class 5 に -123 を入れる
        disk[3][0][5] = -123;
        let rt = build_runtime_weights(&disk);
        assert_eq!(rt[0][5][3], -123);
        // 他は 0
        assert_eq!(rt[0][0][0], 0);
        assert_eq!(rt[1][7][7], 0);
    }

    #[test]
    fn test_evaluate_zero_weights() {
        // 全 weight 0 で evaluate = 0 (= zero-load 時無害の証明)
        let disk: KingDangerWeights = [[[0; 8]; 2]; 8];
        let rt = build_runtime_weights(&disk);
        let pos = make_pos();
        for bucket in 0..KING_DANGER_NUM_BUCKET {
            let v = evaluate_king_danger(&rt, &pos, bucket, Color::Black);
            assert_eq!(v, 0, "zero weight => zero contribution (bucket {bucket})");
        }
    }

    #[test]
    fn test_evaluate_output_scale_linear() {
        // OUTPUT_SCALE を変えると eval が線形変化する (= head 発火の証明)。
        // us class の weight だけ非零にして them=0 を保証。
        let mut disk: KingDangerWeights = [[[0; 8]; 2]; 8];
        // 全 bucket / persp の class 0 に -10 を入れる (= startpos は両者 class 0)
        for bucket_row in disk.iter_mut() {
            for persp_row in bucket_row.iter_mut() {
                persp_row[0] = -10;
            }
        }
        let rt = build_runtime_weights(&disk);
        let pos = make_pos();
        // startpos は black/white 対称 (= 両者 class 0)、 us-them = 0 になるため
        // 非対称局面で測る。 黒玉周辺に敵駒を置いて stm(black) class > 0 にする。
        let mut pos2 = Position::new();
        pos2.set_sfen("4k4/9/9/9/9/9/9/3rp4/4K4 b - 1").unwrap();
        // black class = 4、 white class = 0。 class 4 の weight を設定。
        let mut disk2: KingDangerWeights = [[[0; 8]; 2]; 8];
        for bucket_row in disk2.iter_mut() {
            bucket_row[Color::Black as usize][4] = -10;
        }
        let rt2 = build_runtime_weights(&disk2);

        set_output_scale(1);
        let v1 = evaluate_king_danger(&rt2, &pos2, 0, Color::Black);
        set_output_scale(4);
        let v4 = evaluate_king_danger(&rt2, &pos2, 0, Color::Black);
        set_output_scale(64);
        let v64 = evaluate_king_danger(&rt2, &pos2, 0, Color::Black);
        // default に戻す (= 他 test への副作用防止)
        set_output_scale(KING_DANGER_OUTPUT_SCALE_DEFAULT);

        // us(class4)=-10, them(class0)=0 → (us-them) = -10
        assert_eq!(v1, -10);
        assert_eq!(v4, -40);
        assert_eq!(v64, -640);
        // pos (= startpos) は対称で 0
        let v_sym = evaluate_king_danger(&rt, &pos, 0, Color::Black);
        assert_eq!(v_sym, 0);
    }

    // ---- 解析的補正項 (= analytic) のテスト ----

    #[test]
    fn test_weighted_sum_raw_no_clamp() {
        // 自玉周辺 3x3 に重駒 3 枚 (= 3*3=9)。 class は 7 に clamp されるが
        // 生 weighted_sum は 9 のまま (= clamp なし)。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/3rbg3/4K4 b - 1").unwrap();
        assert_eq!(king_danger_weighted_sum(&pos, Color::Black), 9);
        assert_eq!(compute_king_danger_class(&pos, Color::Black), 7);
        // 敵玉周辺は無し
        assert_eq!(king_danger_weighted_sum(&pos, Color::White), 0);
    }

    #[test]
    fn test_king_danger3_diff_sign() {
        // 黒玉 5九 周辺に白歩(1)+白飛(3) = 4、 白玉 5一 周辺に黒駒なし = 0。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/3rp4/4K4 b - 1").unwrap();
        // stm=Black: diff = self(4) - enemy(0) = +4 (= 自玉危険)
        assert_eq!(king_danger3_diff(&pos, Color::Black), 4);
        // stm=White: diff = self(0) - enemy(4) = -4 (= 相手玉危険)
        assert_eq!(king_danger3_diff(&pos, Color::White), -4);
    }

    #[test]
    fn test_analytic_coef_zero_is_no_op() {
        // COEF=0 (= default off) で寄与 0 (= bit-exact 互換)。
        set_analytic_coef(0);
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/3rp4/4K4 b - 1").unwrap();
        for fv in [1, 16, 24] {
            assert_eq!(analytic_king_danger_raw(&pos, Color::Black, fv), 0);
        }
    }

    #[test]
    fn test_analytic_linear_and_sign() {
        // diff = +4、 COEF=40、 fv_scale=16 →
        //   cp 寄与 = -COEF*diff = -160 cp
        //   raw 寄与 = -COEF*diff*fv = -160*16 = -2560
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/3rp4/4K4 b - 1").unwrap();

        set_analytic_coef(40);
        let raw = analytic_king_danger_raw(&pos, Color::Black, 16);
        assert_eq!(raw, -2560);
        // cp 単位 (= raw / fv_scale) で -160
        assert_eq!(raw / 16, -160);

        // COEF を倍にすると raw も倍 (= 線形性)
        set_analytic_coef(80);
        let raw2 = analytic_king_danger_raw(&pos, Color::Black, 16);
        assert_eq!(raw2, raw * 2);

        // 相手番視点 (= diff 負) では符号反転 (= eval 上がる)
        set_analytic_coef(40);
        let raw_white = analytic_king_danger_raw(&pos, Color::White, 16);
        assert_eq!(raw_white, 2560);

        // default に戻す (= 他 test への副作用防止)
        set_analytic_coef(KING_DANGER_ANALYTIC_COEF_DEFAULT);
    }

    // ---- slider 拡張のテスト ----

    #[test]
    fn test_slider_startpos_diff_zero() {
        // 開幕は左右対称 + slider が玉筋に通っていない → diff = 0、 raw = 0。
        let pos = make_pos();
        assert_eq!(king_danger_slider_diff(&pos, Color::Black), 0);
        assert_eq!(king_danger_slider_diff(&pos, Color::White), 0);
        set_slider_analytic_coef(64);
        assert_eq!(analytic_king_danger_slider_raw(&pos, Color::Black, 14), 0);
        set_slider_analytic_coef(KING_DANGER_SLIDER_ANALYTIC_COEF_DEFAULT);
    }

    #[test]
    fn test_slider_coef_zero_is_no_op() {
        // COEF=0 (= default off) で寄与 0 (= bit-exact 互換)。
        set_slider_analytic_coef(0);
        let mut pos = Position::new();
        // 黒玉 5九、 同じ 5 筋に白飛 5五 (= 玉筋に block 無しで通る)。
        pos.set_sfen("4k4/9/9/9/4r4/9/9/9/4K4 b - 1").unwrap();
        for fv in [1, 14, 16] {
            assert_eq!(analytic_king_danger_slider_raw(&pos, Color::Black, fv), 0);
        }
    }

    #[test]
    fn test_slider_remote_rook_aims_at_king() {
        // 黒玉 5九、 白飛 5五 (= 同 5 筋、 間に駒なし) → 飛の利きが玉 3x3 (5八 含む)
        // に届く。 物理在駒版 (= king_danger3_diff) は zone に駒が無いので 0 だが、
        // slider 版は遠隔の利きを捕捉する。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/4r4/9/9/9/4K4 b - 1").unwrap();

        // 物理在駒版: 玉 3x3 に敵駒なし → 0
        assert_eq!(king_danger3_diff(&pos, Color::Black), 0);
        // slider 版: 白飛 1 枚が玉筋に通る → self=3, enemy=0 → diff=+3
        assert_eq!(king_danger_slider_weighted_sum(&pos, Color::Black), 3);
        assert_eq!(king_danger_slider_diff(&pos, Color::Black), 3);

        // 非 0 効果: COEF=64, fv=14 → raw = -64 * 3 * 14
        set_slider_analytic_coef(64);
        let raw = analytic_king_danger_slider_raw(&pos, Color::Black, 14);
        assert_eq!(raw, -64 * 3 * 14);
        set_slider_analytic_coef(KING_DANGER_SLIDER_ANALYTIC_COEF_DEFAULT);
    }

    #[test]
    fn test_slider_blocked_rook_not_counted() {
        // 黒玉 5九、 白飛 5五 だが間 (= 5七) に駒を挟むと利きが玉に届かない。
        // 5七 に黒歩を置く (= block) → slider 危険は 0。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/4r4/9/4P4/9/4K4 b - 1").unwrap();
        // 5七 の黒歩が飛の縦利きを遮断 → 玉 3x3 に届かない
        assert_eq!(king_danger_slider_weighted_sum(&pos, Color::Black), 0);
        assert_eq!(king_danger_slider_diff(&pos, Color::Black), 0);
    }

    // ---- slider 守備のみ (= 非対称) 版のテスト ----

    #[test]
    fn test_slider_def_startpos_zero() {
        // 開幕は slider が玉筋に通っていない → danger=0 → raw=0。
        let pos = make_pos();
        set_slider_def_analytic_coef(64);
        assert_eq!(analytic_king_danger_slider_def_raw(&pos, Color::Black, 14), 0);
        assert_eq!(analytic_king_danger_slider_def_raw(&pos, Color::White, 14), 0);
        set_slider_def_analytic_coef(KING_DANGER_SLIDER_DEF_ANALYTIC_COEF_DEFAULT);
    }

    #[test]
    fn test_slider_def_coef_zero_is_no_op() {
        // COEF=0 (= default off) で寄与 0 (= bit-exact 互換)。
        set_slider_def_analytic_coef(0);
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/4r4/9/9/9/4K4 b - 1").unwrap();
        for fv in [1, 14, 16] {
            assert_eq!(analytic_king_danger_slider_def_raw(&pos, Color::Black, fv), 0);
        }
    }

    #[test]
    fn test_slider_def_self_danger_lowers_eval() {
        // 黒玉 5九、 白飛 5五 (= 同 5 筋、 間に駒なし) → 飛の利きが自玉 (= stm=Black)
        // 3x3 (5八 含む) に届く → danger=3 → 守備のみ版は eval を下げる (= raw<0)。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/4r4/9/9/9/4K4 b - 1").unwrap();
        // 自玉危険 (= stm) の slider 重み = 飛 1 枚 = 3
        assert_eq!(king_danger_slider_weighted_sum(&pos, Color::Black), 3);

        set_slider_def_analytic_coef(64);
        // raw = -COEF * danger(=3) * fv(=14) < 0
        let raw = analytic_king_danger_slider_def_raw(&pos, Color::Black, 14);
        assert_eq!(raw, -64 * 3 * 14);
        assert!(raw < 0);
        set_slider_def_analytic_coef(KING_DANGER_SLIDER_DEF_ANALYTIC_COEF_DEFAULT);
    }

    #[test]
    fn test_slider_def_enemy_danger_no_boost() {
        // 敵玉だけが狙われ、 自玉 (= stm=Black) は安全な局面 (= 攻めている形)。
        // 黒玉 5九 (= 安全)、 白玉 5一、 黒飛 5五 が白玉筋に通る (= 敵玉危険)。
        // 守備のみ版は自玉項のみを使うため、 攻めでは danger=0 → raw=0 (= 不変)。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1").unwrap();
        // 対称 slider 版では敵玉危険を捕捉 (= diff 負): 自玉 0、 敵玉 3 → diff = -3
        assert_eq!(king_danger_slider_weighted_sum(&pos, Color::Black), 0);
        assert_eq!(king_danger_slider_weighted_sum(&pos, Color::White), 3);
        assert_eq!(king_danger_slider_diff(&pos, Color::Black), -3);

        // 守備のみ版: 自玉 danger=0 → eval 不変 (= 攻めでブーストしない)。
        set_slider_def_analytic_coef(64);
        assert_eq!(analytic_king_danger_slider_def_raw(&pos, Color::Black, 14), 0);
        set_slider_def_analytic_coef(KING_DANGER_SLIDER_DEF_ANALYTIC_COEF_DEFAULT);
    }
}
