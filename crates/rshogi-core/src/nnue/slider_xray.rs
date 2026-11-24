//! Slider X-ray (SXRY) PSQT scalar head (2026-06-28)。
//!
//! 自玉 `K` に対する **敵 slider の「X-ray (= 潜在) 照準」** を重み付きで計数する。
//! 「X-ray 照準」 = 敵 slider と K の間に **ちょうど 1 枚** の駒 (= blocker) が居る
//! 状態の遠方利き (= block が 1 枚どけば直射になる潜在脅威)。 直射 (= blocker 0 枚)
//! は `attack_dev` (= ADEV) が扱う領域なので **計上しない**、 blocker 2 枚以上も計上しない。
//!
//! `min(weighted_sum, 7)` で `0..=7` の 8 class に離散化、
//! 8 class × 8 bucket × 2 persp = **128 i16 weight** の軽量 scalar table で eval 加算する。
//! 構造 (= class/bucket/persp、 weight layout、 ckpt format、 OUTPUT_SCALE 機構) は
//! `king_danger` (= KDGR) / `attack_dev` (= ADEV) の **完全ミラー**。 違うのは
//! feature 計算 (= X-ray 照準) だけ。
//!
//! # 設計
//!
//! ## feature 集合 (= class 計算、 per perspective)
//!
//! perspective `persp` の自玉 `K` に対し、 各 **敵 slider** が以下を満たすとき
//! 「X-ray 照準 1」 とする:
//!
//! 1. その slider の **移動方向** が K を向いている (= aim 成立):
//!    - 飛 / 竜 (= Rook / Dragon): K と同 筋 **または** 同 段 (= 直交 ray)
//!    - 角 / 馬 (= Bishop / Horse): K の斜め線上 (= `|df|==|dr|`)
//!    - 香 (= Lance): K の自軍前方 ray (= `lance_step_effect(persp, K)`) 上
//!      (= 同 筋 かつ K が香の前方)。 竜/馬の 1 マス利きは扱わない (= slider ray のみ)。
//! 2. slider と K の間 (= `between_bb`、 両端含まず) の **占有升数がちょうど 1**。
//!
//! 重み付き合計 = 重 slider (= 飛 / 竜) × 3 + 軽 slider (= 角 / 馬 / 香) × 1
//! (= king_danger の heavy/light weight を踏襲)。 `min(7)` で 8 class に clamp。
//!
//! ## weight layout (= KDGR / ADEV と同形)
//!
//! - disk: `[bucket 8][persp 2][class 8] -> i16` = 128 i16 = 256 byte
//! - runtime: `[persp 2][class 8][bucket 8] -> i32` (= bucket-inner)
//!
//! ## ckpt format
//!
//! ```text
//! sub_magic = "P2SXRY\0\0" (8 byte)
//! version   = u32 LE       (= 1)
//! count     = u32 LE       (= 128)
//! weights   = i16 × 128 LE (= 256 byte)
//! ```
//!
//! 合計 272 byte。 既存 section (= ... / P2KDGR / P2ADEV / ...) の **物理末尾** に append。
//!
//! ## evaluate
//!
//! `(us_acc[bucket] - them_acc[bucket]) × OUTPUT_SCALE`、 stm 視点で反対称合成。
//! KingDanger / AttackDev head と同 convention。
//!
//! `OUTPUT_SCALE` は USI option `SLIDER_XRAY_OUTPUT_SCALE` で runtime 上書き可能
//! (= KDGR / ADEV 同方式、 AtomicI32、 default 4)。

use crate::bitboard::{Bitboard, FILE_BB, RANK_BB, between_bb, bishop_effect, lance_step_effect};
use crate::position::Position;
use crate::types::{Color, PieceType};
use std::sync::atomic::{AtomicI32, Ordering};

pub const SLIDER_XRAY_VERSION: u32 = 1;
pub const SLIDER_XRAY_NUM_CLASS: usize = 8;
pub const SLIDER_XRAY_NUM_BUCKET: usize = 8;
pub const SLIDER_XRAY_NUM_PERSP: usize = 2;
pub const SLIDER_XRAY_WEIGHT_COUNT: usize =
    SLIDER_XRAY_NUM_CLASS * SLIDER_XRAY_NUM_BUCKET * SLIDER_XRAY_NUM_PERSP;

/// 重 slider (= 飛 / 竜) の X-ray weight (= king_danger heavy と同値)。
pub const SLIDER_XRAY_HEAVY_WEIGHT: usize = 3;
/// 軽 slider (= 角 / 馬 / 香) の X-ray weight (= king_danger light と同値)。
pub const SLIDER_XRAY_LIGHT_WEIGHT: usize = 1;

/// `OUTPUT_SCALE` 初期値 (= 4、 KDGR / ADEV と揃える)。
pub const SLIDER_XRAY_OUTPUT_SCALE_DEFAULT: i32 = 4;

/// USI option `SLIDER_XRAY_OUTPUT_SCALE` で runtime 上書き可能な eval scale。
pub static SLIDER_XRAY_OUTPUT_SCALE: AtomicI32 = AtomicI32::new(SLIDER_XRAY_OUTPUT_SCALE_DEFAULT);

/// USI option 経由で eval scale を更新する。
pub fn set_output_scale(scale: i32) {
    SLIDER_XRAY_OUTPUT_SCALE.store(scale, Ordering::Relaxed);
}

/// section magic `P2SXRY\0\0` (= 8 byte)
pub const SLIDER_XRAY_MAGIC: &[u8; 8] = b"P2SXRY\0\0";

/// disk layout: `[bucket][persp][class]`
pub type SliderXrayWeights =
    [[[i16; SLIDER_XRAY_NUM_CLASS]; SLIDER_XRAY_NUM_PERSP]; SLIDER_XRAY_NUM_BUCKET];

/// runtime layout (= bucket-inner): `[persp][class][bucket]`
pub type SliderXrayRuntimeWeights =
    [[[i32; SLIDER_XRAY_NUM_BUCKET]; SLIDER_XRAY_NUM_CLASS]; SLIDER_XRAY_NUM_PERSP];

/// 敵 重 slider mask (= Rook | Dragon)。
#[inline]
pub fn enemy_rook_like_mask(pos: &Position, enemy: Color) -> Bitboard {
    pos.pieces(enemy, PieceType::Rook) | pos.pieces(enemy, PieceType::Dragon)
}

/// 敵 角 / 馬 mask (= Bishop | Horse)。
#[inline]
pub fn enemy_bishop_like_mask(pos: &Position, enemy: Color) -> Bitboard {
    pos.pieces(enemy, PieceType::Bishop) | pos.pieces(enemy, PieceType::Horse)
}

/// 敵 香 mask (= Lance)。
#[inline]
pub fn enemy_lance_mask(pos: &Position, enemy: Color) -> Bitboard {
    pos.pieces(enemy, PieceType::Lance)
}

/// slider `from` と王 `king` の間 (= 両端含まず) に占有升がちょうど 1 枚かを返す。
///
/// `between_bb` は from/king が同一直線 (= 筋 / 段 / 斜め) に無いとき EMPTY を返すため、
/// aim mask で既に直線性を保証した後に呼ぶ前提。 0 枚 = 直射 (= ADEV 領域)、 2 枚以上 = 遮断
/// は X-ray ではないので false。
#[inline]
fn exactly_one_blocker(
    from: crate::types::Square,
    king: crate::types::Square,
    occupied: Bitboard,
) -> bool {
    (between_bb(from, king) & occupied).count() == 1
}

/// perspective 側の自玉 `K` に対する敵 slider の **X-ray 照準 重み付き合計**
/// (= class clamp 前) を返す。
///
/// # 仕様
/// - 飛 / 竜: K の 筋 or 段に乗る敵飛竜 で between 1 枚 → 各 weight 3
/// - 角 / 馬: K の対角線 (= 盤上駒無しの bishop ray) に乗る敵角馬 で between 1 枚 → 各 weight 1
/// - 香: K の自軍前方 ray 上の敵香 で between 1 枚 → 各 weight 1
#[inline]
pub fn slider_xray_weighted_sum(pos: &Position, persp: Color) -> i32 {
    let king_sq = pos.king_square(persp);
    let enemy = persp.opponent();
    let occupied = pos.occupied();

    // aim mask (= 幾何的に K を向く升集合)。 ADEV と同方式で構築し、 piece mask と AND して
    // 候補を絞った後、 各候補について between 1 枚を確認する。
    let rook_line = FILE_BB[king_sq.file().index()] | RANK_BB[king_sq.rank().index()];
    let bishop_line = bishop_effect(king_sq, Bitboard::EMPTY);
    let lance_line = lance_step_effect(persp, king_sq);

    let mut heavy_xray = 0i32;
    let mut light_xray = 0i32;

    for from in (enemy_rook_like_mask(pos, enemy) & rook_line).iter() {
        if exactly_one_blocker(from, king_sq, occupied) {
            heavy_xray += 1;
        }
    }
    for from in (enemy_bishop_like_mask(pos, enemy) & bishop_line).iter() {
        if exactly_one_blocker(from, king_sq, occupied) {
            light_xray += 1;
        }
    }
    for from in (enemy_lance_mask(pos, enemy) & lance_line).iter() {
        if exactly_one_blocker(from, king_sq, occupied) {
            light_xray += 1;
        }
    }

    heavy_xray * (SLIDER_XRAY_HEAVY_WEIGHT as i32) + light_xray * (SLIDER_XRAY_LIGHT_WEIGHT as i32)
}

/// perspective 側の自玉 X-ray 照準を `0..=7` の class で返す。
///
/// [`slider_xray_weighted_sum`] の重み付き合計を `min(7)` で 8 class に clamp。
pub fn compute_slider_xray_class(pos: &Position, persp: Color) -> usize {
    let sum = slider_xray_weighted_sum(pos, persp);
    (sum as usize).min(SLIDER_XRAY_NUM_CLASS - 1)
}

/// flatten index (= disk layout 用): `[bucket][persp][class]` → flat
#[inline]
pub const fn flatten_disk_idx(bucket: usize, persp: usize, class: usize) -> usize {
    bucket * (SLIDER_XRAY_NUM_PERSP * SLIDER_XRAY_NUM_CLASS) + persp * SLIDER_XRAY_NUM_CLASS + class
}

/// runtime weight build (= bucket-inner 転置)
pub fn build_runtime_weights(disk: &SliderXrayWeights) -> SliderXrayRuntimeWeights {
    let mut rt: SliderXrayRuntimeWeights =
        [[[0i32; SLIDER_XRAY_NUM_BUCKET]; SLIDER_XRAY_NUM_CLASS]; SLIDER_XRAY_NUM_PERSP];
    for (bucket, disk_bucket) in disk.iter().enumerate() {
        for (persp, disk_persp) in disk_bucket.iter().enumerate() {
            for (class, &w) in disk_persp.iter().enumerate() {
                rt[persp][class][bucket] = w as i32;
            }
        }
    }
    rt
}

/// evaluate SliderXray scalar head。
///
/// stm 視点で `(us - them) × OUTPUT_SCALE` を返す。
///
/// # Arguments
/// * `weights` - runtime layout の SliderXray weight (= bucket-inner i32)
/// * `pos`     - 局面
/// * `bucket`  - 現在の bucket index (= 0..=7)
/// * `stm`     - 手番 (= side to move)
pub fn evaluate_slider_xray(
    weights: &SliderXrayRuntimeWeights,
    pos: &Position,
    bucket: usize,
    stm: Color,
) -> i32 {
    // OUTPUT_SCALE == 0 のとき最終結果は必ず `× 0 == 0` になるため、
    // bitboard scan (compute_slider_xray_class) を行わず即 0 を返す。
    // `× 0` と早期 return 0 は代数的に同値なので eval は全局面で bit-exact に不変であり、
    // head を無効化 (scale=0) した本番 config で per-node の利き計算を丸ごと省ける純粋な NPS 改善となる。
    let scale = SLIDER_XRAY_OUTPUT_SCALE.load(Ordering::Relaxed);
    if scale == 0 {
        return 0;
    }
    let nstm = stm.opponent();
    let stm_class = compute_slider_xray_class(pos, stm);
    let nstm_class = compute_slider_xray_class(pos, nstm);
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
        assert_eq!(SLIDER_XRAY_MAGIC, b"P2SXRY\0\0");
        assert_eq!(SLIDER_XRAY_VERSION, 1);
        assert_eq!(SLIDER_XRAY_WEIGHT_COUNT, 128);
        assert_eq!(SLIDER_XRAY_NUM_CLASS, 8);
    }

    #[test]
    fn test_startpos_class_zero() {
        // 初期局面: 玉を X-ray する敵 slider は無い → class 0。
        let pos = make_pos();
        assert_eq!(compute_slider_xray_class(&pos, Color::Black), 0);
        assert_eq!(compute_slider_xray_class(&pos, Color::White), 0);
    }

    #[test]
    fn test_xray_lance_behind_one_pawn() {
        // 黒玉 5九。 5七 に白香、 5八 に駒 1 枚 (= 黒歩 = blocker) を置く。
        // 白香は黒玉の前方 同 5 筋、 香と玉の間 (= 5八) に 1 枚 → X-ray 照準 = 1 (= light 1)。
        // SFEN: 段 1..9 (上→下)、 筋 9..1 (左→右、 9 筋から記述)。
        // 5七 = 7 段 5 筋、 5八 = 8 段 5 筋。
        let mut pos = Position::new();
        let sfen = "4k4/9/9/9/9/9/4l4/4P4/4K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        // 黒玉 5九、 白香 5七、 黒歩 5八 (= between 1 枚)。
        assert_eq!(
            slider_xray_weighted_sum(&pos, Color::Black),
            1,
            "lance behind exactly one blocker should be a single light x-ray (weight 1)"
        );
        assert_eq!(compute_slider_xray_class(&pos, Color::Black), 1);
        // 白玉側への黒 slider X-ray は無し。
        assert_eq!(compute_slider_xray_class(&pos, Color::White), 0);
    }

    #[test]
    fn test_direct_attack_not_counted() {
        // 直射 (= blocker 0 枚) は X-ray ではない → class 0 (= ADEV の領域)。
        // 黒玉 5九、 白香 5七、 間 (= 5八) は空 → 直射。
        let mut pos = Position::new();
        let sfen = "4k4/9/9/9/9/9/4l4/9/4K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        assert_eq!(
            slider_xray_weighted_sum(&pos, Color::Black),
            0,
            "direct (0-blocker) attack must not be counted as x-ray"
        );
        assert_eq!(compute_slider_xray_class(&pos, Color::Black), 0);
    }

    #[test]
    fn test_xray_rook_behind_one_pawn_is_heavy() {
        // 黒玉 5九、 5五 に白飛 (= 同 5 筋)、 5七 に駒 1 枚 (= 黒歩 blocker)。
        // 飛は重 slider → X-ray weight 3。
        let mut pos = Position::new();
        let sfen = "4k4/9/9/9/4r4/9/4P4/9/4K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        assert_eq!(
            slider_xray_weighted_sum(&pos, Color::Black),
            3,
            "rook x-ray (one blocker) is heavy weight 3"
        );
        assert_eq!(compute_slider_xray_class(&pos, Color::Black), 3);
    }

    #[test]
    fn test_two_blockers_not_counted() {
        // 間に 2 枚 (= 5七 黒歩 + 5八 黒歩) → X-ray でない (= 0)。
        let mut pos = Position::new();
        let sfen = "4k4/9/9/9/4r4/9/4P4/4P4/4K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        assert_eq!(
            slider_xray_weighted_sum(&pos, Color::Black),
            0,
            "two blockers between slider and king is not an x-ray"
        );
    }

    #[test]
    fn test_bishop_xray_diagonal() {
        // 黒玉 5九、 1五 に白角 (= 斜め線 |df|=4,|dr|=4)、 間 (= 4八/3七/2六) のどこか 1 枚。
        // 3七 (= 7 段 3 筋) に黒歩 1 枚 → 角 X-ray = light 1。
        // SFEN 7 段目 (rank7): 筋 9..1 のうち 3 筋に P → "6P2"。
        let mut pos = Position::new();
        let sfen = "4k4/9/9/9/8b/9/6P2/9/4K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        assert_eq!(
            slider_xray_weighted_sum(&pos, Color::Black),
            1,
            "bishop diagonal x-ray with one blocker is light weight 1"
        );
    }

    #[test]
    fn test_flatten_disk_idx() {
        assert_eq!(flatten_disk_idx(0, 0, 0), 0);
        assert_eq!(flatten_disk_idx(7, 1, 7), 127);
        assert_eq!(flatten_disk_idx(7, 1, 7) + 1, SLIDER_XRAY_WEIGHT_COUNT);
    }

    #[test]
    fn test_build_runtime_weights_layout() {
        let mut disk: SliderXrayWeights = [[[0; 8]; 2]; 8];
        disk[3][0][5] = -123;
        let rt = build_runtime_weights(&disk);
        assert_eq!(rt[0][5][3], -123);
        assert_eq!(rt[0][0][0], 0);
        assert_eq!(rt[1][7][7], 0);
    }

    #[test]
    fn test_evaluate_zero_weights() {
        // 全 weight 0 で evaluate = 0 (= zero-load 時無害の証明)。
        let disk: SliderXrayWeights = [[[0; 8]; 2]; 8];
        let rt = build_runtime_weights(&disk);
        let pos = make_pos();
        for bucket in 0..SLIDER_XRAY_NUM_BUCKET {
            let v = evaluate_slider_xray(&rt, &pos, bucket, Color::Black);
            assert_eq!(v, 0, "zero weight => zero contribution (bucket {bucket})");
        }
    }

    #[test]
    fn test_evaluate_output_scale_linear() {
        // OUTPUT_SCALE を変えると eval が線形変化する (= head 発火の証明)。
        // 非対称局面: 黒玉に白飛が X-ray (= black class 3)、 白玉は照準なし (= class 0)。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/4r4/9/4P4/9/4K4 b - 1").unwrap();
        // class 3 (= black) の weight を全 bucket / persp Black に設定。
        let mut disk: SliderXrayWeights = [[[0; 8]; 2]; 8];
        for bucket_row in disk.iter_mut() {
            bucket_row[Color::Black as usize][3] = -10;
        }
        let rt = build_runtime_weights(&disk);

        set_output_scale(1);
        let v1 = evaluate_slider_xray(&rt, &pos, 0, Color::Black);
        set_output_scale(4);
        let v4 = evaluate_slider_xray(&rt, &pos, 0, Color::Black);
        set_output_scale(64);
        let v64 = evaluate_slider_xray(&rt, &pos, 0, Color::Black);
        set_output_scale(SLIDER_XRAY_OUTPUT_SCALE_DEFAULT);

        // us(class3)=-10, them(class0)=0 → (us-them) = -10。
        assert_eq!(v1, -10);
        assert_eq!(v4, -40);
        assert_eq!(v64, -640);

        // 対称局面 (= startpos) は 0。
        let pos_sym = make_pos();
        let mut disk_sym: SliderXrayWeights = [[[0; 8]; 2]; 8];
        for bucket_row in disk_sym.iter_mut() {
            for persp_row in bucket_row.iter_mut() {
                persp_row[0] = -10;
            }
        }
        let rt_sym = build_runtime_weights(&disk_sym);
        let v_sym = evaluate_slider_xray(&rt_sym, &pos_sym, 0, Color::Black);
        assert_eq!(v_sym, 0);
    }
}
