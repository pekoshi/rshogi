//! Attack Development (ADEV) PSQT scalar head (2026-06-16)。
//!
//! 自玉 (= 升 K) に対する **敵の「照準 (slider aim)」** を重み付きで計数し、
//! `min(weighted_sum, 7)` で `0..=7` の 8 class に離散化、
//! 8 class × 8 bucket × 2 persp = **128 i16 weight** の軽量 scalar table で
//! eval 加算する。
//!
//! 狙い = 「相手の発展中の攻め (= 玉を狙う遠方駒の照準) を過小評価」 を補正する
//! (= move-66 対策)。 玉に対する飛 / 角 / 香の照準が大きいほど stm の eval を
//! **下げる** 悲観方向の補正を学習で獲得させる。
//!
//! # 設計
//!
//! ## feature 集合 (= class 計算、 per perspective)
//!
//! perspective `persp` の自玉 `K` に対し、 **敵 slider** の照準を重み付き合計する。
//! 「照準」 は **block 越しでも計上** する (= 盤上駒を考慮しない幾何的 line / 対角線 /
//! 前方 ray を使う)。
//!
//! - 敵 飛 / 竜 (= Rook / Dragon) で `file(K)==file` **または** `rank(K)==rank`
//!   のもの → 各 weight **2**
//! - 敵 角 / 馬 (= Bishop / Horse) で K の斜め線上 (= `|df|==|dr|`) のもの
//!   → 各 weight **2**
//! - 敵 香 (= Lance) で `file(K)` 上 **かつ** K がその香の前方 (= 攻撃方向)
//!   のもの → 各 weight **1**
//!
//! 合計 = `own_king_aim`。 `min(7)` で `0..=7` の 8 class に clamp。
//!
//! 実装は bitboard 直積で高速化する (= 升単位の `piece_on` ループを避ける):
//! - 飛/竜の照準 mask = `FILE_BB[file(K)] | RANK_BB[rank(K)]`
//! - 角/馬の照準 mask = `bishop_effect(K, EMPTY)` (= block 無し対角 ray)
//! - 香の照準 mask = `lance_step_effect(persp, K)`
//!   (= K から自軍前方への ray。 この ray 上の敵香は K を前方に捉える)
//!
//! ## weight layout (= KDGR / KED と同形)
//!
//! - disk: `[bucket 8][persp 2][class 8] -> i16` = 128 i16 = 256 byte
//! - runtime: `[persp 2][class 8][bucket 8] -> i32` (= bucket-inner、 hot path で
//!   `row[bucket]` の 8 連続 read 可能)
//!
//! ## ckpt format
//!
//! ```text
//! sub_magic = "P2ADEV\0\0" (8 byte)
//! version   = u32 LE       (= 1)
//! count     = u32 LE       (= 128)
//! weights   = i16 × 128 LE (= 256 byte)
//! ```
//!
//! 合計 272 byte。 既存 section (= ... / P2KDGR) の後ろに append。
//!
//! ## evaluate
//!
//! `(us_acc[bucket] - them_acc[bucket]) × OUTPUT_SCALE`、 stm 視点で反対称合成。
//! KingDanger head と同 convention (= `us` = 自玉照準 class の weight、 学習で
//! 負方向に育てば stm を下げる悲観方向)。
//!
//! `OUTPUT_SCALE` は USI option `ATTACK_DEV_OUTPUT_SCALE` で runtime 上書き可能
//! (= ARH_V2 / KDGR 同方式、 AtomicI32、 default 4)。

use crate::bitboard::{Bitboard, FILE_BB, RANK_BB, bishop_effect, lance_step_effect};
use crate::position::Position;
use crate::types::{Color, PieceType};
use std::sync::atomic::{AtomicI32, Ordering};

pub const ATTACK_DEV_VERSION: u32 = 1;
pub const ATTACK_DEV_NUM_CLASS: usize = 8;
pub const ATTACK_DEV_NUM_BUCKET: usize = 8;
pub const ATTACK_DEV_NUM_PERSP: usize = 2;
pub const ATTACK_DEV_WEIGHT_COUNT: usize =
    ATTACK_DEV_NUM_CLASS * ATTACK_DEV_NUM_BUCKET * ATTACK_DEV_NUM_PERSP;

/// 飛 / 竜の照準 weight。
pub const ATTACK_DEV_ROOK_WEIGHT: usize = 2;
/// 角 / 馬の照準 weight。
pub const ATTACK_DEV_BISHOP_WEIGHT: usize = 2;
/// 香の照準 weight。
pub const ATTACK_DEV_LANCE_WEIGHT: usize = 1;

/// `OUTPUT_SCALE` 初期値 (= 4、 ARH_V2 / KDGR と揃える)。
pub const ATTACK_DEV_OUTPUT_SCALE_DEFAULT: i32 = 4;

/// USI option `ATTACK_DEV_OUTPUT_SCALE` で runtime 上書き可能な eval scale。
pub static ATTACK_DEV_OUTPUT_SCALE: AtomicI32 = AtomicI32::new(ATTACK_DEV_OUTPUT_SCALE_DEFAULT);

/// USI option 経由で eval scale を更新する。
pub fn set_output_scale(scale: i32) {
    ATTACK_DEV_OUTPUT_SCALE.store(scale, Ordering::Relaxed);
}

/// section magic `P2ADEV\0\0` (= 8 byte)
pub const ATTACK_DEV_MAGIC: &[u8; 8] = b"P2ADEV\0\0";

/// disk layout: `[bucket][persp][class]`
pub type AttackDevWeights =
    [[[i16; ATTACK_DEV_NUM_CLASS]; ATTACK_DEV_NUM_PERSP]; ATTACK_DEV_NUM_BUCKET];

/// runtime layout (= bucket-inner): `[persp][class][bucket]`
pub type AttackDevRuntimeWeights =
    [[[i32; ATTACK_DEV_NUM_BUCKET]; ATTACK_DEV_NUM_CLASS]; ATTACK_DEV_NUM_PERSP];

/// 敵 飛 / 竜 mask (= Rook | Dragon)。 1 局面につき 1 回構築する想定。
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

/// perspective 側の自玉 `K` に対する敵 slider の **照準 (slider aim) 重み付き合計**
/// (= class clamp 前) を返す。
///
/// block 越しでも照準として計上する (= 盤上駒を考慮しない幾何 mask)。
///
/// # 仕様
/// - 飛 / 竜: `FILE_BB[file(K)] | RANK_BB[rank(K)]` 上の敵飛竜 × 2
/// - 角 / 馬: `bishop_effect(K, EMPTY)` 上の敵角馬 × 2
/// - 香: `lance_step_effect(persp, K)` (= K の自軍前方 ray、 = 敵香の攻撃方向) 上の
///   敵香 × 1
#[inline]
pub fn attack_dev_weighted_sum(pos: &Position, persp: Color) -> i32 {
    let king_sq = pos.king_square(persp);
    let enemy = persp.opponent();

    // 飛 / 竜: K の筋 or 段に乗る敵飛竜 (= block 越し照準)。
    let rook_line = FILE_BB[king_sq.file().index()] | RANK_BB[king_sq.rank().index()];
    let rook_aim = enemy_rook_like_mask(pos, enemy) & rook_line;

    // 角 / 馬: K の対角線 (= 盤上駒無しの bishop ray) に乗る敵角馬。
    let bishop_line = bishop_effect(king_sq, Bitboard::EMPTY);
    let bishop_aim = enemy_bishop_like_mask(pos, enemy) & bishop_line;

    // 香: K の自軍前方 ray (= 敵香が K を前方に捉える升集合) に乗る敵香。
    let lance_line = lance_step_effect(persp, king_sq);
    let lance_aim = enemy_lance_mask(pos, enemy) & lance_line;

    (rook_aim.count() as i32) * (ATTACK_DEV_ROOK_WEIGHT as i32)
        + (bishop_aim.count() as i32) * (ATTACK_DEV_BISHOP_WEIGHT as i32)
        + (lance_aim.count() as i32) * (ATTACK_DEV_LANCE_WEIGHT as i32)
}

/// perspective 側の自玉照準を `0..=7` の class で返す。
///
/// # Arguments
/// * `pos`   - 局面
/// * `persp` - 視点 (= 自玉の色)
///
/// # 仕様
/// [`attack_dev_weighted_sum`] の重み付き合計を `min(7)` で 8 class に clamp。
pub fn compute_attack_dev_class(pos: &Position, persp: Color) -> usize {
    let sum = attack_dev_weighted_sum(pos, persp);
    (sum as usize).min(ATTACK_DEV_NUM_CLASS - 1)
}

/// flatten index (= disk layout 用): `[bucket][persp][class]` → flat
#[inline]
pub const fn flatten_disk_idx(bucket: usize, persp: usize, class: usize) -> usize {
    bucket * (ATTACK_DEV_NUM_PERSP * ATTACK_DEV_NUM_CLASS) + persp * ATTACK_DEV_NUM_CLASS + class
}

/// runtime weight build (= bucket-inner 転置)
pub fn build_runtime_weights(disk: &AttackDevWeights) -> AttackDevRuntimeWeights {
    let mut rt: AttackDevRuntimeWeights =
        [[[0i32; ATTACK_DEV_NUM_BUCKET]; ATTACK_DEV_NUM_CLASS]; ATTACK_DEV_NUM_PERSP];
    for (bucket, disk_bucket) in disk.iter().enumerate() {
        for (persp, disk_persp) in disk_bucket.iter().enumerate() {
            for (class, &w) in disk_persp.iter().enumerate() {
                rt[persp][class][bucket] = w as i32;
            }
        }
    }
    rt
}

/// evaluate AttackDev scalar head。
///
/// stm 視点で `(us - them) × OUTPUT_SCALE` を返す。
///
/// # Arguments
/// * `weights` - runtime layout の AttackDev weight (= bucket-inner i32)
/// * `pos`     - 局面
/// * `bucket`  - 現在の bucket index (= 0..=7)
/// * `stm`     - 手番 (= side to move)
pub fn evaluate_attack_dev(
    weights: &AttackDevRuntimeWeights,
    pos: &Position,
    bucket: usize,
    stm: Color,
) -> i32 {
    // OUTPUT_SCALE == 0 のとき最終結果は必ず `× 0 == 0` になるため、
    // bitboard scan (compute_attack_dev_class) を行わず即 0 を返す。
    // `× 0` と早期 return 0 は代数的に同値なので eval は全局面で bit-exact に不変であり、
    // head を無効化 (scale=0) した本番 config で per-node の利き計算を丸ごと省ける純粋な NPS 改善となる。
    let scale = ATTACK_DEV_OUTPUT_SCALE.load(Ordering::Relaxed);
    if scale == 0 {
        return 0;
    }
    let nstm = stm.opponent();
    let stm_class = compute_attack_dev_class(pos, stm);
    let nstm_class = compute_attack_dev_class(pos, nstm);
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
        assert_eq!(ATTACK_DEV_MAGIC, b"P2ADEV\0\0");
        assert_eq!(ATTACK_DEV_VERSION, 1);
        assert_eq!(ATTACK_DEV_WEIGHT_COUNT, 128);
        assert_eq!(ATTACK_DEV_NUM_CLASS, 8);
    }

    #[test]
    fn test_startpos_class_zero() {
        // 初期局面: 玉に照準する敵 slider は無い (= 自陣に飛角香が居ても
        // block 無し line では玉に乗らない配置) → class 0。
        let pos = make_pos();
        assert_eq!(compute_attack_dev_class(&pos, Color::Black), 0);
        assert_eq!(compute_attack_dev_class(&pos, Color::White), 0);
    }

    #[test]
    fn test_rook_on_king_file_increases_class() {
        // 黒玉 5九、 白飛を同じ 5 筋 (= 5五) に置く → file(K)==file で照準 → weight 2。
        // 盤上に間に駒があっても block 越しで計上されることも確認 (= 5七 に白歩)。
        let mut pos = Position::new();
        // 段 1..9 (上→下)、 筋 9..1 (左→右、 SFEN は 9 筋から記述)。
        // 5九 黒玉、 5五 白飛、 5七 白歩 (= 間の駒)、 5一 白玉。
        let sfen = "4k4/9/9/9/4r4/9/4p4/9/4K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        // 黒玉 5九 と白飛 5五 は同 5 筋 → 飛照準 weight 2 → class 2。
        assert_eq!(
            compute_attack_dev_class(&pos, Color::Black),
            2,
            "rook on king's file should give weight 2 even through a blocker"
        );
        // 白玉 5一 への黒 slider 照準は無し → class 0。
        assert_eq!(compute_attack_dev_class(&pos, Color::White), 0);
    }

    #[test]
    fn test_rook_on_king_rank_increases_class() {
        // 黒玉 5九、 白飛を同じ 9 段 (= 1九) に置く → rank(K)==rank で照準。
        let mut pos = Position::new();
        let sfen = "4k4/9/9/9/9/9/9/9/r3K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        assert_eq!(
            compute_attack_dev_class(&pos, Color::Black),
            2,
            "rook on king's rank should give weight 2"
        );
    }

    #[test]
    fn test_bishop_on_king_diagonal_increases_class() {
        // 黒玉 5九、 白角を斜め線上 (= 1五: |df|=4, |dr|=4) に置く → 角照準 weight 2。
        let mut pos = Position::new();
        // 1五 = 1 筋 5 段。 5九 から見て df=4, dr=4 (= 対角)。
        let sfen = "4k4/9/9/9/8b/9/9/9/4K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        assert_eq!(
            compute_attack_dev_class(&pos, Color::Black),
            2,
            "bishop on king's diagonal should give weight 2"
        );
        // 非対角 (= 制御): 白角を 1 筋 4 段 (= df=4, dr=5、 対角でない) に置くと 0。
        let mut pos2 = Position::new();
        pos2.set_sfen("4k4/9/9/8b/9/9/9/9/4K4 b - 1").unwrap();
        assert_eq!(
            compute_attack_dev_class(&pos2, Color::Black),
            0,
            "bishop off-diagonal should not be counted"
        );
    }

    #[test]
    fn test_lance_in_front_of_king_increases_class() {
        // 黒玉 5九。 白香は黒玉の前方 (= 上方、 段が小さい側) 同 5 筋に居れば
        // 黒玉を前方に捉える。 5五 に白香 → 照準 weight 1 → class 1。
        let mut pos = Position::new();
        let sfen = "4k4/9/9/9/4l4/9/9/9/4K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        assert_eq!(
            compute_attack_dev_class(&pos, Color::Black),
            1,
            "enemy lance in front of king on same file should give weight 1"
        );
        // 白香が黒玉の後方 (= 玉より下の段は盤外なので、 同 5 筋でも玉前方のみ計上)。
        // ここでは黒玉を 5五 に上げ、 白香を 5八 (= 玉の後方) に置くと照準にならないことを確認。
        let mut pos2 = Position::new();
        pos2.set_sfen("4k4/9/9/9/4K4/9/9/4l4/9 b - 1").unwrap();
        assert_eq!(
            compute_attack_dev_class(&pos2, Color::Black),
            0,
            "enemy lance behind black king should not aim at it"
        );
    }

    #[test]
    fn test_class_saturates_at_7() {
        // 黒玉 5九に対し、 飛(筋) + 飛(段) + 角(斜) + 香(前) を多数並べて
        // weighted sum > 7 → clamp 7 を確認。
        let mut pos = Position::new();
        // 5五 白飛(筋,2)、 1九 白飛(段,2)、 1五 白角(斜,2)、 5三 白香(前,1) = 7、
        // さらに 9五 白角(斜,2) で 9 → clamp 7。
        // 5九 から 9五 は df=4, dr=4 で対角。
        let sfen = "4k4/9/4l4/9/b3r3b/9/9/9/r3K4 b - 1";
        pos.set_sfen(sfen).unwrap();
        let cls = compute_attack_dev_class(&pos, Color::Black);
        assert_eq!(cls, 7, "many aiming sliders should saturate to class 7, got {cls}");
    }

    #[test]
    fn test_flatten_disk_idx() {
        assert_eq!(flatten_disk_idx(0, 0, 0), 0);
        assert_eq!(flatten_disk_idx(7, 1, 7), 127);
        assert_eq!(flatten_disk_idx(7, 1, 7) + 1, ATTACK_DEV_WEIGHT_COUNT);
    }

    #[test]
    fn test_build_runtime_weights_layout() {
        let mut disk: AttackDevWeights = [[[0; 8]; 2]; 8];
        disk[3][0][5] = -123;
        let rt = build_runtime_weights(&disk);
        assert_eq!(rt[0][5][3], -123);
        assert_eq!(rt[0][0][0], 0);
        assert_eq!(rt[1][7][7], 0);
    }

    #[test]
    fn test_evaluate_zero_weights() {
        // 全 weight 0 で evaluate = 0 (= zero-load 時無害の証明)。
        let disk: AttackDevWeights = [[[0; 8]; 2]; 8];
        let rt = build_runtime_weights(&disk);
        let pos = make_pos();
        for bucket in 0..ATTACK_DEV_NUM_BUCKET {
            let v = evaluate_attack_dev(&rt, &pos, bucket, Color::Black);
            assert_eq!(v, 0, "zero weight => zero contribution (bucket {bucket})");
        }
    }

    #[test]
    fn test_evaluate_output_scale_linear() {
        // OUTPUT_SCALE を変えると eval が線形変化する (= head 発火の証明)。
        // 非対称局面: 黒玉に白飛が照準 (= black class 2)、 白玉は照準なし (= class 0)。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/4r4/9/9/9/4K4 b - 1").unwrap();
        // class 2 (= black) の weight を全 bucket / persp Black に設定。
        let mut disk: AttackDevWeights = [[[0; 8]; 2]; 8];
        for bucket_row in disk.iter_mut() {
            bucket_row[Color::Black as usize][2] = -10;
        }
        let rt = build_runtime_weights(&disk);

        set_output_scale(1);
        let v1 = evaluate_attack_dev(&rt, &pos, 0, Color::Black);
        set_output_scale(4);
        let v4 = evaluate_attack_dev(&rt, &pos, 0, Color::Black);
        set_output_scale(64);
        let v64 = evaluate_attack_dev(&rt, &pos, 0, Color::Black);
        set_output_scale(ATTACK_DEV_OUTPUT_SCALE_DEFAULT);

        // us(class2)=-10, them(class0)=0 → (us-them) = -10。
        assert_eq!(v1, -10);
        assert_eq!(v4, -40);
        assert_eq!(v64, -640);

        // 対称局面 (= startpos) は 0。
        let pos_sym = make_pos();
        let mut disk_sym: AttackDevWeights = [[[0; 8]; 2]; 8];
        for bucket_row in disk_sym.iter_mut() {
            for persp_row in bucket_row.iter_mut() {
                persp_row[0] = -10;
            }
        }
        let rt_sym = build_runtime_weights(&disk_sym);
        let v_sym = evaluate_attack_dev(&rt_sym, &pos_sym, 0, Color::Black);
        assert_eq!(v_sym, 0);
    }

    #[test]
    fn test_weighted_sum_no_clamp() {
        // 生 weighted_sum は clamp なし。 飛(筋,2) + 角(斜,2) = 4。
        let mut pos = Position::new();
        // 5五 白飛(筋)、 1五 白角(斜)。
        pos.set_sfen("4k4/9/9/9/4r3b/9/9/9/4K4 b - 1").unwrap();
        assert_eq!(attack_dev_weighted_sum(&pos, Color::Black), 4);
        assert_eq!(compute_attack_dev_class(&pos, Color::Black), 4);
        // 敵 (= 白玉) 側への照準は無し。
        assert_eq!(attack_dev_weighted_sum(&pos, Color::White), 0);
    }
}
