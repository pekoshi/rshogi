//! Drop Pressure (DROP) PSQT scalar head (2026-06-16)。
//!
//! 相手の **持駒 (in-hand)** を弾とした **自玉近傍への投下脅威** を重み付きで計数し、
//! `min(weighted_sum, 7)` で `0..=7` の 8 class に離散化、
//! 8 class × 8 bucket × 2 persp = **128 i16 weight** の軽量 scalar table で
//! eval 加算する。
//!
//! 狙い = 負け棋譜分析 (= 9/9 局、 実戦の最頻負け筋) で判明した
//! 「相手の持駒を弾とした自玉近傍投下脅威の過小評価」 を補正する。
//! 既存 head (= king_danger / attack_dev 等は **盤上駒のみ** が入力源) や
//! 失敗 head (= 片側静的カウント) と **直交**する設計であり、
//! **持駒 (in-hand) が唯一の入力源**である点が新しい。
//!
//! 投下脅威が大きい (= class 高) ほど stm の eval を **下げる** 悲観方向の補正を
//! 学習で獲得させる (= memory 法則「危険局面で eval を下げる悲観補正は勝つ」 に適合)。
//! **符号は訓練が学習**する (= ここでは固定しない)。
//!
//! # 設計
//!
//! ## feature 集合 (= class 計算、 per perspective)
//!
//! perspective `persp` の自玉 `K` に対し、 **相手の持駒** を弾とした投下脅威を
//! 重み付き合計する。
//!
//! ```text
//! zone   = king_5x5_mask(K)              // 自玉中心 5x5 (= 盤外除外、 玉自身含む)
//! drop   = zone & ~occupied              // (a) 投下可能な空升
//! target = king_effect(K) | K            // 玉 + 玉近傍 8 升 (= 攻撃対象集合)
//! ```
//!
//! 相手 (= enemy) の各持駒種 (= 歩 / 香 / 桂 / 銀 / 金 / 角 / 飛、 count>0 のもの)
//! について、
//!
//! ```text
//! eff_sq = { sq in drop | drop_attack(piece, enemy, sq) & target != ∅ }
//! sum   += weight(piece) × min(eff_sq.count(), hand_count(piece))
//! ```
//!
//! - `drop_attack` は **投下した駒がその升から利かせる升集合**:
//!   - 歩 / 香 / 桂 = enemy 視点の前方利き (= `pawn_effect` / `lance_effect` /
//!     `knight_effect`、 香は occupancy 込みの遠隔利き)
//!   - 銀 / 金 = 近接利き (= `silver_effect` / `gold_effect`)
//!   - 角 / 飛 = occupancy 込みの遠隔利き (= `bishop_effect` / `rook_effect`、
//!     自陣後方侵入もカバー)
//! - `weight` = 飛 / 角 / 金 / 銀 = **2** (重)、 歩 / 香 / 桂 = **1** (軽)
//! - `min(eff_sq, hand_count)` で「投下可能な弾数」 を上限にして bound する
//!   (= 持駒 1 枚なら効く升が複数あっても投下できるのは 1 マス)
//!
//! 合計 = `opp_drop_pressure`。 `min(7)` で `0..=7` の 8 class に clamp。
//!
//! ## NPS
//!
//! zone 空升 (= 最大 25、 通常はもっと少ない) × 持駒種 (= count>0 のもののみ) を
//! 1 パスで走査し、 bitboard 積 + `is_empty()` で判定する (= hot path heap 割当無し)。
//! drop 升集合と target 集合は 1 局面につき 1 回だけ構築する。
//!
//! ## weight layout (= KDGR / ADEV と同形)
//!
//! - disk: `[bucket 8][persp 2][class 8] -> i16` = 128 i16 = 256 byte
//! - runtime: `[persp 2][class 8][bucket 8] -> i32` (= bucket-inner、 hot path で
//!   `row[bucket]` の 8 連続 read 可能)
//!
//! ## ckpt format
//!
//! ```text
//! sub_magic = "P2DROP\0\0" (8 byte)
//! version   = u32 LE       (= 1)
//! count     = u32 LE       (= 128)
//! weights   = i16 × 128 LE (= 256 byte)
//! ```
//!
//! 合計 272 byte。 既存 section (= ... / P2KDGR / P2ADEV / P2KMOB) の **後ろ**に
//! append する (= 物理末尾、 全 section の最後に probe される)。
//!
//! ## evaluate
//!
//! `(us_acc[bucket] - them_acc[bucket]) × OUTPUT_SCALE`、 stm 視点で反対称合成。
//! KingDanger / AttackDev / KingMobility head と同 convention
//! (= `us` = 自玉への投下脅威 class の weight、 学習で負方向に育てば stm を下げる)。
//!
//! `OUTPUT_SCALE` は USI option `DROP_PRESSURE_OUTPUT_SCALE` で runtime 上書き可能
//! (= ADEV / KDGR / KMOB 同方式、 AtomicI32、 default 4)。

use crate::bitboard::{
    Bitboard, bishop_effect, gold_effect, king_5x5_mask, king_effect, knight_effect, lance_effect,
    pawn_effect, rook_effect, silver_effect,
};
use crate::position::Position;
use crate::types::{Color, PieceType};
use std::sync::atomic::{AtomicI32, Ordering};

pub const DROP_PRESSURE_VERSION: u32 = 1;
pub const DROP_PRESSURE_NUM_CLASS: usize = 8;
pub const DROP_PRESSURE_NUM_BUCKET: usize = 8;
pub const DROP_PRESSURE_NUM_PERSP: usize = 2;
pub const DROP_PRESSURE_WEIGHT_COUNT: usize =
    DROP_PRESSURE_NUM_CLASS * DROP_PRESSURE_NUM_BUCKET * DROP_PRESSURE_NUM_PERSP;

/// 重駒 (= 飛 / 角 / 金 / 銀) の投下 weight。
pub const DROP_PRESSURE_HEAVY_WEIGHT: usize = 2;
/// 軽駒 (= 歩 / 香 / 桂) の投下 weight。
pub const DROP_PRESSURE_LIGHT_WEIGHT: usize = 1;

/// `OUTPUT_SCALE` 初期値 (= 4、 ADEV / KDGR / KMOB と揃える)。
pub const DROP_PRESSURE_OUTPUT_SCALE_DEFAULT: i32 = 4;

/// USI option `DROP_PRESSURE_OUTPUT_SCALE` で runtime 上書き可能な eval scale。
pub static DROP_PRESSURE_OUTPUT_SCALE: AtomicI32 =
    AtomicI32::new(DROP_PRESSURE_OUTPUT_SCALE_DEFAULT);

/// USI option 経由で eval scale を更新する。
pub fn set_output_scale(scale: i32) {
    DROP_PRESSURE_OUTPUT_SCALE.store(scale, Ordering::Relaxed);
}

/// section magic `P2DROP\0\0` (= 8 byte)
pub const DROP_PRESSURE_MAGIC: &[u8; 8] = b"P2DROP\0\0";

/// disk layout: `[bucket][persp][class]`
pub type DropPressureWeights =
    [[[i16; DROP_PRESSURE_NUM_CLASS]; DROP_PRESSURE_NUM_PERSP]; DROP_PRESSURE_NUM_BUCKET];

/// runtime layout (= bucket-inner): `[persp][class][bucket]`
pub type DropPressureRuntimeWeights =
    [[[i32; DROP_PRESSURE_NUM_BUCKET]; DROP_PRESSURE_NUM_CLASS]; DROP_PRESSURE_NUM_PERSP];

/// 投下脅威の対象となる持駒種 (= 歩 / 香 / 桂 / 銀 / 金 / 角 / 飛) と weight の table。
///
/// 玉は持駒になり得ず、 と金等の成駒も持駒に存在しない (= 取れば素の駒種に戻る)。
const DROP_HAND_PIECES: [(PieceType, usize); 7] = [
    (PieceType::Rook, DROP_PRESSURE_HEAVY_WEIGHT),
    (PieceType::Bishop, DROP_PRESSURE_HEAVY_WEIGHT),
    (PieceType::Gold, DROP_PRESSURE_HEAVY_WEIGHT),
    (PieceType::Silver, DROP_PRESSURE_HEAVY_WEIGHT),
    (PieceType::Lance, DROP_PRESSURE_LIGHT_WEIGHT),
    (PieceType::Knight, DROP_PRESSURE_LIGHT_WEIGHT),
    (PieceType::Pawn, DROP_PRESSURE_LIGHT_WEIGHT),
];

/// 投下した駒種 `pt` が升 `sq` から利かせる升集合 (= drop 利き)。
///
/// `dropper` = 投下する側 (= 相手 = enemy) の色。 歩 / 香 / 桂は色依存の前方利き、
/// 角 / 飛 / 香は `occupied` を考慮した遠隔利きを返す。
///
/// # 不変条件
/// 玉以外の持駒種 (= 歩 / 香 / 桂 / 銀 / 金 / 角 / 飛) のみを想定する。
#[inline]
fn drop_attack(
    pt: PieceType,
    dropper: Color,
    sq: crate::types::Square,
    occupied: Bitboard,
) -> Bitboard {
    match pt {
        PieceType::Pawn => pawn_effect(dropper, sq),
        PieceType::Lance => lance_effect(dropper, sq, occupied),
        PieceType::Knight => knight_effect(dropper, sq),
        PieceType::Silver => silver_effect(dropper, sq),
        PieceType::Gold => gold_effect(dropper, sq),
        PieceType::Bishop => bishop_effect(sq, occupied),
        PieceType::Rook => rook_effect(sq, occupied),
        // それ以外 (= 玉 / 成駒) は持駒に存在しないため照準なしとして扱う。
        _ => Bitboard::EMPTY,
    }
}

/// perspective 側の自玉 `K` に対する相手持駒の **投下脅威重み付き合計**
/// (= class clamp 前) を返す。
///
/// # 仕様
/// - `zone = king_5x5_mask(K)`、 `drop = zone & ~occupied` (= 投下可能空升)
/// - `target = king_effect(K) | K` (= 玉 + 玉近傍)
/// - 相手の各持駒種 (count>0) について、
///   `eff = { sq in drop | drop_attack(piece, enemy, sq) & target != ∅ }` を数え、
///   `sum += weight(piece) × min(eff, hand_count(piece))`
/// - 玉は持駒に存在しないため除外
#[inline]
pub fn drop_pressure_weighted_sum(pos: &Position, persp: Color) -> i32 {
    let king_sq = pos.king_square(persp);
    let enemy = persp.opponent();
    let hand = pos.hand(enemy);

    // 持駒が空なら投下脅威ゼロ (= 早期 return で大半の局面を高速化)。
    if hand.is_empty() {
        return 0;
    }

    let occupied = pos.occupied();
    let zone = king_5x5_mask(king_sq);
    // (a) 投下可能な空升 (= zone のうち盤上駒が居ないマス)。
    let drop = zone & !occupied;
    if drop.is_empty() {
        return 0;
    }
    // 攻撃対象集合 = 玉 + 玉近傍 8 升。
    let target = king_effect(king_sq) | Bitboard::from_square(king_sq);

    let mut sum: usize = 0;
    for &(pt, weight) in DROP_HAND_PIECES.iter() {
        let hand_count = hand.count(pt) as usize;
        if hand_count == 0 {
            continue;
        }
        // この駒種を投下して target に利く空升を数える。
        let mut eff: usize = 0;
        for sq in drop.iter() {
            if !(drop_attack(pt, enemy, sq, occupied) & target).is_empty() {
                eff += 1;
            }
        }
        if eff > 0 {
            // 投下できる弾数 (= 持駒数) を上限に評価する。
            sum += weight * eff.min(hand_count);
        }
    }
    sum as i32
}

/// perspective 側の投下脅威を `0..=7` の class で返す。
///
/// # Arguments
/// * `pos`   - 局面
/// * `persp` - 視点 (= 自玉の色)
///
/// # 仕様
/// [`drop_pressure_weighted_sum`] の重み付き合計を `min(7)` で 8 class に clamp。
pub fn compute_drop_pressure_class(pos: &Position, persp: Color) -> usize {
    let sum = drop_pressure_weighted_sum(pos, persp);
    (sum as usize).min(DROP_PRESSURE_NUM_CLASS - 1)
}

/// flatten index (= disk layout 用): `[bucket][persp][class]` → flat
#[inline]
pub const fn flatten_disk_idx(bucket: usize, persp: usize, class: usize) -> usize {
    bucket * (DROP_PRESSURE_NUM_PERSP * DROP_PRESSURE_NUM_CLASS)
        + persp * DROP_PRESSURE_NUM_CLASS
        + class
}

/// runtime weight build (= bucket-inner 転置)
pub fn build_runtime_weights(disk: &DropPressureWeights) -> DropPressureRuntimeWeights {
    let mut rt: DropPressureRuntimeWeights =
        [[[0i32; DROP_PRESSURE_NUM_BUCKET]; DROP_PRESSURE_NUM_CLASS]; DROP_PRESSURE_NUM_PERSP];
    for (bucket, disk_bucket) in disk.iter().enumerate() {
        for (persp, disk_persp) in disk_bucket.iter().enumerate() {
            for (class, &w) in disk_persp.iter().enumerate() {
                rt[persp][class][bucket] = w as i32;
            }
        }
    }
    rt
}

/// evaluate DropPressure scalar head。
///
/// stm 視点で `(us - them) × OUTPUT_SCALE` を返す。
///
/// # Arguments
/// * `weights` - runtime layout の DropPressure weight (= bucket-inner i32)
/// * `pos`     - 局面
/// * `bucket`  - 現在の bucket index (= 0..=7)
/// * `stm`     - 手番 (= side to move)
pub fn evaluate_drop_pressure(
    weights: &DropPressureRuntimeWeights,
    pos: &Position,
    bucket: usize,
    stm: Color,
) -> i32 {
    // OUTPUT_SCALE == 0 のとき最終結果は必ず `× 0 == 0` になるため、
    // bitboard scan (compute_drop_pressure_class) を行わず即 0 を返す。
    // `× 0` と早期 return 0 は代数的に同値なので eval は全局面で bit-exact に不変であり、
    // head を無効化 (scale=0) した本番 config で per-node の利き計算を丸ごと省ける純粋な NPS 改善となる。
    let scale = DROP_PRESSURE_OUTPUT_SCALE.load(Ordering::Relaxed);
    if scale == 0 {
        return 0;
    }
    let nstm = stm.opponent();
    let stm_class = compute_drop_pressure_class(pos, stm);
    let nstm_class = compute_drop_pressure_class(pos, nstm);
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
        assert_eq!(DROP_PRESSURE_MAGIC, b"P2DROP\0\0");
        assert_eq!(DROP_PRESSURE_VERSION, 1);
        assert_eq!(DROP_PRESSURE_WEIGHT_COUNT, 128);
        assert_eq!(DROP_PRESSURE_NUM_CLASS, 8);
    }

    #[test]
    fn test_startpos_class_zero() {
        // 初期局面: 両者持駒ゼロ → 投下脅威ゼロ → class 0。
        let pos = make_pos();
        assert_eq!(compute_drop_pressure_class(&pos, Color::Black), 0);
        assert_eq!(compute_drop_pressure_class(&pos, Color::White), 0);
    }

    #[test]
    fn test_no_hand_is_zero() {
        // 持駒なしの裸玉局面 (= 相手も持駒ゼロ) は投下脅威ゼロ。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/4K4/9/9/9/9 b - 1").unwrap();
        assert_eq!(drop_pressure_weighted_sum(&pos, Color::Black), 0);
        assert_eq!(drop_pressure_weighted_sum(&pos, Color::White), 0);
    }

    #[test]
    fn test_hand_rook_drop_threatens_king() {
        // 黒玉 5九 (= 自陣)、 白 (= 相手) が飛 1 枚持駒。
        // 黒玉近傍の空升へ飛を投下すると玉 or 玉近傍に利く → 投下脅威あり。
        // black 視点 (= 自玉 = 黒玉) で投下脅威 weighted_sum > 0 を確認。
        let mut pos = Position::new();
        // 白玉 5一、 黒玉 5九、 周囲は空。 白 (= gote) 持駒に飛 1 枚 (= 小文字 "r")。
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b r 1").unwrap();
        let raw = drop_pressure_weighted_sum(&pos, Color::Black);
        assert!(raw > 0, "黒玉近傍に白飛を投下できる脅威があるはず (got {raw})");
        // 黒 (= 自分) は持駒ゼロなので白玉への投下脅威はゼロ。
        assert_eq!(drop_pressure_weighted_sum(&pos, Color::White), 0);
    }

    #[test]
    fn test_more_hand_and_exposed_king_increases_class() {
        // 自玉露出 (= 周囲空升多) + 相手持駒多 で class が増えることを確認。
        // 黒玉 5九、 白 (= gote) 持駒 = 飛角金 (= 重 3 枚、 小文字 "rbg")。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b rbg 1").unwrap();
        let cls_many = compute_drop_pressure_class(&pos, Color::Black);
        // 飛 1 枚のみの局面と比較して class が大きいはず。
        let mut pos_few = Position::new();
        pos_few.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b r 1").unwrap();
        let cls_few = compute_drop_pressure_class(&pos_few, Color::Black);
        assert!(
            cls_many >= cls_few,
            "持駒が多いほど投下脅威 class は増える (many {cls_many} vs few {cls_few})"
        );
        assert!(cls_many > 0, "持駒多 + 玉露出で class > 0 のはず (got {cls_many})");
    }

    #[test]
    fn test_remote_bishop_drop_aim() {
        // 角の遠隔 drop 利き確認。 黒玉 5九、 白持駒に角 1 枚。
        // 玉近傍 5x5 内の空升へ角を投下すると、 斜め遠隔で玉 or 玉近傍に利く。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b b 1").unwrap();
        let raw = drop_pressure_weighted_sum(&pos, Color::Black);
        assert!(raw > 0, "角の遠隔投下利きで脅威ありのはず (got {raw})");
        // 角は重駒 weight 2。 少なくとも 2 以上。
        assert!(raw >= DROP_PRESSURE_HEAVY_WEIGHT as i32, "角 weight は 2 以上");
    }

    #[test]
    fn test_class_saturates_at_7() {
        // 相手持駒を多数 (= 飛角金銀香桂歩) 持たせ、 玉露出 5x5 が空 →
        // weighted sum が 7 を超え class 7 に clamp。
        let mut pos = Position::new();
        // 白 (= gote) 持駒 = 飛角金銀香桂歩 各複数 (= 小文字)。 黒玉 5九、 玉周辺空。
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2r2b2g2s2l2n2p 1").unwrap();
        let cls = compute_drop_pressure_class(&pos, Color::Black);
        assert_eq!(cls, 7, "多数の持駒で class 7 に clamp されるはず (got {cls})");
    }

    #[test]
    fn test_flatten_disk_idx() {
        assert_eq!(flatten_disk_idx(0, 0, 0), 0);
        assert_eq!(flatten_disk_idx(7, 1, 7), 127);
        assert_eq!(flatten_disk_idx(7, 1, 7) + 1, DROP_PRESSURE_WEIGHT_COUNT);
    }

    #[test]
    fn test_build_runtime_weights_layout() {
        let mut disk: DropPressureWeights = [[[0; 8]; 2]; 8];
        disk[3][0][5] = -123;
        let rt = build_runtime_weights(&disk);
        assert_eq!(rt[0][5][3], -123);
        assert_eq!(rt[0][0][0], 0);
        assert_eq!(rt[1][7][7], 0);
    }

    #[test]
    fn test_evaluate_zero_weights() {
        // 全 weight 0 で evaluate = 0 (= zero-load 時無害の証明)。
        let disk: DropPressureWeights = [[[0; 8]; 2]; 8];
        let rt = build_runtime_weights(&disk);
        let pos = make_pos();
        for bucket in 0..DROP_PRESSURE_NUM_BUCKET {
            let v = evaluate_drop_pressure(&rt, &pos, bucket, Color::Black);
            assert_eq!(v, 0, "zero weight => zero contribution (bucket {bucket})");
        }
    }

    #[test]
    fn test_evaluate_output_scale_linear() {
        // OUTPUT_SCALE を変えると eval が線形変化する (= head 発火の証明)。
        // 非対称局面: 黒玉に白飛投下脅威 (= black class > 0)、 白玉は脅威なし
        // (= 黒は持駒ゼロ ⇒ white class 0)。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b r 1").unwrap();
        let bc = compute_drop_pressure_class(&pos, Color::Black);
        let wc = compute_drop_pressure_class(&pos, Color::White);
        assert!(bc > 0, "黒玉への投下脅威 class > 0 (got {bc})");
        assert_eq!(wc, 0, "白玉への投下脅威なし class 0 (got {wc})");

        // black class bc の weight を -10、 white class 0 は 0 のまま。
        let mut disk: DropPressureWeights = [[[0; 8]; 2]; 8];
        for bucket_row in disk.iter_mut() {
            bucket_row[Color::Black as usize][bc] = -10;
        }
        let rt = build_runtime_weights(&disk);

        set_output_scale(1);
        let v1 = evaluate_drop_pressure(&rt, &pos, 0, Color::Black);
        set_output_scale(4);
        let v4 = evaluate_drop_pressure(&rt, &pos, 0, Color::Black);
        set_output_scale(64);
        let v64 = evaluate_drop_pressure(&rt, &pos, 0, Color::Black);
        set_output_scale(DROP_PRESSURE_OUTPUT_SCALE_DEFAULT);

        // us(black class bc)=-10, them(white class0)=0 → (us-them) = -10。
        assert_eq!(v1, -10);
        assert_eq!(v4, -40);
        assert_eq!(v64, -640);

        // 対称局面 (= startpos、 両者持駒ゼロ class 0) は 0。
        let pos_sym = make_pos();
        let mut disk_sym: DropPressureWeights = [[[0; 8]; 2]; 8];
        for bucket_row in disk_sym.iter_mut() {
            for persp_row in bucket_row.iter_mut() {
                persp_row[0] = -10;
            }
        }
        let rt_sym = build_runtime_weights(&disk_sym);
        let v_sym = evaluate_drop_pressure(&rt_sym, &pos_sym, 0, Color::Black);
        assert_eq!(v_sym, 0);
    }
}
