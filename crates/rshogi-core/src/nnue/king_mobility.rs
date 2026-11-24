//! King Mobility (KMOB) PSQT scalar head (2026-06-16)。
//!
//! 自玉 (= 升 K) の **「安全な逃げ場マス数」** = K の 8 近傍
//! (= `king_effect(K)`) のうち、
//!
//!   (a) 自駒が居ない (= 空 or 敵駒。 敵駒升へは取って移動可)、 かつ
//!   (b) 敵に利いていない (= 敵の攻撃が無い)
//!
//! を満たすマスの数を数え、 `0..=8` の **9 class** に離散化する
//! (= 8 近傍なので最大 8、 clamp 不要だが安全のため `min(8)`)。
//! 9 class × 8 bucket × 2 persp = **144 i16 weight** の軽量 scalar table で
//! eval 加算する。
//!
//! 守備系特徴であり、 攻め系 head (= ARS1 / attack_dev) と **直交**する設計
//! (= attack_dev は ARS1 と冗長で失敗、 今回は「自玉の広さ (逃げ場)」 を測る守備軸)。
//!
//! 狙い = 「自玉の逃げ場が少ない (= 危険) 局面を過小評価」 を補正する。
//! 逃げ場が少ない class ほど stm の eval を **下げる** 悲観方向の補正を
//! 学習で獲得させる (= memory 法則「守備的盲点の補正は勝つ」 に適合)。
//! **符号は訓練が学習**する (= ここでは固定しない)。
//!
//! # 設計
//!
//! ## feature 集合 (= class 計算、 per perspective)
//!
//! perspective `persp` の自玉 `K` に対し:
//!
//! ```text
//! escape = king_effect(K) & ~own_occ        // (a) 自駒の居ないマス
//! safe   = { sq in escape | 敵に利いていない }  // (b) 敵攻撃の無いマス
//! class  = min(safe.count(), 8)             // 0..=8 の 9 class
//! ```
//!
//! (b) の「敵に利いていない」 判定は升ごとに `attackers_to_c(sq, enemy)` で
//! 敵攻撃の有無を確認する (= 候補は最大 8 升なので安価)。 pin 等の厳密合法性は
//! 考慮しない近似 (= head が重みを学習)。 NPS は多少落ちて良い (= ユーザ方針)。
//!
//! ## weight layout (= KDGR / ADEV と同形、 ただし class が 9)
//!
//! - disk: `[bucket 8][persp 2][class 9] -> i16` = 144 i16 = 288 byte
//! - runtime: `[persp 2][class 9][bucket 8] -> i32` (= bucket-inner、 hot path で
//!   `row[bucket]` の 8 連続 read 可能)
//!
//! ## ckpt format
//!
//! ```text
//! sub_magic = "P2KMOB\0\0" (8 byte)
//! version   = u32 LE       (= 1)
//! count     = u32 LE       (= 144)
//! weights   = i16 × 144 LE (= 288 byte)
//! ```
//!
//! 合計 304 byte。 既存 section (= ... / P2KDGR / P2ADEV) の **後ろ**に append。
//!
//! ## evaluate
//!
//! `(us_acc[bucket] - them_acc[bucket]) × OUTPUT_SCALE`、 stm 視点で反対称合成。
//! KingDanger / AttackDev head と同 convention。
//!
//! `OUTPUT_SCALE` は USI option `KING_MOBILITY_OUTPUT_SCALE` で runtime 上書き可能
//! (= ADEV / KDGR 同方式、 AtomicI32、 default 4)。

use crate::bitboard::king_effect;
use crate::position::Position;
use crate::types::Color;
use std::sync::atomic::{AtomicI32, Ordering};

pub const KING_MOBILITY_VERSION: u32 = 1;
/// class 数 (= 0..=8 の 9 段階。 attack_dev / king_danger の 8 と異なる点に注意)。
pub const KING_MOBILITY_NUM_CLASS: usize = 9;
pub const KING_MOBILITY_NUM_BUCKET: usize = 8;
pub const KING_MOBILITY_NUM_PERSP: usize = 2;
pub const KING_MOBILITY_WEIGHT_COUNT: usize =
    KING_MOBILITY_NUM_CLASS * KING_MOBILITY_NUM_BUCKET * KING_MOBILITY_NUM_PERSP;

/// `OUTPUT_SCALE` 初期値 (= 4、 ADEV / KDGR と揃える)。
pub const KING_MOBILITY_OUTPUT_SCALE_DEFAULT: i32 = 4;

/// USI option `KING_MOBILITY_OUTPUT_SCALE` で runtime 上書き可能な eval scale。
pub static KING_MOBILITY_OUTPUT_SCALE: AtomicI32 =
    AtomicI32::new(KING_MOBILITY_OUTPUT_SCALE_DEFAULT);

/// USI option 経由で eval scale を更新する。
pub fn set_output_scale(scale: i32) {
    KING_MOBILITY_OUTPUT_SCALE.store(scale, Ordering::Relaxed);
}

/// section magic `P2KMOB\0\0` (= 8 byte)
pub const KING_MOBILITY_MAGIC: &[u8; 8] = b"P2KMOB\0\0";

/// disk layout: `[bucket][persp][class]`
pub type KingMobilityWeights =
    [[[i16; KING_MOBILITY_NUM_CLASS]; KING_MOBILITY_NUM_PERSP]; KING_MOBILITY_NUM_BUCKET];

/// runtime layout (= bucket-inner): `[persp][class][bucket]`
pub type KingMobilityRuntimeWeights =
    [[[i32; KING_MOBILITY_NUM_BUCKET]; KING_MOBILITY_NUM_CLASS]; KING_MOBILITY_NUM_PERSP];

/// perspective 側の自玉 `K` の **「安全な逃げ場マス数」** (= class clamp 前) を返す。
///
/// `escape = king_effect(K) & ~own_occ`、 その各升のうち
/// **敵に利いていない**升を数える (= 最大 8)。
///
/// # 仕様
/// - (a) 自駒の居ない近傍升のみ候補 (= 空 or 敵駒 = 取って移動可)
/// - (b) `attackers_to_c(sq, enemy)` が空 (= 敵攻撃無し) の升のみ計上
/// - pin 等の厳密合法性は考慮しない近似
#[inline]
pub fn king_mobility_safe_count(pos: &Position, persp: Color) -> u32 {
    let king_sq = pos.king_square(persp);
    let enemy = persp.opponent();
    let own_occ = pos.pieces_c(persp);

    // (a) 自駒の居ない近傍升 (= 空 or 敵駒)。
    let escape = king_effect(king_sq) & !own_occ;

    // (b) 敵に利いていない升のみ計上。
    let mut safe: u32 = 0;
    for sq in escape.iter() {
        if pos.attackers_to_c(sq, enemy).is_empty() {
            safe += 1;
        }
    }
    safe
}

/// perspective 側の自玉の安全な逃げ場数を `0..=8` の class で返す。
///
/// # Arguments
/// * `pos`   - 局面
/// * `persp` - 視点 (= 自玉の色)
///
/// # 仕様
/// [`king_mobility_safe_count`] を `min(8)` で 9 class に clamp
/// (= 8 近傍なので実質 clamp は発火しないが安全のため)。
pub fn compute_king_mobility_class(pos: &Position, persp: Color) -> usize {
    let count = king_mobility_safe_count(pos, persp) as usize;
    count.min(KING_MOBILITY_NUM_CLASS - 1)
}

/// flatten index (= disk layout 用): `[bucket][persp][class]` → flat
#[inline]
pub const fn flatten_disk_idx(bucket: usize, persp: usize, class: usize) -> usize {
    bucket * (KING_MOBILITY_NUM_PERSP * KING_MOBILITY_NUM_CLASS)
        + persp * KING_MOBILITY_NUM_CLASS
        + class
}

/// runtime weight build (= bucket-inner 転置)
pub fn build_runtime_weights(disk: &KingMobilityWeights) -> KingMobilityRuntimeWeights {
    let mut rt: KingMobilityRuntimeWeights =
        [[[0i32; KING_MOBILITY_NUM_BUCKET]; KING_MOBILITY_NUM_CLASS]; KING_MOBILITY_NUM_PERSP];
    for (bucket, disk_bucket) in disk.iter().enumerate() {
        for (persp, disk_persp) in disk_bucket.iter().enumerate() {
            for (class, &w) in disk_persp.iter().enumerate() {
                rt[persp][class][bucket] = w as i32;
            }
        }
    }
    rt
}

/// evaluate KingMobility scalar head。
///
/// stm 視点で `(us - them) × OUTPUT_SCALE` を返す。
///
/// # Arguments
/// * `weights` - runtime layout の KingMobility weight (= bucket-inner i32)
/// * `pos`     - 局面
/// * `bucket`  - 現在の bucket index (= 0..=7)
/// * `stm`     - 手番 (= side to move)
pub fn evaluate_king_mobility(
    weights: &KingMobilityRuntimeWeights,
    pos: &Position,
    bucket: usize,
    stm: Color,
) -> i32 {
    // OUTPUT_SCALE == 0 のとき最終結果は必ず `× 0 == 0` になるため、
    // bitboard scan (compute_king_mobility_class) を行わず即 0 を返す。
    // `× 0` と早期 return 0 は代数的に同値なので eval は全局面で bit-exact に不変であり、
    // head を無効化 (scale=0) した本番 config で per-node の利き計算 (最重 = attackers_to_c × 16/node)
    // を丸ごと省ける純粋な NPS 改善となる。
    let scale = KING_MOBILITY_OUTPUT_SCALE.load(Ordering::Relaxed);
    if scale == 0 {
        return 0;
    }
    let nstm = stm.opponent();
    let stm_class = compute_king_mobility_class(pos, stm);
    let nstm_class = compute_king_mobility_class(pos, nstm);
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
        assert_eq!(KING_MOBILITY_MAGIC, b"P2KMOB\0\0");
        assert_eq!(KING_MOBILITY_VERSION, 1);
        assert_eq!(KING_MOBILITY_WEIGHT_COUNT, 144);
        assert_eq!(KING_MOBILITY_NUM_CLASS, 9);
    }

    #[test]
    fn test_startpos_mobility() {
        // 初期局面: 黒玉 5九。 8 近傍 = 4八/5八/6八 (= 段 8) と 4九/6九 (= 段 9)。
        // 4八/5八/6八 = 空、 4九/6九 = 自軍金 (= 占有) → escape は 3 升 (= 段 8)。
        // ただし 5八 等は自軍駒に守られ敵利き無し。 初期局面で敵利きは自陣に届かない。
        // → 安全な逃げ場 = 段 8 の 3 升 = class 3 (= 左右対称)。
        let pos = make_pos();
        let b = compute_king_mobility_class(&pos, Color::Black);
        let w = compute_king_mobility_class(&pos, Color::White);
        assert_eq!(b, w, "startpos は黒白対称のはず");
        assert_eq!(b, 3, "初期局面の玉の安全逃げ場は段 8 の 3 升 = class 3");
    }

    #[test]
    fn test_center_king_open_board_max() {
        // 盤中央の黒玉 5五 を裸玉で置く (= 自駒なし、 敵駒なし) →
        // 8 近傍全てが空 + 敵利き無し → 安全逃げ場 8 → class 8 (= 最大)。
        let mut pos = Position::new();
        // 白玉は隅 (= 9一 付近) に置いて敵利きが黒玉近傍に届かないようにする。
        let sfen = "k8/9/9/9/4K4/9/9/9/9 b - 1";
        pos.set_sfen(sfen).unwrap();
        assert_eq!(
            king_mobility_safe_count(&pos, Color::Black),
            8,
            "中央の裸玉は 8 近傍全てが安全逃げ場"
        );
        assert_eq!(compute_king_mobility_class(&pos, Color::Black), 8);
    }

    #[test]
    fn test_corner_king_fewer_escapes() {
        // 隅の黒玉 9九 (= file 8, rank 8) は盤外で近傍が削れる →
        // 有効近傍 = 9八/8八/8九 の 3 升のみ (= 盤内)。 敵利き無しなら 3。
        let mut pos = Position::new();
        let sfen = "k8/9/9/9/9/9/9/9/8K b - 1";
        pos.set_sfen(sfen).unwrap();
        assert_eq!(king_mobility_safe_count(&pos, Color::Black), 3, "隅の玉は盤内近傍 3 升のみ");
        assert_eq!(compute_king_mobility_class(&pos, Color::Black), 3);
    }

    #[test]
    fn test_enemy_attack_reduces_escapes() {
        // 中央の黒玉 5五 (= 裸玉で 8 逃げ場)。
        // 白飛を 5一 (= 同 5 筋) に置くと 5四/5六 (= 玉の上下同筋) が敵利き →
        // ただし 5五 自身が間にあるため飛は片側のみ。 5四 (玉の上、 5一 から見て手前)
        // が敵利き。 5六 (玉の下) は玉が遮るので利かない。
        // → 安全逃げ場 = 8 - 1 = 7。
        let mut pos = Position::new();
        let sfen = "k3r4/9/9/9/4K4/9/9/9/9 b - 1";
        pos.set_sfen(sfen).unwrap();
        let count = king_mobility_safe_count(&pos, Color::Black);
        assert!(count < 8, "敵飛の利きで逃げ場が減るはず (got {count})");
        assert_eq!(count, 7, "5筋の敵飛で 5四 が攻撃され逃げ場 7");
    }

    #[test]
    fn test_own_piece_blocks_escape() {
        // 中央の黒玉 5五 の真上 5四 に自軍歩を置くと、 そのマスは自駒占有で
        // 逃げ場から除外される → 8 - 1 = 7。 敵利きは無し。
        let mut pos = Position::new();
        let sfen = "k8/9/9/4P4/4K4/9/9/9/9 b - 1";
        pos.set_sfen(sfen).unwrap();
        assert_eq!(
            king_mobility_safe_count(&pos, Color::Black),
            7,
            "自駒占有のマスは逃げ場にならない"
        );
    }

    #[test]
    fn test_flatten_disk_idx() {
        assert_eq!(flatten_disk_idx(0, 0, 0), 0);
        // last slot (= bucket=7, persp=1, class=8) = 143
        assert_eq!(flatten_disk_idx(7, 1, 8), 143);
        assert_eq!(flatten_disk_idx(7, 1, 8) + 1, KING_MOBILITY_WEIGHT_COUNT);
    }

    #[test]
    fn test_build_runtime_weights_layout() {
        let mut disk: KingMobilityWeights = [[[0; 9]; 2]; 8];
        disk[3][0][5] = -123;
        let rt = build_runtime_weights(&disk);
        assert_eq!(rt[0][5][3], -123);
        assert_eq!(rt[0][0][0], 0);
        assert_eq!(rt[1][8][7], 0);
    }

    #[test]
    fn test_evaluate_zero_weights() {
        // 全 weight 0 で evaluate = 0 (= zero-load 時無害の証明)。
        let disk: KingMobilityWeights = [[[0; 9]; 2]; 8];
        let rt = build_runtime_weights(&disk);
        let pos = make_pos();
        for bucket in 0..KING_MOBILITY_NUM_BUCKET {
            let v = evaluate_king_mobility(&rt, &pos, bucket, Color::Black);
            assert_eq!(v, 0, "zero weight => zero contribution (bucket {bucket})");
        }
    }

    #[test]
    fn test_evaluate_output_scale_linear() {
        // OUTPUT_SCALE を変えると eval が線形変化する (= head 発火の証明)。
        // 非対称局面: 黒玉中央 (= class 8 = 広い)、 白玉隅 (= class 3 = 狭い)。
        let mut pos = Position::new();
        pos.set_sfen("8k/9/9/9/4K4/9/9/9/9 b - 1").unwrap();
        let bc = compute_king_mobility_class(&pos, Color::Black);
        let wc = compute_king_mobility_class(&pos, Color::White);
        assert_eq!(bc, 8, "中央黒玉 class 8");
        assert_eq!(wc, 3, "隅白玉 class 3");

        // black class 8 の weight を -10、 white class 3 の weight を 0 に設定。
        let mut disk: KingMobilityWeights = [[[0; 9]; 2]; 8];
        for bucket_row in disk.iter_mut() {
            bucket_row[Color::Black as usize][8] = -10;
        }
        let rt = build_runtime_weights(&disk);

        set_output_scale(1);
        let v1 = evaluate_king_mobility(&rt, &pos, 0, Color::Black);
        set_output_scale(4);
        let v4 = evaluate_king_mobility(&rt, &pos, 0, Color::Black);
        set_output_scale(64);
        let v64 = evaluate_king_mobility(&rt, &pos, 0, Color::Black);
        set_output_scale(KING_MOBILITY_OUTPUT_SCALE_DEFAULT);

        // us(black class8)=-10, them(white class3)=0 → (us-them) = -10。
        assert_eq!(v1, -10);
        assert_eq!(v4, -40);
        assert_eq!(v64, -640);

        // 対称局面 (= startpos) は 0。
        let pos_sym = make_pos();
        let mut disk_sym: KingMobilityWeights = [[[0; 9]; 2]; 8];
        for bucket_row in disk_sym.iter_mut() {
            for persp_row in bucket_row.iter_mut() {
                // 両 persp の class 3 (= startpos の class) に同値を入れる。
                persp_row[3] = -10;
            }
        }
        let rt_sym = build_runtime_weights(&disk_sym);
        let v_sym = evaluate_king_mobility(&rt_sym, &pos_sym, 0, Color::Black);
        assert_eq!(v_sym, 0);
    }
}
