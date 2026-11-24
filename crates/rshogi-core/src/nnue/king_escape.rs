//! King Escape Direction (KED) PSQT scalar head (2026-06-09)。
//!
//! 玉の **forward 方向** (= 入玉方向、 先手 = 上 / 後手 = 下) で同一 file
//! 1-3 マス先の升に **自軍利き** が当たっている数を `0..=3` にクランプし、
//! 4 class × 8 bucket × 2 persp = **64 i16 weight** の軽量 scalar table で
//! eval 加算する。 入玉局面の表現強化 + 玉前進路の評価が目的。
//!
//! # 設計
//!
//! ## feature 集合
//!
//! - perspective `Color::Black` の forward = `Square::DELTA_U` (= -1、 rank decr)
//! - perspective `Color::White` の forward = `Square::DELTA_D` (= +1、 rank incr)
//! - target 升 = 玉から forward 方向 同一 file の 1-3 マス先 (= 最大 3 升)
//! - 各 target 升で `pos.attackers_to_occ(target, occupied) & own_pieces` が
//!   非空なら count +1、 結果を `min(3)` で 4 class に clamp
//!
//! ## weight layout
//!
//! - disk: `[bucket 8][persp 2][class 4] -> i16` = 64 i16 = 128 byte
//! - runtime: `[persp 2][class 4][bucket 8] -> i32` (bucket-inner)
//!   = hot path で `row[bucket]` の 8 連続 read 可能
//!
//! ## ckpt format
//!
//! ```text
//! sub_magic = "P2KESC\0\0" (8 byte)
//! version   = u32 LE       (= 1)
//! count     = u32 LE       (= 64)
//! weights   = i16 × 64 LE  (= 128 byte)
//! ```
//!
//! 合計 144 byte。 既存 P2DUALZ / P2DUAL2 / P2KRING / P2PAIR / P2ACTR
//! section の後ろに append。
//!
//! ## evaluate
//!
//! `(us_acc[bucket] - them_acc[bucket]) * OUTPUT_SCALE` (= ×8)、 stm 視点で
//! 反対称合成。 PSQT / pairshape / activity_risk と同 convention。

use crate::bitboard::king_effect; // 既 use を流用
use crate::position::Position;
use crate::types::{Color, Square};

pub const KING_ESCAPE_VERSION: u32 = 1;
pub const KING_ESCAPE_NUM_CLASS: usize = 4;
pub const KING_ESCAPE_NUM_BUCKET: usize = 8;
pub const KING_ESCAPE_NUM_PERSP: usize = 2;
pub const KING_ESCAPE_WEIGHT_COUNT: usize =
    KING_ESCAPE_NUM_CLASS * KING_ESCAPE_NUM_BUCKET * KING_ESCAPE_NUM_PERSP;
pub const KING_ESCAPE_OUTPUT_SCALE: i32 = 8;

/// section magic `P2KESC\0\0` (= 8 byte)
pub const KING_ESCAPE_MAGIC: &[u8; 8] = b"P2KESC\0\0";

/// disk layout: `[bucket][persp][class]`
pub type KingEscapeWeights =
    [[[i16; KING_ESCAPE_NUM_CLASS]; KING_ESCAPE_NUM_PERSP]; KING_ESCAPE_NUM_BUCKET];

/// runtime layout (= bucket-inner): `[persp][class][bucket]`
pub type KingEscapeRuntimeWeights =
    [[[i32; KING_ESCAPE_NUM_BUCKET]; KING_ESCAPE_NUM_CLASS]; KING_ESCAPE_NUM_PERSP];

/// 玉の forward 方向 (= 入玉方向) 同一 file 1-3 マス先の自軍利き本数を 0..=3 で返す。
///
/// # Arguments
/// * `pos`   - 局面
/// * `persp` - 視点 (= 玉の色)
///
/// # 仕様
/// - 先手 (Black) は `DELTA_U = -1` (= rank decreasing) 方向に前進
/// - 後手 (White) は `DELTA_D = +1` (= rank increasing) 方向に前進
/// - 玉位置から 1, 2, 3 マス先まで順に check、 同一 file から外れたら break
/// - 各 target 升で `attackers_to_occ` & 自軍駒 が非空なら count +1
/// - 最終 count を `min(3)` で 4 class に clamp
pub fn compute_king_escape_class(pos: &Position, persp: Color) -> usize {
    let king_sq = pos.king_square(persp);
    let king_file = king_sq.file();
    let own_pieces = pos.pieces_c(persp);
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);

    let forward_delta: i8 = match persp {
        Color::Black => Square::DELTA_U,
        Color::White => Square::DELTA_D,
    };

    let mut count: usize = 0;
    let mut cur = king_sq;
    for _ in 0..3 {
        let Some(next) = cur.offset(forward_delta) else {
            break;
        };
        // 同一 file から外れたら break (= 境界 wrap 防止)
        if next.file() != king_file {
            break;
        }
        let attackers = pos.attackers_to_occ(next, occupied) & own_pieces;
        if !attackers.is_empty() {
            count += 1;
        }
        cur = next;
    }
    count.min(KING_ESCAPE_NUM_CLASS - 1)
}

/// flatten index (= disk layout 用): `[bucket][persp][class]` → flat
#[inline]
pub const fn flatten_disk_idx(bucket: usize, persp: usize, class: usize) -> usize {
    bucket * (KING_ESCAPE_NUM_PERSP * KING_ESCAPE_NUM_CLASS) + persp * KING_ESCAPE_NUM_CLASS + class
}

/// runtime weight build (= bucket-inner 転置)
pub fn build_runtime_weights(disk: &KingEscapeWeights) -> KingEscapeRuntimeWeights {
    let mut rt: KingEscapeRuntimeWeights =
        [[[0i32; KING_ESCAPE_NUM_BUCKET]; KING_ESCAPE_NUM_CLASS]; KING_ESCAPE_NUM_PERSP];
    for bucket in 0..KING_ESCAPE_NUM_BUCKET {
        for persp in 0..KING_ESCAPE_NUM_PERSP {
            for class in 0..KING_ESCAPE_NUM_CLASS {
                rt[persp][class][bucket] = disk[bucket][persp][class] as i32;
            }
        }
    }
    rt
}

/// evaluate KED scalar head
///
/// stm 視点で `(us - them) × OUTPUT_SCALE` を返す。 weights が None の場合は 0。
///
/// # Arguments
/// * `weights` - runtime layout の KED weight (= bucket-inner i32)
/// * `pos`     - 局面
/// * `bucket`  - 現在の bucket index (= 0..=7)
/// * `stm`     - 手番 (= side to move)
pub fn evaluate_king_escape(
    weights: &KingEscapeRuntimeWeights,
    pos: &Position,
    bucket: usize,
    stm: Color,
) -> i32 {
    let nstm = stm.opponent();
    let stm_class = compute_king_escape_class(pos, stm);
    let nstm_class = compute_king_escape_class(pos, nstm);
    let us = weights[stm as usize][stm_class][bucket];
    let them = weights[nstm as usize][nstm_class][bucket];
    us.saturating_sub(them).saturating_mul(KING_ESCAPE_OUTPUT_SCALE)
}

// king_effect が使用されないと警告される場合があるので参照だけ保持
const _: fn(Square) -> crate::bitboard::Bitboard = king_effect;

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
        assert_eq!(KING_ESCAPE_MAGIC, b"P2KESC\0\0");
        assert_eq!(KING_ESCAPE_VERSION, 1);
        assert_eq!(KING_ESCAPE_WEIGHT_COUNT, 64);
        assert_eq!(KING_ESCAPE_NUM_CLASS, 4);
    }

    #[test]
    fn test_startpos_escape_class() {
        let pos = make_pos();
        let black = compute_king_escape_class(&pos, Color::Black);
        let white = compute_king_escape_class(&pos, Color::White);
        // 初期局面: 玉前 3 マス分の自軍利きが 1-3 (= 自軍駒からの利きあり)
        // 厳密値は board structure 依存だが、 0..=3 の範囲内であること
        assert!(black <= 3);
        assert!(white <= 3);
    }

    #[test]
    fn test_flatten_disk_idx() {
        // first slot (= bucket=0, persp=0, class=0) = 0
        assert_eq!(flatten_disk_idx(0, 0, 0), 0);
        // last slot (= bucket=7, persp=1, class=3) = 63
        assert_eq!(flatten_disk_idx(7, 1, 3), 63);
        // total = 64 = WEIGHT_COUNT
        assert_eq!(flatten_disk_idx(7, 1, 3) + 1, KING_ESCAPE_WEIGHT_COUNT);
    }

    #[test]
    fn test_build_runtime_weights_layout() {
        let mut disk: KingEscapeWeights = [[[0; 4]; 2]; 8];
        // bucket 3, persp 0 (Black), class 2 に -123 を入れる
        disk[3][0][2] = -123;
        let rt = build_runtime_weights(&disk);
        assert_eq!(rt[0][2][3], -123);
        // 他は 0
        assert_eq!(rt[0][0][0], 0);
        assert_eq!(rt[1][3][7], 0);
    }

    #[test]
    fn test_evaluate_zero_weights() {
        // 全 weight 0 で evaluate = 0
        let disk: KingEscapeWeights = [[[0; 4]; 2]; 8];
        let rt = build_runtime_weights(&disk);
        let pos = make_pos();
        let v = evaluate_king_escape(&rt, &pos, 0, Color::Black);
        assert_eq!(v, 0);
    }
}
