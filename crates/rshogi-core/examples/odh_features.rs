//! ODH (Optimism Damp Head) class 軸候補の per-sfen 幾何特徴ダンプ (= 2026-07-02)。
//!
//! stdin から 1 行 1 sfen (= `position sfen ...` / `sfen ...` 接頭辞可) を読み、
//! **先手(B) / 後手(W) 両視点**で ODH class 候補の生の幾何指標を計算して出力する。
//! NNUE net 非依存 (= 局面幾何のみ、 weight 不要)。
//!
//! 呼び出し側 (Python) が sign(main_raw) から「優勢側」を選び、 その視点の指標で
//! own_king_risk / force / commitment class を組む。
//!
//! 出力 (tab 区切り):
//!   `<sfen>\tB:<15 csv>\tW:<15 csv>`
//! 各視点 15 指標の意味は [`persp_metrics`] のコメント参照。

use rshogi_core::bitboard::{Bitboard, king_5x5_mask, king_effect};
use rshogi_core::nnue::attack_dev::compute_attack_dev_class;
use rshogi_core::nnue::king_danger::{
    compute_king_danger_class, king_danger_slider_weighted_sum, king_danger_weighted_sum,
};
use rshogi_core::position::Position;
use rshogi_core::types::{Color, PieceType, Square};
use std::io::{self, BufRead, Write};

const HAND_TYPES: [PieceType; 7] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Gold,
    PieceType::Bishop,
    PieceType::Rook,
];

/// zone 内の各マスに利いている `c` 側の駒数の総和 (= 攻撃圧の総量)。
fn zone_attack_pressure(pos: &Position, mut zone: Bitboard, c: Color) -> u32 {
    let mut sum = 0u32;
    while zone.is_not_empty() {
        let sq = zone.pop();
        sum += (pos.attackers_to(sq) & pos.pieces_c(c)).count();
    }
    sum
}

/// 玉隣接 (king_effect) の逃げ升数 = 空 かつ 敵利き無し の隣接マス。
fn king_escape_squares(pos: &Position, king_sq: Square, enemy: Color) -> u32 {
    let mut adj = king_effect(king_sq);
    let occ = pos.occupied();
    let mut n = 0u32;
    while adj.is_not_empty() {
        let sq = adj.pop();
        let empty = (occ & Bitboard::from_square(sq)).is_empty();
        let attacked = (pos.attackers_to(sq) & pos.pieces_c(enemy)).is_not_empty();
        if empty && !attacked {
            n += 1;
        }
    }
    n
}

/// 手駒の駒種数 (0..=7) = 打ちの多様性 proxy。
fn hand_type_count(pos: &Position, c: Color) -> u32 {
    let h = pos.hand(c);
    HAND_TYPES.iter().filter(|&&pt| h.count(pt) > 0).count() as u32
}

/// 手駒総数。
fn hand_total(pos: &Position, c: Color) -> u32 {
    let h = pos.hand(c);
    HAND_TYPES.iter().map(|&pt| h.count(pt)).sum()
}

/// `c` 側の金銀 (Gold+Silver) bitboard。
fn gold_silver_bb(pos: &Position, c: Color) -> Bitboard {
    pos.pieces(c, PieceType::Gold) | pos.pieces(c, PieceType::Silver)
}

/// 1 視点 (persp = 優勢候補側) の 15 指標を返す。
///
/// own_king_risk 群 (自玉の危うさ、 大きいほど危険):
///  0 kd_wsum        : 自玉 3x3 内の敵重/軽駒重み付き数 (king_danger_weighted_sum)
///  1 kd_slider      : 自玉 3x3 に利く遠隔敵 slider 重み付き数
///  2 own_zone_press : 自玉 5x5 各升への敵利き総量
///  3 king_escape    : 自玉逃げ升数 (空 & 敵利き無し) — 少ないほど危険
///  4 enemy_hand_ty  : 敵手駒の駒種数 (打ち王手ポテンシャル proxy)
///  5 defenders5     : 自玉 5x5 内の自軍金銀数 (守備) — 多いほど安全
///  6 kd_class       : compute_king_danger_class(persp)
/// force 群 (敵玉への攻めの強制力、 大きいほど強制的):
///  7 ek_press       : 敵玉 5x5 各升への自軍利き総量
///  8 ek_escape      : 敵玉逃げ升数 (空 & 自利き無し) — 少ないほど強制
///  9 own_hand_ty    : 自手駒の駒種数 (打ち王手ポテンシャル)
/// 10 ad_class       : compute_attack_dev_class(persp)
/// 11 att_in_ek5     : 敵玉 5x5 内に居る自軍駒数 (玉除く)
/// commitment 群 (攻めへの投資 / 反動リスク):
/// 12 gs_far         : 自玉 5x5 外の自軍金銀数 (守備を離れた金銀 = 攻め投資)
/// 13 own_def3       : 自玉 3x3 内の自軍金銀数 (残存守備)
/// 14 own_hand_total : 自手駒総数 (少ないほど投入済み = commitment 高)
fn persp_metrics(pos: &Position, persp: Color) -> [i32; 15] {
    let enemy = persp.opponent();
    let own_king = pos.king_square(persp);
    let enemy_king = pos.king_square(enemy);

    let own_zone5 = king_5x5_mask(own_king);
    let enemy_zone5 = king_5x5_mask(enemy_king);
    let own_zone3 = king_effect(own_king) | Bitboard::from_square(own_king);

    let gs_own = gold_silver_bb(pos, persp);
    let defenders5 = (gs_own & own_zone5).count() as i32;
    let own_def3 = (gs_own & own_zone3).count() as i32;
    let gs_far = gs_own.count() as i32 - defenders5;

    // 敵玉 5x5 内の自軍駒数 (玉自身は persp 側の玉なので含まれない)
    let att_in_ek5 = (pos.pieces_c(persp) & enemy_zone5).count() as i32;

    [
        king_danger_weighted_sum(pos, persp),
        king_danger_slider_weighted_sum(pos, persp),
        zone_attack_pressure(pos, own_zone5, enemy) as i32,
        king_escape_squares(pos, own_king, enemy) as i32,
        hand_type_count(pos, enemy) as i32,
        defenders5,
        compute_king_danger_class(pos, persp) as i32,
        zone_attack_pressure(pos, enemy_zone5, persp) as i32,
        king_escape_squares(pos, enemy_king, persp) as i32,
        hand_type_count(pos, persp) as i32,
        compute_attack_dev_class(pos, persp) as i32,
        att_in_ek5,
        gs_far,
        own_def3,
        hand_total(pos, persp) as i32,
    ]
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let body = line
            .strip_prefix("position sfen ")
            .or_else(|| line.strip_prefix("sfen "))
            .unwrap_or(line);
        let mut pos = Position::new();
        if pos.set_sfen(body).is_err() {
            writeln!(out, "{body}\tERR")?;
            continue;
        }
        let b = persp_metrics(&pos, Color::Black);
        let w = persp_metrics(&pos, Color::White);
        let csv = |a: &[i32; 15]| a.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
        writeln!(out, "{body}\tB:{}\tW:{}", csv(&b), csv(&w))?;
    }
    Ok(())
}
