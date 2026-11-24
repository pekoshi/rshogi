//! Attack Relay Head (ARH) scalar head (= 2026-06-09)。
//!
//! 敵玉中心 **5x5** を **inner (= 3x3 玉自身含む)** と **outer ring** に分け、
//! 攻撃側 perspective の駒について次の 3 量を class 化し eval 加算する:
//!
//! - **contact** = inner zone に利く自軍駒のユニーク数 (= 0/1/2/3+)
//! - **reserve** = outer ring に利き かつ contact 集合に含まれない自軍駒数
//! - **relay**   = contact 駒のうち、 その現在升へ別の自軍駒 (= 自玉除く) が
//!   利いている駒数
//!
//! 玉自身は contact / reserve / relay のいずれからも除外する。 利きは pseudo
//! attack (= pin / 王手放置を考慮しない、 現在 occupancy)。
//!
//! 詳細仕様は `docs/attack_relay_head_design.md`。
//!
//! # 次元と容量
//!
//! - cell 数: 4 contact × 4 reserve × 4 relay = **64** / (persp, bucket)
//! - `ARH_DIM_PER_PERSP` = 8 bucket × 64 = **512** (= 1 perspective 分の cell)
//! - `ARH_WEIGHT_COUNT` = 512 × 2 persp = **1024 i16 = 2 KiB**
//!
//! # disk layout (= ckpt 上の sequential 順)
//!
//! `[bucket][persp][contact][reserve][relay]` row-major i16。
//!
//! ```text
//! sub_magic = "P2AREL\0\0"   (= 8 byte)
//! version   = u32 LE         (= 1)
//! count     = u32 LE         (= 1024)
//! weights   = i16 LE × 1024
//! ```
//!
//! # 視点規約 (= §3.6)
//!
//! - **訓練 graph**: `black - white` を main に統合 (= black - white = stm 視点
//!   ではなく black 絶対視点)、 KED と同 pattern。
//! - **推論 evaluate**: `stm - nstm` 反対称合成で stm 視点 cp を返す。
//!
//! 同一局面に対し訓練式と推論式は等価。

use crate::bitboard::{
    Bitboard, bishop_effect, dragon_effect, gold_effect, horse_effect, king_5x5_mask, king_effect,
    knight_effect, lance_effect, pawn_effect, rook_effect, silver_effect,
};
use crate::position::Position;
use crate::types::{Color, PieceType, Square};

// =============================================================================
// 定数
// =============================================================================

/// bucket 数 (= LayerStacks の progress8kpabs と共有)。 値域 `0..=7`。
pub const ARH_NUM_BUCKETS: usize = 8;

/// perspective 数 (= Black / White)。
pub const ARH_NUM_PERSP: usize = 2;

/// contact class 数 (= 0 / 1 / 2 / 3+)。
pub const ARH_NUM_CONTACT: usize = 4;

/// reserve class 数 (= 0 / 1 / 2 / 3+)。
pub const ARH_NUM_RESERVE: usize = 4;

/// relay class 数 (= 0 / 1 / 2 / 3+)。
pub const ARH_NUM_RELAY: usize = 4;

/// 1 perspective × 全 bucket の cell 数 (= 8 × 4 × 4 × 4 = 512)。
pub const ARH_DIM_PER_PERSP: usize =
    ARH_NUM_BUCKETS * ARH_NUM_CONTACT * ARH_NUM_RESERVE * ARH_NUM_RELAY;

/// ckpt 上の総 weight 数 (= ARH_DIM_PER_PERSP × 2 = 1024)。
pub const ARH_WEIGHT_COUNT: usize = ARH_DIM_PER_PERSP * ARH_NUM_PERSP;

/// section sub-magic (= 8 byte、 他 section と衝突しない)。
pub const ARH_MAGIC: &[u8; 8] = b"P2AREL\0\0";

/// section format version (= 1、 初版)。
pub const ARH_VERSION: u32 = 1;

/// 評価合成時の output scale (= 初版 4)。
///
/// 2026-06-10 OUTPUT_SCALE=16 calibration 試行: A sb 12 ckpt で 4→16 変更しても
/// 100 SFEN diff mean -1.86 不変 = startpos noise レベル。 真因は class 軸 (= contact/
/// reserve/relay) の中盤 invariance (= class (0,0,0) 支配で異 ckpt 同 cell)、 OUTPUT_SCALE
/// 拡大は対症療法。 = 初版 4 維持、 真の改善は C path (= v2 仕様、 attackers/overlap/
/// supported) で実施。
pub const ARH_OUTPUT_SCALE: i32 = 4;

const _: () = {
    assert!(ARH_DIM_PER_PERSP == 512);
    assert!(ARH_WEIGHT_COUNT == 1024);
};

// =============================================================================
// 型定義
// =============================================================================

/// disk layout: `[bucket][persp][contact][reserve][relay] -> i16` (= ckpt と同一)。
pub type ArhWeights =
    [[[[[i16; ARH_NUM_RELAY]; ARH_NUM_RESERVE]; ARH_NUM_CONTACT]; ARH_NUM_PERSP]; ARH_NUM_BUCKETS];

/// runtime layout: `[persp][contact][reserve][relay][bucket] -> i32`
/// (= bucket-inner、 hot path で `row[bucket]` 連続 read 可能、 KED と同 pattern)。
pub type ArhRuntimeWeights =
    [[[[[i32; ARH_NUM_BUCKETS]; ARH_NUM_RELAY]; ARH_NUM_RESERVE]; ARH_NUM_CONTACT]; ARH_NUM_PERSP];

// =============================================================================
// index helper
// =============================================================================

/// disk 上の flatten index (= `[bucket][persp][contact][reserve][relay]` row-major)。
#[inline]
pub const fn flatten_disk_idx(
    bucket: usize,
    persp: usize,
    contact: usize,
    reserve: usize,
    relay: usize,
) -> usize {
    let inner = ((contact * ARH_NUM_RESERVE) + reserve) * ARH_NUM_RELAY + relay;
    let pb = persp * (ARH_NUM_CONTACT * ARH_NUM_RESERVE * ARH_NUM_RELAY) + inner;
    bucket * (ARH_NUM_PERSP * ARH_NUM_CONTACT * ARH_NUM_RESERVE * ARH_NUM_RELAY) + pb
}

/// 学習側 sparse input の local idx (= 1 perspective 内、 `0..ARH_DIM_PER_PERSP`)。
///
/// `bucket × 64 + contact × 16 + reserve × 4 + relay`。 bullet 側
/// `attack_relay_features.rs::local_idx` と **完全同一** にする。
#[inline]
pub const fn local_idx(bucket: usize, contact: usize, reserve: usize, relay: usize) -> usize {
    ((bucket * ARH_NUM_CONTACT + contact) * ARH_NUM_RESERVE + reserve) * ARH_NUM_RELAY + relay
}

// =============================================================================
// 特徴量計算 (= §3.2-3.4 の厳密定義)
// =============================================================================

/// 指定 piece の attack bitboard を返す (= own 攻撃用の per-piece effect lookup)。
///
/// 成り駒 (= ProPawn / ProLance / ProKnight / ProSilver) は **gold と同じ動き**。
/// King は ARH の attacker / supporter から常に除外するため、 King を渡しても
/// `king_effect` を返すが ARH 仕様上呼ばれない (= 自玉除外 + relay supporter 玉除外)。
#[inline]
fn piece_attack_bb(pt: PieceType, color: Color, sq: Square, occupied: Bitboard) -> Bitboard {
    match pt {
        PieceType::Pawn => pawn_effect(color, sq),
        PieceType::Lance => lance_effect(color, sq, occupied),
        PieceType::Knight => knight_effect(color, sq),
        PieceType::Silver => silver_effect(color, sq),
        PieceType::Gold
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver => gold_effect(color, sq),
        PieceType::Bishop => bishop_effect(sq, occupied),
        PieceType::Rook => rook_effect(sq, occupied),
        PieceType::Horse => horse_effect(sq, occupied),
        PieceType::Dragon => dragon_effect(sq, occupied),
        PieceType::King => king_effect(sq),
    }
}

/// 攻撃側 perspective `p` の 3 軸 class を計算して返す。
///
/// 「p が敵玉 (= opp(p) の玉) を攻める」 観点で、 contact / reserve / relay を
/// それぞれ 4 class (= 0 / 1 / 2 / 3+) で返す。
///
/// # 仕様要約
///
/// - **inner zone** = `king_effect(opp_king_sq) | from_square(opp_king_sq)` (= 9 sq)
/// - **outer ring** = `king_5x5_mask(opp_king_sq) & !inner_zone` (= 最大 16 sq)
/// - 全 attacker から **自玉** (= `pos.king_square(p)`) を除外する
/// - **contact** = inner に利く自軍駒の **ユニーク** bitboard の popcount
/// - **reserve** = outer に利く自軍駒のうち contact に含まれないものの popcount
/// - **relay** = contact 駒の現在升への supporter (= 自分自身 / 自玉除く) が
///   1 枚以上ある contact 駒の数
///
/// 全 class は `min(count, 3)` で 4 class に clamp。
///
/// # 実装 (= 2026-06-10、 NPS 最適化版)
///
/// 旧版 (= 25 sq × `attackers_to_occ`) は 1 eval あたり 25 回の逆引きで重い。 本版は
/// **own_pieces を iterate して per-piece attack bb を計算 + zone & で intersection
/// 判定** に変更する (= ~20 駒 × 1 effect call)。 attack_bb は inner / outer の和集合
/// (= zone 5x5) と一度に AND をとり、 inner と排他なら reserve、 そうでなければ contact。
/// relay は元の `attackers_to_occ(piece_sq) & own_pieces` 方式を維持 (= contact 駒数
/// は平均 0-2 個で軽量)。
#[inline]
pub fn compute_arh_classes(pos: &Position, persp: Color) -> (usize, usize, usize) {
    let opp = persp.opponent();
    let opp_king_sq = pos.king_square(opp);
    let own_king_sq = pos.king_square(persp);
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);
    let own_pieces = pos.pieces_c(persp);

    let inner_zone = king_effect(opp_king_sq) | Bitboard::from_square(opp_king_sq);
    let outer_ring = king_5x5_mask(opp_king_sq) & !inner_zone;
    let own_king_bb = Bitboard::from_square(own_king_sq);

    // own_pieces を iterate (= 自玉除く) して各駒の attack_bb で zone 接触判定。
    let candidates = own_pieces & !own_king_bb;
    let mut contact_attackers = Bitboard::EMPTY;
    let mut reserve_attackers = Bitboard::EMPTY;
    let mut iter = candidates;
    while iter.is_not_empty() {
        let sq = iter.pop();
        let pt = pos.piece_on(sq).piece_type();
        let attack_bb = piece_attack_bb(pt, persp, sq, occupied);
        if !(attack_bb & inner_zone).is_empty() {
            contact_attackers |= Bitboard::from_square(sq);
        } else if !(attack_bb & outer_ring).is_empty() {
            reserve_attackers |= Bitboard::from_square(sq);
        }
    }
    let contact_class = (contact_attackers.count() as usize).min(ARH_NUM_CONTACT - 1);
    let reserve_class = (reserve_attackers.count() as usize).min(ARH_NUM_RESERVE - 1);

    // relay: contact 駒数は平均 0-2 個と少ないため、 元の `attackers_to_occ` 経由が軽い。
    let mut relay_count: usize = 0;
    let mut iter_c = contact_attackers;
    while iter_c.is_not_empty() {
        let piece_sq = iter_c.pop();
        let raw_supporters = pos.attackers_to_occ(piece_sq, occupied) & own_pieces;
        let supporters = raw_supporters & !Bitboard::from_square(piece_sq) & !own_king_bb;
        if supporters.is_not_empty() {
            relay_count += 1;
        }
    }
    let relay_class = relay_count.min(ARH_NUM_RELAY - 1);

    (contact_class, reserve_class, relay_class)
}

// =============================================================================
// 量子化 weight build / evaluate
// =============================================================================

/// `ArhWeights` (= disk format、 bucket-outer i16) を `ArhRuntimeWeights`
/// (= bucket-inner i32) に転置して構築する。 ckpt read 直後に **1 回だけ** 呼ぶ。
pub fn build_runtime_weights(disk: &ArhWeights) -> Box<ArhRuntimeWeights> {
    let mut rt: Box<ArhRuntimeWeights> = Box::new(
        [[[[[0i32; ARH_NUM_BUCKETS]; ARH_NUM_RELAY]; ARH_NUM_RESERVE]; ARH_NUM_CONTACT];
            ARH_NUM_PERSP],
    );
    for (bucket, b_arr) in disk.iter().enumerate() {
        for (persp, p_arr) in b_arr.iter().enumerate() {
            for (contact, c_arr) in p_arr.iter().enumerate() {
                for (reserve, r_arr) in c_arr.iter().enumerate() {
                    for (relay, &w) in r_arr.iter().enumerate() {
                        rt[persp][contact][reserve][relay][bucket] = w as i32;
                    }
                }
            }
        }
    }
    rt
}

/// ARH を評価して **stm 視点 cp 寄与 (= fv_scale 前 raw score)** を返す。
///
/// `(us - them) × OUTPUT_SCALE` で反対称合成。 `weights` が無い (= 旧 ckpt /
/// section 不在) 場合の処理は呼出側で行う (= 既存 KED と同方針)。
///
/// # Arguments
/// * `rt`     - runtime 形式の ARH weight
/// * `pos`    - 局面
/// * `bucket` - 現在の bucket index (= `0..ARH_NUM_BUCKETS`)
/// * `stm`    - 手番 (= side to move)
#[inline]
pub fn evaluate_arh(rt: &ArhRuntimeWeights, pos: &Position, bucket: usize, stm: Color) -> i32 {
    debug_assert!(bucket < ARH_NUM_BUCKETS);
    let nstm = stm.opponent();
    let (sc, sr, sre) = compute_arh_classes(pos, stm);
    let (nc, nr, nre) = compute_arh_classes(pos, nstm);
    let us = rt[stm as usize][sc][sr][sre][bucket];
    let them = rt[nstm as usize][nc][nr][nre][bucket];
    us.saturating_sub(them).saturating_mul(ARH_OUTPUT_SCALE)
}

// king_5x5_mask が utils 経由でしか使われない feature gate 配置だと未参照警告に
// なる場合があるので関数 pointer で参照を保持。
const _: fn(Square) -> Bitboard = king_5x5_mask;

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Position, SFEN_HIRATE};

    fn startpos() -> Position {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        pos
    }

    #[test]
    fn test_dims_and_magic() {
        assert_eq!(ARH_NUM_BUCKETS, 8);
        assert_eq!(ARH_NUM_PERSP, 2);
        assert_eq!(ARH_NUM_CONTACT, 4);
        assert_eq!(ARH_NUM_RESERVE, 4);
        assert_eq!(ARH_NUM_RELAY, 4);
        assert_eq!(ARH_DIM_PER_PERSP, 512);
        assert_eq!(ARH_WEIGHT_COUNT, 1024);
        assert_eq!(ARH_MAGIC, b"P2AREL\0\0");
        assert_eq!(ARH_VERSION, 1);
    }

    #[test]
    fn test_flatten_disk_idx_bounds() {
        // first slot
        assert_eq!(flatten_disk_idx(0, 0, 0, 0, 0), 0);
        // last slot = ARH_WEIGHT_COUNT - 1
        assert_eq!(
            flatten_disk_idx(
                ARH_NUM_BUCKETS - 1,
                ARH_NUM_PERSP - 1,
                ARH_NUM_CONTACT - 1,
                ARH_NUM_RESERVE - 1,
                ARH_NUM_RELAY - 1,
            ),
            ARH_WEIGHT_COUNT - 1
        );
        // unique idx 確認
        let mut seen = std::collections::HashSet::new();
        for b in 0..ARH_NUM_BUCKETS {
            for p in 0..ARH_NUM_PERSP {
                for c in 0..ARH_NUM_CONTACT {
                    for r in 0..ARH_NUM_RESERVE {
                        for re in 0..ARH_NUM_RELAY {
                            assert!(seen.insert(flatten_disk_idx(b, p, c, r, re)));
                        }
                    }
                }
            }
        }
        assert_eq!(seen.len(), ARH_WEIGHT_COUNT);
    }

    #[test]
    fn test_local_idx_bounds() {
        // 1 perspective の最初 / 最後
        assert_eq!(local_idx(0, 0, 0, 0), 0);
        assert_eq!(
            local_idx(
                ARH_NUM_BUCKETS - 1,
                ARH_NUM_CONTACT - 1,
                ARH_NUM_RESERVE - 1,
                ARH_NUM_RELAY - 1,
            ),
            ARH_DIM_PER_PERSP - 1
        );
        // unique 確認
        let mut seen = std::collections::HashSet::new();
        for b in 0..ARH_NUM_BUCKETS {
            for c in 0..ARH_NUM_CONTACT {
                for r in 0..ARH_NUM_RESERVE {
                    for re in 0..ARH_NUM_RELAY {
                        let v = local_idx(b, c, r, re);
                        assert!(v < ARH_DIM_PER_PERSP);
                        assert!(seen.insert(v));
                    }
                }
            }
        }
        assert_eq!(seen.len(), ARH_DIM_PER_PERSP);
    }

    #[test]
    fn test_compute_arh_classes_startpos_in_range() {
        let pos = startpos();
        let (bc, br, bre) = compute_arh_classes(&pos, Color::Black);
        let (wc, wr, wre) = compute_arh_classes(&pos, Color::White);
        // 全 class は `0..=3` の範囲内であること
        for v in [bc, br, bre, wc, wr, wre] {
            assert!(v < ARH_NUM_CONTACT, "class out of range: {v}");
        }
    }

    #[test]
    fn test_compute_arh_classes_startpos_no_contact() {
        // 平手初期局面: 自軍の利きが敵玉 3x3 に届くことはない (= 双方 玉前は
        // 自陣最深、 自軍利きが敵陣まで届くのは飛香でしか不可能だが対称配置)。
        // contact = 0 を期待。 reserve / relay は実装一致性のみ確認。
        let pos = startpos();
        let (bc, _, bre) = compute_arh_classes(&pos, Color::Black);
        let (wc, _, wre) = compute_arh_classes(&pos, Color::White);
        assert_eq!(bc, 0);
        assert_eq!(wc, 0);
        // contact=0 なら relay=0 (= relay ≤ contact 自然制約)
        assert_eq!(bre, 0);
        assert_eq!(wre, 0);
    }

    #[test]
    fn test_evaluate_arh_zero_weights_is_zero() {
        // 全 weight 0 で stm 視点 score = 0
        let disk: Box<ArhWeights> = Box::new(
            [[[[[0i16; ARH_NUM_RELAY]; ARH_NUM_RESERVE]; ARH_NUM_CONTACT]; ARH_NUM_PERSP];
                ARH_NUM_BUCKETS],
        );
        let rt = build_runtime_weights(&disk);
        let pos = startpos();
        for bucket in 0..ARH_NUM_BUCKETS {
            assert_eq!(evaluate_arh(&rt, &pos, bucket, Color::Black), 0);
            assert_eq!(evaluate_arh(&rt, &pos, bucket, Color::White), 0);
        }
    }

    #[test]
    fn test_evaluate_arh_stm_sign_antisymmetry() {
        // 設計書 §10.1 注: 全 bucket 同 weight で bucket index 変動の影響を消し、
        // stm を反転させた呼出で sign 反転を確認。 同一 weight tensor を build、
        // 同一 SFEN に対し stm=Black と stm=White で evaluate して逆符号を確認。
        let mut disk: Box<ArhWeights> = Box::new(
            [[[[[0i16; ARH_NUM_RELAY]; ARH_NUM_RESERVE]; ARH_NUM_CONTACT]; ARH_NUM_PERSP];
                ARH_NUM_BUCKETS],
        );
        // 全 bucket で同一 weight (= class index で一意な + ピース別)
        for bucket in 0..ARH_NUM_BUCKETS {
            for persp in 0..ARH_NUM_PERSP {
                for c in 0..ARH_NUM_CONTACT {
                    for r in 0..ARH_NUM_RESERVE {
                        for re in 0..ARH_NUM_RELAY {
                            // persp で差が出るように Black=+v、 White=-v とする
                            let v: i16 = ((c * 16 + r * 4 + re) as i16) + 1;
                            disk[bucket][persp][c][r][re] = if persp == 0 { v } else { -v };
                        }
                    }
                }
            }
        }
        let rt = build_runtime_weights(&disk);
        let pos = startpos();
        let v_black = evaluate_arh(&rt, &pos, 0, Color::Black);
        let v_white = evaluate_arh(&rt, &pos, 0, Color::White);
        assert_eq!(
            v_black, -v_white,
            "stm 視点 score は stm 反転で符号反転すべき (black={v_black}, white={v_white})"
        );
    }
}
