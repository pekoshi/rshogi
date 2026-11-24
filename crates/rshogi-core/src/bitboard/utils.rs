//! Bitboard ユーティリティ関数

use crate::bitboard::core::Bitboard;
use crate::types::Square;

/// 玉中心 (2*R+1)x(2*R+1) 領域 mask テーブルを const 構築する。
///
/// 盤端は clip、 中心マス自身を含む。 `Bitboard` の内部表現
/// (= p\[0\]: 1-7 筋 bit 0-62、 p\[1\]: 8-9 筋 bit 0-17) に直接 bit を立てる。
const fn build_king_zone_table(radius: i32) -> [Bitboard; 81] {
    let mut table = [Bitboard::EMPTY; 81];
    let mut idx = 0usize;
    while idx < 81 {
        let kf = (idx / 9) as i32;
        let kr = (idx % 9) as i32;
        let mut p0 = 0u64;
        let mut p1 = 0u64;
        let mut df = -radius;
        while df <= radius {
            let f = kf + df;
            if 0 <= f && f < 9 {
                let mut dr = -radius;
                while dr <= radius {
                    let r = kr + dr;
                    if 0 <= r && r < 9 {
                        let sq_idx = (f * 9 + r) as usize;
                        if sq_idx < 63 {
                            p0 |= 1u64 << sq_idx;
                        } else {
                            p1 |= 1u64 << (sq_idx - 63);
                        }
                    }
                    dr += 1;
                }
            }
            df += 1;
        }
        table[idx] = Bitboard::new(p0, p1);
        idx += 1;
    }
    table
}

/// `king_5x5_mask` 用 const テーブル (= 81 マス分)。
static KING_5X5_TABLE: [Bitboard; 81] = build_king_zone_table(2);

/// `king_7x7_mask` 用 const テーブル (= 81 マス分)。
static KING_7X7_TABLE: [Bitboard; 81] = build_king_zone_table(3);

/// 玉中心 5x5 領域 bitboard (= 盤外除外、 玉自身を含む)。
///
/// `king_sq` の file / rank ± 2 の範囲、 盤端は clip。 ARH (attack_relay)
/// および activity_risk 双方の zone 定義に使う共通 helper。
/// const テーブル参照 1 回で返す (= 旧実装はループ構築で evaluate hot path で重かった)。
///
/// `nnue-attack-relay` / `nnue-activity-risk-psqt` 各 feature とは独立に
/// 常時 build される (= feature gate 下に置かない)。
#[inline]
pub fn king_5x5_mask(king_sq: Square) -> Bitboard {
    KING_5X5_TABLE[king_sq.index()]
}

/// 玉中心 7x7 領域 bitboard (= 盤外除外、 玉自身を含む、 最大 49 sq)。
///
/// `king_sq` の file / rank ± 3 の範囲、 盤端は clip。 ARH v2 構造化 (= 2026-06-11
/// 改訂、 zone 拡張版) で 5x5 → 7x7 攻撃 zone として使う。 const テーブル参照 1 回。
#[inline]
pub fn king_7x7_mask(king_sq: Square) -> Bitboard {
    KING_7X7_TABLE[king_sq.index()]
}

/// 64bit整数の最上位ビット位置を返す（MSB: Most Significant Bit）
///
/// # Arguments
/// * `x` - 入力値
///
/// # Returns
/// - `x != 0`: 最上位ビットの位置（0-63）
/// - `x == 0`: 0（未定義動作を避けるため）
///
/// # Examples
/// ```ignore
/// assert_eq!(msb64(0b1000), 3);
/// assert_eq!(msb64(0x8000_0000_0000_0000), 63);
/// assert_eq!(msb64(0), 0);
/// ```
#[inline(always)]
pub fn msb64(x: u64) -> u32 {
    if x == 0 { 0 } else { 63 - x.leading_zeros() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msb64_zero() {
        assert_eq!(msb64(0), 0);
    }

    #[test]
    fn test_msb64_single_bit() {
        assert_eq!(msb64(1), 0);
        assert_eq!(msb64(2), 1);
        assert_eq!(msb64(4), 2);
        assert_eq!(msb64(8), 3);
        assert_eq!(msb64(0x80), 7);
        assert_eq!(msb64(0x8000_0000_0000_0000), 63);
    }

    #[test]
    fn test_msb64_multiple_bits() {
        assert_eq!(msb64(0b1111), 3);
        assert_eq!(msb64(0xFF), 7);
        assert_eq!(msb64(0xFFFF_FFFF_FFFF_FFFF), 63);
    }

    #[test]
    fn test_msb64_power_of_two() {
        for i in 0..64 {
            assert_eq!(msb64(1u64 << i), i);
        }
    }
}
