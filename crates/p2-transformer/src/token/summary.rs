//! Summary token (phase / material / ply / check) 用 bucket utility。
//!
//! 値はすべて `const fn` で計算可能 (cache table 不要)、 inline-able。
//! Transformer 側の embedding lookup 用 vocab id を返す。

use super::vocab::{
    CHK_NO, CHK_YES, MATERIAL_BASE, MATERIAL_BUCKET_COUNT, PHASE_END, PHASE_MIDDLE, PHASE_OPEN,
    PLY_BASE, PLY_BUCKET_COUNT, SIDE_BLACK, SIDE_WHITE,
};
use rshogi_core::types::Color;

/// 手番 → vocab id。
#[inline]
pub const fn side_token(color: Color) -> u16 {
    match color {
        Color::Black => SIDE_BLACK,
        Color::White => SIDE_WHITE,
    }
}

/// 王手の有無 → vocab id。
#[inline]
pub const fn check_token(in_check: bool) -> u16 {
    if in_check { CHK_YES } else { CHK_NO }
}

/// game_ply (1-based) → ply bucket (16 段)。
///
/// 配置:
/// - 0..16 手: bucket 0..15 を均等割 (1 手 = 1 bucket)
/// - >= 16 手: 16 手以降は log scale で bucket を切る。
///   > 具体的には `min(15, ply/8 + 7)` のような単純式とし、 64 手 = bucket 15。
///
/// const fn で fold するため簡潔な式とする。
#[inline]
pub const fn ply_bucket(ply: i32) -> u16 {
    // 0..8 手:   bucket 0..7  (1 手 = 1 bucket)
    // 8..16 手:  bucket 8..11 (2 手 = 1 bucket)
    // 16..32 手: bucket 12..15 (4 手 = 1 bucket)
    // 32+ 手:    bucket 15 saturate
    let p = if ply < 0 { 0 } else { ply };
    let b = if p < 8 {
        p as u32
    } else if p < 16 {
        8 + ((p - 8) / 2) as u32
    } else if p < 32 {
        12 + ((p - 16) / 4) as u32
    } else {
        (PLY_BUCKET_COUNT as u32) - 1
    };
    let bb = if b >= PLY_BUCKET_COUNT as u32 {
        (PLY_BUCKET_COUNT as u32) - 1
    } else {
        b
    };
    PLY_BASE + bb as u16
}

/// material balance (Black 視点の点数差) → 17 段 bucket。
///
/// 入力 `material_score` は cp 単位の差分 (Black - White)。
/// bucket は -8..=+8 の 17 段で、 1 段 ≒ 1 駒 (約 300cp) を目安に切る。
///
/// 配置:
/// - <= -2400: bucket -8 (index 0)
/// - >= +2400: bucket +8 (index 16)
/// - それ以外: `(score / 300)` を clamp して中央 (bucket 8) からの offset とする。
#[inline]
pub const fn material_bucket(material_score: i32) -> u16 {
    let raw = material_score / 300;
    let clamped = if raw < -8 {
        -8
    } else if raw > 8 {
        8
    } else {
        raw
    };
    let idx = (clamped + 8) as u32; // 0..=16
    debug_assert!(idx < MATERIAL_BUCKET_COUNT as u32);
    MATERIAL_BASE + idx as u16
}

/// 局面進行度 phase を決定する。 序盤 (Open) / 中盤 (Middle) / 終盤 (End)。
///
/// 大まかな heuristic:
/// - 序盤: ply < 30 かつ 双方とも手駒が空に近い
/// - 中盤: それ以外
/// - 終盤: total material が大きく削られた状態
///
/// ここでは ply ベースの簡易判定とする。 PUCT 側で別途 PhaseTracker を持っており、
/// summary token は粗い hint で十分。
#[inline]
pub const fn phase_token(ply: i32, total_hand_count: u32) -> u16 {
    if ply < 30 && total_hand_count < 4 {
        PHASE_OPEN
    } else if total_hand_count >= 10 {
        PHASE_END
    } else {
        PHASE_MIDDLE
    }
}

#[cfg(test)]
mod tests {
    use super::super::vocab::{CHK_YES, MATERIAL_BUCKET_COUNT, PLY_BUCKET_COUNT};
    use super::*;

    #[test]
    fn side_tokens() {
        assert_eq!(side_token(Color::Black), SIDE_BLACK);
        assert_eq!(side_token(Color::White), SIDE_WHITE);
    }

    #[test]
    fn check_tokens() {
        assert_eq!(check_token(true), CHK_YES);
        assert_eq!(check_token(false), CHK_NO);
    }

    #[test]
    fn ply_bucket_monotone_within_range() {
        // 単調非減少を確認 (絶対値ではなく bucket id が ply 増加に対し非減少)。
        let mut prev = ply_bucket(0);
        for ply in 1..=300 {
            let cur = ply_bucket(ply);
            assert!(cur >= prev, "ply_bucket regressed at ply={ply}");
            assert!((cur - PLY_BASE) < PLY_BUCKET_COUNT, "ply_bucket overflow at ply={ply}");
            prev = cur;
        }
        // 最大値到達確認
        assert_eq!(ply_bucket(1000), PLY_BASE + PLY_BUCKET_COUNT - 1);
    }

    #[test]
    fn material_bucket_clamps() {
        assert_eq!(material_bucket(0), MATERIAL_BASE + 8);
        assert_eq!(material_bucket(-10000), MATERIAL_BASE);
        assert_eq!(material_bucket(10000), MATERIAL_BASE + 16);
        // すべての bucket が範囲内
        for s in (-3000..=3000).step_by(50) {
            let id = material_bucket(s);
            assert!((MATERIAL_BASE..MATERIAL_BASE + MATERIAL_BUCKET_COUNT).contains(&id));
        }
    }

    #[test]
    fn phase_open_for_startpos() {
        assert_eq!(phase_token(1, 0), PHASE_OPEN);
        // 大量の手駒で終盤
        assert_eq!(phase_token(80, 12), PHASE_END);
        // 中盤
        assert_eq!(phase_token(50, 5), PHASE_MIDDLE);
    }
}
