//! target label generation
//!
//! 主要な label モード:
//!
//! - [`LabelMode::PlyRatio`]:
//!   `progress(t) = clamp(game_ply(t) / total_plies(game), 0, 1)`
//!   完全 game 系列 (csa 等) から計算。 推奨。
//! - [`LabelMode::Heuristic`]:
//!   `compute_endgame_score(pos)` をそのまま label 化。
//!   PSV のような単発レコード source 用の疑似 label。
//! - [`LabelMode::PlyRatioOrHeuristic`]: PlyRatio 入力なら ply、 fallback heuristic。

use p2_search::phase::{compute_endgame_score, compute_features as compute_endgame_features};
use rshogi_core::position::Position;

/// label モード列挙
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelMode {
    /// 完全 game 系列前提: ply / total_plies
    PlyRatio,
    /// レコード単発前提: rule-based score を流用 (擬似 label)
    Heuristic,
    /// total_plies が既知ならば PlyRatio、 さもなくば Heuristic に fallback
    PlyRatioOrHeuristic,
}

/// 進行度 label を計算する。
///
/// - `total_plies`: ゲーム終局までの総手数 (PlyRatio 系で必須)。 不明なら `None`。
/// - 出力は `[0, 1]` に clamp 済。
pub fn compute_label(pos: &Position, total_plies: Option<u32>, mode: LabelMode) -> f32 {
    match mode {
        LabelMode::PlyRatio => ply_ratio(pos, total_plies),
        LabelMode::Heuristic => heuristic_score(pos),
        LabelMode::PlyRatioOrHeuristic => match total_plies {
            Some(tp) if tp > 0 => ply_ratio(pos, Some(tp)),
            _ => heuristic_score(pos),
        },
    }
}

fn ply_ratio(pos: &Position, total_plies: Option<u32>) -> f32 {
    let total = total_plies.unwrap_or(256).max(1);
    let cur = pos.game_ply() as f32;
    (cur / total as f32).clamp(0.0, 1.0)
}

fn heuristic_score(pos: &Position) -> f32 {
    let ef = compute_endgame_features(pos, 0);
    compute_endgame_score(&ef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::position::Position;

    #[test]
    fn ply_ratio_startpos_is_low() {
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let y = compute_label(&pos, Some(120), LabelMode::PlyRatio);
        assert!(y < 0.01, "ply ratio at ply 1 should be ~0, got {}", y);
    }

    #[test]
    fn ply_ratio_endgame_is_high() {
        let mut pos = Position::new();
        // 王と金のみ、 120 手目想定
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2G2g 120").unwrap();
        let y = compute_label(&pos, Some(120), LabelMode::PlyRatio);
        assert!(y > 0.99, "ply ratio at final ply should be ~1, got {}", y);
    }

    #[test]
    fn heuristic_label_in_range() {
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2G2g 120").unwrap();
        let y = compute_label(&pos, None, LabelMode::Heuristic);
        assert!((0.0..=1.0).contains(&y), "got {}", y);
    }

    #[test]
    fn fallback_uses_heuristic_when_no_total() {
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let y = compute_label(&pos, None, LabelMode::PlyRatioOrHeuristic);
        // startpos heuristic = ほぼ 0 (material 満タン)
        assert!(y < 0.30, "heuristic at startpos should be low, got {}", y);
    }
}
