//! 評価値正規化 + cp スケール検出
//!
//! ソース別の score 分布から cp スケールを自動検出し、
//! 必要に応じて正規化を行う。

use crate::dataset::types::{CpScale, SourceStats};

/// P99 値からスケールを自動検出する
pub fn detect_scale(stats: &SourceStats) -> CpScale {
    let p99 = stats.percentile99();

    if p99 < 2 {
        // P99 < 2.0 → tanh スケール
        CpScale::Tanh
    } else if p99 > 100 {
        // P99 > 100 → cp スケール
        CpScale::Centipawn
    } else {
        CpScale::Unknown
    }
}

/// スケールに応じて score を cp に変換する
pub fn normalize_to_cp(score: i32, scale: CpScale) -> i32 {
    match scale {
        CpScale::Centipawn => score,
        CpScale::Tanh => {
            // tanh スケール（-1.0 〜 1.0 が i32 で来る場合の近似逆変換）
            // 実際の tanh スケールでは score は浮動小数の 1000 倍等で格納されていることが多い
            // ここでは score * 600 / 1000 の近似を使う
            let clamped = score.clamp(-999, 999);
            // atanh 近似: cp ≈ 600 * atanh(score/1000)
            let x = clamped as f64 / 1000.0;
            let atanh = 0.5 * ((1.0 + x) / (1.0 - x)).ln();
            (atanh * 600.0) as i32
        }
        CpScale::Unknown => score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_centipawn_scale() {
        let mut stats = SourceStats::new(40000);
        // 大きな値を多数追加
        for i in 0..1000 {
            stats.record_score(i);
        }
        assert_eq!(detect_scale(&stats), CpScale::Centipawn);
    }

    #[test]
    fn normalize_cp_passthrough() {
        assert_eq!(normalize_to_cp(500, CpScale::Centipawn), 500);
        assert_eq!(normalize_to_cp(-300, CpScale::Centipawn), -300);
    }
}
