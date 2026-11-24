//! 局面フィルタリング
//!
//! 評価値・手数・合法性等に基づくフィルタリングを行う。

use crate::dataset::config::FilterConfig;
use crate::dataset::types::{FilterReasons, RawRecord};

/// 局面フィルタ
pub struct RecordFilter {
    config: FilterConfig,
}

impl RecordFilter {
    /// フィルタを作成する
    pub fn new(config: FilterConfig) -> Self {
        Self { config }
    }

    /// レコードをフィルタリングする。
    ///
    /// 通過した場合は `Ok(clipped_score)` を返す。
    /// 除外された場合は `Err(reason)` を返す。
    pub fn check(
        &self,
        record: &RawRecord,
        reasons: &mut FilterReasons,
    ) -> Result<i32, &'static str> {
        // best_move チェック
        if record.best_move.is_empty() || record.best_move == "none" {
            reasons.missing_best_move += 1;
            return Err("missing_best_move");
        }

        // 手数チェック
        if record.ply < self.config.ply_min || record.ply > self.config.ply_max {
            reasons.ply_out_of_range += 1;
            return Err("ply_out_of_range");
        }

        let abs_score = record.score.abs();

        // mate 除外
        if self.config.exclude_mate && abs_score >= self.config.mate_threshold {
            reasons.mate_excluded += 1;
            return Err("mate_excluded");
        }

        // score 範囲チェック
        if abs_score > self.config.score_abs_max {
            reasons.score_out_of_range += 1;
            return Err("score_out_of_range");
        }

        // score クリップ
        let clipped = record.score.clamp(-self.config.score_clip, self.config.score_clip);

        Ok(clipped)
    }
}
