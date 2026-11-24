//! データセットパイプラインの基盤型定義
//!
//! 学習用 JSONL の統一レコード形式と中間形式を定義する。

use serde::{Deserialize, Serialize};

/// 出力 JSONL の統一レコード形式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedRecord {
    /// SFEN 文字列（手番・持ち駒・手数含む）
    pub sfen: String,

    /// USI 形式の最善手
    #[serde(rename = "move")]
    pub best_move: String,

    /// 合法手リスト（USI 形式）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub legal_moves: Vec<String>,

    /// Policy ターゲット（onehot or soft）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_target: Option<PolicyTarget>,

    /// centipawn スケールの評価値
    pub value_cp: i32,

    /// 対局結果（1=勝ち, 0=引き分け, -1=負け）
    pub result: i8,

    /// 手数
    pub ply: u32,

    /// データソース名
    pub source: String,

    /// フェーズ分類
    pub phase: String,

    /// タグ（追加メタデータ）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

/// Policy ターゲット
///
/// `moves` は `UnifiedRecord.legal_moves` と一致するため、
/// JSONL 出力時のみ含める。メモリ保持時は `probs` のみ。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyTarget {
    /// ターゲット種別: "onehot" | "soft"
    #[serde(rename = "type")]
    pub target_type: PolicyType,

    /// 各手に対応する確率（legal_moves と同じ順序）
    pub probs: Vec<f32>,
}

/// Policy ターゲット種別
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolicyType {
    Onehot,
    Soft,
}

/// フェーズ分類
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Opening,
    Middle,
    Endgame,
}

impl Phase {
    /// 文字列表現
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Opening => "opening",
            Phase::Middle => "middle",
            Phase::Endgame => "endgame",
        }
    }
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Ingester が生成する中間レコード
#[derive(Debug, Clone)]
pub struct RawRecord {
    /// SFEN 文字列
    pub sfen: String,

    /// USI 形式の最善手
    pub best_move: String,

    /// 評価値（ソース固有のスケール）
    pub score: i32,

    /// 対局結果（1=勝ち, 0=引き分け, -1=負け）
    pub game_result: i8,

    /// 手数
    pub ply: u32,

    /// データソース名
    pub source_name: String,
}

/// 入力フォーマット
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// PackedSfenValue（.bin, .psv）
    Psv,
    /// SFEN テキスト（.sfen, .txt）
    Sfen,
    /// CSA 棋譜（.csa）
    Csa,
    /// KIF 棋譜（.kif, .kifu）
    Kif,
    /// JSONL（.jsonl）
    Jsonl,
    /// CSV/TSV（.csv, .tsv）
    CsvTsv,
}

impl std::fmt::Display for InputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            InputFormat::Psv => "PSV",
            InputFormat::Sfen => "SFEN",
            InputFormat::Csa => "CSA",
            InputFormat::Kif => "KIF",
            InputFormat::Jsonl => "JSONL",
            InputFormat::CsvTsv => "CSV/TSV",
        };
        f.write_str(name)
    }
}

/// value bucket の区間定義
#[derive(Debug, Clone)]
pub struct ValueBucket {
    /// |cp| の上限
    pub abs_max: i32,
    /// この区間の目標比率
    pub ratio: f64,
}

/// ソース別統計（Pass 1 で集計）
#[derive(Debug, Default)]
pub struct SourceStats {
    /// 読み込みレコード数
    pub ingested: u64,
    /// dedup 後のレコード数
    pub after_dedup: u64,
    /// score の最小値
    pub score_min: i32,
    /// score の最大値
    pub score_max: i32,
    /// score の合計（平均計算用）
    pub score_sum: i64,
    /// score の二乗和（分散計算用）
    pub score_sq_sum: i64,
    /// P99 推定用のヒストグラム（|score| → 件数）
    pub abs_score_histogram: Vec<u64>,
}

impl SourceStats {
    /// 新規作成（ヒストグラムのビン数を指定）
    pub fn new(histogram_bins: usize) -> Self {
        Self {
            abs_score_histogram: vec![0; histogram_bins],
            score_min: i32::MAX,
            score_max: i32::MIN,
            ..Default::default()
        }
    }

    /// score を集計に追加
    pub fn record_score(&mut self, score: i32) {
        self.score_min = self.score_min.min(score);
        self.score_max = self.score_max.max(score);
        self.score_sum += score as i64;
        self.score_sq_sum += (score as i64) * (score as i64);

        let abs = score.unsigned_abs() as usize;
        let bin = abs.min(self.abs_score_histogram.len() - 1);
        self.abs_score_histogram[bin] += 1;
    }

    /// P99 値を推定する
    pub fn percentile99(&self) -> i32 {
        let total: u64 = self.abs_score_histogram.iter().sum();
        if total == 0 {
            return 0;
        }
        let threshold = (total as f64 * 0.99).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.abs_score_histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= threshold {
                return i as i32;
            }
        }
        self.abs_score_histogram.len() as i32 - 1
    }
}

/// cp スケール検出結果
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CpScale {
    /// centipawn スケール（P99 > 100）
    Centipawn,
    /// tanh / sigmoid スケール（P99 < 2.0）
    Tanh,
    /// 不明
    Unknown,
}

/// パイプライン実行中の統計（レポート用）
#[derive(Debug, Default)]
pub struct PipelineStats {
    /// ソース別統計
    pub source_stats: Vec<(String, SourceStats)>,
    /// フィルタ除外理由別件数
    pub filter_reasons: FilterReasons,
    /// 最終 train/valid/test 件数
    pub train_count: u64,
    pub valid_count: u64,
    pub test_count: u64,
    /// conflict 件数
    pub conflict_count: u64,
    /// 処理時間（秒）
    pub elapsed_secs: f64,
}

/// フィルタ除外理由別件数
#[derive(Debug, Default, Serialize)]
pub struct FilterReasons {
    pub score_out_of_range: u64,
    pub ply_out_of_range: u64,
    pub mate_excluded: u64,
    pub illegal_position: u64,
    pub missing_best_move: u64,
}
