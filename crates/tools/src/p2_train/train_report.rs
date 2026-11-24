//! 学習レポート出力
//!
//! RUN_REPORT.md と report.json を出力する。

use std::fs;
use std::io;
use std::path::Path;

use serde::Serialize;

/// 学習実行のサマリ情報
#[derive(Debug, Serialize)]
pub struct TrainReport {
    /// 学習データファイル
    pub data_file: String,
    /// 学習サンプル数
    pub train_samples: usize,
    /// 検証サンプル数
    pub val_samples: usize,
    /// エポック数
    pub epochs: u32,
    /// バッチサイズ
    pub batch_size: usize,
    /// 学習率
    pub lr: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// ランダムシード
    pub seed: Option<u64>,
    /// 最終 train loss
    pub final_train_loss: f32,
    /// 最終 val loss
    pub final_val_loss: f32,
    /// エポックごとの loss 履歴
    pub epoch_history: Vec<EpochRecord>,
    /// 出力ディレクトリ
    pub output_dir: String,
    /// 学習時間（秒）
    pub elapsed_secs: f64,
}

/// エポックごとの記録
#[derive(Debug, Serialize)]
pub struct EpochRecord {
    pub epoch: u32,
    pub train_loss: f32,
    pub val_loss: f32,
}

/// RUN_REPORT.md を出力
pub fn write_report_md(path: &Path, report: &TrainReport) -> io::Result<()> {
    let mut md = String::new();
    md.push_str("# P2-NNUE Training Report\n\n");

    md.push_str("## Configuration\n\n");
    md.push_str(&format!("- Data: `{}`\n", report.data_file));
    md.push_str(&format!("- Train samples: {}\n", report.train_samples));
    md.push_str(&format!("- Val samples: {}\n", report.val_samples));
    md.push_str(&format!("- Epochs: {}\n", report.epochs));
    md.push_str(&format!("- Batch size: {}\n", report.batch_size));
    md.push_str(&format!("- Learning rate: {}\n", report.lr));
    md.push_str(&format!("- Weight decay: {}\n", report.weight_decay));
    if let Some(seed) = report.seed {
        md.push_str(&format!("- Seed: {seed}\n"));
    }
    md.push_str(&format!("- Elapsed: {:.1}s\n", report.elapsed_secs));

    md.push_str("\n## Results\n\n");
    md.push_str(&format!("- Final train loss: {:.6}\n", report.final_train_loss));
    md.push_str(&format!("- Final val loss: {:.6}\n", report.final_val_loss));

    md.push_str("\n## Epoch History\n\n");
    md.push_str("| Epoch | Train Loss | Val Loss |\n");
    md.push_str("|-------|-----------|----------|\n");
    for rec in &report.epoch_history {
        md.push_str(&format!("| {} | {:.6} | {:.6} |\n", rec.epoch, rec.train_loss, rec.val_loss));
    }

    fs::write(path, md)
}

/// report.json を出力
pub fn write_report_json(path: &Path, report: &TrainReport) -> io::Result<()> {
    let json = serde_json::to_string_pretty(report).map_err(io::Error::other)?;
    fs::write(path, json)
}
