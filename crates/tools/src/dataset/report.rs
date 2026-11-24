//! データセットレポート生成
//!
//! DATASET_REPORT.md と dataset_manifest.json を生成する。

use crate::dataset::types::{PipelineStats, UnifiedRecord};
use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// データセットマニフェスト（JSON 出力用）
#[derive(Serialize)]
pub struct DatasetManifest {
    pub created_at: String,
    pub seed: u64,
    pub total_records: u64,
    pub train_records: u64,
    pub valid_records: u64,
    pub test_records: u64,
    pub conflict_records: u64,
    pub sources: Vec<SourceSummary>,
    pub phase_distribution: HashMap<String, u64>,
    pub elapsed_secs: f64,
}

#[derive(Serialize)]
pub struct SourceSummary {
    pub name: String,
    pub ingested: u64,
    pub after_dedup: u64,
    pub after_sample: u64,
}

/// DATASET_REPORT.md を生成する
pub fn write_report_md(
    path: &Path,
    stats: &PipelineStats,
    train: &[UnifiedRecord],
    valid: &[UnifiedRecord],
    test: &[UnifiedRecord],
    seed: u64,
) -> Result<()> {
    let mut f = std::fs::File::create(path)?;

    writeln!(f, "# Dataset Report")?;
    writeln!(f)?;
    writeln!(f, "Generated: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(f, "Seed: {seed}")?;
    writeln!(f, "Elapsed: {:.1}s", stats.elapsed_secs)?;
    writeln!(f)?;

    // ソース別統計
    writeln!(f, "## Source Statistics")?;
    writeln!(f)?;
    writeln!(f, "| Source | Ingested | After Dedup | After Sample |")?;
    writeln!(f, "|--------|----------|-------------|--------------|")?;
    for (name, src_stats) in &stats.source_stats {
        writeln!(f, "| {} | {} | {} | - |", name, src_stats.ingested, src_stats.after_dedup,)?;
    }
    writeln!(f)?;

    // フェーズ別
    writeln!(f, "## Phase Distribution")?;
    writeln!(f)?;
    let mut phase_counts: HashMap<&str, u64> = HashMap::new();
    for rec in train.iter().chain(valid.iter()).chain(test.iter()) {
        *phase_counts.entry(&rec.phase).or_default() += 1;
    }
    writeln!(f, "| Phase | Count | Ratio |")?;
    writeln!(f, "|-------|-------|-------|")?;
    let total = train.len() + valid.len() + test.len();
    for phase in &["opening", "middle", "endgame"] {
        let count = phase_counts.get(phase).copied().unwrap_or(0);
        let ratio = if total > 0 {
            count as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        writeln!(f, "| {phase} | {count} | {ratio:.1}% |")?;
    }
    writeln!(f)?;

    // Value 分布
    writeln!(f, "## Value Distribution")?;
    writeln!(f)?;
    let all: Vec<_> = train.iter().chain(valid.iter()).chain(test.iter()).collect();
    if !all.is_empty() {
        let values: Vec<i32> = all.iter().map(|r| r.value_cp).collect();
        let min = values.iter().min().copied().unwrap_or(0);
        let max = values.iter().max().copied().unwrap_or(0);
        let mean = values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64;
        let var =
            values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = var.sqrt();
        writeln!(f, "- min: {min}")?;
        writeln!(f, "- max: {max}")?;
        writeln!(f, "- mean: {mean:.1}")?;
        writeln!(f, "- std: {std:.1}")?;
    }
    writeln!(f)?;

    // フィルタ除外
    writeln!(f, "## Filter Exclusions")?;
    writeln!(f)?;
    writeln!(f, "| Reason | Count |")?;
    writeln!(f, "|--------|-------|")?;
    writeln!(f, "| score_out_of_range | {} |", stats.filter_reasons.score_out_of_range)?;
    writeln!(f, "| ply_out_of_range | {} |", stats.filter_reasons.ply_out_of_range)?;
    writeln!(f, "| mate_excluded | {} |", stats.filter_reasons.mate_excluded)?;
    writeln!(f, "| illegal_position | {} |", stats.filter_reasons.illegal_position)?;
    writeln!(f, "| missing_best_move | {} |", stats.filter_reasons.missing_best_move)?;
    writeln!(f)?;

    // 最終件数
    writeln!(f, "## Final Counts")?;
    writeln!(f)?;
    writeln!(f, "- train: {}", train.len())?;
    writeln!(f, "- valid: {}", valid.len())?;
    writeln!(f, "- test: {}", test.len())?;
    writeln!(f, "- conflicts: {}", stats.conflict_count)?;
    writeln!(f)?;

    // サンプル SFEN
    writeln!(f, "## Sample SFENs")?;
    writeln!(f)?;
    for phase in &["opening", "middle", "endgame"] {
        writeln!(f, "### {phase}")?;
        let samples: Vec<_> = all.iter().filter(|r| r.phase == *phase).take(3).collect();
        for s in samples {
            writeln!(f, "- `{}` (move: {}, cp: {})", s.sfen, s.best_move, s.value_cp)?;
        }
        writeln!(f)?;
    }

    Ok(())
}

/// dataset_manifest.json を生成する
pub fn write_manifest_json(
    path: &Path,
    stats: &PipelineStats,
    train_count: u64,
    valid_count: u64,
    test_count: u64,
    seed: u64,
) -> Result<()> {
    let mut sources = Vec::new();
    for (name, src_stats) in &stats.source_stats {
        sources.push(SourceSummary {
            name: name.clone(),
            ingested: src_stats.ingested,
            after_dedup: src_stats.after_dedup,
            after_sample: 0,
        });
    }

    let manifest = DatasetManifest {
        created_at: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
        seed,
        total_records: train_count + valid_count + test_count,
        train_records: train_count,
        valid_records: valid_count,
        test_records: test_count,
        conflict_records: stats.conflict_count,
        sources,
        phase_distribution: HashMap::new(),
        elapsed_secs: stats.elapsed_secs,
    };

    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(path, json)?;
    Ok(())
}
