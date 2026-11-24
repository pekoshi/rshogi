//! 3パス統合パイプライン
//!
//! Pass 1: Ingest + Dedup + 統計集計
//! Pass 2: Filter + Enrich + Sample
//! Pass 3: Shuffle + Split + Report

use crate::common::io;
use crate::dataset::config::DatasetConfig;
use crate::dataset::conflict::ConflictDetector;
use crate::dataset::dedup::DedupFilter;
use crate::dataset::detect::detect_format;
use crate::dataset::filter::RecordFilter;
use crate::dataset::ingest::create_ingester;
use crate::dataset::normalize::{detect_scale, normalize_to_cp};
use crate::dataset::phase::classify_phase;
use crate::dataset::policy::{generate_legal_moves_usi, generate_onehot_policy};
use crate::dataset::report;
use crate::dataset::sample::{StratifiedSampler, StratumKey};
use crate::dataset::shuffle::shuffle_and_split;
use crate::dataset::types::*;
use anyhow::{Context, Result};
use rshogi_core::position::Position;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// パイプラインを実行する
pub fn run_pipeline(config: &DatasetConfig, limit: Option<u64>) -> Result<PathBuf> {
    run_pipeline_inner(config, limit, None)
}

/// 既存の中間ファイルから Pass 2 以降を再開する
pub fn resume_pipeline(config: &DatasetConfig, resume_dir: &Path) -> Result<PathBuf> {
    run_pipeline_inner(config, None, Some(resume_dir))
}

fn run_pipeline_inner(
    config: &DatasetConfig,
    limit: Option<u64>,
    resume_dir: Option<&Path>,
) -> Result<PathBuf> {
    let start = std::time::Instant::now();

    let output_dir = if let Some(dir) = resume_dir {
        dir.to_path_buf()
    } else {
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        config.output_dir.join(&timestamp)
    };
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("出力ディレクトリ作成失敗: {}", output_dir.display()))?;

    eprintln!("=== P2-NNUE Dataset Pipeline ===");
    eprintln!("Output: {}", output_dir.display());

    let (intermediate_paths, mut stats) = if resume_dir.is_some() {
        eprintln!("\n=== Resuming from existing intermediates ===");
        rebuild_stats_from_intermediates(config, &output_dir)?
    } else {
        // --- Pass 1: Ingest + Dedup + 統計集計 ---
        eprintln!("\n=== Pass 1: Ingest + Dedup ===");
        pass1_ingest_dedup(config, &output_dir, limit)?
    };

    // --- Pass 2: Filter + Enrich + Sample ---
    eprintln!("\n=== Pass 2: Filter + Enrich + Sample ===");
    let sampled_records = pass2_filter_sample(config, &intermediate_paths, &mut stats)?;
    eprintln!("Sampled: {} records", sampled_records.len());

    // --- Pass 3: Shuffle + Split + Report ---
    eprintln!("\n=== Pass 3: Shuffle + Split + Report ===");
    stats.elapsed_secs = start.elapsed().as_secs_f64();
    pass3_shuffle_split_report(config, sampled_records, &output_dir, &stats)?;

    eprintln!("\nPipeline complete in {:.1}s", stats.elapsed_secs);
    eprintln!("Output: {}", output_dir.display());

    Ok(output_dir)
}

/// Pass 1: 全入力をストリーミング読み込み → Dedup → 中間 JSONL 出力
fn pass1_ingest_dedup(
    config: &DatasetConfig,
    output_dir: &Path,
    limit: Option<u64>,
) -> Result<(Vec<(String, PathBuf)>, PipelineStats)> {
    // 推定レコード数を計算
    let estimated_total = estimate_total_records(config)?;
    eprintln!("Estimated total records: {estimated_total}");

    // Bloom filter の構築
    let mut dedup = DedupFilter::new(estimated_total, config.bloom.fpr);

    let mut stats = PipelineStats::default();
    let mut intermediate_paths = Vec::new();
    let mut global_count = 0u64;
    let global_limit = limit.unwrap_or(u64::MAX);

    for source in &config.sources {
        eprintln!("  Source: {} ({})", source.name, source.path.display());

        let input_paths = resolve_source_paths(source)?;
        if input_paths.is_empty() {
            eprintln!("    Warning: ファイルが見つかりません");
            continue;
        }

        let intermediate_path = output_dir.join(format!("_intermediate_{}.jsonl.gz", source.name));
        let mut writer = io::open_writer(&intermediate_path)?;

        let mut source_stats = SourceStats::new(40000);

        for input_path in &input_paths {
            let format = detect_format(input_path)
                .with_context(|| format!("フォーマット検出失敗: {}", input_path.display()))?;
            eprintln!("    {} ({})", input_path.display(), format);

            let ingester = create_ingester(format);

            ingester.ingest(input_path, &source.name, &mut |record| {
                if global_count >= global_limit {
                    return false;
                }

                source_stats.ingested += 1;

                // Bloom filter で dedup
                if dedup.insert_or_check(&record.sfen) {
                    return true; // 重複 → スキップ
                }

                source_stats.after_dedup += 1;
                source_stats.record_score(record.score);

                // 中間 JSONL 出力
                let json = serde_json::to_string(&IntermediateRecord {
                    sfen: &record.sfen,
                    best_move: &record.best_move,
                    score: record.score,
                    game_result: record.game_result,
                    ply: record.ply,
                    source: &record.source_name,
                })
                .unwrap();
                let _ = writeln!(writer, "{json}");

                global_count += 1;

                if source_stats.after_dedup.is_multiple_of(1_000_000) {
                    eprintln!(
                        "    {} ingested, {} after dedup",
                        source_stats.ingested, source_stats.after_dedup,
                    );
                }

                true
            })?;
        }

        writer.close()?;

        eprintln!(
            "    Total: {} ingested, {} after dedup",
            source_stats.ingested, source_stats.after_dedup,
        );

        intermediate_paths.push((source.name.clone(), intermediate_path));
        stats.source_stats.push((source.name.clone(), source_stats));
    }

    Ok((intermediate_paths, stats))
}

/// 中間ファイルから統計情報を再構築する（resume 用）
fn rebuild_stats_from_intermediates(
    config: &DatasetConfig,
    output_dir: &Path,
) -> Result<(Vec<(String, PathBuf)>, PipelineStats)> {
    let mut stats = PipelineStats::default();
    let mut intermediate_paths = Vec::new();

    for source in &config.sources {
        let intermediate_path = output_dir.join(format!("_intermediate_{}.jsonl.gz", source.name));
        if !intermediate_path.exists() {
            eprintln!("  Warning: 中間ファイルが見つかりません: {}", intermediate_path.display());
            continue;
        }

        eprintln!("  Scanning intermediate for stats: {}", intermediate_path.display());
        let reader = io::open_reader(&intermediate_path)?;
        let mut source_stats = SourceStats::new(40000);

        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            if let Ok(rec) = serde_json::from_str::<IntermediateRecord<'_>>(&line) {
                source_stats.after_dedup += 1;
                source_stats.record_score(rec.score);
            }
        }
        source_stats.ingested = source_stats.after_dedup; // resume ではdedup前の数は不明
        eprintln!("    {} records found", source_stats.after_dedup);

        intermediate_paths.push((source.name.clone(), intermediate_path));
        stats.source_stats.push((source.name.clone(), source_stats));
    }

    Ok((intermediate_paths, stats))
}

/// Pass 2: 中間 JSONL を読み戻し → フィルタ + 正規化 + サンプリング
fn pass2_filter_sample(
    config: &DatasetConfig,
    intermediate_paths: &[(String, PathBuf)],
    stats: &mut PipelineStats,
) -> Result<Vec<UnifiedRecord>> {
    // スケール検出
    let mut source_scales: HashMap<String, CpScale> = HashMap::new();
    if config.normalize.auto_detect_scale {
        for (name, src_stats) in &stats.source_stats {
            let scale = detect_scale(src_stats);
            eprintln!("  Scale detection for '{name}': {scale:?}");
            source_scales.insert(name.clone(), scale);
        }
    }

    let filter = RecordFilter::new(config.filter.clone());
    let source_names: Vec<String> = config.sources.iter().map(|s| s.name.clone()).collect();
    let mut sampler: StratifiedSampler<UnifiedRecord> =
        StratifiedSampler::new(&source_names, &config.sampling, config.seed);

    for (source_name, path) in intermediate_paths {
        eprintln!("  Reading intermediate: {}", path.display());
        let reader = io::open_reader(path)?;
        let scale = source_scales.get(source_name.as_str()).copied().unwrap_or(CpScale::Centipawn);

        let mut line_count = 0u64;
        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            let rec: IntermediateRecord<'_> = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(_) => continue,
            };

            let raw = RawRecord {
                sfen: rec.sfen.to_string(),
                best_move: rec.best_move.to_string(),
                score: rec.score,
                game_result: rec.game_result,
                ply: rec.ply,
                source_name: rec.source.to_string(),
            };

            // フィルタ
            let clipped_cp = match filter.check(&raw, &mut stats.filter_reasons) {
                Ok(cp) => cp,
                Err(_) => continue,
            };

            // 正規化
            let value_cp = normalize_to_cp(clipped_cp, scale);

            // フェーズ分類
            let phase = classify_phase(&raw.sfen, raw.ply);

            // Policy 生成（サンプリング前は軽量な分類のみ）
            let bucket = sampler.value_bucket_index(value_cp);
            let key = StratumKey {
                source: source_name.clone(),
                phase,
                value_bucket: bucket,
            };

            // legal_moves と policy_target はサンプリング採用後に生成
            let unified = UnifiedRecord {
                sfen: raw.sfen,
                best_move: raw.best_move,
                legal_moves: Vec::new(),
                policy_target: None,
                value_cp,
                result: raw.game_result,
                ply: raw.ply,
                source: source_name.clone(),
                phase: phase.to_string(),
                tags: Vec::new(),
            };

            sampler.add(key, unified);

            line_count += 1;
            if line_count.is_multiple_of(1_000_000) {
                eprintln!("    {line_count} records processed");
            }
        }
        eprintln!("    Total: {line_count} records processed");
    }

    let records = sampler.into_all_items();

    // サンプリング後に conflict 検出（メモリ節約のため sampled records のみ）
    let mut conflict_detector = ConflictDetector::new();
    for rec in &records {
        conflict_detector.check(&rec.sfen, &rec.best_move);
    }
    stats.conflict_count = conflict_detector.conflict_count() as u64;

    // conflict JSONL 出力
    let conflicts = conflict_detector.into_conflicts();
    if !conflicts.is_empty() {
        let conflict_path = intermediate_paths
            .first()
            .map(|(_, p)| p.parent().unwrap_or(Path::new(".")))
            .unwrap_or(Path::new("."))
            .join("conflict.jsonl");
        let mut writer = io::open_writer(&conflict_path)?;
        for c in &conflicts {
            let json = serde_json::json!({
                "sfen": c.sfen,
                "move_a": c.move_a,
                "move_b": c.move_b,
            });
            writeln!(writer, "{}", json)?;
        }
        writer.close()?;
    }

    // legal_moves / policy_target は Pass 3 の JSONL 書き出し時にストリーミング生成
    // （10M件 × ~50合法手のメモリ保持を回避）

    Ok(records)
}

/// Pass 3: シャッフル + 分割 + 出力 + レポート
fn pass3_shuffle_split_report(
    config: &DatasetConfig,
    records: Vec<UnifiedRecord>,
    output_dir: &Path,
    stats: &PipelineStats,
) -> Result<()> {
    let split = shuffle_and_split(records, &config.split, config.seed.wrapping_add(1));

    eprintln!(
        "  train: {}, valid: {}, test: {}",
        split.train.len(),
        split.valid.len(),
        split.test.len(),
    );

    let gen_policy = config.policy.generate_legal_moves;
    if gen_policy {
        eprintln!("  Generating legal moves + policy (streaming)...");
    }

    // JSONL 出力（legal_moves/policy をストリーミング生成）
    write_jsonl(&output_dir.join("train.jsonl"), &split.train, gen_policy)?;
    eprintln!("  train.jsonl written");
    write_jsonl(&output_dir.join("valid.jsonl"), &split.valid, gen_policy)?;
    write_jsonl(&output_dir.join("test.jsonl"), &split.test, gen_policy)?;
    eprintln!("  All JSONL written");

    // レポート
    report::write_report_md(
        &output_dir.join("DATASET_REPORT.md"),
        stats,
        &split.train,
        &split.valid,
        &split.test,
        config.seed,
    )?;

    report::write_manifest_json(
        &output_dir.join("dataset_manifest.json"),
        stats,
        split.train.len() as u64,
        split.valid.len() as u64,
        split.test.len() as u64,
        config.seed,
    )?;

    // 中間ファイルを削除
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("_intermediate_") {
            std::fs::remove_file(entry.path())?;
        }
    }

    Ok(())
}

/// JSONL ファイルを書き出す（legal_moves + policy_target をストリーミング生成）
fn write_jsonl(path: &Path, records: &[UnifiedRecord], generate_policy: bool) -> Result<()> {
    let mut writer = io::open_writer(path)?;
    for rec in records {
        if generate_policy {
            let mut enriched = rec.clone();
            let mut pos = Position::new();
            if pos.set_sfen(&rec.sfen).is_ok() {
                let legal = generate_legal_moves_usi(&pos);
                if let Some(policy) = generate_onehot_policy(&legal, &rec.best_move) {
                    enriched.policy_target = Some(policy);
                }
                enriched.legal_moves = legal;
            }
            let json = serde_json::to_string(&enriched)?;
            writeln!(writer, "{json}")?;
        } else {
            let json = serde_json::to_string(rec)?;
            writeln!(writer, "{json}")?;
        }
    }
    writer.close()?;
    eprintln!("  Wrote: {} ({} records)", path.display(), records.len());
    Ok(())
}

/// ソースの入力パスを解決する
fn resolve_source_paths(source: &crate::dataset::config::SourceConfig) -> Result<Vec<PathBuf>> {
    let path = &source.path;
    if path.is_file() {
        return Ok(vec![path.clone()]);
    }
    if path.is_dir() {
        return crate::common::dedup::collect_input_paths(None, Some(path), &source.pattern)
            .map_err(Into::into);
    }
    // glob パターンとして試す
    let pattern = path.to_string_lossy();
    let paths: Vec<PathBuf> = glob::glob(&pattern)
        .map_err(|e| anyhow::anyhow!("glob パターン解析失敗: {e}"))?
        .filter_map(|r| r.ok())
        .filter(|p| p.is_file())
        .collect();
    Ok(paths)
}

/// 全ソースの推定レコード数を計算する
fn estimate_total_records(config: &DatasetConfig) -> Result<u64> {
    let mut total = 0u64;
    for source in &config.sources {
        let paths = resolve_source_paths(source)?;
        for path in &paths {
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

            // PSV: 40 バイト/レコード、テキスト: 平均 100 バイト/行
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
            let records_per_file = match ext {
                "bin" | "psv" => size / 40,
                _ => size / 100,
            };
            total += records_per_file;
        }
    }
    Ok(total.max(1000))
}

/// 中間 JSONL レコード（Serialize/Deserialize 用）
#[derive(serde::Serialize, serde::Deserialize)]
struct IntermediateRecord<'a> {
    #[serde(borrow)]
    sfen: &'a str,
    #[serde(borrow)]
    best_move: &'a str,
    score: i32,
    game_result: i8,
    ply: u32,
    #[serde(borrow)]
    source: &'a str,
}
