//! Phase A: 学習データの value target / cp 分布検査
//!
//! - cp / target の分布 (min/max/mean/std/percentiles)
//! - saturation rate (|target - 0.5| > thresholds)
//! - CP_SCALE 候補比較 (saturation rate, std)
//! - source / phase 別件数
//! - SFEN 重複率, bestmove 欠損率, 異常 SFEN 件数
//! - cp ↔ target 相関 (Pearson)
//!
//! 出力:
//!   <out>/PHASE_A_REPORT.md
//!   <out>/phase_a_summary.json

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;

#[derive(Parser, Debug)]
#[command(
    name = "check_value_targets",
    about = "Phase A: value target / cp 分布検査"
)]
struct Args {
    /// JSONL 入力ファイル
    #[arg(long)]
    input: PathBuf,

    /// 出力ディレクトリ (PHASE_A_REPORT.md / phase_a_summary.json を保存)
    #[arg(long)]
    output: PathBuf,

    /// 読み込む最大レコード数 (0 = 全件)
    #[arg(long, default_value = "0")]
    max_records: usize,

    /// CP_SCALE 候補 (カンマ区切り)
    #[arg(long, default_value = "600,800,1000,1200")]
    cp_scale_candidates: String,

    /// cp clamp (絶対値の上限)
    #[arg(long, default_value = "2000")]
    cp_clamp: i32,
}

#[derive(Debug, Deserialize)]
struct InputRecord {
    sfen: String,
    #[serde(rename = "move", default)]
    best_move: String,
    #[serde(default)]
    legal_moves: Vec<String>,
    value_cp: i32,
    #[serde(default)]
    result: i8,
    #[serde(default)]
    source: String,
    #[serde(default)]
    phase: String,
}

#[derive(Default)]
struct Counters {
    total: u64,
    parse_errors: u64,
    empty_sfen: u64,
    empty_bestmove: u64,
    empty_legal: u64,
    unique_sfens: u64,
    duplicate_sfens: u64,
    source_counts: HashMap<String, u64>,
    phase_counts: HashMap<String, u64>,
    result_counts: HashMap<i32, u64>,
}

#[inline]
fn cp_to_winrate(cp: i32, cp_clamp: i32, cp_scale: f32) -> f32 {
    let clamped = (cp.clamp(-cp_clamp, cp_clamp)) as f32;
    1.0 / (1.0 + (-clamped / cp_scale).exp())
}

/// 1D 分布統計
struct Stats {
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
    p1: f64,
    p5: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

fn compute_stats(values: &mut [f64]) -> Stats {
    if values.is_empty() {
        return Stats {
            min: f64::NAN,
            max: f64::NAN,
            mean: f64::NAN,
            std: f64::NAN,
            p1: f64::NAN,
            p5: f64::NAN,
            p50: f64::NAN,
            p95: f64::NAN,
            p99: f64::NAN,
        };
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let min = values[0];
    let max = values[n - 1];
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    let pct = |p: f64| -> f64 {
        let idx = ((p * (n - 1) as f64).round() as usize).min(n - 1);
        values[idx]
    };
    Stats {
        min,
        max,
        mean,
        std,
        p1: pct(0.01),
        p5: pct(0.05),
        p50: pct(0.50),
        p95: pct(0.95),
        p99: pct(0.99),
    }
}

/// CP_SCALE 候補ごとの簡易統計
struct ScaleScan {
    scale: f32,
    saturation_95: f64, // |target - 0.5| > 0.45 比率
    saturation_80: f64, // |target - 0.5| > 0.30 比率
    std: f64,
    mean: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output)
        .with_context(|| format!("出力ディレクトリ作成失敗: {:?}", args.output))?;

    let cp_scale_candidates: Vec<f32> = args
        .cp_scale_candidates
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    if cp_scale_candidates.is_empty() {
        anyhow::bail!("cp_scale_candidates が空です");
    }
    println!("CP_SCALE 候補: {:?}", cp_scale_candidates);

    // ストリーミング読み込み + 集計
    let file = File::open(&args.input)
        .with_context(|| format!("入力ファイルオープン失敗: {:?}", args.input))?;
    let reader = BufReader::with_capacity(8 * 1024 * 1024, file);

    let mut counters = Counters::default();
    let mut cps: Vec<f64> = Vec::new();
    // 各 CP_SCALE で target 値を蓄積
    let mut targets_by_scale: Vec<Vec<f64>> =
        (0..cp_scale_candidates.len()).map(|_| Vec::new()).collect();
    // 重複検出 (sfen を ハッシュ) — 完全一致のみ
    let mut sfen_set: HashMap<u64, u32> = HashMap::with_capacity(1_000_000);
    let max = if args.max_records == 0 {
        usize::MAX
    } else {
        args.max_records
    };

    let load_start = Instant::now();

    for line in reader.lines() {
        if counters.total as usize >= max {
            break;
        }
        let line = line.context("読み込みエラー")?;
        if line.is_empty() {
            continue;
        }
        let rec: InputRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => {
                counters.parse_errors += 1;
                continue;
            }
        };

        counters.total += 1;
        if rec.sfen.is_empty() {
            counters.empty_sfen += 1;
        }
        if rec.best_move.is_empty() {
            counters.empty_bestmove += 1;
        }
        if rec.legal_moves.is_empty() {
            counters.empty_legal += 1;
        }

        // 重複 (高速化のため hash のみで判定)
        let h = sfen_hash(&rec.sfen);
        let entry = sfen_set.entry(h).or_insert(0);
        if *entry == 0 {
            counters.unique_sfens += 1;
        } else {
            counters.duplicate_sfens += 1;
        }
        *entry += 1;

        // source / phase / result
        if !rec.source.is_empty() {
            *counters.source_counts.entry(rec.source.clone()).or_insert(0) += 1;
        }
        if !rec.phase.is_empty() {
            *counters.phase_counts.entry(rec.phase.clone()).or_insert(0) += 1;
        }
        *counters.result_counts.entry(rec.result as i32).or_insert(0) += 1;

        // cp / target 蓄積
        let cp_f = rec.value_cp as f64;
        cps.push(cp_f);

        for (i, &scale) in cp_scale_candidates.iter().enumerate() {
            let t = cp_to_winrate(rec.value_cp, args.cp_clamp, scale) as f64;
            targets_by_scale[i].push(t);
        }

        if counters.total % 200_000 == 0 {
            println!(
                "  {} records loaded ({:.1}s)",
                counters.total,
                load_start.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "読み込み完了: {} records ({:.1}s)",
        counters.total,
        load_start.elapsed().as_secs_f64()
    );

    // 相関計算は compute_stats (sort 破壊) の前に実施
    let chosen_idx = 0usize;
    let pearson = pearson_correlation(&cps, &targets_by_scale[chosen_idx]);

    // 統計計算 (cps を sort して破壊)
    let cp_stats = compute_stats(&mut cps);

    // CP_SCALE スキャン
    let mut scans: Vec<ScaleScan> = Vec::with_capacity(cp_scale_candidates.len());
    for (i, &scale) in cp_scale_candidates.iter().enumerate() {
        let targets = &mut targets_by_scale[i];
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut sat_95 = 0u64;
        let mut sat_80 = 0u64;
        for &t in targets.iter() {
            sum += t;
            sum_sq += t * t;
            let off = (t - 0.5).abs();
            if off > 0.45 {
                sat_95 += 1;
            }
            if off > 0.30 {
                sat_80 += 1;
            }
        }
        let n = targets.len() as f64;
        let mean = sum / n;
        let var = sum_sq / n - mean * mean;
        let std = var.max(0.0).sqrt();
        scans.push(ScaleScan {
            scale,
            saturation_95: sat_95 as f64 / n,
            saturation_80: sat_80 as f64 / n,
            std,
            mean,
        });
    }

    // 採用候補 (saturation_95 が中庸かつ std が大きい)
    let recommended = scans
        .iter()
        .min_by(|a, b| {
            // 飽和 < 5% を満たし、std 最大を選ぶ。満たさないなら飽和最小。
            let a_ok = a.saturation_95 < 0.05;
            let b_ok = b.saturation_95 < 0.05;
            match (a_ok, b_ok) {
                (true, true) => b.std.partial_cmp(&a.std).unwrap_or(std::cmp::Ordering::Equal),
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => a
                    .saturation_95
                    .partial_cmp(&b.saturation_95)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        })
        .map(|s| s.scale)
        .unwrap_or(800.0);

    // ----------------------------------------------------------------
    // PHASE_A_REPORT.md
    // ----------------------------------------------------------------
    let report_path = args.output.join("PHASE_A_REPORT.md");
    let mut md = String::new();
    md.push_str("# Phase A: Value Target / CP 分布検査\n\n");
    md.push_str(&format!("- Input: `{}`\n", args.input.display()));
    md.push_str(&format!("- Records loaded: {}\n", counters.total));
    md.push_str(&format!("- Parse errors: {}\n", counters.parse_errors));
    md.push_str(&format!("- Empty SFEN: {}\n", counters.empty_sfen));
    md.push_str(&format!("- Empty bestmove: {}\n", counters.empty_bestmove));
    md.push_str(&format!("- Empty legal_moves: {}\n", counters.empty_legal));
    md.push_str(&format!("- Unique SFEN (hash-based): {}\n", counters.unique_sfens));
    md.push_str(&format!("- Duplicate SFEN occurrences: {}\n", counters.duplicate_sfens));
    md.push_str(&format!(
        "- Duplicate ratio: {:.4}\n\n",
        counters.duplicate_sfens as f64 / counters.total.max(1) as f64
    ));

    md.push_str("## CP 分布 (raw)\n\n");
    md.push_str(&format!("| min | p1 | p5 | p50 | p95 | p99 | max | mean | std |\n"));
    md.push_str("|---|---|---|---|---|---|---|---|---|\n");
    md.push_str(&format!(
        "| {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} | {:.2} |\n\n",
        cp_stats.min,
        cp_stats.p1,
        cp_stats.p5,
        cp_stats.p50,
        cp_stats.p95,
        cp_stats.p99,
        cp_stats.max,
        cp_stats.mean,
        cp_stats.std,
    ));

    md.push_str("## CP_SCALE 候補比較\n\n");
    md.push_str(
        "`target = sigmoid(clamp(cp, ±2000) / cp_scale)` で saturation 率と std を評価。\n\n",
    );
    md.push_str("| cp_scale | mean | std | sat (>0.45) | sat (>0.30) |\n");
    md.push_str("|---|---|---|---|---|\n");
    for s in &scans {
        md.push_str(&format!(
            "| {:.0} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            s.scale, s.mean, s.std, s.saturation_95, s.saturation_80,
        ));
    }
    md.push_str(&format!("\n**推奨 CP_SCALE: {:.0}**\n", recommended));
    md.push_str("(saturation < 5% を満たす中で std 最大のものを選択)\n\n");

    md.push_str(&format!(
        "## CP ↔ Target Pearson 相関 (cp_scale={:.0})\n\n",
        cp_scale_candidates[chosen_idx]
    ));
    md.push_str(&format!("- r = {:.6}\n\n", pearson));

    md.push_str("## Source 別件数\n\n");
    md.push_str("| source | count |\n|---|---|\n");
    let mut sources: Vec<_> = counters.source_counts.iter().collect();
    sources.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in &sources {
        md.push_str(&format!("| {} | {} |\n", k, v));
    }
    md.push_str("\n");

    md.push_str("## Phase 別件数\n\n");
    md.push_str("| phase | count |\n|---|---|\n");
    let mut phases: Vec<_> = counters.phase_counts.iter().collect();
    phases.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in &phases {
        md.push_str(&format!("| {} | {} |\n", k, v));
    }
    md.push_str("\n");

    md.push_str("## game_result 別件数\n\n");
    md.push_str("| result | count |\n|---|---|\n");
    let mut results: Vec<_> = counters.result_counts.iter().collect();
    results.sort_by(|a, b| a.0.cmp(b.0));
    for (k, v) in &results {
        md.push_str(&format!("| {} | {} |\n", k, v));
    }
    md.push_str("\n");

    let mut f = File::create(&report_path)
        .with_context(|| format!("レポート書き出し失敗: {report_path:?}"))?;
    f.write_all(md.as_bytes())?;
    println!("PHASE_A_REPORT.md → {}", report_path.display());

    // ----------------------------------------------------------------
    // phase_a_summary.json
    // ----------------------------------------------------------------
    let summary_path = args.output.join("phase_a_summary.json");
    let summary = json!({
      "input": args.input.display().to_string(),
      "total_records": counters.total,
      "parse_errors": counters.parse_errors,
      "empty_sfen": counters.empty_sfen,
      "empty_bestmove": counters.empty_bestmove,
      "empty_legal_moves": counters.empty_legal,
      "duplicate_sfen_ratio": counters.duplicate_sfens as f64 / counters.total.max(1) as f64,
      "cp_stats": {
        "min": cp_stats.min,
        "max": cp_stats.max,
        "mean": cp_stats.mean,
        "std": cp_stats.std,
        "p1": cp_stats.p1,
        "p5": cp_stats.p5,
        "p50": cp_stats.p50,
        "p95": cp_stats.p95,
        "p99": cp_stats.p99,
      },
      "cp_scale_scan": scans.iter().map(|s| json!({
        "scale": s.scale,
        "mean": s.mean,
        "std": s.std,
        "saturation_above_0_45": s.saturation_95,
        "saturation_above_0_30": s.saturation_80,
      })).collect::<Vec<_>>(),
      "recommended_cp_scale": recommended,
      "cp_target_pearson_at_first_scale": pearson,
      "source_counts": counters.source_counts,
      "phase_counts": counters.phase_counts,
      "result_counts": counters.result_counts.iter()
        .map(|(k, v)| (k.to_string(), v)).collect::<HashMap<_,_>>(),
    });
    let mut sf = File::create(&summary_path)
        .with_context(|| format!("summary 書き出し失敗: {summary_path:?}"))?;
    sf.write_all(serde_json::to_string_pretty(&summary)?.as_bytes())?;
    println!("phase_a_summary.json → {}", summary_path.display());

    println!("\n=== Phase A 完了 ===");
    println!("  読み込み: {} records", counters.total);
    println!("  推奨 CP_SCALE: {:.0}", recommended);
    println!(
        "  cp p99: {:.0}, mean: {:.1}, std: {:.1}",
        cp_stats.p99, cp_stats.mean, cp_stats.std
    );

    Ok(())
}

/// 高速 SFEN ハッシュ (FNV-1a)
fn sfen_hash(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Pearson 相関係数
fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return f64::NAN;
    }
    let mean_x = xs.iter().take(n).sum::<f64>() / n as f64;
    let mean_y = ys.iter().take(n).sum::<f64>() / n as f64;
    let mut cov = 0.0f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom <= 0.0 { f64::NAN } else { cov / denom }
}
