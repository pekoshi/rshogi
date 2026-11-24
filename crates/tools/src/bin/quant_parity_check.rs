//! Phase D: quantization parity check
//!
//! 学習中の f32 weights (checkpoint .bin) と、量子化往復後の f32 weights
//! (.p2nn → P2Network → F32Weights::from_network) を同一 SFEN で forward 比較し、
//! logit 差分の分布を計測する。
//!
//! 出力:
//!   <out>/QUANT_PARITY_REPORT.md
//!   <out>/quant_parity_summary.json
//!   exit code 2: max_relative_diff 超過 (REJECT)

use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use serde_json::json;

use p2_nnue::features::extract_features;
use p2_nnue::network::P2Network;
use p2_nnue::types::P2_INPUT_DIM;
use rshogi_core::position::Position;
use tools::p2_train::checkpoint::load_checkpoint;
use tools::p2_train::data_loader::{JsonlRecord, load_jsonl_records};
use tools::p2_train::forward_f32::{F32Weights, forward};
use tools::p2_train::optimizer::AdamWConfig;

#[derive(Parser, Debug)]
#[command(
    name = "quant_parity_check",
    about = "Phase D: f32 vs 量子化後 f32 の出力差分計測"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,

    /// 学習中 f32 重みを持つ checkpoint (.bin)
    #[arg(long)]
    checkpoint: PathBuf,

    /// 量子化済み (.p2nn)
    #[arg(long)]
    model: PathBuf,

    /// SFEN サンプル元 (validation JSONL)
    #[arg(long)]
    valid_jsonl: Option<PathBuf>,

    #[arg(long)]
    output: PathBuf,

    /// サンプル数 override
    #[arg(long)]
    num_positions: Option<usize>,

    #[arg(long, default_value = "0")]
    seed: u64,
}

#[derive(Deserialize)]
struct Config {
    data: DataCfg,
    value_target: ValueTargetCfg,
    quant_parity: QuantParityCfg,
}

#[derive(Deserialize)]
struct DataCfg {
    train_jsonl: PathBuf,
}

#[derive(Deserialize)]
struct ValueTargetCfg {
    cp_scale: f32,
}

#[derive(Deserialize)]
struct QuantParityCfg {
    num_positions: usize,
    max_relative_diff: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output)?;

    let cfg_text = fs::read_to_string(&args.config)?;
    let cfg: Config = toml::from_str(&cfg_text)?;
    let n = args.num_positions.unwrap_or(cfg.quant_parity.num_positions);
    let max_rel = cfg.quant_parity.max_relative_diff;
    let cp_scale = cfg.value_target.cp_scale;

    println!("Loading checkpoint (training f32): {:?}", args.checkpoint);
    let (_meta, train_weights, _opt) = load_checkpoint(&args.checkpoint, AdamWConfig::default())
        .with_context(|| format!("checkpoint ロード失敗: {:?}", args.checkpoint))?;

    println!("Loading quantized .p2nn: {:?}", args.model);
    let f = File::open(&args.model)?;
    let mut reader = BufReader::new(f);
    let net = P2Network::load(&mut reader)?;
    let quant_weights_flat = net.weights_to_f32();

    let train_w = F32Weights::from_flat(&train_weights);
    let quant_w = F32Weights::from_flat(&quant_weights_flat);

    // SFEN サンプル
    let valid_jsonl = args.valid_jsonl.clone().unwrap_or_else(|| cfg.data.train_jsonl.clone());
    println!("Sampling {} positions from {}", n, valid_jsonl.display());
    let records = load_jsonl_records(&valid_jsonl, Some(n * 3))?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut shuffled: Vec<JsonlRecord> = records;
    shuffled.shuffle(&mut rng);

    let mut diffs: Vec<f32> = Vec::with_capacity(n);
    let mut rel_diffs: Vec<f32> = Vec::with_capacity(n);
    let mut samples_used = 0usize;

    for rec in shuffled.iter() {
        if samples_used >= n {
            break;
        }
        let mut pos = Position::new();
        if pos.set_sfen(&rec.sfen).is_err() {
            continue;
        }
        let input = extract_features(&pos);
        let mut features = [0.0f32; P2_INPUT_DIM];
        for (i, &v) in input.piece_features.iter().enumerate() {
            features[i] = v as f32;
        }
        let s_train = forward(&train_w, &features);
        let s_quant = forward(&quant_w, &features);
        let diff = s_train.logit - s_quant.logit;
        let rel = diff.abs() / s_train.logit.abs().max(1.0);
        diffs.push(diff);
        rel_diffs.push(rel);
        samples_used += 1;
    }

    if diffs.is_empty() {
        anyhow::bail!("有効サンプルが 0 件");
    }

    let stats = stats_of(&diffs);
    let rel_stats = stats_of(&rel_diffs);

    println!(
        "  diff:     mean={:.4} std={:.4} max={:.4}",
        stats.mean, stats.std, stats.max_abs
    );
    println!(
        "  rel_diff: mean={:.4} std={:.4} max={:.4}",
        rel_stats.mean, rel_stats.std, rel_stats.max_abs
    );

    let pass = rel_stats.max_abs <= max_rel as f64;
    println!("  Quantization parity: {}", if pass { "PASS" } else { "REJECT" });

    // Report
    let mut md = String::new();
    md.push_str("# Phase D: Quantization Parity Check\n\n");
    md.push_str(&format!("- Checkpoint: `{}`\n", args.checkpoint.display()));
    md.push_str(&format!("- Model:      `{}`\n", args.model.display()));
    md.push_str(&format!("- cp_scale:   {}\n", cp_scale));
    md.push_str(&format!("- Samples:    {}\n", samples_used));
    md.push_str(&format!("- Threshold:  rel_diff <= {}\n", max_rel));
    md.push_str(&format!("- Verdict:    **{}**\n\n", if pass { "PASS" } else { "REJECT" }));

    md.push_str("## logit diff (train_f32 - quant_f32)\n\n");
    md.push_str("| mean | std | max(abs) | min | max | p50 | p95 | p99 |\n");
    md.push_str("|---|---|---|---|---|---|---|---|\n");
    md.push_str(&format!(
        "| {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n\n",
        stats.mean, stats.std, stats.max_abs, stats.min, stats.max, stats.p50, stats.p95, stats.p99
    ));

    md.push_str("## relative |diff| / max(1, |train_logit|)\n\n");
    md.push_str("| mean | std | max | p50 | p95 | p99 |\n");
    md.push_str("|---|---|---|---|---|---|\n");
    md.push_str(&format!(
        "| {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n\n",
        rel_stats.mean,
        rel_stats.std,
        rel_stats.max_abs,
        rel_stats.p50,
        rel_stats.p95,
        rel_stats.p99
    ));

    let report_path = args.output.join("QUANT_PARITY_REPORT.md");
    let mut f = File::create(&report_path)?;
    f.write_all(md.as_bytes())?;

    let summary_path = args.output.join("quant_parity_summary.json");
    let summary = json!({
      "checkpoint": args.checkpoint.display().to_string(),
      "model": args.model.display().to_string(),
      "samples": samples_used,
      "max_relative_diff_threshold": max_rel,
      "verdict": if pass { "PASS" } else { "REJECT" },
      "diff": {
        "mean": stats.mean, "std": stats.std, "max_abs": stats.max_abs,
        "min": stats.min, "max": stats.max,
        "p50": stats.p50, "p95": stats.p95, "p99": stats.p99,
      },
      "relative_diff": {
        "mean": rel_stats.mean, "std": rel_stats.std, "max_abs": rel_stats.max_abs,
        "p50": rel_stats.p50, "p95": rel_stats.p95, "p99": rel_stats.p99,
      },
    });
    let mut sf = File::create(&summary_path)?;
    sf.write_all(serde_json::to_string_pretty(&summary)?.as_bytes())?;

    println!("QUANT_PARITY_REPORT.md → {}", report_path.display());
    println!("quant_parity_summary.json → {}", summary_path.display());

    if !pass {
        std::process::exit(2);
    }
    Ok(())
}

struct DiffStats {
    mean: f64,
    std: f64,
    max_abs: f64,
    min: f64,
    max: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

fn stats_of(values: &[f32]) -> DiffStats {
    let mut sorted: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    let var = sorted.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    let max_abs = values.iter().map(|v| v.abs() as f64).fold(0.0f64, f64::max);
    let pct = |p: f64| -> f64 {
        let idx = ((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
        sorted[idx]
    };
    DiffStats {
        mean,
        std,
        max_abs,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        p50: pct(0.50),
        p95: pct(0.95),
        p99: pct(0.99),
    }
}
