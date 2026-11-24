//! Phase E: sanity check / adoption gate
//!
//! 学習済み .p2nn を読み込み、以下を検査:
//!   - 固定 7 局面の value (logit / winrate / cp_estimate) と policy top-5
//!   - validation set から N 件ランダム抽出した eval 分布
//!   - 採用ゲート判定 (startpos eval、std 範囲、outlier ratio)
//!
//! 出力:
//!   <out>/SANITY_REPORT.md
//!   <out>/sanity_summary.json
//!   <out>/ADOPT.txt or REJECT.txt

use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use serde_json::json;

use p2_nnue::features::extract_features;
use p2_nnue::network::P2Network;
use p2_nnue::types::P2_INPUT_DIM;
use rshogi_core::movegen::{MoveList, generate_legal_all};
use rshogi_core::position::Position;
use tools::dlshogi_features::make_move_label;
use tools::p2_train::data_loader::{JsonlRecord, load_jsonl_records};
use tools::p2_train::forward_f32::{F32Weights, forward, sigmoid};

#[derive(Parser, Debug)]
#[command(
    name = "eval_sanity_check",
    about = "Phase E: sanity check / adoption gate"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,

    /// 検査するモデル (.p2nn)
    #[arg(long)]
    model: PathBuf,

    /// validation JSONL (random eval 抽出元)。 省略時は config の train_jsonl を使用
    #[arg(long)]
    valid_jsonl: Option<PathBuf>,

    /// 出力先ディレクトリ
    #[arg(long)]
    output: PathBuf,

    /// random eval サンプル数 override
    #[arg(long)]
    random_samples: Option<usize>,

    /// 採用判定の seed
    #[arg(long, default_value = "0")]
    seed: u64,
}

#[derive(Deserialize)]
struct Config {
    data: DataCfg,
    value_target: ValueTargetCfg,
    sanity: SanityCfg,
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
struct SanityCfg {
    fixed_positions: Vec<String>,
    random_sample_size: usize,
    gate: GateCfg,
}

#[derive(Deserialize)]
struct GateCfg {
    startpos_cp_min: f32,
    startpos_cp_max: f32,
    random_cp_std_min: f32,
    random_cp_std_max: f32,
    extreme_outlier_ratio_max: f32,
}

struct PositionResult {
    sfen: String,
    logit: f32,
    winrate: f32,
    cp_est: f32,
    outlook_logit: f32,
    outlook_winrate: f32,
    top5: Vec<(String, f32, f32)>, // (usi, masked_prob, raw_logit)
    best_usi: String,
    legal_count: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output)
        .with_context(|| format!("出力ディレクトリ作成失敗: {:?}", args.output))?;

    let cfg_text = fs::read_to_string(&args.config)
        .with_context(|| format!("config 読み込み失敗: {:?}", args.config))?;
    let cfg: Config = toml::from_str(&cfg_text).context("TOML パース失敗")?;
    let cp_scale = cfg.value_target.cp_scale;
    let valid_jsonl = args.valid_jsonl.clone().unwrap_or_else(|| cfg.data.train_jsonl.clone());
    let n_random = args.random_samples.unwrap_or(cfg.sanity.random_sample_size);

    println!("Loading model: {:?}", args.model);
    let f =
        File::open(&args.model).with_context(|| format!(".p2nn オープン失敗: {:?}", args.model))?;
    let mut reader = BufReader::new(f);
    let net = P2Network::load(&mut reader).context(".p2nn ロード失敗")?;
    let weights = F32Weights::from_network(&net);

    // 固定局面評価
    println!("\n=== Fixed positions ===");
    let mut fixed_results: Vec<PositionResult> = Vec::new();
    for sfen in &cfg.sanity.fixed_positions {
        match eval_position(sfen, &weights, cp_scale) {
            Some(r) => {
                println!(
                    "  cp={:+.0} winrate={:.3} outlook={:.3} best={} legal={}",
                    r.cp_est, r.winrate, r.outlook_winrate, r.best_usi, r.legal_count
                );
                fixed_results.push(r);
            }
            None => {
                eprintln!("  [skip] sfen parse failed: {sfen}");
            }
        }
    }

    if fixed_results.is_empty() {
        bail!("固定局面の評価に全件失敗しました");
    }

    // 通常 startpos は最初に置いておく想定
    let startpos = &fixed_results[0];

    // ランダム局面評価
    println!("\n=== Random positions from {} ===", valid_jsonl.display());
    let records = load_jsonl_records(&valid_jsonl, Some(n_random * 3))
        .context("validation JSONL 読み込み失敗")?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut shuffled: Vec<JsonlRecord> = records;
    shuffled.shuffle(&mut rng);

    let mut random_logits: Vec<f32> = Vec::with_capacity(n_random);
    let mut random_cps: Vec<f32> = Vec::with_capacity(n_random);
    for rec in shuffled.iter().take(n_random * 3) {
        if random_cps.len() >= n_random {
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
        let state = forward(&weights, &features);
        random_logits.push(state.logit);
        random_cps.push(state.logit * cp_scale);
    }
    println!("  collected {} samples", random_cps.len());

    let stats = compute_basic_stats(&random_cps);
    let extreme_count = random_cps.iter().filter(|&&v| v.abs() > 3000.0).count() as f32;
    let extreme_ratio = if random_cps.is_empty() {
        0.0
    } else {
        extreme_count / random_cps.len() as f32
    };

    println!(
        "  cp distribution: mean={:.1} std={:.1} p1={:.1} p99={:.1} extreme(|cp|>3000)={:.3}",
        stats.mean, stats.std, stats.p1, stats.p99, extreme_ratio
    );

    // ----------------------------------------------------------------
    // Adoption gate
    // ----------------------------------------------------------------
    let mut gate_failures: Vec<String> = Vec::new();
    if startpos.cp_est < cfg.sanity.gate.startpos_cp_min
        || startpos.cp_est > cfg.sanity.gate.startpos_cp_max
    {
        gate_failures.push(format!(
            "startpos cp={:.1} 範囲外 [{}, {}]",
            startpos.cp_est, cfg.sanity.gate.startpos_cp_min, cfg.sanity.gate.startpos_cp_max
        ));
    }
    if stats.std < cfg.sanity.gate.random_cp_std_min as f64 {
        gate_failures.push(format!(
            "random cp std={:.1} 下限 {} 未満 (狭すぎ)",
            stats.std, cfg.sanity.gate.random_cp_std_min
        ));
    }
    if stats.std > cfg.sanity.gate.random_cp_std_max as f64 {
        gate_failures.push(format!(
            "random cp std={:.1} 上限 {} 超過 (暴走)",
            stats.std, cfg.sanity.gate.random_cp_std_max
        ));
    }
    if extreme_ratio > cfg.sanity.gate.extreme_outlier_ratio_max {
        gate_failures.push(format!(
            "|cp|>3000 比率 {:.3} 上限 {:.3} 超過",
            extreme_ratio, cfg.sanity.gate.extreme_outlier_ratio_max
        ));
    }

    let adopt = gate_failures.is_empty();
    println!("\n=== Adoption gate ===");
    if adopt {
        println!("  ADOPT: 全 gate 通過");
    } else {
        println!("  REJECT:");
        for f in &gate_failures {
            println!("    - {}", f);
        }
    }

    // ----------------------------------------------------------------
    // Reports
    // ----------------------------------------------------------------
    let report_path = args.output.join("SANITY_REPORT.md");
    let mut md = String::new();
    md.push_str("# Phase E: Sanity Check / Adoption Gate\n\n");
    md.push_str(&format!("- Model: `{}`\n", args.model.display()));
    md.push_str(&format!("- Config: `{}`\n", args.config.display()));
    md.push_str(&format!("- cp_scale: {}\n", cp_scale));
    md.push_str(&format!("- Valid JSONL: `{}`\n", valid_jsonl.display()));
    md.push_str(&format!("- Random samples: {} (collected)\n\n", random_cps.len()));

    md.push_str("## Adoption verdict\n\n");
    md.push_str(if adopt {
        "**ADOPT**\n\n"
    } else {
        "**REJECT**\n\n"
    });
    if !gate_failures.is_empty() {
        md.push_str("Failure reasons:\n");
        for f in &gate_failures {
            md.push_str(&format!("- {}\n", f));
        }
        md.push_str("\n");
    }

    md.push_str("## Fixed positions\n\n");
    md.push_str("| # | value_logit | value_wr | cp_est | outlook_wr | best_usi | top-5 |\n");
    md.push_str("|---|---|---|---|---|---|---|\n");
    for (i, r) in fixed_results.iter().enumerate() {
        let top5_str = r
            .top5
            .iter()
            .map(|(u, p, l)| format!("{u}: p={p:.3} l={l:.2}"))
            .collect::<Vec<_>>()
            .join("<br>");
        md.push_str(&format!(
            "| {} | {:.3} | {:.4} | {:+.1} | {:.4} | {} | {} |\n",
            i, r.logit, r.winrate, r.cp_est, r.outlook_winrate, r.best_usi, top5_str
        ));
    }
    md.push_str("\n");

    md.push_str("Fixed SFENs:\n\n");
    for (i, r) in fixed_results.iter().enumerate() {
        md.push_str(&format!("- [{}] `{}`\n", i, r.sfen));
    }
    md.push_str("\n");

    md.push_str("## Random eval distribution (cp)\n\n");
    md.push_str(&format!(
        "| count | min | p1 | p5 | p50 | p95 | p99 | max | mean | std | extreme>3000 |\n"
    ));
    md.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");
    md.push_str(&format!(
        "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.4} |\n",
        random_cps.len(),
        stats.min,
        stats.p1,
        stats.p5,
        stats.p50,
        stats.p95,
        stats.p99,
        stats.max,
        stats.mean,
        stats.std,
        extreme_ratio
    ));

    let mut f = File::create(&report_path)?;
    f.write_all(md.as_bytes())?;
    println!("\nSANITY_REPORT.md → {}", report_path.display());

    // JSON summary
    let summary = json!({
      "model": args.model.display().to_string(),
      "cp_scale": cp_scale,
      "adopt": adopt,
      "gate_failures": gate_failures,
      "startpos": {
        "logit": startpos.logit,
        "winrate": startpos.winrate,
        "cp_est": startpos.cp_est,
        "outlook_logit": startpos.outlook_logit,
        "outlook_winrate": startpos.outlook_winrate,
        "best_usi": startpos.best_usi,
      },
      "fixed_positions": fixed_results.iter().map(|r| json!({
        "sfen": r.sfen,
        "logit": r.logit,
        "winrate": r.winrate,
        "cp_est": r.cp_est,
        "outlook_logit": r.outlook_logit,
        "outlook_winrate": r.outlook_winrate,
        "best_usi": r.best_usi,
        "legal_count": r.legal_count,
        "top5": r.top5.iter().map(|(u,p,l)| json!({"usi": u, "prob": p, "logit": l})).collect::<Vec<_>>(),
      })).collect::<Vec<_>>(),
      "random_distribution": {
        "count": random_cps.len(),
        "min": stats.min, "max": stats.max, "mean": stats.mean, "std": stats.std,
        "p1": stats.p1, "p5": stats.p5, "p50": stats.p50, "p95": stats.p95, "p99": stats.p99,
        "extreme_ratio_above_3000cp": extreme_ratio,
      },
      "gate_thresholds": {
        "startpos_cp_min": cfg.sanity.gate.startpos_cp_min,
        "startpos_cp_max": cfg.sanity.gate.startpos_cp_max,
        "random_cp_std_min": cfg.sanity.gate.random_cp_std_min,
        "random_cp_std_max": cfg.sanity.gate.random_cp_std_max,
        "extreme_outlier_ratio_max": cfg.sanity.gate.extreme_outlier_ratio_max,
      },
    });
    let summary_path = args.output.join("sanity_summary.json");
    let mut sf = File::create(&summary_path)?;
    sf.write_all(serde_json::to_string_pretty(&summary)?.as_bytes())?;
    println!("sanity_summary.json → {}", summary_path.display());

    // Adoption marker file
    let marker_name = if adopt { "ADOPT.txt" } else { "REJECT.txt" };
    let marker_path = args.output.join(marker_name);
    let mut mf = File::create(&marker_path)?;
    mf.write_all(format!("model: {}\n", args.model.display()).as_bytes())?;
    for f in &gate_failures {
        mf.write_all(format!("- {}\n", f).as_bytes())?;
    }
    println!("{} → {}", marker_name, marker_path.display());

    if !adopt {
        std::process::exit(2);
    }
    Ok(())
}

fn eval_position(sfen: &str, weights: &F32Weights, cp_scale: f32) -> Option<PositionResult> {
    let mut pos = Position::new();
    pos.set_sfen(sfen).ok()?;
    let color = pos.side_to_move();

    let input = extract_features(&pos);
    let mut features = [0.0f32; P2_INPUT_DIM];
    for (i, &v) in input.piece_features.iter().enumerate() {
        features[i] = v as f32;
    }
    let state = forward(weights, &features);

    let logit = state.logit;
    let winrate = sigmoid(logit);
    let cp_est = logit * cp_scale;
    let outlook_logit = state.outlook_logit;
    let outlook_winrate = sigmoid(outlook_logit);

    // 合法手の policy logits を集めて softmax
    let mut legal = MoveList::new();
    generate_legal_all(&pos, &mut legal);

    let mut entries: Vec<(usize, f32, String)> = Vec::with_capacity(legal.len());
    for &mv in legal.iter() {
        let label = make_move_label(mv, color);
        let raw = state.policy_logits[label];
        entries.push((label, raw, mv.to_usi()));
    }

    // softmax over legal
    let mut max_l = f32::NEG_INFINITY;
    for &(_, l, _) in &entries {
        if l > max_l {
            max_l = l;
        }
    }
    let mut exps: Vec<f32> = entries.iter().map(|&(_, l, _)| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }

    // top 5
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| exps[b].partial_cmp(&exps[a]).unwrap_or(std::cmp::Ordering::Equal));

    let top5: Vec<(String, f32, f32)> = order
        .iter()
        .take(5)
        .map(|&i| (entries[i].2.clone(), exps[i], entries[i].1))
        .collect();

    let best_usi = top5.first().map(|t| t.0.clone()).unwrap_or_else(|| "(none)".to_string());

    Some(PositionResult {
        sfen: sfen.to_string(),
        logit,
        winrate,
        cp_est,
        outlook_logit,
        outlook_winrate,
        top5,
        best_usi,
        legal_count: legal.len(),
    })
}

struct BasicStats {
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

fn compute_basic_stats(values: &[f32]) -> BasicStats {
    if values.is_empty() {
        return BasicStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            p1: 0.0,
            p5: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
        };
    }
    let mut sorted: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let var = sorted.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    let pct = |p: f64| -> f64 {
        let idx = ((p * (n - 1) as f64).round() as usize).min(n - 1);
        sorted[idx]
    };
    BasicStats {
        min: sorted[0],
        max: sorted[n - 1],
        mean,
        std,
        p1: pct(0.01),
        p5: pct(0.05),
        p50: pct(0.50),
        p95: pct(0.95),
        p99: pct(0.99),
    }
}
