//! Stage 4-Hybrid 学習用に JSONL から features tensor + labels を一括書き出す
//!
//! 出力 (バイナリ、 Python numpy で読み込み):
//!   features.bin           (N, 9728) i8
//!   policy_labels.bin      (N,) i32
//!   value_cps.bin          (N,) i32
//!   value_cps_aobannue.bin (N,) i32  - 任意 (jsonl に value_cp_aobannue があれば出力)
//!   results.bin            (N,) i8
//!   plys.bin               (N,) u16
//!   phase_idxs.bin         (N,) u8
//!   legal_offsets.bin      (N+1,) u32 (累積)
//!   legal_indices.bin      (sum_legal,) i32
//!   meta.json              {"n_records": N, "has_aobannue": bool, ...}

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;

use p2_nnue::features::{P2_FEATURE_DIM, extract_features};
use rshogi_core::position::Position;
use rshogi_core::types::Move;
use tools::dlshogi_features::make_move_label;

#[derive(Parser, Debug)]
#[command(name = "prepare_v3_tensors")]
struct Args {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value = "0")]
    max_records: usize,
}

#[derive(Deserialize)]
struct InputRecord {
    sfen: String,
    #[serde(rename = "move")]
    best_move: String,
    #[serde(default)]
    legal_moves: Vec<String>,
    value_cp: i32,
    /// AobaNNUE eval (任意)。 jsonl に存在すれば value_cps_aobannue.bin に出力。
    #[serde(default)]
    value_cp_aobannue: Option<i32>,
    #[serde(default)]
    result: i8,
    #[serde(default)]
    ply: u32,
    #[serde(default)]
    phase: String,
}

fn phase_to_idx(p: &str) -> u8 {
    match p {
        "opening" => 0,
        "middle" => 1,
        "midgame" => 1,
        "endgame" => 2,
        _ => 3, // unknown
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output)?;

    let f = File::open(&args.input)
        .with_context(|| format!("入力 JSONL オープン失敗: {:?}", args.input))?;
    let reader = BufReader::with_capacity(8 * 1024 * 1024, f);

    let max = if args.max_records == 0 {
        usize::MAX
    } else {
        args.max_records
    };

    // 出力 writer
    let mut features_w =
        BufWriter::with_capacity(8 * 1024 * 1024, File::create(args.output.join("features.bin"))?);
    let mut policy_labels_w = BufWriter::new(File::create(args.output.join("policy_labels.bin"))?);
    let mut value_cps_w = BufWriter::new(File::create(args.output.join("value_cps.bin"))?);
    let mut value_cps_aoba_w =
        BufWriter::new(File::create(args.output.join("value_cps_aobannue.bin"))?);
    let mut results_w = BufWriter::new(File::create(args.output.join("results.bin"))?);
    let mut plys_w = BufWriter::new(File::create(args.output.join("plys.bin"))?);
    let mut phase_idxs_w = BufWriter::new(File::create(args.output.join("phase_idxs.bin"))?);
    let mut legal_offsets_w = BufWriter::new(File::create(args.output.join("legal_offsets.bin"))?);
    let mut legal_indices_w = BufWriter::new(File::create(args.output.join("legal_indices.bin"))?);

    let mut n_records: u64 = 0;
    let mut n_skipped: u64 = 0;
    let mut n_aobannue: u64 = 0;
    let mut total_legal: u64 = 0;
    let start = Instant::now();

    // legal_offsets[0] = 0
    legal_offsets_w.write_all(&0u32.to_le_bytes())?;

    for line in reader.lines() {
        if n_records as usize >= max {
            break;
        }
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let rec: InputRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => {
                n_skipped += 1;
                continue;
            }
        };

        // Position パース
        let mut pos = Position::new();
        if pos.set_sfen(&rec.sfen).is_err() {
            n_skipped += 1;
            continue;
        }
        let color = pos.side_to_move();

        // best move label
        let best_mv = match Move::from_usi(&rec.best_move) {
            Some(m) => m,
            None => {
                n_skipped += 1;
                continue;
            }
        };
        let target_label = make_move_label(best_mv, color) as i32;

        // legal labels
        if rec.legal_moves.is_empty() {
            n_skipped += 1;
            continue;
        }
        let legal_labels: Vec<i32> = rec
            .legal_moves
            .iter()
            .filter_map(|s| Move::from_usi(s))
            .map(|mv| make_move_label(mv, color) as i32)
            .collect();
        if legal_labels.is_empty() || !legal_labels.iter().any(|&l| l == target_label) {
            n_skipped += 1;
            continue;
        }

        // features 抽出
        let input = extract_features(&pos);
        // i32 → i8 (binary 0/1)
        for v in input.piece_features.iter() {
            let b: i8 = if *v != 0 { 1 } else { 0 };
            features_w.write_all(&b.to_le_bytes())?;
        }

        // labels
        policy_labels_w.write_all(&target_label.to_le_bytes())?;
        value_cps_w.write_all(&rec.value_cp.to_le_bytes())?;
        let aoba_cp = rec.value_cp_aobannue.unwrap_or(i32::MIN); // Sentinel = データなし
        if rec.value_cp_aobannue.is_some() {
            n_aobannue += 1;
        }
        value_cps_aoba_w.write_all(&aoba_cp.to_le_bytes())?;
        results_w.write_all(&rec.result.to_le_bytes())?;
        plys_w.write_all(&(rec.ply as u16).to_le_bytes())?;
        phase_idxs_w.write_all(&[phase_to_idx(&rec.phase)])?;

        // legal indices
        for &lbl in &legal_labels {
            legal_indices_w.write_all(&lbl.to_le_bytes())?;
        }
        total_legal += legal_labels.len() as u64;
        legal_offsets_w.write_all(&(total_legal as u32).to_le_bytes())?;

        n_records += 1;
        if n_records.is_multiple_of(50_000) {
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "  {} records ({:.0} rec/s, {:.0}s elapsed, skipped={})",
                n_records,
                n_records as f64 / elapsed,
                elapsed,
                n_skipped,
            );
        }
    }

    features_w.flush()?;
    policy_labels_w.flush()?;
    value_cps_w.flush()?;
    value_cps_aoba_w.flush()?;
    results_w.flush()?;
    plys_w.flush()?;
    phase_idxs_w.flush()?;
    legal_offsets_w.flush()?;
    legal_indices_w.flush()?;

    let meta = json!({
      "n_records": n_records,
      "n_skipped": n_skipped,
      "feature_dim": P2_FEATURE_DIM,
      "total_legal": total_legal,
      "n_aobannue": n_aobannue,
      "has_aobannue": n_aobannue > 0,
      "elapsed_secs": start.elapsed().as_secs_f64(),
      "input": args.input.display().to_string(),
    });
    fs::write(args.output.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;

    println!("\n=== prepare_v3_tensors 完了 ===");
    println!("  records: {}", n_records);
    println!("  skipped: {}", n_skipped);
    println!("  total_legal: {}", total_legal);
    println!("  output: {}", args.output.display());
    Ok(())
}
