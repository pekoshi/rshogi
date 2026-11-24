//! jsonl_to_psv_dakai - tournament jsonl から「劣勢局面 + 実戦 outcome」を抽出して PSV 化
//!
//! 手筋打開 (じわ負け打開) 学習 pilot 用。
//! teacher eval ではなく実戦 outcome (WDL) を学習信号にするため、
//! 各 move 局面の side_to_move 視点の score_cp が劣勢帯にあるものを抽出し、
//! その対局の最終 outcome を side_to_move 視点の game_result (1/0/-1) として埋める。
//!
//! - PSV.score       = move の eval.score_cp (手番側視点、 i16 clamp)
//! - PSV.game_result = 1 (手番側勝ち) / 0 (引分) / -1 (負け)
//! - PSV.game_ply    = move の ply
//! - PSV.move16      = 0 (未使用)
//!
//! trainer 側で `--wdl` を高め (outcome 支配) にして fine-tune する想定。
//!
//! # 使用例
//! ```bash
//! cargo run --release -p tools --bin jsonl_to_psv_dakai -- \
//!   --input "runs/dakai_pilot/gen1/*.jsonl" \
//!   --output runs/dakai_pilot/dakai.psv \
//!   --cp-min -700 --cp-max -100
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use rshogi_core::position::Position;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use tools::packed_sfen::{PackedSfenValue, pack_position};

#[derive(Parser)]
#[command(
    name = "jsonl_to_psv_dakai",
    about = "tournament jsonl から劣勢局面 + 実戦 outcome を抽出して PSV 化"
)]
struct Cli {
    /// 入力 jsonl の glob パターン (複数可)
    #[arg(long, num_args = 1..)]
    input: Vec<String>,
    /// 出力 PSV (.bin)
    #[arg(long)]
    output: PathBuf,
    /// 抽出する score_cp 下限 (手番側視点、 default -700)
    #[arg(long, default_value_t = -700)]
    cp_min: i32,
    /// 抽出する score_cp 上限 (手番側視点、 default -100)
    #[arg(long, default_value_t = -100)]
    cp_max: i32,
}

#[derive(Deserialize)]
struct Record {
    #[serde(rename = "type")]
    kind: String,
    game_id: u32,
    // move 用
    ply: Option<u32>,
    side_to_move: Option<char>,
    sfen_before: Option<String>,
    eval: Option<EvalField>,
    // result 用
    outcome: Option<String>,
}

#[derive(Deserialize)]
struct EvalField {
    score_cp: Option<i32>,
}

/// side_to_move 視点の game_result を返す。1=勝ち, 0=引分, -1=負け。
fn stm_result(outcome: &str, side_to_move: char) -> i8 {
    match outcome {
        "draw" => 0,
        "black_win" => {
            if side_to_move == 'b' {
                1
            } else {
                -1
            }
        }
        "white_win" => {
            if side_to_move == 'w' {
                1
            } else {
                -1
            }
        }
        _ => 0,
    }
}

fn expand_inputs(patterns: &[String]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for pat in patterns {
        let mut matched = false;
        for entry in glob::glob(pat).with_context(|| format!("bad glob: {pat}"))? {
            let path = entry?;
            if path.is_file() {
                files.push(path);
                matched = true;
            }
        }
        if !matched {
            // glob にマッチしない場合は素のパスとして扱う
            let p = PathBuf::from(pat);
            if p.is_file() {
                files.push(p);
            }
        }
    }
    Ok(files)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let files = expand_inputs(&cli.input)?;
    if files.is_empty() {
        anyhow::bail!("no input files matched: {:?}", cli.input);
    }

    let mut writer = BufWriter::new(File::create(&cli.output)?);

    // 統計
    let mut total_moves: u64 = 0;
    let mut total_results: u64 = 0;
    let mut disadv_moves: u64 = 0; // 劣勢帯に入った move 数
    let mut written: u64 = 0; // PSV 書き出し数
    let mut recovery_examples: u64 = 0; // 劣勢→そのまま手番側が勝った例
    let mut parse_errs: u64 = 0;

    for file in &files {
        // 1 pass 目: game_id -> outcome を集める
        let mut outcomes: HashMap<u32, String> = HashMap::new();
        let reader = BufReader::new(File::open(file)?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(rec) = serde_json::from_str::<Record>(&line) {
                if rec.kind == "result" {
                    total_results += 1;
                    if let Some(oc) = rec.outcome {
                        outcomes.insert(rec.game_id, oc);
                    }
                }
            }
        }

        // 2 pass 目: move を走査して劣勢局面を PSV 化
        let reader = BufReader::new(File::open(file)?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let rec: Record = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(_) => {
                    parse_errs += 1;
                    continue;
                }
            };
            if rec.kind != "move" {
                continue;
            }
            total_moves += 1;

            let (Some(stm), Some(sfen), Some(eval)) =
                (rec.side_to_move, rec.sfen_before.as_ref(), rec.eval.as_ref())
            else {
                continue;
            };
            let Some(cp) = eval.score_cp else {
                continue;
            };
            // 劣勢帯フィルタ (手番側視点)
            if cp < cli.cp_min || cp > cli.cp_max {
                continue;
            }
            disadv_moves += 1;

            let Some(outcome) = outcomes.get(&rec.game_id) else {
                continue;
            };
            let result = stm_result(outcome, stm);
            if result > 0 {
                recovery_examples += 1;
            }

            // SFEN を pack
            let mut pos = Position::default();
            if pos.set_sfen(sfen).is_err() {
                parse_errs += 1;
                continue;
            }
            let game_ply = rec.ply.unwrap_or_else(|| {
                sfen.split_whitespace().nth(3).and_then(|s| s.parse::<u32>().ok()).unwrap_or(1)
            }) as u16;
            let packed = pack_position(&pos);
            let psv = PackedSfenValue {
                sfen: packed,
                score: cp.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                move16: 0,
                game_ply,
                game_result: result,
                padding: 0,
            };
            writer.write_all(&psv.to_bytes())?;
            written += 1;
        }
    }

    writer.flush()?;

    eprintln!("=== jsonl_to_psv_dakai 完了 ===");
    eprintln!("入力ファイル数: {}", files.len());
    eprintln!("総 move 行: {total_moves}");
    eprintln!("総 result 行: {total_results}");
    eprintln!("劣勢帯 [{}, {}] move 数: {disadv_moves}", cli.cp_min, cli.cp_max);
    eprintln!("PSV 書き出し数: {written}");
    eprintln!(
        "うち recovery 例 (劣勢→手番側勝ち): {recovery_examples} ({:.1}%)",
        if written > 0 {
            recovery_examples as f64 / written as f64 * 100.0
        } else {
            0.0
        }
    );
    eprintln!("parse error: {parse_errs}");

    Ok(())
}
