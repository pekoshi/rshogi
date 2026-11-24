//! sfen_to_psv - SFEN リスト (.txt / .txt.gz) を PSV (.bin) に変換
//!
//! distillation pipeline 用、 score / move / result は 0 init。
//! rescore_psv に input として渡して 外部 USI engine で rescore する用途。

use anyhow::{Context, Result};
use clap::Parser;
use flate2::read::GzDecoder;
use rshogi_core::position::Position;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use tools::packed_sfen::{PackedSfenValue, pack_position};

#[derive(Parser)]
#[command(
    name = "sfen_to_psv",
    about = "SFEN list を PSV (.bin) に変換 (score=0 init)"
)]
struct Cli {
    /// 入力 SFEN list (.txt or .txt.gz、 1 行 1 SFEN)
    #[arg(long)]
    input: PathBuf,
    /// 出力 PSV (.bin)
    #[arg(long)]
    output: PathBuf,
}

fn open_reader(path: &PathBuf) -> Result<Box<dyn BufRead>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let gz = path.extension().and_then(|s| s.to_str()) == Some("gz");
    let r: Box<dyn Read> = if gz {
        Box::new(GzDecoder::new(f))
    } else {
        Box::new(f)
    };
    Ok(Box::new(BufReader::new(r)))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let reader = open_reader(&cli.input)?;
    let mut writer = BufWriter::new(File::create(&cli.output)?);
    let mut count: u64 = 0;
    let mut errs: u64 = 0;
    for line in reader.lines() {
        let sfen = line?;
        let sfen = sfen.trim();
        if sfen.is_empty() {
            continue;
        }
        let mut pos = Position::default();
        if pos.set_sfen(sfen).is_err() {
            errs += 1;
            continue;
        }
        let game_ply =
            sfen.split_whitespace().nth(3).and_then(|s| s.parse::<u16>().ok()).unwrap_or(1);
        let packed = pack_position(&pos);
        let psv = PackedSfenValue {
            sfen: packed,
            score: 0,
            move16: 0,
            game_ply,
            game_result: 0,
            padding: 0,
        };
        writer.write_all(&psv.to_bytes())?;
        count += 1;
        if count.is_multiple_of(100_000) {
            eprintln!("  packed {count} SFENs ({errs} parse errors)");
        }
    }
    writer.flush()?;
    eprintln!("Done: {count} PSV records written ({errs} parse errors)");
    Ok(())
}
