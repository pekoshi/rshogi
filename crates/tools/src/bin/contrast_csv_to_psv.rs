//! contrast_csv_to_psv - child_sfen / teacher_child_score CSV を score付き PSV に変換

use anyhow::{Context, Result, bail};
use clap::Parser;
use rshogi_core::position::Position;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use tools::packed_sfen::{PackedSfenValue, pack_position};

#[derive(Parser)]
#[command(name = "contrast_csv_to_psv")]
struct Cli {
    /// confirm_floodgate_contrasts.py の出力CSV
    #[arg(long)]
    input: PathBuf,
    /// select_strong_contrast_roots.py の出力CSV。指定時は該当rootのみ出力
    #[arg(long)]
    roots: Option<PathBuf>,
    /// 出力 PSV
    #[arg(long)]
    output: PathBuf,
}

fn header_index(header: &[String], name: &str) -> Result<usize> {
    header
        .iter()
        .position(|field| field == name)
        .with_context(|| format!("missing CSV column: {name}"))
}

fn load_roots(path: &PathBuf) -> Result<HashSet<String>> {
    let mut roots = HashSet::new();
    let reader =
        BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut lines = reader.lines();
    let header: Vec<String> = lines
        .next()
        .transpose()?
        .context("empty roots CSV")?
        .split(',')
        .map(str::to_owned)
        .collect();
    let sfen_idx = header_index(&header, "sfen")?;
    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split(',').collect();
        if let Some(sfen) = fields.get(sfen_idx) {
            roots.insert((*sfen).to_owned());
        }
    }
    Ok(roots)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let roots = match &cli.roots {
        Some(path) => Some(load_roots(path)?),
        None => None,
    };
    let reader = BufReader::new(
        File::open(&cli.input).with_context(|| format!("open {}", cli.input.display()))?,
    );
    let mut lines = reader.lines();
    let header: Vec<String> = lines
        .next()
        .transpose()?
        .context("empty contrast CSV")?
        .split(',')
        .map(str::to_owned)
        .collect();
    let root_idx = header_index(&header, "root_sfen")?;
    let child_idx = header_index(&header, "child_sfen")?;
    let score_idx = header_index(&header, "teacher_child_score")?;
    let mut writer = BufWriter::new(File::create(&cli.output)?);
    let mut written = 0u64;
    let mut skipped = 0u64;
    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split(',').collect();
        let root_sfen = fields.get(root_idx).context("short row: root_sfen")?;
        if roots.as_ref().is_some_and(|set| !set.contains(*root_sfen)) {
            continue;
        }
        let child_sfen = fields.get(child_idx).context("short row: child_sfen")?;
        let score: i16 = fields
            .get(score_idx)
            .context("short row: teacher_child_score")?
            .parse::<i32>()?
            .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let mut pos = Position::default();
        if pos.set_sfen(child_sfen).is_err() {
            skipped += 1;
            continue;
        }
        let game_ply = child_sfen
            .split_whitespace()
            .nth(3)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(1);
        let psv = PackedSfenValue {
            sfen: pack_position(&pos),
            score,
            move16: 0,
            game_ply,
            game_result: 0,
            padding: 0,
        };
        writer.write_all(&psv.to_bytes())?;
        written += 1;
    }
    writer.flush()?;
    if written == 0 {
        bail!("no PSV records written");
    }
    eprintln!("Done: {written} PSV records written ({skipped} parse errors)");
    Ok(())
}
