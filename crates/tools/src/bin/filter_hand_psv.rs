//! filter_hand_psv - PSVから既存教師ラベルを保持したまま持ち駒局面を選別する。
//!
//! PackedSfenValueの40バイトレコードをそのまま出力するため、score・move16・
//! game_ply・game_result・paddingを変更しない。教師値の再評価やクリップは行わない。

use anyhow::{Context, Result};
use clap::Parser;
use rshogi_core::position::Position;
use rshogi_core::types::{Color, PieceType};
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use tools::packed_sfen::{PackedSfenValue, unpack_sfen};

const HAND_TYPES: [PieceType; 7] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Gold,
    PieceType::Bishop,
    PieceType::Rook,
];

#[derive(Parser, Debug)]
#[command(
    name = "filter_hand_psv",
    about = "既存DLSuishoラベルを保持したまま、持ち駒構造でPSVを選別する"
)]
struct Cli {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    ply_min: Option<u16>,
    #[arg(long)]
    score_abs_max: Option<i16>,
    #[arg(long, default_value_t = 1)]
    hand_count_min: u32,
    #[arg(long)]
    require_major_hand: bool,
    #[arg(long)]
    exclude_in_check: bool,
    #[arg(long, default_value_t = 0)]
    limit: u64,
    #[arg(long)]
    stats: Option<PathBuf>,
}

#[derive(Debug, Default, Serialize)]
struct Stats {
    input_records: u64,
    output_records: u64,
    filtered_ply_min: u64,
    filtered_score_abs_max: u64,
    filtered_hand_count: u64,
    filtered_major_hand: u64,
    filtered_in_check: u64,
    decode_errors: u64,
}

fn total_hand_pieces(pos: &Position) -> u32 {
    [Color::Black, Color::White]
        .into_iter()
        .map(|color| {
            let hand = pos.hand(color);
            HAND_TYPES.into_iter().map(|pt| hand.count(pt)).sum::<u32>()
        })
        .sum()
}

fn has_major_hand(pos: &Position) -> bool {
    [Color::Black, Color::White].into_iter().any(|color| {
        let hand = pos.hand(color);
        hand.has(PieceType::Bishop) || hand.has(PieceType::Rook)
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.hand_count_min == 0 {
        anyhow::bail!("--hand-count-min must be at least 1");
    }
    if !cli.input.is_file() {
        anyhow::bail!("Input file not found: {}", cli.input.display());
    }
    if cli.output.exists() {
        anyhow::bail!("Refusing to overwrite output: {}", cli.output.display());
    }
    if cli.input == cli.output {
        anyhow::bail!("Input and output must be different files");
    }
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    let input = File::open(&cli.input)
        .with_context(|| format!("Failed to open input: {}", cli.input.display()))?;
    let output = File::create(&cli.output)
        .with_context(|| format!("Failed to create output: {}", cli.output.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, input);
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, output);
    let mut stats = Stats::default();

    loop {
        let mut record = [0u8; PackedSfenValue::SIZE];
        let mut filled = 0usize;
        while filled < record.len() {
            let n = reader.read(&mut record[filled..])?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled == 0 {
            break;
        }
        if filled != record.len() {
            anyhow::bail!("Input has a trailing partial PSV record ({filled} bytes)");
        }
        stats.input_records += 1;

        let Some(psv) = PackedSfenValue::from_bytes(&record) else {
            stats.decode_errors += 1;
            continue;
        };
        if let Some(min_ply) = cli.ply_min {
            if psv.game_ply < min_ply {
                stats.filtered_ply_min += 1;
                continue;
            }
        }
        if let Some(max_abs) = cli.score_abs_max {
            if psv.score.unsigned_abs() > max_abs as u16 {
                stats.filtered_score_abs_max += 1;
                continue;
            }
        }
        let sfen = match unpack_sfen(&psv.sfen) {
            Ok(value) => value,
            Err(_) => {
                stats.decode_errors += 1;
                continue;
            }
        };
        let mut pos = Position::new();
        if pos.set_sfen(&sfen).is_err() {
            stats.decode_errors += 1;
            continue;
        }
        if cli.exclude_in_check && pos.in_check() {
            stats.filtered_in_check += 1;
            continue;
        }
        if total_hand_pieces(&pos) < cli.hand_count_min {
            stats.filtered_hand_count += 1;
            continue;
        }
        if cli.require_major_hand && !has_major_hand(&pos) {
            stats.filtered_major_hand += 1;
            continue;
        }

        writer.write_all(&record)?;
        stats.output_records += 1;
        if cli.limit > 0 && stats.output_records >= cli.limit {
            break;
        }
    }
    writer.flush()?;

    if let Some(stats_path) = cli.stats {
        if let Some(parent) = stats_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let payload = serde_json::to_vec_pretty(&stats)?;
        std::fs::write(&stats_path, payload)
            .with_context(|| format!("Failed to write statistics: {}", stats_path.display()))?;
    }
    eprintln!(
        "filter_hand_psv: input={} output={} hand_count_min={} major_hand={} in_check_excluded={}",
        stats.input_records,
        stats.output_records,
        cli.hand_count_min,
        cli.require_major_hand,
        cli.exclude_in_check
    );
    Ok(())
}
