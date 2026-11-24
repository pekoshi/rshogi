//! Profile15用の再配分PSVを構築する。
//!
//! 入力のdedup済みPackedSfenValue（40 bytes）を1回だけ走査し、次の相互排他的な
//! score×plyカテゴリごとにreservoir samplingを行う。カテゴリを排他的にするため、
//! 入力がdedup済みであれば出力でもcross-category duplicateは発生しない。
//!
//! 目的は、±300cpを厚く、301..500cpをtransitionとして次に厚くしつつ、
//! |score|>500cpのearly/mid/late文脈を残して全局面進行帯を捨てないことである。

use anyhow::{Context, Result};
use clap::Parser;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

const RECORD_SIZE: usize = 40;
const READ_CHUNK_BYTES: usize = 16 * 1024 * 1024 / RECORD_SIZE * RECORD_SIZE;
const MATE_LIKE_ABS: i32 = 30_000;

#[derive(Parser, Debug)]
#[command(
    name = "build_profile15_rebalanced_psv",
    about = "Profile15向けscore×ply再配分PSVを1-pass reservoir samplingで構築"
)]
struct Cli {
    /// 入力PSV（40-byte PackedSfenValue、dedup済みを前提）
    #[arg(long)]
    input: PathBuf,
    /// 再配分した非shuffle PSVの出力先
    #[arg(long)]
    output: PathBuf,
    /// 構築manifest JSON
    #[arg(long)]
    manifest: PathBuf,
    /// 乱数シード
    #[arg(long, default_value_t = 20260827)]
    seed: u64,
    /// |score| <= core_abs の厚化目標件数
    #[arg(long, default_value_t = 45_000_000)]
    core_count: usize,
    /// core_abs < |score| <= transition_abs の目標件数
    #[arg(long, default_value_t = 20_000_000)]
    transition_count: usize,
    /// |score| > transition_abs かつ ply <= early_ply_max の目標件数
    #[arg(long, default_value_t = 10_000_000)]
    early_context_count: usize,
    /// |score| > transition_abs かつ mid_ply_min..=mid_ply_max の目標件数
    #[arg(long, default_value_t = 10_000_000)]
    mid_context_count: usize,
    /// |score| > transition_abs かつ ply >= late_ply_min の目標件数
    #[arg(long, default_value_t = 15_000_000)]
    late_context_count: usize,
    /// 中心帯の絶対score上限（cp）
    #[arg(long, default_value_t = 300)]
    core_abs: i32,
    /// transition帯の絶対score上限（cp）
    #[arg(long, default_value_t = 500)]
    transition_abs: i32,
    /// early contextの最大ply
    #[arg(long, default_value_t = 39)]
    early_ply_max: u16,
    /// mid contextの最小ply
    #[arg(long, default_value_t = 40)]
    mid_ply_min: u16,
    /// mid contextの最大ply
    #[arg(long, default_value_t = 119)]
    mid_ply_max: u16,
    /// late contextの最小ply
    #[arg(long, default_value_t = 120)]
    late_ply_min: u16,
}

#[derive(Debug, Serialize)]
struct SegmentManifest {
    name: String,
    rule: String,
    target_records: usize,
    eligible_records: u64,
    selected_records: usize,
}

#[derive(Debug, Serialize)]
struct Manifest {
    candidate: String,
    source: String,
    source_record_size: usize,
    source_records_read: u64,
    skipped_mate_like: u64,
    seed: u64,
    output: String,
    output_records: usize,
    output_bytes: u64,
    source_dedup_assumption: String,
    category_exclusivity: String,
    segments: Vec<SegmentManifest>,
}

struct Segment {
    name: &'static str,
    rule: String,
    target: usize,
    eligible: u64,
    reservoir: Vec<[u8; RECORD_SIZE]>,
}

impl Segment {
    fn new(name: &'static str, rule: String, target: usize) -> Self {
        Self {
            name,
            rule,
            target,
            eligible: 0,
            reservoir: Vec::with_capacity(target),
        }
    }

    fn offer(&mut self, record: [u8; RECORD_SIZE], rng: &mut ChaCha8Rng) {
        self.eligible += 1;
        if self.reservoir.len() < self.target {
            self.reservoir.push(record);
            return;
        }
        let index = rng.random_range(0..self.eligible);
        if (index as usize) < self.target {
            self.reservoir[index as usize] = record;
        }
    }
}

#[inline]
fn score_of(record: &[u8; RECORD_SIZE]) -> i16 {
    i16::from_le_bytes([record[32], record[33]])
}

#[inline]
fn ply_of(record: &[u8; RECORD_SIZE]) -> u16 {
    u16::from_le_bytes([record[36], record[37]])
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.core_abs <= 0 || cli.transition_abs <= cli.core_abs {
        anyhow::bail!("Require 0 < core_abs < transition_abs");
    }
    if cli.early_ply_max >= cli.mid_ply_min
        || cli.mid_ply_min > cli.mid_ply_max
        || cli.mid_ply_max >= cli.late_ply_min
    {
        anyhow::bail!("Ply bands must be disjoint and ordered");
    }
    if cli.output.exists() || cli.manifest.exists() {
        anyhow::bail!("Refusing to overwrite output or manifest");
    }

    let mut core = Segment::new(
        "core_abs_0_300",
        format!("abs(score) <= {} ; any ply", cli.core_abs),
        cli.core_count,
    );
    let mut transition = Segment::new(
        "transition_abs_301_500",
        format!("{} < abs(score) <= {} ; any ply", cli.core_abs, cli.transition_abs),
        cli.transition_count,
    );
    let mut early = Segment::new(
        "early_context_abs_gt_500",
        format!("abs(score) > {} ; ply <= {}", cli.transition_abs, cli.early_ply_max),
        cli.early_context_count,
    );
    let mut mid = Segment::new(
        "mid_context_abs_gt_500",
        format!(
            "abs(score) > {} ; {} <= ply <= {}",
            cli.transition_abs, cli.mid_ply_min, cli.mid_ply_max
        ),
        cli.mid_context_count,
    );
    let mut late = Segment::new(
        "late_context_abs_gt_500",
        format!("abs(score) > {} ; ply >= {}", cli.transition_abs, cli.late_ply_min),
        cli.late_context_count,
    );

    let file = File::open(&cli.input).with_context(|| format!("open {}", cli.input.display()))?;
    let mut reader = BufReader::with_capacity(READ_CHUNK_BYTES, file);
    let mut buffer = [0u8; RECORD_SIZE];
    let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);
    let mut read_records = 0u64;
    let mut skipped_mate_like = 0u64;

    loop {
        match reader.read_exact(&mut buffer) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        }
        read_records += 1;
        let score = score_of(&buffer) as i32;
        let abs_score = score.abs();
        if abs_score >= MATE_LIKE_ABS {
            skipped_mate_like += 1;
            continue;
        }
        let record = buffer;
        if abs_score <= cli.core_abs {
            core.offer(record, &mut rng);
        } else if abs_score <= cli.transition_abs {
            transition.offer(record, &mut rng);
        } else {
            let ply = ply_of(&record);
            if ply <= cli.early_ply_max {
                early.offer(record, &mut rng);
            } else if ply >= cli.mid_ply_min && ply <= cli.mid_ply_max {
                mid.offer(record, &mut rng);
            } else if ply >= cli.late_ply_min {
                late.offer(record, &mut rng);
            } else {
                anyhow::bail!("Record ply {ply} did not match any configured context band");
            }
        }

        if read_records.is_multiple_of(10_000_000) {
            eprintln!(
                "read={} core={}/{} transition={}/{} early={}/{} mid={}/{} late={}/{}",
                read_records,
                core.reservoir.len(),
                core.target,
                transition.reservoir.len(),
                transition.target,
                early.reservoir.len(),
                early.target,
                mid.reservoir.len(),
                mid.target,
                late.reservoir.len(),
                late.target,
            );
        }
    }

    for segment in [&core, &transition, &early, &mid, &late] {
        if segment.reservoir.len() < segment.target {
            anyhow::bail!(
                "Segment {} has only {} eligible records; need {}",
                segment.name,
                segment.reservoir.len(),
                segment.target
            );
        }
    }

    let output_file =
        File::create(&cli.output).with_context(|| format!("create {}", cli.output.display()))?;
    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, output_file);
    let mut output_records = 0usize;
    for segment in [&core, &transition, &early, &mid, &late] {
        for record in &segment.reservoir {
            writer.write_all(record)?;
            output_records += 1;
        }
    }
    writer.flush()?;

    let segments = [&core, &transition, &early, &mid, &late]
        .into_iter()
        .map(|segment| SegmentManifest {
            name: segment.name.to_string(),
            rule: segment.rule.clone(),
            target_records: segment.target,
            eligible_records: segment.eligible,
            selected_records: segment.reservoir.len(),
        })
        .collect::<Vec<_>>();
    let manifest = Manifest {
        candidate: "profile15_score_ply_rebalance_v1".to_string(),
        source: cli.input.display().to_string(),
        source_record_size: RECORD_SIZE,
        source_records_read: read_records,
        skipped_mate_like,
        seed: cli.seed,
        output: cli.output.display().to_string(),
        output_records,
        output_bytes: output_records as u64 * RECORD_SIZE as u64,
        source_dedup_assumption: "Input is dual3v2_all_priority_dedup PSV; this builder does not repeat a record within any category.".to_string(),
        category_exclusivity: "core, transition, early_context, mid_context, and late_context are mutually exclusive by score/ply rule; therefore a source-unique record cannot occur in two output categories.".to_string(),
        segments,
    };
    let manifest_text = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&cli.manifest, manifest_text)
        .with_context(|| format!("write {}", cli.manifest.display()))?;

    eprintln!(
        "completed: records={} output={} manifest={}",
        output_records,
        cli.output.display(),
        cli.manifest.display()
    );
    Ok(())
}
