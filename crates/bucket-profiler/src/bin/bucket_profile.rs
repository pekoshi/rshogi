//! `bucket-profile` — PSV file の `progress8kpabs` bucket 分布を診断する CLI。
//!
//! ## 背景
//!
//! rshogi 作者 (ramu_shogi_dev) の WCSC36 6 位記事
//! <https://note.com/ramu_shogi_dev/n/n6b294199a7db> の最重要知見:
//!
//! > 「教師データの 8 割くらいが特定のバケットに集中していた」
//!
//! これは **玉位置バケット** で起きた偏りで、 進行度バケット (progress8kpabs)
//! への移行で解決した。 我々は既に progress8kpabs を採用しているが、 教師
//! データの実分布が均等かを別途確認する手段がなかった。
//!
//! 本ツールは PSV file を読み込み、 各 record の局面を構築して
//! `progress8kpabs` bucket idx を計算し、 8 bucket の出現頻度を histogram
//! 表示する。
//!
//! ## 使い方
//!
//! ```text
//! bucket-profile --psv path/to/teacher.bin \
//!                [--progress E:/shogi/p2shogi/bullet_checkpoints/_progress/progress.bin] \
//!                [--sample 100000] [--output result.json] [--quiet]
//! ```
//!
//! `--sample 0` で全件処理。 `--output <PATH>` 指定時は stats を JSON で
//! 保存し、 stdout には ASCII histogram も併記する。

use anyhow::{anyhow, Context, Result};
use bucket_profiler::{
    load_progress_kpabs_weights, progress_bucket_for_record, BucketHistogram, BucketStats,
    PackedSfenValue, NUM_BUCKETS, PSV_RECORD_SIZE,
};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rshogi_core::position::Position;
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

/// CLI 引数。
#[derive(Parser, Debug)]
#[command(
    name = "bucket-profile",
    about = "PSV file の progress8kpabs bucket 分布を診断する",
    version
)]
struct Args {
    /// PSV file (40 byte/record、 YaneuraOu / Bullet 互換)。
    #[arg(long)]
    psv: PathBuf,

    /// progress.bin (KP-absolute 線形回帰重み、 f64 配列)。
    #[arg(
        long,
        default_value = r"E:\shogi\p2shogi\bullet_checkpoints\_progress\progress.bin"
    )]
    progress: PathBuf,

    /// 最大処理 record 数。 0 で全件。
    #[arg(long, default_value_t = 0)]
    sample: u64,

    /// 結果 JSON 出力先 (省略時は stdout のみ)。
    #[arg(long)]
    output: Option<PathBuf>,

    /// progress bar を抑制。
    #[arg(long, default_value_t = false)]
    quiet: bool,
}

/// JSON 出力スキーマ。
#[derive(Debug, Serialize)]
struct ReportJson<'a> {
    psv_path: &'a str,
    progress_path: &'a str,
    sample: u64,
    counts: [u64; NUM_BUCKETS],
    ratios: [f64; NUM_BUCKETS],
    total: u64,
    failed: u64,
    stats: BucketStats,
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(&args)
}

fn run(args: &Args) -> Result<()> {
    let weights = load_progress_kpabs_weights(&args.progress)
        .with_context(|| format!("progress.bin の読み込みに失敗: {}", args.progress.display()))?;
    eprintln!(
        "[bucket-profile] loaded progress.bin: {} ({} f32 weights)",
        args.progress.display(),
        weights.len()
    );

    let (hist, processed) = scan_psv(&args.psv, &weights, args.sample, args.quiet)?;

    print_report(&args.psv, &args.progress, args.sample, processed, &hist);

    if let Some(out_path) = args.output.as_ref() {
        write_report_json(out_path, args, &hist)?;
        eprintln!("[bucket-profile] JSON 出力: {}", out_path.display());
    }

    Ok(())
}

/// PSV file を読み込み、 bucket 集計を行う。
fn scan_psv(
    psv_path: &Path,
    weights: &[f32],
    sample: u64,
    quiet: bool,
) -> Result<(BucketHistogram, u64)> {
    let file =
        File::open(psv_path).with_context(|| format!("PSV open 失敗: {}", psv_path.display()))?;
    let file_len = file
        .metadata()
        .map(|m| m.len())
        .with_context(|| format!("PSV metadata 失敗: {}", psv_path.display()))?;

    if file_len % PSV_RECORD_SIZE as u64 != 0 {
        return Err(anyhow!(
            "PSV size {} は {} の倍数ではありません ({})",
            file_len,
            PSV_RECORD_SIZE,
            psv_path.display()
        ));
    }
    let total_records = file_len / PSV_RECORD_SIZE as u64;
    let target = if sample == 0 {
        total_records
    } else {
        sample.min(total_records)
    };

    eprintln!(
        "[bucket-profile] PSV {}: {} records (target = {})",
        psv_path.display(),
        total_records,
        target
    );

    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(target);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        pb
    };

    let mut reader = BufReader::with_capacity(8 << 20, file);
    let mut buf = [0u8; PSV_RECORD_SIZE];
    let mut hist = BucketHistogram::new();
    let mut scratch = Position::new();
    let mut processed: u64 = 0;

    while processed < target {
        match reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(anyhow::Error::new(e).context("PSV read 失敗")),
        }
        match PackedSfenValue::from_bytes(&buf) {
            Some(psv) => match progress_bucket_for_record(&psv, weights, &mut scratch) {
                Ok(b) => hist.record(b),
                Err(_) => hist.record_failure(),
            },
            None => hist.record_failure(),
        }
        processed += 1;
        if processed.is_multiple_of(50_000) {
            pb.set_position(processed);
        }
    }
    pb.set_position(processed);
    pb.finish_and_clear();
    Ok((hist, processed))
}

/// stdout に ASCII histogram + stats を表示。
fn print_report(
    psv_path: &Path,
    progress_path: &Path,
    sample: u64,
    processed: u64,
    hist: &BucketHistogram,
) {
    let ratios = hist.ratios();
    let stats = hist.stats();
    let psv_name = psv_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| psv_path.display().to_string());

    println!("=== bucket distribution (PSV: {}, N={}) ===", psv_name, hist.total);

    // ASCII bar (1 文字 ≒ 1%、 最大 50 文字)。
    let max_chars = 50usize;
    for (b, &r) in ratios.iter().enumerate() {
        let pct = r * 100.0;
        let bar_len = (pct.round() as usize).min(max_chars);
        let bar = "#".repeat(bar_len);
        println!("bucket {b} [{pct:>5.1}%] {bar}");
    }

    println!();
    println!("stats:");
    println!("  min ratio:   {:>5.1}% (bucket {})", stats.min_ratio * 100.0, stats.min_bucket);
    println!("  max ratio:   {:>5.1}% (bucket {})", stats.max_ratio * 100.0, stats.max_bucket);
    println!("  stdev:       {:>5.2}%", stats.stdev * 100.0);
    println!("  mean (ideal):{:>5.1}%", stats.mean * 100.0);
    println!("  judgment:    {}", stats.judgment);
    println!(
        "  processed:   {} (failed decode: {}, sample arg: {})",
        processed, hist.failed, sample
    );
    let _ = progress_path; // 既に冒頭で eprintln 済み
}

fn write_report_json(out_path: &Path, args: &Args, hist: &BucketHistogram) -> Result<()> {
    let ratios = hist.ratios();
    let stats = hist.stats();
    let report = ReportJson {
        psv_path: &args.psv.display().to_string(),
        progress_path: &args.progress.display().to_string(),
        sample: args.sample,
        counts: hist.counts,
        ratios,
        total: hist.total,
        failed: hist.failed,
        stats,
    };
    let json =
        serde_json::to_string_pretty(&report).with_context(|| "JSON serialize 失敗".to_string())?;
    std::fs::write(out_path, json)
        .with_context(|| format!("JSON 書き込み失敗: {}", out_path.display()))?;
    Ok(())
}
