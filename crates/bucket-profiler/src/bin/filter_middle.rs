//! `filter-middle` — PSV file を scan して中盤帯 bucket 3/4/5 のみ抽出する CLI。
//!
//! ## 背景
//!
//! `bucket-profile` での実測 (前回 mekitto 100K scan) では bucket 3=17.9%、
//! bucket 4=16.0%、 bucket 5=14.3% の合計 ~48% を中盤帯と推定。 Stage 3 で
//! 中盤精度を強化するため、 既存 PSV 5-7B records から中盤 bucket のみ抽出
//! した PSV を生成する。
//!
//! ## 設計 ( `filter_endgame.rs` から fork)
//!
//! - 入力 file ごとに worker thread を立て、 PSV を 40 byte ずつ stream read。
//! - 各 worker が PSV unpack + bucket 計算 → matched record を crossbeam
//!   bounded channel 経由で writer thread に送信。
//! - writer thread が単一の `BufWriter` (`bucket345.bin`) に append。
//! - 出力は元 PSV と byte-level で互換 (40 byte/record)。
//!
//! 走行中 Stage 2 訓練と並行可能 (read のみ、 write 先は別 file)。

use anyhow::{anyhow, Context, Result};
use bucket_profiler::{
    load_progress_kpabs_weights, progress_bucket_for_record, PackedSfenValue, PSV_RECORD_SIZE,
};
use clap::Parser;
use crossbeam_channel::{bounded, Sender};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rshogi_core::position::Position;
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// 抽出対象 bucket (中盤帯 3,4,5)。
const BUCKET_TARGETS: [usize; 3] = [3, 4, 5];
const WRITE_BUF_BYTES: usize = 4 << 20; // 4 MB
const CHANNEL_CAP: usize = 64 * 1024; // record 単位 (1 record = 40 byte ペイロード)

/// CLI 引数。
#[derive(Parser, Debug)]
#[command(
    name = "filter-middle",
    about = "PSV file から bucket 3/4/5 (中盤帯) のみ抽出して新規 PSV を生成する",
    version
)]
struct Args {
    /// 入力 PSV (カンマ区切りで複数指定可)。
    #[arg(long, value_delimiter = ',')]
    inputs: Vec<PathBuf>,

    /// 追加で glob 指定 (例: `E:/shogi/p2shogi/data/_psv/gct/*.psv`)。
    #[arg(long)]
    inputs_glob: Option<String>,

    /// progress.bin (KP-absolute 線形回帰重み、 f64 配列)。
    #[arg(
        long,
        default_value = r"E:\shogi\p2shogi\bullet_checkpoints\_progress\progress.bin"
    )]
    progress: PathBuf,

    /// bucket 3+4+5 出力先。
    #[arg(long)]
    output_bucket345: PathBuf,

    /// 並列 read worker 数 (file 単位の並列度)。
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// progress bar を抑制。
    #[arg(long, default_value_t = false)]
    quiet: bool,

    /// 動作確認用: 各 input file から最大 N record だけ処理 (0 で全件)。
    #[arg(long, default_value_t = 0)]
    max_records: u64,

    /// 既存出力ファイルを上書きする。
    #[arg(long, default_value_t = false)]
    force: bool,
}

/// writer thread に渡す message。
enum WriteMsg {
    Record { record: [u8; PSV_RECORD_SIZE] },
}

/// 1 file 分の集計。
#[derive(Debug, Clone, Copy, Default)]
struct FileStats {
    processed: u64,
    failed: u64,
    /// bucket 別 hit (中盤帯 only 関心、 但し統計のため全 bucket 記録)。
    bucket_counts: [u64; 8],
}

impl FileStats {
    fn merge(&mut self, other: &FileStats) {
        self.processed += other.processed;
        self.failed += other.failed;
        for (a, b) in self.bucket_counts.iter_mut().zip(other.bucket_counts.iter()) {
            *a += *b;
        }
    }

    fn middle_total(&self) -> u64 {
        BUCKET_TARGETS.iter().map(|&b| self.bucket_counts[b]).sum()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(&args)
}

fn run(args: &Args) -> Result<()> {
    // 1. 入力 file collection
    let inputs = collect_inputs(args)?;
    if inputs.is_empty() {
        return Err(anyhow!(
            "入力 PSV が 1 件もありません (--inputs か --inputs-glob を指定してください)"
        ));
    }

    // 2. progress.bin load
    let weights = load_progress_kpabs_weights(&args.progress)
        .with_context(|| format!("progress.bin の読み込みに失敗: {}", args.progress.display()))?;
    eprintln!(
        "[filter-middle] loaded progress.bin: {} ({} f32 weights)",
        args.progress.display(),
        weights.len()
    );
    let weights = Arc::new(weights);

    // 3. 出力 file 安全チェック
    ensure_output_path(&args.output_bucket345, args.force)?;

    // 4. writer thread を起動
    let (tx, rx) = bounded::<WriteMsg>(CHANNEL_CAP);
    let out_path = args.output_bucket345.clone();
    let writer_handle = thread::spawn(move || -> Result<u64> {
        let f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&out_path)
            .with_context(|| format!("output_bucket345 open 失敗: {}", out_path.display()))?;
        let mut w = BufWriter::with_capacity(WRITE_BUF_BYTES, f);
        let mut n: u64 = 0;
        while let Ok(msg) = rx.recv() {
            match msg {
                WriteMsg::Record { record } => {
                    w.write_all(&record).context("output_bucket345 write 失敗")?;
                    n += 1;
                }
            }
        }
        w.flush().context("output_bucket345 flush 失敗")?;
        Ok(n)
    });

    // 5. worker pool (scoped thread) で各 file を並列処理
    let mp = if args.quiet {
        None
    } else {
        Some(MultiProgress::new())
    };
    let global_processed = Arc::new(AtomicU64::new(0));
    let global_failed = Arc::new(AtomicU64::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let max_records = args.max_records;
    let threads = args.threads.max(1);
    let started = Instant::now();

    eprintln!(
        "[filter-middle] starting: {} input files, threads={}, max_records/file={}",
        inputs.len(),
        threads,
        if max_records == 0 {
            "all".to_string()
        } else {
            max_records.to_string()
        }
    );

    let file_stats: Vec<FileStats> = thread::scope(|scope| -> Result<Vec<FileStats>> {
        let (job_tx, job_rx) = bounded::<(usize, PathBuf)>(inputs.len());
        for (idx, path) in inputs.iter().cloned().enumerate() {
            job_tx.send((idx, path)).expect("job channel send");
        }
        drop(job_tx);

        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let job_rx = job_rx.clone();
            let tx = tx.clone();
            let weights = Arc::clone(&weights);
            let global_processed = Arc::clone(&global_processed);
            let global_failed = Arc::clone(&global_failed);
            let stop_flag = Arc::clone(&stop_flag);
            let mp_clone = mp.clone();
            let handle = scope.spawn(move || -> Vec<(usize, FileStats)> {
                let mut local_results = Vec::new();
                while let Ok((idx, path)) = job_rx.recv() {
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let stats = match process_file(
                        &path,
                        &weights,
                        &tx,
                        max_records,
                        mp_clone.as_ref(),
                        &global_processed,
                        &global_failed,
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[filter-middle][error] {} : {:#}", path.display(), e);
                            stop_flag.store(true, Ordering::Relaxed);
                            FileStats::default()
                        }
                    };
                    local_results.push((idx, stats));
                }
                local_results
            });
            handles.push(handle);
        }

        let mut collected: Vec<(usize, FileStats)> = Vec::new();
        for h in handles {
            let part = h.join().map_err(|_| anyhow!("worker thread panicked"))?;
            collected.extend(part);
        }
        collected.sort_by_key(|(idx, _)| *idx);
        Ok(collected.into_iter().map(|(_, s)| s).collect())
    })?;

    // 6. writer 終了
    drop(tx);
    let written = writer_handle.join().map_err(|_| anyhow!("writer thread panicked"))??;

    // 7. stats 集計 + print
    let mut total = FileStats::default();
    for (path, stats) in inputs.iter().zip(file_stats.iter()) {
        total.merge(stats);
        eprintln!(
            "[filter-middle] {} : processed={} failed={} bucket3={} bucket4={} bucket5={} (middle={})",
            path.display(),
            stats.processed,
            stats.failed,
            stats.bucket_counts[3],
            stats.bucket_counts[4],
            stats.bucket_counts[5],
            stats.middle_total(),
        );
    }

    let elapsed = started.elapsed();
    let out_size = std::fs::metadata(&args.output_bucket345).map(|m| m.len()).unwrap_or(0);

    println!("=== filter-middle summary ===");
    println!("inputs:          {}", inputs.len());
    println!("processed total: {}", total.processed);
    println!("failed decode:   {}", total.failed);
    let middle_total = total.middle_total();
    println!(
        "bucket=3 hits:   {} ({:.3}%)",
        total.bucket_counts[3],
        ratio_pct(total.bucket_counts[3], total.processed),
    );
    println!(
        "bucket=4 hits:   {} ({:.3}%)",
        total.bucket_counts[4],
        ratio_pct(total.bucket_counts[4], total.processed),
    );
    println!(
        "bucket=5 hits:   {} ({:.3}%)",
        total.bucket_counts[5],
        ratio_pct(total.bucket_counts[5], total.processed),
    );
    println!(
        "bucket 3+4+5:    {} ({:.3}%) -> {} ({:.2} MB)",
        middle_total,
        ratio_pct(middle_total, total.processed),
        args.output_bucket345.display(),
        out_size as f64 / (1024.0 * 1024.0),
    );
    println!("writer wrote:    bucket345={} (channel consistency check)", written);
    println!("elapsed:         {:.2?}", elapsed);

    // writer と worker の件数一致を sanity check
    if written != middle_total {
        return Err(anyhow!(
            "writer count mismatch: writer345={} worker345={}",
            written,
            middle_total,
        ));
    }

    Ok(())
}

fn ratio_pct(num: u64, denom: u64) -> f64 {
    if denom == 0 {
        0.0
    } else {
        (num as f64) * 100.0 / (denom as f64)
    }
}

/// `--inputs` + `--inputs-glob` を統合し、 重複除去 + 存在チェック。
fn collect_inputs(args: &Args) -> Result<Vec<PathBuf>> {
    let mut set: BTreeSet<PathBuf> = BTreeSet::new();
    for p in &args.inputs {
        set.insert(p.clone());
    }
    if let Some(g) = args.inputs_glob.as_ref() {
        let pattern = g.as_str();
        let entries =
            glob::glob(pattern).with_context(|| format!("invalid glob pattern: {pattern}"))?;
        for entry in entries {
            match entry {
                Ok(p) => {
                    set.insert(p);
                }
                Err(e) => eprintln!("[filter-middle][warn] glob entry error: {e}"),
            }
        }
    }
    let mut out: Vec<PathBuf> = Vec::with_capacity(set.len());
    for p in set {
        if !p.exists() {
            return Err(anyhow!("input file not found: {}", p.display()));
        }
        let len = std::fs::metadata(&p)
            .with_context(|| format!("metadata 失敗: {}", p.display()))?
            .len();
        if len % PSV_RECORD_SIZE as u64 != 0 {
            return Err(anyhow!(
                "PSV size {} は {} の倍数ではありません ({})",
                len,
                PSV_RECORD_SIZE,
                p.display()
            ));
        }
        out.push(p);
    }
    Ok(out)
}

/// 出力 file の安全チェック。
fn ensure_output_path(out: &Path, force: bool) -> Result<()> {
    if out.exists() {
        if !force {
            return Err(anyhow!(
                "output file already exists: {} (use --force to overwrite)",
                out.display()
            ));
        } else {
            eprintln!("[filter-middle][warn] overwriting existing file: {}", out.display());
        }
    }
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(anyhow!("output parent directory missing: {}", parent.display()));
        }
    }
    Ok(())
}

/// 1 file を stream read + bucket 計算 + 送信。
fn process_file(
    path: &Path,
    weights: &[f32],
    tx: &Sender<WriteMsg>,
    max_records: u64,
    mp: Option<&MultiProgress>,
    global_processed: &AtomicU64,
    global_failed: &AtomicU64,
) -> Result<FileStats> {
    let file = File::open(path).with_context(|| format!("PSV open 失敗: {}", path.display()))?;
    let file_len = file
        .metadata()
        .map(|m| m.len())
        .with_context(|| format!("PSV metadata 失敗: {}", path.display()))?;
    let total_records = file_len / PSV_RECORD_SIZE as u64;
    let target = if max_records == 0 {
        total_records
    } else {
        max_records.min(total_records)
    };

    let pb_opt = mp.map(|mp| {
        let pb = mp.add(ProgressBar::new(target));
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} ({eta}) {msg}",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        pb.set_message(name);
        pb
    });

    let mut reader = BufReader::with_capacity(8 << 20, file);
    let mut buf = [0u8; PSV_RECORD_SIZE];
    let mut scratch = Position::new();
    let mut stats = FileStats::default();

    while stats.processed < target {
        match reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(anyhow::Error::new(e).context("PSV read 失敗")),
        }

        let mut hit_middle = false;
        match PackedSfenValue::from_bytes(&buf) {
            Some(psv) => match progress_bucket_for_record(&psv, weights, &mut scratch) {
                Ok(b) => {
                    if b < stats.bucket_counts.len() {
                        stats.bucket_counts[b] += 1;
                    }
                    if BUCKET_TARGETS.contains(&b) {
                        hit_middle = true;
                    }
                }
                Err(_) => {
                    stats.failed += 1;
                    global_failed.fetch_add(1, Ordering::Relaxed);
                }
            },
            None => {
                stats.failed += 1;
                global_failed.fetch_add(1, Ordering::Relaxed);
            }
        }

        if hit_middle {
            tx.send(WriteMsg::Record { record: buf })
                .map_err(|_| anyhow!("writer channel closed early"))?;
        }

        stats.processed += 1;
        let gp = global_processed.fetch_add(1, Ordering::Relaxed) + 1;
        if stats.processed.is_multiple_of(50_000) {
            if let Some(pb) = pb_opt.as_ref() {
                pb.set_position(stats.processed);
            }
        }
        if gp.is_multiple_of(1_000_000) {
            eprintln!(
                "[filter-middle] global processed={} failed={}",
                gp,
                global_failed.load(Ordering::Relaxed),
            );
        }
    }
    if let Some(pb) = pb_opt {
        pb.set_position(stats.processed);
        pb.finish_and_clear();
    }
    Ok(stats)
}
