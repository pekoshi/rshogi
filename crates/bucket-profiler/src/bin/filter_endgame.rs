//! `filter-endgame` — PSV file を scan して bucket 6/7 のみ抽出する CLI。
//!
//! ## 背景
//!
//! `bucket-profile` での実測で bucket 7 (終盤) が 3.7%、 bucket 6 が 10.7%
//! と中盤帯比で 6-9 倍少ないことが判明した。 Stage 2/3 で終盤教師を補強する
//! ため、 既存 PSV から bucket 6/7 のみ抽出した PSV を生成する。
//!
//! ## 設計
//!
//! - 入力 file ごとに worker thread を立て、 PSV を 40 byte ずつ stream read。
//! - 各 worker が PSV unpack + bucket 計算 → matched record を crossbeam
//!   bounded channel 経由で writer thread に送信。
//! - writer thread が 2 系統 (bucket7 / bucket67) の `BufWriter` に append。
//! - 出力は元 PSV と byte-level で互換 (40 byte/record)。
//!
//! 走行中 Stage 1 訓練と並行可能 (read のみ、 write 先は別 file)。

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

const BUCKET_TARGET_HIGH: usize = 7;
const BUCKET_TARGET_LOW: usize = 6;
const WRITE_BUF_BYTES: usize = 4 << 20; // 4 MB
const CHANNEL_CAP: usize = 64 * 1024; // record 単位 (1 record = 41 byte ペイロード)

/// CLI 引数。
#[derive(Parser, Debug)]
#[command(
    name = "filter-endgame",
    about = "PSV file から bucket 6/7 のみ抽出して新規 PSV を生成する",
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

    /// bucket=7 のみ出力先。
    #[arg(long)]
    output_bucket7: PathBuf,

    /// bucket 6+7 出力先。
    #[arg(long)]
    output_bucket67: PathBuf,

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
    /// `targets` bit 0=bucket7, bit 1=bucket67 (両方立つこともある)。
    Record {
        record: [u8; PSV_RECORD_SIZE],
        targets: u8,
    },
}

const TARGET_BUCKET7: u8 = 0b01;
const TARGET_BUCKET67: u8 = 0b10;

/// 1 file 分の集計。
#[derive(Debug, Clone, Copy, Default)]
struct FileStats {
    processed: u64,
    failed: u64,
    bucket7: u64,
    bucket67: u64,
}

impl FileStats {
    fn merge(&mut self, other: &FileStats) {
        self.processed += other.processed;
        self.failed += other.failed;
        self.bucket7 += other.bucket7;
        self.bucket67 += other.bucket67;
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
        "[filter-endgame] loaded progress.bin: {} ({} f32 weights)",
        args.progress.display(),
        weights.len()
    );
    let weights = Arc::new(weights);

    // 3. 出力 file 安全チェック
    ensure_output_paths(&args.output_bucket7, &args.output_bucket67, args.force)?;

    // 4. writer thread を起動
    let (tx, rx) = bounded::<WriteMsg>(CHANNEL_CAP);
    let out7_path = args.output_bucket7.clone();
    let out67_path = args.output_bucket67.clone();
    let writer_handle = thread::spawn(move || -> Result<(u64, u64)> {
        let f7 = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&out7_path)
            .with_context(|| format!("output_bucket7 open 失敗: {}", out7_path.display()))?;
        let f67 = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&out67_path)
            .with_context(|| format!("output_bucket67 open 失敗: {}", out67_path.display()))?;
        let mut w7 = BufWriter::with_capacity(WRITE_BUF_BYTES, f7);
        let mut w67 = BufWriter::with_capacity(WRITE_BUF_BYTES, f67);
        let mut n7: u64 = 0;
        let mut n67: u64 = 0;
        while let Ok(msg) = rx.recv() {
            match msg {
                WriteMsg::Record { record, targets } => {
                    if targets & TARGET_BUCKET7 != 0 {
                        w7.write_all(&record).context("output_bucket7 write 失敗")?;
                        n7 += 1;
                    }
                    if targets & TARGET_BUCKET67 != 0 {
                        w67.write_all(&record).context("output_bucket67 write 失敗")?;
                        n67 += 1;
                    }
                }
            }
        }
        w7.flush().context("output_bucket7 flush 失敗")?;
        w67.flush().context("output_bucket67 flush 失敗")?;
        Ok((n7, n67))
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
        "[filter-endgame] starting: {} input files, threads={}, max_records/file={}",
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
                            eprintln!("[filter-endgame][error] {} : {:#}", path.display(), e);
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
    let (written7, written67) =
        writer_handle.join().map_err(|_| anyhow!("writer thread panicked"))??;

    // 7. stats 集計 + print
    let mut total = FileStats::default();
    for (path, stats) in inputs.iter().zip(file_stats.iter()) {
        total.merge(stats);
        eprintln!(
            "[filter-endgame] {} : processed={} failed={} bucket7={} bucket67={}",
            path.display(),
            stats.processed,
            stats.failed,
            stats.bucket7,
            stats.bucket67,
        );
    }

    let elapsed = started.elapsed();
    let out7_size = std::fs::metadata(&args.output_bucket7).map(|m| m.len()).unwrap_or(0);
    let out67_size = std::fs::metadata(&args.output_bucket67).map(|m| m.len()).unwrap_or(0);

    println!("=== filter-endgame summary ===");
    println!("inputs:          {}", inputs.len());
    println!("processed total: {}", total.processed);
    println!("failed decode:   {}", total.failed);
    println!(
        "bucket=7 hits:   {} ({:.3}%) -> {} ({:.2} MB)",
        total.bucket7,
        ratio_pct(total.bucket7, total.processed),
        args.output_bucket7.display(),
        out7_size as f64 / (1024.0 * 1024.0),
    );
    println!(
        "bucket>=6 hits:  {} ({:.3}%) -> {} ({:.2} MB)",
        total.bucket67,
        ratio_pct(total.bucket67, total.processed),
        args.output_bucket67.display(),
        out67_size as f64 / (1024.0 * 1024.0),
    );
    println!(
        "writer wrote:    bucket7={} bucket67={} (channel consistency check)",
        written7, written67
    );
    println!("elapsed:         {:.2?}", elapsed);

    // writer と worker の件数一致を sanity check
    if written7 != total.bucket7 || written67 != total.bucket67 {
        return Err(anyhow!(
            "writer count mismatch: writer7={} worker7={} writer67={} worker67={}",
            written7,
            total.bucket7,
            written67,
            total.bucket67
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
                Err(e) => eprintln!("[filter-endgame][warn] glob entry error: {e}"),
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
fn ensure_output_paths(out7: &Path, out67: &Path, force: bool) -> Result<()> {
    if out7 == out67 {
        return Err(anyhow!("--output-bucket7 と --output-bucket67 は別 path にしてください"));
    }
    for p in [out7, out67] {
        if p.exists() {
            if !force {
                return Err(anyhow!(
                    "output file already exists: {} (use --force to overwrite)",
                    p.display()
                ));
            } else {
                eprintln!("[filter-endgame][warn] overwriting existing file: {}", p.display());
            }
        }
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(anyhow!("output parent directory missing: {}", parent.display()));
            }
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

        let mut targets: u8 = 0;
        match PackedSfenValue::from_bytes(&buf) {
            Some(psv) => match progress_bucket_for_record(&psv, weights, &mut scratch) {
                Ok(b) => {
                    if b == BUCKET_TARGET_HIGH {
                        targets |= TARGET_BUCKET7;
                    }
                    if b >= BUCKET_TARGET_LOW {
                        targets |= TARGET_BUCKET67;
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

        if targets != 0 {
            if targets & TARGET_BUCKET7 != 0 {
                stats.bucket7 += 1;
            }
            if targets & TARGET_BUCKET67 != 0 {
                stats.bucket67 += 1;
            }
            tx.send(WriteMsg::Record {
                record: buf,
                targets,
            })
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
                "[filter-endgame] global processed={} failed={}",
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
