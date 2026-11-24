//! `filter-bucket` — PSV file を scan して **指定 bucket** のみ抽出する CLI。
//! 2026-09-15: filter_middle を fork し、 `--buckets` と `--record-range` を足した。
//! ★抽出は bucket (盤上の進行度) だけで決める。 score/gamePly は一切見ない
//!   (難例 -70 Elo の失敗はラベルで選んだことが原因)。
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
// 2026-09-15: 実行時に --buckets で指定する (既定は従来どおり 3,4,5)
static BUCKET_TARGETS: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
fn bucket_targets() -> &'static Vec<usize> {
    BUCKET_TARGETS.get_or_init(|| vec![3, 4, 5])
}
// 2026-09-15: 読むレコード範囲 (0 基点、 end 排他)。 未指定なら全件。
static RECORD_RANGE: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
// 2026-09-15: 背景を残す割合 (百万分率)。 未設定 = 背景を書かない。
static OTHER_RATE_PERMIL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
static WANT_OTHER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// 決定的な間引き。 record の連番を Knuth hash して上位 20 bit を見る。
/// ★乱数を使わないので、 同じ引数なら何度でも同じ集合が出る。
#[inline]
fn keep_other(idx: u64) -> bool {
    let r = OTHER_RATE_PERMIL.get().copied().unwrap_or(0);
    if r >= 1_000_000 {
        return true;
    }
    if r == 0 {
        return false;
    }
    let h = idx.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> 44) < ((r << 20) / 1_000_000)
}
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

    /// 抽出する bucket (カンマ区切り、 例: 0 / 3,4)。
    #[arg(long, value_delimiter = ',', default_value = "3,4,5")]
    buckets: Vec<usize>,
    /// 処理するレコード範囲 start:end (0 基点、 end 排他。 例: 300000000:1200000000)。
    #[arg(long)]
    record_range: Option<String>,
    /// 背景 (= --buckets に含まれない帯) の出力先。 指定すると 1 パスで両方書ける。
    #[arg(long)]
    output_other: Option<PathBuf>,
    /// 背景を残す割合 (0.0-1.0、 負で背景を書かない)。 決定的な間引き (record 連番の hash) なので再現する。
    /// --output-other を省くと背景は**主出力に混ぜて**書かれる (元ファイルの順序のまま = 自然に混ざる)。
    #[arg(long, default_value_t = -1.0)]
    other_rate: f64,
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
    /// `to_other=false` で前景 (--buckets)、 `true` で背景 (--output-other)。
    Record {
        record: [u8; PSV_RECORD_SIZE],
        to_other: bool,
    },
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
        bucket_targets().iter().map(|&b| self.bucket_counts[b]).sum()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    // 2026-09-15: 抽出 bucket を実行時に決める
    BUCKET_TARGETS.set(args.buckets.clone()).ok();
    // 2026-09-15: レコード範囲 start:end (0 基点、 end 排他)
    if let Some(ref r) = args.record_range {
        let mut it = r.split(':');
        let a: u64 = it.next().unwrap_or("0").parse().unwrap_or(0);
        let b: u64 = it.next().unwrap_or("0").parse().unwrap_or(u64::MAX);
        RECORD_RANGE.set((a, b)).ok();
        eprintln!("[filter-bucket] record-range {}..{} (= {} 件)", a, b, b.saturating_sub(a));
    }
    WANT_OTHER.set(args.output_other.is_some() || args.other_rate >= 0.0).ok();
    eprintln!("[filter-bucket] 抽出 bucket = {:?}", args.buckets);
    if args.other_rate >= 0.0 {
        match args.output_other {
            Some(ref o) => eprintln!(
                "[filter-bucket] 背景 -> {} (残す割合 {:.4})",
                o.display(),
                args.other_rate
            ),
            None => {
                eprintln!("[filter-bucket] 背景は主出力に混ぜる (残す割合 {:.4})", args.other_rate)
            }
        }
    }
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
    if let Some(ref o) = args.output_other {
        ensure_output_path(o, args.force)?;
    }
    // 2026-09-15: 背景の間引き率をグローバルに (worker から参照する)
    if args.other_rate >= 0.0 {
        OTHER_RATE_PERMIL.set((args.other_rate.min(1.0) * 1_000_000.0) as u64).ok();
    }

    // 4. writer thread を起動
    let (tx, rx) = bounded::<WriteMsg>(CHANNEL_CAP);
    let out_path = args.output_bucket345.clone();
    let other_path = args.output_other.clone();
    let writer_handle = thread::spawn(move || -> Result<u64> {
        let open = |p: &Path| -> Result<BufWriter<File>> {
            let f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(p)
                .with_context(|| format!("出力 open 失敗: {}", p.display()))?;
            Ok(BufWriter::with_capacity(WRITE_BUF_BYTES, f))
        };
        let mut w = open(&out_path)?;
        let mut w2 = match other_path {
            Some(ref p) => Some(open(p)?),
            None => None,
        };
        let mut n: u64 = 0;
        let mut n2: u64 = 0;
        while let Ok(msg) = rx.recv() {
            match msg {
                WriteMsg::Record { record, to_other } => {
                    if to_other && w2.is_some() {
                        let ww = w2.as_mut().unwrap();
                        ww.write_all(&record).context("output_other write 失敗")?;
                        n2 += 1;
                    } else {
                        w.write_all(&record).context("output_bucket345 write 失敗")?;
                        n += 1;
                    }
                }
            }
        }
        w.flush().context("output_bucket345 flush 失敗")?;
        if let Some(mut ww) = w2 {
            ww.flush().context("output_other flush 失敗")?;
            eprintln!("[filter-bucket] 背景 (output-other) {} 件", n2);
        }
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
    // 2026-09-15: 背景を主出力に混ぜたときは writer の件数 > 前景件数 になるので比較しない
    let mixed_into_main = args.other_rate >= 0.0 && args.output_other.is_none();
    if mixed_into_main {
        println!(
            "  (うち前景 bucket{:?} = {} 件 = {:.3}%)",
            args.buckets,
            middle_total,
            100.0 * middle_total as f64 / written.max(1) as f64
        );
    }
    if !mixed_into_main && written != middle_total {
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
    // 2026-09-15: --record-range が指定されていれば start まで seek し、 end で打ち切る
    let (rng_start, rng_end) = RECORD_RANGE.get().copied().unwrap_or((0, u64::MAX));
    if rng_start > 0 {
        use std::io::{Seek, SeekFrom};
        reader
            .seek(SeekFrom::Start(rng_start * PSV_RECORD_SIZE as u64))
            .context("record-range の seek 失敗")?;
    }
    let range_limit = rng_end.saturating_sub(rng_start);
    let mut buf = [0u8; PSV_RECORD_SIZE];
    let mut scratch = Position::new();
    let mut stats = FileStats::default();

    while stats.processed < target && stats.processed < range_limit {
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
                    if bucket_targets().contains(&b) {
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
            tx.send(WriteMsg::Record {
                record: buf,
                to_other: false,
            })
            .map_err(|_| anyhow!("writer channel closed early"))?;
        } else if WANT_OTHER.get().copied().unwrap_or(false)
            && keep_other(rng_start + stats.processed)
        {
            // 2026-09-15: 前景に入らなかった record を、 決定的に間引いて背景へ
            tx.send(WriteMsg::Record {
                record: buf,
                to_other: true,
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
