//! `hard-mining` — Stage 1 ckpt の予測 cp と PSV 内 真 cp の差が大きい
//! "hard" 局面を抽出する CLI (skeleton 実装)。
//!
//! ## 背景
//!
//! Stage 3 で NNUE が苦手な局面を集中投入することで中盤精度を強化する。
//! 既存 rescore_psv は外部 USI engine を呼ぶため低速 (~1k eval/s)。
//! hard_mining は内部 NNUE evaluation を直接呼んで ~10-50k eval/s を狙う。
//!
//! ## 現状 (skeleton)
//!
//! - PSV stream read + Position 構築 + writer thread の骨組みのみ実装。
//! - **NNUE evaluation 部分は TODO**: `predict_cp = 0` の placeholder で
//!   動作確認だけ可能。 threshold=0 なら全件 hit、 threshold>0 なら 0 件 hit。
//! - 本格実装は別 task (1-2 日工)、 `rshogi_core::nnue::evaluate_dispatch`
//!   API + `init_nnue(eval_file)` で置換予定。 詳細は
//!   `docs/design/hard_sample_mining.md` 参照。

use anyhow::{anyhow, Context, Result};
use bucket_profiler::{PackedSfenValue, PSV_RECORD_SIZE};
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

use bucket_profiler::unpack_sfen;

const WRITE_BUF_BYTES: usize = 4 << 20; // 4 MB
const CHANNEL_CAP: usize = 64 * 1024;

/// CLI 引数。
#[derive(Parser, Debug)]
#[command(
    name = "hard-mining",
    about = "Stage 1 ckpt 予測 cp と真 cp の差が大きい hard 局面を抽出する (skeleton)",
    version
)]
struct Args {
    /// 入力 PSV (カンマ区切りで複数指定可)。
    #[arg(long, value_delimiter = ',')]
    inputs: Vec<PathBuf>,

    /// 追加で glob 指定。
    #[arg(long)]
    inputs_glob: Option<String>,

    /// Stage 1 / Stage 2 NNUE ckpt (.bin、 quantised 形式)。
    /// NOTE: 現 skeleton では参照のみ (load せず、 placeholder の predict_cp=0 を返す)。
    #[arg(long)]
    eval_file: PathBuf,

    /// progress.bin (将来的に bucket 別 threshold を変えるための参照用、 skeleton では未使用)。
    #[arg(
        long,
        default_value = r"E:\shogi\p2shogi\bullet_checkpoints\_progress\progress.bin"
    )]
    progress: PathBuf,

    /// hard 局面のみ出力先 PSV。
    #[arg(long)]
    output_hard: PathBuf,

    /// |predict_cp - true_cp| > threshold_cp を hard とみなす。
    #[arg(long, default_value_t = 500)]
    threshold_cp: i32,

    /// 並列 read worker 数。
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

enum WriteMsg {
    Record { record: [u8; PSV_RECORD_SIZE] },
}

#[derive(Debug, Clone, Copy, Default)]
struct FileStats {
    processed: u64,
    failed: u64,
    hard_hits: u64,
}

impl FileStats {
    fn merge(&mut self, other: &FileStats) {
        self.processed += other.processed;
        self.failed += other.failed;
        self.hard_hits += other.hard_hits;
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(&args)
}

fn run(args: &Args) -> Result<()> {
    let inputs = collect_inputs(args)?;
    if inputs.is_empty() {
        return Err(anyhow!(
            "入力 PSV が 1 件もありません (--inputs か --inputs-glob を指定してください)"
        ));
    }

    // eval_file の存在チェックのみ実施。 本格実装で load する。
    if !args.eval_file.exists() {
        return Err(anyhow!("eval_file not found: {}", args.eval_file.display()));
    }
    eprintln!(
        "[hard-mining][TODO] eval_file 指定: {} (skeleton では未 load、 predict_cp=0)",
        args.eval_file.display()
    );

    // progress.bin は将来の bucket 別 threshold 用、 skeleton では存在チェックのみ。
    if !args.progress.exists() {
        eprintln!(
            "[hard-mining][warn] progress.bin not found: {} (skeleton では未使用)",
            args.progress.display()
        );
    }

    ensure_output_path(&args.output_hard, args.force)?;

    let (tx, rx) = bounded::<WriteMsg>(CHANNEL_CAP);
    let out_path = args.output_hard.clone();
    let writer_handle = thread::spawn(move || -> Result<u64> {
        let f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&out_path)
            .with_context(|| format!("output_hard open 失敗: {}", out_path.display()))?;
        let mut w = BufWriter::with_capacity(WRITE_BUF_BYTES, f);
        let mut n: u64 = 0;
        while let Ok(msg) = rx.recv() {
            match msg {
                WriteMsg::Record { record } => {
                    w.write_all(&record).context("output_hard write 失敗")?;
                    n += 1;
                }
            }
        }
        w.flush().context("output_hard flush 失敗")?;
        Ok(n)
    });

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
    let threshold = args.threshold_cp;
    let started = Instant::now();

    eprintln!(
        "[hard-mining] starting: {} input files, threads={}, threshold_cp={}, max_records/file={}",
        inputs.len(),
        threads,
        threshold,
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
                        &tx,
                        max_records,
                        threshold,
                        mp_clone.as_ref(),
                        &global_processed,
                        &global_failed,
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[hard-mining][error] {} : {:#}", path.display(), e);
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

    drop(tx);
    let written = writer_handle.join().map_err(|_| anyhow!("writer thread panicked"))??;

    let mut total = FileStats::default();
    for (path, stats) in inputs.iter().zip(file_stats.iter()) {
        total.merge(stats);
        eprintln!(
            "[hard-mining] {} : processed={} failed={} hard={}",
            path.display(),
            stats.processed,
            stats.failed,
            stats.hard_hits,
        );
    }

    let elapsed = started.elapsed();
    let out_size = std::fs::metadata(&args.output_hard).map(|m| m.len()).unwrap_or(0);

    println!("=== hard-mining summary (SKELETON) ===");
    println!("inputs:          {}", inputs.len());
    println!("processed total: {}", total.processed);
    println!("failed decode:   {}", total.failed);
    println!(
        "hard hits:       {} ({:.3}%) -> {} ({:.2} MB)",
        total.hard_hits,
        ratio_pct(total.hard_hits, total.processed),
        args.output_hard.display(),
        out_size as f64 / (1024.0 * 1024.0),
    );
    println!("writer wrote:    hard={} (channel consistency check)", written);
    println!("elapsed:         {:.2?}", elapsed);
    println!("NOTE: NNUE evaluation 未実装、 predict_cp=0 placeholder で動作");

    if written != total.hard_hits {
        return Err(anyhow!(
            "writer count mismatch: writer={} worker={}",
            written,
            total.hard_hits,
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
                Err(e) => eprintln!("[hard-mining][warn] glob entry error: {e}"),
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

fn ensure_output_path(out: &Path, force: bool) -> Result<()> {
    if out.exists() {
        if !force {
            return Err(anyhow!(
                "output file already exists: {} (use --force to overwrite)",
                out.display()
            ));
        } else {
            eprintln!("[hard-mining][warn] overwriting existing file: {}", out.display());
        }
    }
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(anyhow!("output parent directory missing: {}", parent.display()));
        }
    }
    Ok(())
}

/// Stage 1 ckpt の予測 cp を返す。
///
/// **TODO (skeleton)**: 現状は常に 0 を返す placeholder。
/// 本格実装では `rshogi_core::nnue::evaluate_dispatch(pos, stack, ...)` を呼ぶ。
/// 移行手順:
///   1. main() 冒頭で `rshogi_core::nnue::init_nnue(&args.eval_file)?` を呼ぶ
///   2. process_file の thread local に NNUE accumulator stack を確保
///   3. 各 record で `set_sfen` 後に `evaluate_dispatch` で cp 算出
///   4. SCALE 換算は scale=600 想定 (Stage 1 ckpt 設定に従う)
fn predict_cp_placeholder(_pos: &Position) -> i32 {
    0
}

/// 1 file を stream read + 予測 cp 計算 + hard 判定 + 送信。
fn process_file(
    path: &Path,
    tx: &Sender<WriteMsg>,
    max_records: u64,
    threshold_cp: i32,
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

        let mut hard = false;
        match PackedSfenValue::from_bytes(&buf) {
            Some(psv) => {
                let decode_ok = match unpack_sfen(&psv.sfen) {
                    Ok(sfen) => scratch.set_sfen(&sfen).is_ok(),
                    Err(_) => false,
                };
                if decode_ok {
                    let predict = predict_cp_placeholder(&scratch);
                    let true_cp = psv.score as i32;
                    if (predict - true_cp).abs() > threshold_cp {
                        hard = true;
                    }
                } else {
                    stats.failed += 1;
                    global_failed.fetch_add(1, Ordering::Relaxed);
                }
            }
            None => {
                stats.failed += 1;
                global_failed.fetch_add(1, Ordering::Relaxed);
            }
        }

        if hard {
            stats.hard_hits += 1;
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
                "[hard-mining] global processed={} failed={}",
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
