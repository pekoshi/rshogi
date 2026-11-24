//! PSV → MATE_THREAT 2 bit 注釈付与 tool。
//!
//! 各 PSV record で `mate_probe_3ply` を実行、 my-mate / opponent-mate を検出して
//! 1 byte/record の 別 file に出力。 もともと LMAS bit 18/19 用に開発したが、
//! LMAS は 2026-05-25 廃止。 sidecar format 自体は generic な mate 注釈として
//! 今後 dataset 強化 (= 詰めろ局面の loss 加重等) に再利用できるため tool は維持する。
//!
//! ## 出力 format
//! 各 record = 1 byte:
//! - bit 0: my mate detected (= 自手番で 3 手詰めあり)
//! - bit 1: opponent mate detected (= 相手手番で 3 手詰めあり)
//! - bit 2-7: reserved
//!
//! ## 並列化
//! - default で rayon の global pool (= nproc 自動検出) を使った chunk 並列
//! - chunk size records を main thread で seq read → par_iter で並列処理 →
//!   出力 Vec<u8> を chunk 順 (= 入力 PSV 順) で seq write することで
//!   **output 順序が PSV と完全 1:1 対応** を保つ。
//! - `--thread-count 1` で従来の single thread mode に戻せる (= 互換性確保)。
//!
//! ## 使用例
//! ```
//! preprocess-mate-threat \
//!   --input /data/dataset/dlsuisho15b/shuffled_01.bin \
//!   --output /data/dataset/dlsuisho15b/shuffled_01.mate_threat.bin \
//!   --max-nodes 100
//! ```

use anyhow::{anyhow, Context, Result};
use bucket_profiler::{unpack_sfen, PackedSfenValue, PSV_RECORD_SIZE};
use clap::Parser;
use p2_search::mate_probe::{mate_probe_3ply, MateResult};
use rayon::prelude::*;
use rshogi_core::position::Position;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "preprocess-mate-threat", version)]
struct Args {
    /// 入力 PSV file
    #[arg(long)]
    input: PathBuf,
    /// 出力 MATE_THREAT 注釈 file (= 1 byte/record、 PSV と record 数一致)
    #[arg(long)]
    output: PathBuf,
    /// mate_probe_3ply max nodes (= ノード上限)
    #[arg(long, default_value = "100")]
    max_nodes: u32,
    /// progress log 間隔 (= records 毎)
    #[arg(long, default_value = "100000")]
    log_interval: u64,
    /// 最大処理 records (= 0 で無制限、 smoke test 用)
    #[arg(long, default_value = "0")]
    max_records: u64,
    /// 並列 thread 数 (= 0 で rayon default = nproc 自動検出、 1 で single thread mode)
    #[arg(long, default_value = "0")]
    thread_count: usize,
    /// 1 chunk あたりの records 数 (= 並列単位、 大きすぎると tail 残り、 小さすぎると overhead)
    #[arg(long, default_value = "10000")]
    chunk_size: usize,
}

/// 1 record 分の処理 (= PSV bytes → mate bits)。
/// 戻り値: (bits, my_mate_detected, opp_mate_detected, both_detected)
fn process_record(
    scratch: &mut Position,
    record: &[u8],
    max_nodes: u32,
) -> Result<(u8, bool, bool)> {
    let psv = PackedSfenValue::from_bytes(record).ok_or_else(|| anyhow!("PSV parse failed"))?;
    let sfen = unpack_sfen(&psv.sfen).map_err(|e| anyhow!("unpack_sfen: {e}"))?;
    scratch.set_sfen(&sfen).map_err(|e| anyhow!("set_sfen: {e}"))?;

    let mut bits = 0u8;

    // bit 0: my mate detected (= 自手番で 3 手詰めあり)
    let my_mate = matches!(mate_probe_3ply(scratch, max_nodes), MateResult::MateFound { .. });
    if my_mate {
        bits |= 0x01;
    }

    // bit 1: opponent mate detected (= null move で相手手番に切替、 3 手詰め検出)
    // do_null_move は in_check 局面では panic する場合があるので、
    // 安全のため check 中なら skip (= opponent mate detection skipped)。
    let mut opp_mate = false;
    if !scratch.in_check() {
        scratch.do_null_move();
        opp_mate = matches!(mate_probe_3ply(scratch, max_nodes), MateResult::MateFound { .. });
        if opp_mate {
            bits |= 0x02;
        }
        scratch.undo_null_move();
    }

    Ok((bits, my_mate, opp_mate))
}

fn main() -> Result<()> {
    let args = Args::parse();

    // rayon thread pool 構成 (= thread_count > 0 で明示指定、 0 で default = nproc)
    if args.thread_count > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.thread_count)
            .build_global()
            .context("failed to build rayon thread pool")?;
    }
    let active_threads = rayon::current_num_threads();

    let in_file =
        File::open(&args.input).with_context(|| format!("failed to open {:?}", args.input))?;
    let in_size = in_file.metadata()?.len();
    let total_records = in_size / PSV_RECORD_SIZE as u64;
    eprintln!("Input: {:?} ({} bytes, {} records)", args.input, in_size, total_records);
    eprintln!("Output: {:?}", args.output);
    eprintln!("max_nodes: {}", args.max_nodes);
    eprintln!(
        "Parallelism: {} threads, chunk_size={} records",
        active_threads, args.chunk_size
    );

    let limit = if args.max_records > 0 {
        args.max_records.min(total_records)
    } else {
        total_records
    };

    let mut reader = BufReader::with_capacity(16 << 20, in_file);
    let mut writer = BufWriter::with_capacity(
        16 << 20,
        File::create(&args.output)
            .with_context(|| format!("failed to create {:?}", args.output))?,
    );

    let chunk_records = args.chunk_size.max(1);
    let chunk_bytes = chunk_records * PSV_RECORD_SIZE;

    // counters (= atomic で thread 間共有)
    let my_mate_count = AtomicU64::new(0);
    let opp_mate_count = AtomicU64::new(0);
    let both_count = AtomicU64::new(0);

    let mut count: u64 = 0;
    let mut chunk_buf: Vec<u8> = vec![0u8; chunk_bytes];
    let mut next_log_threshold = args.log_interval;
    let started = Instant::now();

    while count < limit {
        let remaining = (limit - count) as usize;
        let this_records = chunk_records.min(remaining);
        let this_bytes = this_records * PSV_RECORD_SIZE;

        // chunk を seq read (= input 順 を保つ起点)
        match reader.read_exact(&mut chunk_buf[..this_bytes]) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }

        // chunk 内 records を rayon par で並列処理。
        // output は index 順 (= record index) の Vec<u8> に直接書き込むため、
        // chunk 内順序は保証される (= write は chunk 全体を seq write するので
        // 全体順序も input PSV と完全 1:1 対応)。
        let max_nodes = args.max_nodes;
        let chunk_input = &chunk_buf[..this_bytes];
        let result: Result<Vec<u8>> = (0..this_records)
            .into_par_iter()
            .map_init(Position::new, |scratch, idx| -> Result<u8> {
                let offset = idx * PSV_RECORD_SIZE;
                let record = &chunk_input[offset..offset + PSV_RECORD_SIZE];
                let (bits, my_mate, opp_mate) = process_record(scratch, record, max_nodes)
                    .with_context(|| format!("record idx (chunk-local) = {idx}"))?;
                if my_mate {
                    my_mate_count.fetch_add(1, Ordering::Relaxed);
                }
                if opp_mate {
                    opp_mate_count.fetch_add(1, Ordering::Relaxed);
                }
                if my_mate && opp_mate {
                    both_count.fetch_add(1, Ordering::Relaxed);
                }
                Ok(bits)
            })
            .collect();

        let bits_vec = result?;
        writer.write_all(&bits_vec)?;

        count += this_records as u64;

        // progress log (= log_interval 越え時)
        if count >= next_log_threshold {
            let elapsed = started.elapsed().as_secs_f64();
            let rate = count as f64 / elapsed;
            let eta_secs = (total_records.saturating_sub(count)) as f64 / rate.max(1.0);
            eprintln!(
                "[{:>6.1}s] {:>10}/{} ({:>3.1}%) {:>7.0}/s | my_mate={} opp_mate={} both={} | ETA {:>6.1}s",
                elapsed,
                count,
                total_records,
                100.0 * count as f64 / total_records as f64,
                rate,
                my_mate_count.load(Ordering::Relaxed),
                opp_mate_count.load(Ordering::Relaxed),
                both_count.load(Ordering::Relaxed),
                eta_secs,
            );
            // 次の log threshold (= log_interval 単位で切り上げ)
            next_log_threshold = ((count / args.log_interval) + 1) * args.log_interval;
        }
    }

    writer.flush()?;
    let elapsed = started.elapsed().as_secs_f64();
    let my_mate_final = my_mate_count.load(Ordering::Relaxed);
    let opp_mate_final = opp_mate_count.load(Ordering::Relaxed);
    let both_final = both_count.load(Ordering::Relaxed);
    let pct = |n: u64| 100.0 * n as f64 / count.max(1) as f64;
    eprintln!("\n=== Summary ===");
    eprintln!(
        "Total: {} records in {:.1}s ({:.0}/s, {} threads)",
        count,
        elapsed,
        count as f64 / elapsed,
        active_threads,
    );
    eprintln!("My mate detected:       {} ({:.2}%)", my_mate_final, pct(my_mate_final));
    eprintln!("Opponent mate detected: {} ({:.2}%)", opp_mate_final, pct(opp_mate_final));
    eprintln!("Both detected:          {} ({:.2}%)", both_final, pct(both_final));
    Ok(())
}
