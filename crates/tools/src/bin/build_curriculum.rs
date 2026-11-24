//! build_curriculum - カリキュラム PSV を訓練用に構築する汎用ツール
//!
//! `--seg NAME=PATH:PLYMIN:PLYMAX:COUNT` を優先度順 (先頭ほど高優先) に複数指定する。
//! 各セグメントを game_ply でフィルタし reservoir sampling で COUNT 件抽出、
//! 32byte packed sfen をキーに record 単位 dedup して結合する。
//!
//! - 先頭セグメントを先に処理 → global_seen に登録 → 後続で同一 packed sfen は skip
//!   (= 先頭セグメント版を保持。定跡を先頭に置けば「定跡優先 dedup」になる)。
//! - 全体で packed sfen は unique (record 単位 dedup)。
//! - mate/異常 score (|score| >= 30000) は skip。
//!
//! 出力は unshuffled。別途 shuffle_psv でシャッフルする。
//!
//! 例 (Stage1 核 = 定跡優先 + rin 主導):
//! ```bash
//! build_curriculum \
//!   --seg petabook=petabook_12m.psv:0:max:6000000 \
//!   --seg rin=rin_head.psv:0:max:43600000 \
//!   --seg entering=suisho5_entering_king:0:max:12000000 \
//!   --out stage1_core_unshuf.psv
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

const RECORD_SIZE: usize = 40;
/// 読み込みバッファ (16MB, 40 の倍数)
const READ_CHUNK: usize = 16 * 1024 * 1024 / RECORD_SIZE * RECORD_SIZE;
/// mate/異常 score 閾値 (これ以上の |score| は skip)
const MATE_THRESHOLD: i32 = 30000;

#[derive(Parser)]
#[command(
    name = "build_curriculum",
    about = "カリキュラム PSV を構築 (セグメント別 ply フィルタ + reservoir sampling + 先頭優先 dedup)"
)]
struct Cli {
    /// セグメント指定 (優先度順, 先頭ほど高優先=dedup で残る)。
    /// 書式: `NAME=PATH:PLYMIN:PLYMAX:COUNT`
    ///   PATH   = ファイル or ディレクトリ(ディレクトリは *.bin を全採用)
    ///   PLYMIN = 最小 game_ply (含む)
    ///   PLYMAX = 最大 game_ply (含む, 空 or `max` で上限なし)
    ///   COUNT  = reservoir 目標件数 (dedup 前)
    #[arg(long = "seg", required = true)]
    seg: Vec<String>,
    /// 出力 (unshuffled PSV)
    #[arg(long)]
    out: PathBuf,
    /// 乱数シード
    #[arg(long, default_value_t = 20260707u64)]
    seed: u64,
}

/// セグメント spec parse 結果
struct SegSpec {
    name: String,
    paths: Vec<PathBuf>,
    ply_min: u16,
    ply_max: u16,
    count: usize,
}

fn parse_seg(spec: &str) -> Result<SegSpec> {
    let (name, rest) = spec
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("seg '{spec}' に '=' がありません"))?;
    // PATH は `E:/...` のように ':' を含みうるため、末尾 3 フィールドを右から切り出す。
    let mut it = rest.rsplitn(4, ':');
    let count_s = it.next();
    let plymax_s = it.next();
    let plymin_s = it.next();
    let path_s = it.next();
    let (Some(path_s), Some(plymin_s), Some(plymax_s), Some(count_s)) =
        (path_s, plymin_s, plymax_s, count_s)
    else {
        anyhow::bail!("seg '{spec}' は NAME=PATH:PLYMIN:PLYMAX:COUNT 形式が必要");
    };
    let parts = [path_s, plymin_s, plymax_s, count_s];
    let path = PathBuf::from(parts[0]);
    let paths = if std::fs::metadata(&path)
        .with_context(|| format!("stat {}", path.display()))?
        .is_dir()
    {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("bin"))
            .collect();
        v.sort();
        v
    } else {
        vec![path]
    };
    let ply_min: u16 = parts[1].parse().context("PLYMIN parse")?;
    let ply_max: u16 = if parts[2].is_empty() || parts[2] == "max" {
        u16::MAX
    } else {
        parts[2].parse().context("PLYMAX parse")?
    };
    let count: usize = parts[3].parse().context("COUNT parse")?;
    Ok(SegSpec {
        name: name.to_string(),
        paths,
        ply_min,
        ply_max,
        count,
    })
}

/// 1 セグメント bucket
struct Bucket {
    ply_min: u16,
    ply_max: u16,
    target: usize,
    reservoir: Vec<[u8; RECORD_SIZE]>,
    /// filter + global_seen 通過数 (reservoir 母数)
    passed: u64,
}

impl Bucket {
    fn new(ply_min: u16, ply_max: u16, target: usize) -> Self {
        Self {
            ply_min,
            ply_max,
            target,
            reservoir: Vec::with_capacity(target.min(4_000_000)),
            passed: 0,
        }
    }

    #[inline]
    fn matches(&self, ply: u16) -> bool {
        ply >= self.ply_min && ply <= self.ply_max
    }

    /// Algorithm R reservoir sampling
    #[inline]
    fn offer(&mut self, rec: &[u8; RECORD_SIZE], rng: &mut StdRng) {
        self.passed += 1;
        if self.reservoir.len() < self.target {
            self.reservoir.push(*rec);
        } else if self.target > 0 {
            let j = rng.random_range(0..self.passed);
            if (j as usize) < self.target {
                self.reservoir[j as usize] = *rec;
            }
        }
    }
}

#[inline]
fn ply_of(rec: &[u8]) -> u16 {
    u16::from_le_bytes([rec[36], rec[37]])
}

#[inline]
fn score_of(rec: &[u8]) -> i16 {
    i16::from_le_bytes([rec[32], rec[33]])
}

#[inline]
fn key_of(rec: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 32];
    k.copy_from_slice(&rec[0..32]);
    k
}

/// 異常/mate record か
#[inline]
fn is_bad(rec: &[u8]) -> bool {
    let s = score_of(rec) as i32;
    s.abs() >= MATE_THRESHOLD
}

/// 1 record を処理 (filter → bucket 振り分け → reservoir)
struct Counters {
    read: u64,
    bad: u64,
    seen_skip: u64,
}

#[inline]
fn handle_record(
    rec: &[u8; RECORD_SIZE],
    buckets: &mut [Bucket],
    seen: &HashSet<[u8; 32]>,
    rng: &mut StdRng,
    c: &mut Counters,
) {
    c.read += 1;
    if is_bad(rec) {
        c.bad += 1;
        return;
    }
    let ply = ply_of(rec);
    let mut idx = usize::MAX;
    for (bi, b) in buckets.iter().enumerate() {
        if b.matches(ply) {
            idx = bi;
            break;
        }
    }
    if idx == usize::MAX {
        return;
    }
    if seen.contains(&key_of(rec)) {
        c.seen_skip += 1;
        return;
    }
    buckets[idx].offer(rec, rng);
}

/// ファイル群 (複数 bucket 同時) を 1 pass で reservoir sampling
fn process_group(
    paths: &[PathBuf],
    buckets: &mut [Bucket],
    seen: &HashSet<[u8; 32]>,
    rng: &mut StdRng,
) -> Result<(u64, u64, u64)> {
    let mut c = Counters {
        read: 0,
        bad: 0,
        seen_skip: 0,
    };
    let mut buf = vec![0u8; READ_CHUNK];
    let mut carry: Vec<u8> = Vec::with_capacity(RECORD_SIZE);

    for path in paths {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut reader = BufReader::with_capacity(READ_CHUNK, f);
        carry.clear();
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            let mut start = 0usize;
            // 前回の端数 record を完成させる
            if !carry.is_empty() {
                let need = RECORD_SIZE - carry.len();
                let take = need.min(n);
                carry.extend_from_slice(&buf[..take]);
                start = take;
                if carry.len() == RECORD_SIZE {
                    let rec: [u8; RECORD_SIZE] = carry[..].try_into().unwrap();
                    handle_record(&rec, buckets, seen, rng, &mut c);
                    carry.clear();
                }
            }
            let data = &buf[start..n];
            let full = data.len() / RECORD_SIZE;
            for i in 0..full {
                let mut rec = [0u8; RECORD_SIZE];
                rec.copy_from_slice(&data[i * RECORD_SIZE..(i + 1) * RECORD_SIZE]);
                handle_record(&rec, buckets, seen, rng, &mut c);
            }
            let rem = data.len() - full * RECORD_SIZE;
            if rem > 0 {
                carry.extend_from_slice(&data[full * RECORD_SIZE..]);
            }
        }
    }
    Ok((c.read, c.bad, c.seen_skip))
}

/// reservoir を dedup (内部) → global_seen に登録 → 出力
fn commit_bucket(
    b: &mut Bucket,
    seen: &mut HashSet<[u8; 32]>,
    writer: &mut BufWriter<File>,
) -> Result<(u64, u64)> {
    let mut written: u64 = 0;
    let mut intra_dup: u64 = 0;
    for rec in b.reservoir.drain(..) {
        let key = key_of(&rec);
        if seen.contains(&key) {
            intra_dup += 1;
            continue;
        }
        seen.insert(key);
        writer.write_all(&rec)?;
        written += 1;
    }
    b.reservoir.shrink_to_fit();
    Ok((written, intra_dup))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut rng = StdRng::seed_from_u64(cli.seed);

    let specs: Vec<SegSpec> = cli.seg.iter().map(|s| parse_seg(s)).collect::<Result<_>>()?;
    let total_target: usize = specs.iter().map(|s| s.count).sum();

    eprintln!("=== build_curriculum (generic, priority-ordered) ===");
    for s in &specs {
        eprintln!(
            "  seg {:20} files={:>3} ply=[{}..{}] target={}",
            s.name,
            s.paths.len(),
            s.ply_min,
            if s.ply_max == u16::MAX {
                "max".to_string()
            } else {
                s.ply_max.to_string()
            },
            s.count
        );
    }

    let mut seen: HashSet<[u8; 32]> = HashSet::with_capacity(total_target + total_target / 4);
    let out_file =
        File::create(&cli.out).with_context(|| format!("create {}", cli.out.display()))?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, out_file);

    // (name, passed, written, intra/cross_dup)
    let mut report: Vec<(String, u64, u64, u64)> = Vec::new();

    for s in &specs {
        let mut b = Bucket::new(s.ply_min, s.ply_max, s.count);
        let (rd, bad, sn) = process_group(&s.paths, std::slice::from_mut(&mut b), &seen, &mut rng)?;
        let passed = b.passed;
        let (w, dup) = commit_bucket(&mut b, &mut seen, &mut writer)?;
        eprintln!(
            "[{}] read={rd} bad={bad} seen_skip={sn} passed={passed} written={w} intra/cross_dup={dup}",
            s.name
        );
        report.push((s.name.clone(), passed, w, dup));
    }

    writer.flush()?;
    let total_written: u64 = report.iter().map(|r| r.2).sum();

    eprintln!("\n=== FINAL (unique, first-seg priority) ===");
    eprintln!("total written = {total_written}");
    for (name, passed, written, dup) in &report {
        let pct = if total_written > 0 {
            *written as f64 / total_written as f64 * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {name:20} written={written:>11} ({pct:5.2}%)  passed={passed} intra/cross_dup={dup}"
        );
    }
    eprintln!("output: {}", cli.out.display());
    Ok(())
}
