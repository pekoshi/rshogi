//! bucket_jump_measure - bucket 境界 jump を実局面で計測する診断 tool (= 2026-06-09)。
//!
//! 同一局面に対して通常 bucket k と 隣接 bucket (k-1, k+1) を強制投入し、
//! eval Δcp 分布を計測する。 head 別 (= L3 main / PSQT / D2 cross / PairShape / AR / KED)
//! 分解 + 境界別 / eval 値帯別の統計集計を出力する。
//!
//! 対局生成は行わず、 既存 SFEN list を入力とする。
//!
//! # 使用方法
//!
//! ```bash
//! cargo run --release -p tools --bin bucket_jump_measure -- \
//!   --nnue path/to/quantised.bin \
//!   --sfens path/to/sfens.txt \
//!   --progress-coeff path/to/progress.bin \
//!   --output-csv out.csv \
//!   --output-md out.md \
//!   --count 1000
//! ```

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::mem::size_of;
use std::path::PathBuf;

use rshogi_core::nnue::{
    AccumulatorLayerStacks, LayerStackBucketMode, LayerStacksNetwork, NNUENetwork,
    NetworkLayerStacks, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
    compute_layer_stack_progress8kpabs_bucket_index, get_layer_stack_progress_kpabs_weights,
    set_layer_stack_bucket_mode, set_layer_stack_progress_kpabs_weights,
};
use rshogi_core::position::Position;

/// bucket 境界 jump 計測 tool の CLI。
#[derive(Parser)]
#[command(
    name = "bucket_jump_measure",
    about = "bucket 境界 jump を実局面で計測する診断 tool"
)]
struct Cli {
    /// NNUE quantised ckpt path
    #[arg(long)]
    nnue: PathBuf,

    /// SFEN file (= 1 行 1 SFEN)
    #[arg(long)]
    sfens: PathBuf,

    /// progress.bin (= bucket 計算用 KP-absolute 重み)
    #[arg(long)]
    progress_coeff: PathBuf,

    /// 最大評価局面数 (= 0 で全件)
    #[arg(long, default_value_t = 1000)]
    count: usize,

    /// 出力 CSV path (= 全 record 行)
    #[arg(long)]
    output_csv: Option<PathBuf>,

    /// 出力 Markdown path (= 集計レポート)
    #[arg(long)]
    output_md: Option<PathBuf>,

    /// Bucket mode (現状 progress8kpabs のみ)
    #[arg(long, value_enum, default_value = "progress8kpabs")]
    bucket_mode: BucketMode,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BucketMode {
    Progress8kpabs,
}

/// progress.bin (= KP-absolute 重み) を f32 配列で読み込む。
fn load_progress_coeff_kpabs(path: &PathBuf) -> Result<Box<[f32]>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read --progress-coeff '{}'", path.display()))?;
    let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f64>();
    anyhow::ensure!(
        bytes.len() == expected,
        "progress.bin size mismatch: got {} bytes, expected {}",
        bytes.len(),
        expected
    );
    let weights: Vec<f32> = bytes
        .chunks_exact(size_of::<f64>())
        .map(|chunk| {
            f64::from_le_bytes(chunk.try_into().expect("chunk size guaranteed by chunks_exact"))
                as f32
        })
        .collect();
    Ok(weights.into_boxed_slice())
}

/// 1 record = 1 SFEN × 1 比較 bucket。
#[derive(Clone, Copy, Debug)]
struct JumpRecord {
    sfen_idx: usize,
    normal_bucket: usize,
    compare_bucket: usize,
    eval_normal: i32,
    eval_compare: i32,
    delta_cp: i32,
    abs_delta_cp: i32,
    sign_flipped: bool,
    // head 別 Δcp (= compare - normal)
    delta_raw: i32,
    delta_psqt: i32,
    delta_dual_cross: i32,
    delta_pairshape: i32,
    delta_activity_risk: i32,
    delta_king_escape: i32,
}

/// eval 値帯 (= |eval| 範囲)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum EvalBand {
    Under100,
    From100To300,
    From300To1000,
    Above1000,
}

impl EvalBand {
    fn from_eval(eval: i32) -> Self {
        let a = eval.unsigned_abs();
        if a < 100 {
            Self::Under100
        } else if a < 300 {
            Self::From100To300
        } else if a < 1000 {
            Self::From300To1000
        } else {
            Self::Above1000
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Under100 => "|eval|<100",
            Self::From100To300 => "100-300",
            Self::From300To1000 => "300-1000",
            Self::Above1000 => "1000+",
        }
    }
}

/// 集計統計 (= 1 group 単位)。
#[derive(Clone, Debug, Default)]
struct AggStats {
    count: usize,
    sum_signed: i64,
    sum_abs: i64,
    abs_values: Vec<i32>,
    sign_flipped_count: usize,
}

impl AggStats {
    fn add(&mut self, r: &JumpRecord) {
        self.count += 1;
        self.sum_signed += r.delta_cp as i64;
        self.sum_abs += r.abs_delta_cp as i64;
        self.abs_values.push(r.abs_delta_cp);
        if r.sign_flipped {
            self.sign_flipped_count += 1;
        }
    }

    fn finalize(&mut self) -> AggSummary {
        self.abs_values.sort_unstable();
        let n = self.abs_values.len();
        let pct = |p: f64| -> i32 {
            if n == 0 {
                return 0;
            }
            let idx = ((p * n as f64) as usize).min(n - 1);
            self.abs_values[idx]
        };
        AggSummary {
            count: self.count,
            mean_signed: if n > 0 {
                self.sum_signed as f64 / n as f64
            } else {
                0.0
            },
            mean_abs: if n > 0 {
                self.sum_abs as f64 / n as f64
            } else {
                0.0
            },
            median: pct(0.50),
            p90: pct(0.90),
            p95: pct(0.95),
            p99: pct(0.99),
            max: self.abs_values.last().copied().unwrap_or(0),
            sign_flip_rate: if n > 0 {
                self.sign_flipped_count as f64 / n as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Clone, Debug, Default)]
struct AggSummary {
    count: usize,
    mean_signed: f64,
    mean_abs: f64,
    median: i32,
    p90: i32,
    p95: i32,
    p99: i32,
    max: i32,
    sign_flip_rate: f64,
}

/// 計測本体: 各 SFEN で normal / adjacent bucket を強制 evaluate して record を返す。
fn measure_for_network<
    const L1: usize,
    const LS_L1_OUT: usize,
    const LS_L2_IN: usize,
    const LS_L2_PADDED_INPUT: usize,
    const LS_L3: usize,
>(
    cli: &Cli,
    network: &NetworkLayerStacks<L1, LS_L1_OUT, LS_L2_IN, LS_L2_PADDED_INPUT, LS_L3>,
) -> Result<Vec<JumpRecord>> {
    let file =
        File::open(&cli.sfens).with_context(|| format!("Failed to open: {:?}", cli.sfens))?;
    let reader = BufReader::new(file);

    let mut pos = Position::new();
    let mut acc = AccumulatorLayerStacks::<L1>::new();
    let mut records = Vec::new();

    let max = if cli.count == 0 {
        usize::MAX
    } else {
        cli.count
    };

    for (i, line) in reader.lines().enumerate() {
        if records.len() >= max {
            break;
        }
        let raw_line = line?;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // 「sfen <board> ...」 prefix 形式 (= USI position 命令形式) と素 SFEN の両対応
        let sfen = trimmed.strip_prefix("sfen ").unwrap_or(trimmed);

        if let Err(e) = pos.set_sfen(sfen) {
            eprintln!("warn: skip invalid SFEN at line {i}: {e:?}");
            continue;
        }
        network.refresh_accumulator(&pos, &mut acc);

        let side_to_move = pos.side_to_move();
        let weights = get_layer_stack_progress_kpabs_weights();
        let normal_bucket =
            compute_layer_stack_progress8kpabs_bucket_index(&pos, side_to_move, weights);

        let bd_normal = network.evaluate_with_breakdown(&mut pos, &acc, normal_bucket);

        let adjacent: [i32; 2] = [normal_bucket as i32 - 1, normal_bucket as i32 + 1];
        for adj_signed in adjacent.iter() {
            if !(0..=7).contains(adj_signed) {
                continue;
            }
            let compare_bucket = *adj_signed as usize;
            let bd_compare = network.evaluate_with_breakdown(&mut pos, &acc, compare_bucket);

            let delta_cp = bd_compare.total_cp.saturating_sub(bd_normal.total_cp);
            let sign_flipped =
                (bd_normal.total_cp >= 0) != (bd_compare.total_cp >= 0) && delta_cp != 0;

            records.push(JumpRecord {
                sfen_idx: i,
                normal_bucket,
                compare_bucket,
                eval_normal: bd_normal.total_cp,
                eval_compare: bd_compare.total_cp,
                delta_cp,
                abs_delta_cp: delta_cp.unsigned_abs() as i32,
                sign_flipped,
                delta_raw: bd_compare.raw_score.saturating_sub(bd_normal.raw_score),
                delta_psqt: bd_compare.psqt_value.saturating_sub(bd_normal.psqt_value),
                delta_dual_cross: bd_compare
                    .dual_cross_value
                    .saturating_sub(bd_normal.dual_cross_value),
                delta_pairshape: bd_compare
                    .pairshape_value
                    .saturating_sub(bd_normal.pairshape_value),
                delta_activity_risk: bd_compare
                    .activity_risk_value
                    .saturating_sub(bd_normal.activity_risk_value),
                delta_king_escape: bd_compare
                    .king_escape_value
                    .saturating_sub(bd_normal.king_escape_value),
            });
        }
    }

    eprintln!(
        "measured {} records from {:?}",
        records.len(),
        cli.sfens.file_name().unwrap_or_default()
    );
    Ok(records)
}

/// 全 record を CSV (= header + 行) 形式で書き出す。
fn write_csv(records: &[JumpRecord], path: &PathBuf) -> Result<()> {
    let mut f =
        File::create(path).with_context(|| format!("failed to create CSV: {}", path.display()))?;
    writeln!(
        f,
        "sfen_idx,normal_bucket,compare_bucket,eval_normal_cp,eval_compare_cp,\
         delta_cp,abs_delta_cp,sign_flipped,\
         delta_raw,delta_psqt,delta_dual_cross,delta_pairshape,delta_activity_risk,delta_king_escape"
    )?;
    for r in records {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.sfen_idx,
            r.normal_bucket,
            r.compare_bucket,
            r.eval_normal,
            r.eval_compare,
            r.delta_cp,
            r.abs_delta_cp,
            r.sign_flipped,
            r.delta_raw,
            r.delta_psqt,
            r.delta_dual_cross,
            r.delta_pairshape,
            r.delta_activity_risk,
            r.delta_king_escape,
        )?;
    }
    Ok(())
}

/// 集計レポート Markdown を書き出す。
fn write_markdown(records: &[JumpRecord], cli: &Cli, path: &PathBuf) -> Result<()> {
    let mut f = File::create(path)
        .with_context(|| format!("failed to create Markdown: {}", path.display()))?;

    writeln!(f, "# bucket 境界 jump 計測レポート")?;
    writeln!(f)?;
    writeln!(f, "## 入力")?;
    writeln!(f)?;
    writeln!(f, "- nnue: `{}`", cli.nnue.display())?;
    writeln!(f, "- sfens: `{}`", cli.sfens.display())?;
    writeln!(f, "- progress_coeff: `{}`", cli.progress_coeff.display())?;
    writeln!(f, "- record 数: {}", records.len())?;
    writeln!(f)?;

    // 全体集計
    let mut all = AggStats::default();
    for r in records {
        all.add(r);
    }
    let all_summary = all.finalize();
    writeln!(f, "## 全体集計")?;
    writeln!(f)?;
    write_summary_table(&mut f, "all", &all_summary)?;
    writeln!(f)?;

    // 境界別集計
    writeln!(f, "## 境界別集計 (= normal_bucket → compare_bucket)")?;
    writeln!(f)?;
    writeln!(
        f,
        "| boundary | count | mean signed | mean |Δcp| | median | p90 | p95 | p99 | max | sign flip% |"
    )?;
    writeln!(f, "|---|---|---|---|---|---|---|---|---|---|")?;
    for k in 0..8 {
        for &dir in &[-1i32, 1i32] {
            let compare = k as i32 + dir;
            if !(0..=7).contains(&compare) {
                continue;
            }
            let mut agg = AggStats::default();
            for r in records {
                if r.normal_bucket == k && r.compare_bucket == compare as usize {
                    agg.add(r);
                }
            }
            let s = agg.finalize();
            if s.count > 0 {
                writeln!(
                    f,
                    "| b{}→b{} | {} | {:.2} | {:.2} | {} | {} | {} | {} | {} | {:.1}% |",
                    k,
                    compare,
                    s.count,
                    s.mean_signed,
                    s.mean_abs,
                    s.median,
                    s.p90,
                    s.p95,
                    s.p99,
                    s.max,
                    s.sign_flip_rate * 100.0
                )?;
            }
        }
    }
    writeln!(f)?;

    // eval 値帯別集計
    writeln!(f, "## 通常評価値帯別集計")?;
    writeln!(f)?;
    writeln!(
        f,
        "| band | count | mean signed | mean |Δcp| | median | p90 | p95 | p99 | max | sign flip% |"
    )?;
    writeln!(f, "|---|---|---|---|---|---|---|---|---|---|")?;
    for band in [
        EvalBand::Under100,
        EvalBand::From100To300,
        EvalBand::From300To1000,
        EvalBand::Above1000,
    ] {
        let mut agg = AggStats::default();
        for r in records {
            if EvalBand::from_eval(r.eval_normal) == band {
                agg.add(r);
            }
        }
        let s = agg.finalize();
        if s.count > 0 {
            writeln!(
                f,
                "| {} | {} | {:.2} | {:.2} | {} | {} | {} | {} | {} | {:.1}% |",
                band.label(),
                s.count,
                s.mean_signed,
                s.mean_abs,
                s.median,
                s.p90,
                s.p95,
                s.p99,
                s.max,
                s.sign_flip_rate * 100.0
            )?;
        }
    }
    writeln!(f)?;

    // head 別 mean |Δcp|
    writeln!(f, "## head 別 mean |Δcp| (= compare - normal の各 head 寄与)")?;
    writeln!(f)?;
    writeln!(f, "| head | mean |Δcp| (cp) | max |Δcp| (cp) |")?;
    writeln!(f, "|---|---|---|")?;
    let head_specs: [(&str, fn(&JumpRecord) -> i32); 6] = [
        ("raw (= L3 main)", |r| r.delta_raw),
        ("PSQT", |r| r.delta_psqt),
        ("D2 cross threat", |r| r.delta_dual_cross),
        ("PairShape", |r| r.delta_pairshape),
        ("ActivityRisk", |r| r.delta_activity_risk),
        ("KED", |r| r.delta_king_escape),
    ];
    for (label, getter) in head_specs {
        let n = records.len();
        if n == 0 {
            continue;
        }
        let sum_abs: i64 = records.iter().map(|r| getter(r).unsigned_abs() as i64).sum();
        let max_abs: i32 =
            records.iter().map(|r| getter(r).unsigned_abs() as i32).max().unwrap_or(0);
        writeln!(f, "| {} | {:.2} | {} |", label, sum_abs as f64 / n as f64, max_abs)?;
    }

    Ok(())
}

fn write_summary_table(f: &mut File, group: &str, s: &AggSummary) -> Result<()> {
    writeln!(
        f,
        "| group | count | mean signed | mean |Δcp| | median | p90 | p95 | p99 | max | sign flip% |"
    )?;
    writeln!(f, "|---|---|---|---|---|---|---|---|---|---|")?;
    writeln!(
        f,
        "| {} | {} | {:.2} | {:.2} | {} | {} | {} | {} | {} | {:.1}% |",
        group,
        s.count,
        s.mean_signed,
        s.mean_abs,
        s.median,
        s.p90,
        s.p95,
        s.p99,
        s.max,
        s.sign_flip_rate * 100.0
    )?;
    Ok(())
}

/// Tool entry point。
pub fn run() -> Result<()> {
    let cli = Cli::parse();

    // bucket mode の設定 (= progress8kpabs)
    let progress_weights = load_progress_coeff_kpabs(&cli.progress_coeff)
        .map_err(|e| anyhow::anyhow!("progress.bin load error: {e:?}"))?;
    set_layer_stack_progress_kpabs_weights(progress_weights)
        .map_err(|e| anyhow::anyhow!("set_layer_stack_progress_kpabs_weights failed: {e:?}"))?;
    set_layer_stack_bucket_mode(LayerStackBucketMode::Progress8KPAbs);

    eprintln!("loading nnue: {:?}", cli.nnue);
    let bytes = std::fs::read(&cli.nnue)
        .with_context(|| format!("failed to read --nnue '{}'", cli.nnue.display()))?;
    let net = NNUENetwork::from_bytes(&bytes)
        .with_context(|| "NNUENetwork::from_bytes failed (= ckpt format mismatch?)")?;
    let layerstack_net = match net {
        NNUENetwork::LayerStacks(n) => n,
        _ => anyhow::bail!(
            "unsupported NNUE network kind for bucket_jump_measure (= LayerStacks 系のみ)"
        ),
    };
    let records = match layerstack_net {
        LayerStacksNetwork::L1536x16x32(n) => measure_for_network(&cli, &n)?,
        other => anyhow::bail!(
            "unsupported LayerStacks variant for bucket_jump_measure (= L1536x16x32 のみ対応): {:?}",
            std::mem::discriminant(&other)
        ),
    };

    if let Some(p) = &cli.output_csv {
        write_csv(&records, p)?;
        eprintln!("wrote CSV: {}", p.display());
    }
    if let Some(p) = &cli.output_md {
        write_markdown(&records, &cli, p)?;
        eprintln!("wrote Markdown: {}", p.display());
    }

    Ok(())
}
