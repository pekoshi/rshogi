//! bullet ckpt directory (1536-8-96) を 1536-16-96 形式に拡張する。
//!
//! ## 目的
//!
//! bullet `--resume <ckpt_dir>` は `<ckpt_dir>/optimiser_state/` を読み込んで
//! 訓練を継続する。 そのため L1=8 → L1=16 の fine-tune を bullet で resume するには
//! optimiser_state 内の f32 weight tensor を新 shape (l1=16) に展開した directory が必要。
//!
//! 既存の `convert_layerstack_l1_8_to_16` は **推論用** quantised.bin のみ生成するが、
//! bullet resume の入口は quantised.bin ではなく optimiser_state ディレクトリのため、
//! この binary では optimiser_state の各 .bin ファイル (raw f32 weight) を直接展開する。
//!
//! ## bullet ckpt directory 構造 (調査結果)
//!
//! ```text
//! <ckpt>/
//!   quantised.bin                   <- 推論用 (existing tool が生成)
//!   raw.bin                          <- bullet が生成するが現状空 (resume では不要)
//!   log.txt
//!   optimiser_state/                 <- ★ bullet --resume が読む
//!     weights.bin                    <- graph primary weights (主)
//!     momentum.bin                   <- RAdam 1st moment
//!     velocity.bin                   <- RAdam 2nd moment
//!     slow.bin                       <- Ranger lookahead slow params
//!     step.txt                       <- RAdam step counter (per tensor)
//!     step_ranger.txt                <- Ranger lookahead step counter (per tensor)
//! ```
//!
//! ## f32 weight ファイル format (= bullet `utils::write_weights_to_file`)
//!
//! 全 weight tensor を 1 file に連結:
//!
//! ```text
//! [tensor_1][tensor_2]...
//! tensor = <id_ascii> '\n' <single_size: u64 LE 8 bytes> <data: single_size * f32 LE>
//! ```
//!
//! `step.txt` / `step_ranger.txt` は単純な `<id>,<step>\n` の連続。
//!
//! ## 対象 tensor (1536-8-96)
//!
//! 確認済み tensor 一覧 (lmas_p3_full_middle_stage3_15-10 の weights.bin から):
//!
//! | id                          | size       | shape                          | 拡張要否           |
//! |-----------------------------|------------|--------------------------------|--------------------|
//! | l0w                         | 112596480  | [1536, 73305] = ft_out * inp   | 不変               |
//! | l0b                         | 1536       | [1536]                         | 不変               |
//! | l1w                         | 98304      | [NUM_BUCKETS*l1_out, ft_out]   | ★ l1_out 8→16     |
//! | l1b                         | 64         | [NUM_BUCKETS*l1_out]           | ★ l1_out 8→16     |
//! | l1fw                        | 12288      | [l1_out, ft_out]               | ★ l1_out 8→16     |
//! | l1fb                        | 8          | [l1_out]                       | ★ l1_out 8→16     |
//! | l2w                         | 10752      | [NUM_BUCKETS*l2_out, l2_in]    | ★ l2_in 14→30     |
//! | l2b                         | 768        | [NUM_BUCKETS*l2_out]           | 不変               |
//! | l3w                         | 768        | [NUM_BUCKETS, l2_out]          | 不変               |
//! | l3b                         | 8          | [NUM_BUCKETS]                  | 不変               |
//! | lmas_sla_w / sla_b          | 112 / 8    |                                | 不変               |
//! | lmas_hml_w / hml_b          | 56 / 8     |                                | 不変               |
//! | lmas_zpsqt_w_black/white    | 1008 / 1008|                                | 不変               |
//! | lmas_zpsqt_b                | 8          |                                | 不変               |
//! | lmas_zpsqt_b_white_unused   | 8          |                                | 不変               |
//!
//! ## L1 拡張 layout (bullet graph 定義に基づく)
//!
//! - `l1_effective = l1_out - 1` (= 7 for src, = 15 for dst)
//! - `l1_out_t.slice_rows(0, l1_effective)` = **main** (SqrCReLU + CReLU で L2 入力に)
//! - `l1_out_t.slice_rows(l1_effective, l1_out)` = **skip** (l3 出力に加算)
//!
//! src l1_out=8: row 0..7 = main, row 7 = skip
//! dst l1_out=16: row 0..15 = main, row 15 = skip
//!
//! 拡張規則:
//! - `l1w` (column-major `value[in_idx * (NUM_BUCKETS*l1_out) + bucket*l1_out + out_idx]`):
//!   - 旧 main (out 0..7) → 新 main (out 0..7) コピー
//!   - 新 main 拡張 (out 7..15) → 小ガウス乱数 (std ≈ 0.04)
//!   - 旧 skip (out 7) → 新 skip (out 15) コピー
//! - `l1b` (bucket ごとに l1_out 並ぶ):
//!   - 旧 bucket*8+0..7 → 新 bucket*16+0..7 コピー
//!   - 新 bucket*16+7..15 → 0
//!   - 旧 bucket*8+7 → 新 bucket*16+15 コピー
//! - `l1fw` (shape `[l1_out, ft_out]` column-major、 shared):
//!   - 旧 out 0..7 → 新 out 0..7 コピー
//!   - 新 out 7..15 → 0 (新規 main は shared なし)
//!   - 旧 out 7 → 新 out 15 コピー
//! - `l1fb` (shape `[l1_out]`):
//!   - 旧 0..7 → 新 0..7 コピー
//!   - 新 7..15 → 0
//!   - 旧 7 → 新 15 コピー
//!
//! ## L2 拡張 layout
//!
//! l2_input = `l1_sqr.concat(l1_main).crelu()` = [sqr_main(l1_effective) | crelu_main(l1_effective)]
//! src l2_in=14 = sqr(7) + crelu(7)
//! dst l2_in=30 = sqr(15) + crelu(15)
//!
//! `l2w` (column-major `value[in_idx * (NUM_BUCKETS*l2_out) + bucket*l2_out + out_idx]`):
//! - dst col 0..7 = src col 0..7 (sqr main 0..7) コピー
//! - dst col 7..15 = 0 (新規 sqr main 8..15)
//! - dst col 15..22 = src col 7..14 (crelu main 0..7) コピー
//! - dst col 22..30 = 0 (新規 crelu main 8..15)
//!
//! ## momentum/velocity/slow の扱い
//!
//! 拡張部 (新 main row) の statistic はないので 0 init。
//! 既存部 (旧 main 0..7、 旧 skip 7 → 新 skip 15) は元の値をコピー。
//! step.txt / step_ranger.txt は元の値をそのままコピー (resume として step は維持)。
//!
//! ## 使用例
//!
//! ```text
//! cargo run --release -p nnue-convert --bin expand_ckpt_l1_8_to_16 -- \
//!     --input-dir  E:/shogi/p2shogi/bullet_checkpoints/.../lmas_..._15-10 \
//!     --output-dir E:/shogi/p2shogi/bullet_checkpoints/lmas_..._15-10_l1to16 \
//!     --seed 20260514
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

const NUM_BUCKETS: usize = 8;
const FT_OUT: usize = 1536;
const SRC_L1_OUT: usize = 8;
const DST_L1_OUT: usize = 16;
const SRC_L2_IN: usize = 14;
const DST_L2_IN: usize = 30;
const L2_OUT: usize = 96;

/// 新 main 行の f32 初期化標準偏差。
///
/// i8 量子化 (scale=64) で ±5 ≒ 0.08 程度に対応。 新層が学習開始時に強く効きすぎないため、
/// やや控えめの std=0.04 を採用。
const INIT_STD: f32 = 0.04;

#[derive(Parser, Debug)]
#[command(
    name = "expand_ckpt_l1_8_to_16",
    about = "bullet ckpt directory を 1536-8-96 → 1536-16-96 に展開 (raw f32 weights + optimiser_state)"
)]
struct Cli {
    /// 入力 ckpt directory (例: lmas_p3_full_middle_stage3_15-10)
    #[arg(long)]
    input_dir: PathBuf,

    /// 出力 ckpt directory
    #[arg(long)]
    output_dir: PathBuf,

    /// 乱数シード
    #[arg(long, default_value = "20260514")]
    seed: u64,

    /// 新 main 拡張部の f32 std (default: 0.04)
    #[arg(long, default_value_t = INIT_STD)]
    init_std: f32,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    run(&cli)
}

fn run(cli: &Cli) -> Result<()> {
    let in_opt = cli.input_dir.join("optimiser_state");
    if !in_opt.is_dir() {
        bail!("入力 ckpt directory に optimiser_state/ がありません: {:?}", in_opt);
    }
    fs::create_dir_all(&cli.output_dir)
        .with_context(|| format!("出力 directory が作れません: {:?}", cli.output_dir))?;
    let out_opt = cli.output_dir.join("optimiser_state");
    fs::create_dir_all(&out_opt)
        .with_context(|| format!("出力 directory が作れません: {:?}", out_opt))?;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(cli.seed);

    // weight bin (weights.bin, momentum.bin, velocity.bin, slow.bin) を変換
    // momentum/velocity/slow は拡張部を 0 init、 weights は新 main 部のみ乱数
    for (name, init_random_main) in [
        ("weights.bin", true),
        ("momentum.bin", false),
        ("velocity.bin", false),
        ("slow.bin", false),
    ] {
        let src = in_opt.join(name);
        let dst = out_opt.join(name);
        expand_weight_file(&src, &dst, &mut rng, cli.init_std, init_random_main)
            .with_context(|| format!("{name} の展開に失敗"))?;
    }

    // step.txt / step_ranger.txt はそのままコピー
    for name in ["step.txt", "step_ranger.txt"] {
        let src = in_opt.join(name);
        let dst = out_opt.join(name);
        if src.exists() {
            fs::copy(&src, &dst)
                .with_context(|| format!("{name} のコピー失敗 {:?} -> {:?}", src, dst))?;
            println!("[copy] {name}");
        } else {
            println!("[skip] {name} (入力に存在せず)");
        }
    }

    // top-level log.txt はあればコピー (bullet 出力互換のため)
    {
        let name = "log.txt";
        let src = cli.input_dir.join(name);
        let dst = cli.output_dir.join(name);
        if src.exists() {
            fs::copy(&src, &dst)
                .with_context(|| format!("{name} のコピー失敗 {:?} -> {:?}", src, dst))?;
            println!("[copy-top] {name}");
        }
    }
    // raw.bin: touch (空) — bullet save_to_checkpoint 互換 (load_from_checkpoint は読まない)
    let _ = fs::File::create(cli.output_dir.join("raw.bin"))?;
    println!("[touch] raw.bin (empty placeholder)");

    println!("[done] expanded ckpt: {:?}", cli.output_dir);
    Ok(())
}

/// f32 weight ファイル全体を読み込み、 L1/L2 関係 tensor のみ拡張して書き出す。
fn expand_weight_file(
    src_path: &Path,
    dst_path: &Path,
    rng: &mut Xoshiro256PlusPlus,
    init_std: f32,
    init_random_main: bool,
) -> Result<()> {
    let raw =
        fs::read(src_path).with_context(|| format!("入力ファイルが読めません: {:?}", src_path))?;
    let tensors = parse_tensors(&raw)?;
    let mut out = Vec::with_capacity(raw.len() * 2);
    for t in &tensors {
        let new_values = expand_tensor(&t.id, &t.values, rng, init_std, init_random_main)?;
        write_tensor(&mut out, &t.id, &new_values)?;
    }
    let mut f = fs::File::create(dst_path)
        .with_context(|| format!("出力ファイルが作れません: {:?}", dst_path))?;
    f.write_all(&out)?;
    println!(
        "[expand] {:?} ({} bytes) -> {:?} ({} bytes)",
        src_path.file_name().unwrap_or_default(),
        raw.len(),
        dst_path.file_name().unwrap_or_default(),
        out.len()
    );
    Ok(())
}

/// 1 tensor の f32 weight を新 shape に拡張する。 知らない id はそのまま返す。
fn expand_tensor(
    id: &str,
    src: &[f32],
    rng: &mut Xoshiro256PlusPlus,
    init_std: f32,
    init_random_main: bool,
) -> Result<Vec<f32>> {
    match id {
        "l1w" => expand_l1w(src, rng, init_std, init_random_main),
        "l1b" => expand_l1b(src),
        "l1fw" => expand_l1fw(src),
        "l1fb" => expand_l1fb(src),
        "l2w" => expand_l2w(src),
        // 全て不変
        "l0w" | "l0b" | "l2b" | "l3w" | "l3b" => Ok(src.to_vec()),
        // PSQT / Threat / Pin / DualPSQT 等は L1/L2 拡張と無関係なので passthrough
        _ => {
            eprintln!(
                "[warn] 未知 tensor id={s} (size={n}): そのまま passthrough",
                s = id,
                n = src.len()
            );
            Ok(src.to_vec())
        }
    }
}

/// l1w: column-major [NUM_BUCKETS * l1_out, ft_out]
///   src size = NUM_BUCKETS * SRC_L1_OUT * FT_OUT = 8 * 8 * 1536 = 98304
///   dst size = NUM_BUCKETS * DST_L1_OUT * FT_OUT = 8 * 16 * 1536 = 196608
///   index: value[in_idx * (NUM_BUCKETS*l1_out) + bucket*l1_out + out_idx]
fn expand_l1w(
    src: &[f32],
    rng: &mut Xoshiro256PlusPlus,
    init_std: f32,
    init_random_main: bool,
) -> Result<Vec<f32>> {
    let src_rows = NUM_BUCKETS * SRC_L1_OUT;
    let dst_rows = NUM_BUCKETS * DST_L1_OUT;
    let expected_src = src_rows * FT_OUT;
    if src.len() != expected_src {
        bail!("l1w size 不整合: got {}, expected {expected_src}", src.len());
    }
    let mut dst = vec![0.0f32; dst_rows * FT_OUT];
    let src_main = SRC_L1_OUT - 1; // 7
    let dst_main = DST_L1_OUT - 1; // 15
    for in_idx in 0..FT_OUT {
        let src_col_off = in_idx * src_rows;
        let dst_col_off = in_idx * dst_rows;
        for bucket in 0..NUM_BUCKETS {
            // 旧 main (out 0..7) → 新 main (out 0..7) コピー
            for out_idx in 0..src_main {
                let s = src[src_col_off + bucket * SRC_L1_OUT + out_idx];
                dst[dst_col_off + bucket * DST_L1_OUT + out_idx] = s;
            }
            // 新 main 拡張 (out 7..15)
            for out_idx in src_main..dst_main {
                dst[dst_col_off + bucket * DST_L1_OUT + out_idx] = if init_random_main {
                    sample_normal(rng, init_std)
                } else {
                    0.0
                };
            }
            // 旧 skip (out 7) → 新 skip (out 15) コピー
            let s = src[src_col_off + bucket * SRC_L1_OUT + src_main];
            dst[dst_col_off + bucket * DST_L1_OUT + dst_main] = s;
        }
    }
    Ok(dst)
}

/// l1b: [NUM_BUCKETS * l1_out]
fn expand_l1b(src: &[f32]) -> Result<Vec<f32>> {
    let expected_src = NUM_BUCKETS * SRC_L1_OUT;
    if src.len() != expected_src {
        bail!("l1b size 不整合: got {}, expected {expected_src}", src.len());
    }
    let mut dst = vec![0.0f32; NUM_BUCKETS * DST_L1_OUT];
    let src_main = SRC_L1_OUT - 1; // 7
    let dst_main = DST_L1_OUT - 1; // 15
    for bucket in 0..NUM_BUCKETS {
        for out_idx in 0..src_main {
            dst[bucket * DST_L1_OUT + out_idx] = src[bucket * SRC_L1_OUT + out_idx];
        }
        // out_idx 7..15 は 0 init
        dst[bucket * DST_L1_OUT + dst_main] = src[bucket * SRC_L1_OUT + src_main]; // skip
    }
    Ok(dst)
}

/// l1fw: column-major [l1_out, ft_out]
///   src size = SRC_L1_OUT * FT_OUT = 8 * 1536 = 12288
///   dst size = DST_L1_OUT * FT_OUT = 16 * 1536 = 24576
///   index: value[in_idx * l1_out + out_idx]
fn expand_l1fw(src: &[f32]) -> Result<Vec<f32>> {
    let expected_src = SRC_L1_OUT * FT_OUT;
    if src.len() != expected_src {
        bail!("l1fw size 不整合: got {}, expected {expected_src}", src.len());
    }
    let mut dst = vec![0.0f32; DST_L1_OUT * FT_OUT];
    let src_main = SRC_L1_OUT - 1; // 7
    let dst_main = DST_L1_OUT - 1; // 15
    for in_idx in 0..FT_OUT {
        let src_col_off = in_idx * SRC_L1_OUT;
        let dst_col_off = in_idx * DST_L1_OUT;
        // 旧 main 0..7 → 新 main 0..7
        dst[dst_col_off..dst_col_off + src_main]
            .copy_from_slice(&src[src_col_off..src_col_off + src_main]);
        // 新 main 拡張 7..15 = 0 (shared 部は新規 dim には持たせない)
        // 旧 skip 7 → 新 skip 15
        dst[dst_col_off + dst_main] = src[src_col_off + src_main];
    }
    Ok(dst)
}

/// l1fb: [l1_out]
fn expand_l1fb(src: &[f32]) -> Result<Vec<f32>> {
    if src.len() != SRC_L1_OUT {
        bail!("l1fb size 不整合: got {}, expected {SRC_L1_OUT}", src.len());
    }
    let mut dst = vec![0.0f32; DST_L1_OUT];
    let src_main = SRC_L1_OUT - 1; // 7
    let dst_main = DST_L1_OUT - 1; // 15
    dst[..src_main].copy_from_slice(&src[..src_main]);
    dst[dst_main] = src[src_main];
    Ok(dst)
}

/// l2w: column-major [NUM_BUCKETS * l2_out, l2_in]
///   src size = NUM_BUCKETS * L2_OUT * SRC_L2_IN = 8 * 96 * 14 = 10752
///   dst size = NUM_BUCKETS * L2_OUT * DST_L2_IN = 8 * 96 * 30 = 23040
///   index: value[in_idx * (NUM_BUCKETS*l2_out) + bucket*l2_out + out_idx]
///
/// L2 入力 layout: [sqr_main(l1_effective) | crelu_main(l1_effective)]
///   src col 0..7 = sqr (l1 main 0..7)、 col 7..14 = crelu (l1 main 0..7)
///   dst col 0..15 = sqr (l1 main 0..15)、 col 15..30 = crelu (l1 main 0..15)
fn expand_l2w(src: &[f32]) -> Result<Vec<f32>> {
    let rows = NUM_BUCKETS * L2_OUT;
    let expected_src = rows * SRC_L2_IN;
    if src.len() != expected_src {
        bail!("l2w size 不整合: got {}, expected {expected_src}", src.len());
    }
    let mut dst = vec![0.0f32; rows * DST_L2_IN];
    let src_main = SRC_L1_OUT - 1; // 7
    let dst_main = DST_L1_OUT - 1; // 15
    // src col 0..7 (sqr main 0..7) → dst col 0..7
    for src_col in 0..src_main {
        let dst_col = src_col;
        let src_off = src_col * rows;
        let dst_off = dst_col * rows;
        dst[dst_off..dst_off + rows].copy_from_slice(&src[src_off..src_off + rows]);
    }
    // dst col 7..15 (新規 sqr main 8..15) は 0 (既に 0 init)
    // src col 7..14 (crelu main 0..7) → dst col 15..22
    for k in 0..src_main {
        let src_col = src_main + k; // 7..14
        let dst_col = dst_main + k; // 15..22
        let src_off = src_col * rows;
        let dst_off = dst_col * rows;
        dst[dst_off..dst_off + rows].copy_from_slice(&src[src_off..src_off + rows]);
    }
    // dst col 22..30 (新規 crelu main 8..15) は 0
    Ok(dst)
}

struct Tensor {
    id: String,
    values: Vec<f32>,
}

fn parse_tensors(buf: &[u8]) -> Result<Vec<Tensor>> {
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < buf.len() {
        // id (ASCII) + '\n'
        let nl = buf[offset..]
            .iter()
            .position(|&b| b == b'\n')
            .with_context(|| format!("tensor header に '\\n' がない (offset={offset})"))?
            + offset;
        let id = std::str::from_utf8(&buf[offset..nl])
            .with_context(|| format!("tensor id が UTF-8 でない (offset={offset})"))?
            .to_string();
        offset = nl + 1;
        if offset + 8 > buf.len() {
            bail!("tensor size header が読めません (offset={offset})");
        }
        let mut size_bytes = [0u8; 8];
        size_bytes.copy_from_slice(&buf[offset..offset + 8]);
        let size = usize::from_le_bytes(size_bytes);
        offset += 8;
        let data_bytes = size * 4;
        if offset + data_bytes > buf.len() {
            bail!(
                "tensor data 不足: id={id} size={size} need {data_bytes} byte (remaining {})",
                buf.len() - offset
            );
        }
        let mut values = vec![0.0f32; size];
        for (chunk, v) in buf[offset..offset + data_bytes].chunks_exact(4).zip(values.iter_mut()) {
            let mut b = [0u8; 4];
            b.copy_from_slice(chunk);
            *v = f32::from_le_bytes(b);
        }
        offset += data_bytes;
        out.push(Tensor { id, values });
    }
    Ok(out)
}

fn write_tensor(out: &mut Vec<u8>, id: &str, values: &[f32]) -> Result<()> {
    if !id.is_ascii() {
        bail!("tensor id は ASCII のみ: {id:?}");
    }
    if id.contains('\n') {
        bail!("tensor id に '\\n' を含めません: {id:?}");
    }
    out.extend_from_slice(id.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(())
}

/// Box-Muller で正規分布乱数 (mean=0、 std=std) を 1 つ返す。
/// rand_distr に依存しないための簡易実装。
fn sample_normal(rng: &mut Xoshiro256PlusPlus, std: f32) -> f32 {
    let u1 = u32_to_unit(rng.next_u32());
    let u2 = u32_to_unit(rng.next_u32());
    // u1 が 0 だと ln(0) になるためクランプ
    let u1 = u1.max(f32::MIN_POSITIVE);
    let r = (-2.0f32 * u1.ln()).sqrt();
    let theta = 2.0f32 * std::f32::consts::PI * u2;
    r * theta.cos() * std
}

/// u32 → (0, 1) の f32。 0 は除外。
fn u32_to_unit(x: u32) -> f32 {
    // 24bit mantissa を使って (0, 1)。
    let bits = (x >> 8) | 1; // LSB に 1 を立てて 0 を除外
    bits as f32 / (1u32 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_l1b() {
        // src: bucket 0 = [10, 11, 12, 13, 14, 15, 16, 17 (skip)]
        // src: bucket 1 = [20, 21, 22, 23, 24, 25, 26, 27 (skip)]
        // ... 8 buckets
        let mut src = vec![0.0f32; NUM_BUCKETS * SRC_L1_OUT];
        for bucket in 0..NUM_BUCKETS {
            for out in 0..SRC_L1_OUT {
                src[bucket * SRC_L1_OUT + out] = (10 * (bucket + 1) + out) as f32;
            }
        }
        let dst = expand_l1b(&src).unwrap();
        assert_eq!(dst.len(), NUM_BUCKETS * DST_L1_OUT);
        for bucket in 0..NUM_BUCKETS {
            // 新 main 0..7 = 旧 main 0..7
            for out in 0..7 {
                assert_eq!(dst[bucket * DST_L1_OUT + out], src[bucket * SRC_L1_OUT + out]);
            }
            // 新 main 7..15 = 0
            for out in 7..15 {
                assert_eq!(dst[bucket * DST_L1_OUT + out], 0.0);
            }
            // 新 skip 15 = 旧 skip 7
            assert_eq!(dst[bucket * DST_L1_OUT + 15], src[bucket * SRC_L1_OUT + 7]);
        }
    }

    #[test]
    fn test_expand_l1fb() {
        let src: Vec<f32> = (0..SRC_L1_OUT).map(|i| (i + 100) as f32).collect();
        let dst = expand_l1fb(&src).unwrap();
        assert_eq!(dst.len(), DST_L1_OUT);
        // 0..7 = src 0..7
        for (i, (&d, &s)) in dst.iter().zip(src.iter()).enumerate().take(7) {
            assert_eq!(d, s, "i={i}");
        }
        // 7..15 = 0
        for (i, &v) in dst.iter().enumerate().take(15).skip(7) {
            assert_eq!(v, 0.0, "i={i}");
        }
        // 15 = src[7] (skip)
        assert_eq!(dst[15], src[7]);
    }

    #[test]
    fn test_expand_l1w_copies_main_and_skip() {
        // simple pattern: src[in_idx * (8*8) + bucket*8 + out] = encode(in_idx, bucket, out)
        let src_rows = NUM_BUCKETS * SRC_L1_OUT;
        let mut src = vec![0.0f32; src_rows * FT_OUT];
        for in_idx in 0..FT_OUT {
            for bucket in 0..NUM_BUCKETS {
                for out in 0..SRC_L1_OUT {
                    let v = (in_idx as f32) * 1e-4 + (bucket as f32) * 1e-2 + (out as f32) * 1e-1;
                    src[in_idx * src_rows + bucket * SRC_L1_OUT + out] = v;
                }
            }
        }
        // 乱数なし (init_random_main = false) で確認
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let dst = expand_l1w(&src, &mut rng, 0.0, false).unwrap();
        let dst_rows = NUM_BUCKETS * DST_L1_OUT;
        assert_eq!(dst.len(), dst_rows * FT_OUT);
        // spot check at in_idx=100, bucket=3
        let in_idx = 100usize;
        let bucket = 3usize;
        for out in 0..7 {
            // 新 main 0..7 = 旧 main 0..7
            assert_eq!(
                dst[in_idx * dst_rows + bucket * DST_L1_OUT + out],
                src[in_idx * src_rows + bucket * SRC_L1_OUT + out]
            );
        }
        // 新 main 7..15 (init_random_main=false) = 0
        for out in 7..15 {
            assert_eq!(dst[in_idx * dst_rows + bucket * DST_L1_OUT + out], 0.0);
        }
        // 新 skip 15 = 旧 skip 7
        assert_eq!(
            dst[in_idx * dst_rows + bucket * DST_L1_OUT + 15],
            src[in_idx * src_rows + bucket * SRC_L1_OUT + 7]
        );
    }

    #[test]
    fn test_expand_l1w_random_main_amplitude() {
        // 旧 main / skip コピー部は乱数で書き換えないこと、 新 main は std に従うこと
        let src_rows = NUM_BUCKETS * SRC_L1_OUT;
        let src = vec![5.0f32; src_rows * FT_OUT]; // 全て 5.0
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(20260514);
        let std = 0.04f32;
        let dst = expand_l1w(&src, &mut rng, std, true).unwrap();
        let dst_rows = NUM_BUCKETS * DST_L1_OUT;
        // 旧 main 0..7 全部 5.0、 skip 15 も 5.0
        for in_idx in [0usize, 100, 1535] {
            for bucket in 0..NUM_BUCKETS {
                for out in 0..7 {
                    assert_eq!(dst[in_idx * dst_rows + bucket * DST_L1_OUT + out], 5.0);
                }
                assert_eq!(dst[in_idx * dst_rows + bucket * DST_L1_OUT + 15], 5.0);
            }
        }
        // 新 main 7..15 の sample 平均と std
        let mut samples = Vec::new();
        for in_idx in 0..FT_OUT {
            for bucket in 0..NUM_BUCKETS {
                for out in 7..15 {
                    samples.push(dst[in_idx * dst_rows + bucket * DST_L1_OUT + out]);
                }
            }
        }
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let var: f32 =
            samples.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        let observed_std = var.sqrt();
        // n が十分大きいので mean は 0 付近、 std は 0.04 付近
        assert!(mean.abs() < 0.005, "mean too large: {mean}");
        assert!((observed_std - std).abs() < 0.005, "std off: {observed_std} (expected {std})");
    }

    #[test]
    fn test_expand_l2w() {
        // src col 0..7 = sqr (main 0..7)、 col 7..14 = crelu (main 0..7)
        // dst col 0..15 = sqr (main 0..15)、 col 15..30 = crelu (main 0..15)
        let rows = NUM_BUCKETS * L2_OUT;
        let mut src = vec![0.0f32; rows * SRC_L2_IN];
        for col in 0..SRC_L2_IN {
            for r in 0..rows {
                src[col * rows + r] = (col + 1) as f32 * 1000.0 + r as f32 * 0.001;
            }
        }
        let dst = expand_l2w(&src).unwrap();
        assert_eq!(dst.len(), rows * DST_L2_IN);
        // dst col 0..7 = src col 0..7
        for col in 0..7 {
            for r in 0..rows {
                assert_eq!(dst[col * rows + r], src[col * rows + r]);
            }
        }
        // dst col 7..15 = 0
        for col in 7..15 {
            for r in 0..rows {
                assert_eq!(dst[col * rows + r], 0.0);
            }
        }
        // dst col 15..22 = src col 7..14
        for k in 0..7 {
            let dst_col = 15 + k;
            let src_col = 7 + k;
            for r in 0..rows {
                assert_eq!(dst[dst_col * rows + r], src[src_col * rows + r]);
            }
        }
        // dst col 22..30 = 0
        for col in 22..30 {
            for r in 0..rows {
                assert_eq!(dst[col * rows + r], 0.0);
            }
        }
    }

    #[test]
    fn test_expand_l1fw() {
        // src layout: in_idx * 8 + out, out=7 が skip
        let mut src = vec![0.0f32; SRC_L1_OUT * FT_OUT];
        for in_idx in 0..FT_OUT {
            for out in 0..SRC_L1_OUT {
                src[in_idx * SRC_L1_OUT + out] = in_idx as f32 + out as f32 * 0.01;
            }
        }
        let dst = expand_l1fw(&src).unwrap();
        assert_eq!(dst.len(), DST_L1_OUT * FT_OUT);
        for in_idx in [0usize, 7, 1535] {
            // 0..7 コピー
            for out in 0..7 {
                assert_eq!(dst[in_idx * DST_L1_OUT + out], src[in_idx * SRC_L1_OUT + out]);
            }
            // 7..15 = 0
            for out in 7..15 {
                assert_eq!(dst[in_idx * DST_L1_OUT + out], 0.0);
            }
            // 15 = src[in_idx*8 + 7]
            assert_eq!(dst[in_idx * DST_L1_OUT + 15], src[in_idx * SRC_L1_OUT + 7]);
        }
    }

    #[test]
    fn test_parse_and_write_roundtrip() {
        // 2 tensor を書いて parse して同じ tensor が戻ること
        let mut buf = Vec::new();
        let a_values = vec![1.0f32, 2.0, 3.0];
        let b_values = vec![-1.0f32, 0.5];
        write_tensor(&mut buf, "alpha", &a_values).unwrap();
        write_tensor(&mut buf, "beta", &b_values).unwrap();
        let tensors = parse_tensors(&buf).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].id, "alpha");
        assert_eq!(tensors[0].values, a_values);
        assert_eq!(tensors[1].id, "beta");
        assert_eq!(tensors[1].values, b_values);
    }

    #[test]
    fn test_sample_normal_distribution() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let std = 0.04f32;
        let n = 20000;
        let samples: Vec<f32> = (0..n).map(|_| sample_normal(&mut rng, std)).collect();
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let var: f32 = samples.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        let observed_std = var.sqrt();
        assert!(mean.abs() < 0.005, "mean = {mean}");
        assert!((observed_std - std).abs() < 0.005, "std = {observed_std}");
    }
}
