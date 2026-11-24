//! candle ckpt (.safetensors) → rshogi shogi2344_v8 quantised.bin export
//!
//! Format (rshogi/crates/rshogi-core/src/nnue/shogi2344_v8/bin_loader.rs 仕様):
//!   ftw  | 2344 × 512  i16 | × 255       (FT weight)
//!   ftb  | 512         i16 | × 255       (FT bias)
//!   l1w  | 1024 × 64   i16 | × 64  (T)   (PyTorch [out=64, in=1024] → 転置)
//!   l1b  | 64          i16 | × 16320     (= 255 × 64)
//!   l2w  | 16 × 768    i16 | × 64  (T)
//!   l2b  | 768         i16 | × 16320
//!   l3w  | 96 × 8      i16 | × 64  (T)
//!   l3b  | 8           i16 | × 16320
//!   + 48 byte 0x00 padding → total 2,560,192 bytes

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use nnue_trainer_rs::model::{Config, Shogi2344V8};
use std::path::PathBuf;

const QUANTISED_SIZE: usize = 2_560_192;

#[derive(Parser, Debug)]
struct Args {
    /// candle ckpt (.safetensors)
    #[arg(long)]
    ckpt: PathBuf,
    /// output quantised.bin
    #[arg(long)]
    out: PathBuf,
}

/// Tensor (f32) を i16 量子化、 overflow 警告
fn quantise(t: &Tensor, scale: f32, name: &str) -> Result<Vec<i16>> {
    let data: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut out = Vec::with_capacity(data.len());
    let mut overflows = 0usize;
    let mut max_abs = 0f32;
    for v in data {
        let q = (v * scale).round();
        max_abs = max_abs.max(q.abs());
        let q_i16 = if q > 32767.0 {
            overflows += 1;
            32767i16
        } else if q < -32768.0 {
            overflows += 1;
            -32768i16
        } else {
            q as i16
        };
        out.push(q_i16);
    }
    if overflows > 0 {
        eprintln!("  WARN [{}]: {} elements overflow i16 (max_abs={})", name, overflows, max_abs);
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = Device::Cpu;
    let cfg = Config::shogi2344_v8();
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = Shogi2344V8::new(cfg.clone(), vb)?;
    varmap.load(&args.ckpt).context("load ckpt")?;
    eprintln!("[export] loaded {}", args.ckpt.display());

    // varmap から weight 取得 (key は VarBuilder の path)
    let vars = varmap.data().lock().unwrap();
    let get = |k: &str| -> Result<Tensor> {
        vars.get(k).map(|v| v.as_tensor().clone()).context(format!("missing var {}", k))
    };
    let ft_w = get("ft_weight")?; // [2344, 512]
    let ft_b = get("ft_bias")?; // [512]
    let l1_w = get("l1.weight")?; // [64, 1024]   (out, in)
    let l1_b = get("l1.bias")?; // [64]
    let l2_w = get("l2.weight")?; // [768, 16]
    let l2_b = get("l2.bias")?; // [768]
    let l3_w = get("l3.weight")?; // [8, 96]
    let l3_b = get("l3.bias")?; // [8]
    drop(vars);

    // Bullet quantised: l1w/l2w/l3w は転置保存 ([in, out]) なので .t() 適用
    let ftw_q = quantise(&ft_w, 255.0, "ftw")?; // [2344, 512]、 そのまま
    let ftb_q = quantise(&ft_b, 255.0, "ftb")?; // [512]
    let l1w_q = quantise(&l1_w.t()?.contiguous()?, 64.0, "l1w")?; // [1024, 64]
    let l1b_q = quantise(&l1_b, 255.0 * 64.0, "l1b")?; // [64]
    let l2w_q = quantise(&l2_w.t()?.contiguous()?, 64.0, "l2w")?; // [16, 768]
    let l2b_q = quantise(&l2_b, 255.0 * 64.0, "l2b")?; // [768]
    let l3w_q = quantise(&l3_w.t()?.contiguous()?, 64.0, "l3w")?; // [96, 8]
    let l3b_q = quantise(&l3_b, 255.0 * 64.0, "l3b")?; // [8]

    let _ = cfg;

    // Section size 検証
    assert_eq!(ftw_q.len() * 2, 2_400_256, "ftw size");
    assert_eq!(ftb_q.len() * 2, 1_024, "ftb");
    assert_eq!(l1w_q.len() * 2, 131_072, "l1w");
    assert_eq!(l1b_q.len() * 2, 128, "l1b");
    assert_eq!(l2w_q.len() * 2, 24_576, "l2w");
    assert_eq!(l2b_q.len() * 2, 1_536, "l2b");
    assert_eq!(l3w_q.len() * 2, 1_536, "l3w");
    assert_eq!(l3b_q.len() * 2, 16, "l3b");

    // Byte buffer 構築
    let mut blob = Vec::with_capacity(QUANTISED_SIZE);
    let push = |blob: &mut Vec<u8>, q: &[i16]| {
        for v in q {
            blob.extend_from_slice(&v.to_le_bytes());
        }
    };
    push(&mut blob, &ftw_q);
    push(&mut blob, &ftb_q);
    push(&mut blob, &l1w_q);
    push(&mut blob, &l1b_q);
    push(&mut blob, &l2w_q);
    push(&mut blob, &l2b_q);
    push(&mut blob, &l3w_q);
    push(&mut blob, &l3b_q);
    // 48 byte 0x00 padding
    blob.extend(std::iter::repeat(0u8).take(48));

    assert_eq!(blob.len(), QUANTISED_SIZE, "total size mismatch");

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out, &blob)?;
    eprintln!("[export] wrote {} ({} bytes)", args.out.display(), blob.len());

    Ok(())
}
