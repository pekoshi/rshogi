//! 訓練済 NNUE-B ckpt (.safetensors) → ONNX stub 変換 binary。
//!
//! 使い方:
//!
//! ```bash
//! export_onnx --ckpt checkpoints/nnue_b/nnue_b-ep1.safetensors \
//!             --output nnue_b.onnx
//! ```
//!
//! 出力は **重みのみ含むスタブ ONNX** (graph.node が空)。
//! 完全 graph 化は Python 側 (`scripts/onnx_export_nnue_b.py` 想定) で行うこと。
//! 詳細は `crates/nnue-b/src/onnx.rs` の module doc を参照。

use anyhow::{anyhow, Result};
use candle_core::Device;
use clap::Parser;
use nnue_b::model::{NnueBConfig, NnueBModel};
use nnue_b::onnx::export_onnx_stub;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "export_onnx", about = "NNUE-B safetensors → ONNX stub export")]
struct Args {
    /// 入力 ckpt
    #[arg(long)]
    ckpt: Option<PathBuf>,

    /// 出力 ONNX path (拡張子は `.onnx` 必須)
    #[arg(long)]
    output: PathBuf,

    /// random init (ckpt なしで構造だけ export)
    #[arg(long, default_value_t = false)]
    random: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    let cfg = NnueBConfig::nnue_b_default();
    let model = if args.random {
        eprintln!("[export_onnx] random init で export");
        NnueBModel::new(cfg, device)?
    } else {
        let ckpt = args
            .ckpt
            .as_ref()
            .ok_or_else(|| anyhow!("--ckpt または --random を指定してください"))?;
        eprintln!("[export_onnx] load {}", ckpt.display());
        NnueBModel::load_safetensors(cfg, ckpt, device)?
    };

    let all = model.varmap().all_vars();
    let owned: Vec<(String, candle_core::Tensor)> = all
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("var{i}"), v.as_tensor().clone()))
        .collect();
    let refs: Vec<(&str, &candle_core::Tensor)> =
        owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
    export_onnx_stub(&model, &refs, &args.output)?;

    let size = std::fs::metadata(&args.output)?.len();
    eprintln!(
        "[export_onnx] wrote {} ({} bytes, {} initializers)",
        args.output.display(),
        size,
        refs.len()
    );
    Ok(())
}
