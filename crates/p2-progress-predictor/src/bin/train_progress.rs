//! Progress predictor 訓練 binary (candle CPU/CUDA)
//!
//! 使用例 (smoke):
//!   train_progress --synthetic 10000 --epochs 5 --batch 256 --output progress.bin
//!
//! 使用例 (実 csa pipeline、 未実装):
//!   train_progress --data dataset.jsonl --label-mode ply-ratio --output progress.bin
//!
//! ## label モード
//!
//! - `ply-ratio`: game_ply / total_plies (csa 由来想定)
//! - `heuristic`: compute_endgame_score 流用 (PSV 由来想定)
//!
//! ## TODO (今回スコープ外)
//!
//! - 実 csa 28 GB pipeline からの label 抽出
//! - JSONL / hcpe / PSV loader 実装 (現在は synthetic mode のみ)
//! - eval / validation split

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, loss, AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use p2_progress_predictor::features::FEATURE_DIM;
use p2_progress_predictor::model::{ProgressWeights, HIDDEN1, HIDDEN2, OUTPUT};
use p2_progress_predictor::weights;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "train_progress",
    about = "p2-progress-predictor MLP trainer (candle)"
)]
struct Args {
    /// 訓練データ (jsonl / psv / 未実装の場合は --synthetic 推奨)
    #[arg(long, default_value = "")]
    data: String,

    /// label mode: ply-ratio | heuristic | ply-or-heuristic
    #[arg(long, default_value = "ply-ratio")]
    label_mode: String,

    /// 出力 path (.bin)
    #[arg(long, default_value = "progress.bin")]
    output: PathBuf,

    /// synthetic データ件数 (smoke test 用、 0 で無効)
    #[arg(long, default_value_t = 0)]
    synthetic: usize,

    /// epochs
    #[arg(long, default_value_t = 5)]
    epochs: usize,

    /// batch size
    #[arg(long, default_value_t = 256)]
    batch: usize,

    /// learning rate
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// device: cpu | cuda
    #[arg(long, default_value = "cpu")]
    device: String,

    /// rng seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn pick_device(name: &str) -> Result<Device> {
    match name {
        "cuda" => Device::new_cuda(0).map_err(|e| anyhow::anyhow!("CUDA init failed: {}", e)),
        _ => Ok(Device::Cpu),
    }
}

/// 合成データ生成: 32 次元 random feature + label = w·x + b (sigmoid 噛ませる)
/// 真の重みは ply / 256 風に「29 番 (side)」と「31 番 (ply)」を強調。
fn make_synthetic(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut x = vec![0.0f32; n * FEATURE_DIM];
    let mut y = vec![0.0f32; n];
    let mut rng = seed;
    for (i, y_i) in y.iter_mut().enumerate().take(n) {
        for v in x.iter_mut().skip(i * FEATURE_DIM).take(FEATURE_DIM) {
            // xorshift32
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng & 0xFFFF) as f32 / 65536.0;
            *v = u;
        }
        // 「真の」 progress = sigmoid(2.0 * (ply_norm - 0.5) + small noise on material)
        let ply_norm = x[i * FEATURE_DIM + 31];
        let mat = x[i * FEATURE_DIM]; // material_phase
        let logit = 2.0 * (ply_norm - 0.5) + 1.5 * (mat - 0.5);
        *y_i = 1.0 / (1.0 + (-logit).exp());
    }
    (x, y)
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!("[train_progress] args: {:?}", args);

    let device = pick_device(&args.device)?;
    eprintln!("[train_progress] device: {:?}", device);

    // データ準備
    let (x_flat, y_vec) = if args.synthetic > 0 {
        eprintln!("[train_progress] synthetic mode: {} samples", args.synthetic);
        make_synthetic(args.synthetic, args.seed)
    } else if !args.data.is_empty() {
        // TODO: 実 csa/hcpe/psv pipeline。 今回スコープ外、 明示 bail で TODO 警告。
        bail!(
            "実データ loader は未実装です ({}). --synthetic で smoke test してください。",
            args.data
        );
    } else {
        bail!("--synthetic か --data のいずれかを指定してください。");
    };

    let n = y_vec.len();
    let x = Tensor::from_vec(x_flat, (n, FEATURE_DIM), &device)?;
    let y = Tensor::from_vec(y_vec, (n, 1), &device)?;
    eprintln!("[train_progress] dataset: {} samples", n);

    // Model (candle 側で構築、 最終的に ProgressWeights に dump する)
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let l1 = linear(FEATURE_DIM, HIDDEN1, vb.pp("l1"))?;
    let l2 = linear(HIDDEN1, HIDDEN2, vb.pp("l2"))?;
    let l3 = linear(HIDDEN2, OUTPUT, vb.pp("l3"))?;

    let forward = |x: &Tensor| -> Result<Tensor> {
        let h = l1.forward(x)?.relu()?;
        let h = l2.forward(&h)?.relu()?;
        let logit = l3.forward(&h)?;
        // sigmoid
        let sig = candle_nn::ops::sigmoid(&logit)?;
        Ok(sig)
    };

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )?;

    let batches = n.div_ceil(args.batch);
    for epoch in 0..args.epochs {
        let mut running_loss = 0.0f32;
        for b in 0..batches {
            let start = b * args.batch;
            let end = (start + args.batch).min(n);
            let bx = x.narrow(0, start, end - start)?;
            let by = y.narrow(0, start, end - start)?;
            let pred = forward(&bx)?;
            let l = loss::mse(&pred, &by)?;
            opt.backward_step(&l)?;
            running_loss += l.to_scalar::<f32>()?;
        }
        eprintln!("[train_progress] epoch {} loss {:.6}", epoch + 1, running_loss / batches as f32);
    }

    // candle weight → ProgressWeights
    // l1.weight: [HIDDEN1, FEATURE_DIM]、 l1.bias: [HIDDEN1]
    let w1 = l1.weight().to_vec2::<f32>()?;
    let b1 = l1.bias().ok_or_else(|| anyhow::anyhow!("l1 has no bias"))?.to_vec1::<f32>()?;
    let w2 = l2.weight().to_vec2::<f32>()?;
    let b2 = l2.bias().ok_or_else(|| anyhow::anyhow!("l2 has no bias"))?.to_vec1::<f32>()?;
    let w3 = l3.weight().to_vec2::<f32>()?;
    let b3 = l3.bias().ok_or_else(|| anyhow::anyhow!("l3 has no bias"))?.to_vec1::<f32>()?;

    let mut weights_out = ProgressWeights::zeros();
    for (j, row) in weights_out.w1.iter_mut().enumerate().take(HIDDEN1) {
        row.copy_from_slice(&w1[j][..FEATURE_DIM]);
    }
    weights_out.b1.copy_from_slice(&b1);
    for (j, row) in weights_out.w2.iter_mut().enumerate().take(HIDDEN2) {
        row.copy_from_slice(&w2[j][..HIDDEN1]);
    }
    weights_out.b2.copy_from_slice(&b2);
    for (j, row) in weights_out.w3.iter_mut().enumerate().take(OUTPUT) {
        row.copy_from_slice(&w3[j][..HIDDEN2]);
    }
    weights_out.b3.copy_from_slice(&b3);

    weights::save(&args.output, &weights_out)?;
    eprintln!("[train_progress] saved {}", args.output.display());
    Ok(())
}
