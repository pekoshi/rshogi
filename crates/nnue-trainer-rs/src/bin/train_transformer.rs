//! NNUE-B Candle Transformer 訓練 binary (PUCT 用 multi-task value+policy)。
//!
//! 入力: psv_to_tokens 事前変換済み `.tokens` ファイル (TOKEN_RECORD_SIZE byte/record)
//! 出力: ckpt (.safetensors) + 量子化 / ORT export は別 binary で
//!
//! Loss:
//!   - value: sigmoid(score / SCALE) と result の混合 (WDL λ で interpolate) を target に MSE
//!   - policy: cross_entropy(policy_logits, policy_idx)、 policy_idx = -1 (sentinel) は mask
//!
//! 使用例:
//!   train_transformer --tokens path/to/train.tokens --batch 1024 --superbatches 50 \
//!     --wdl 0.5 --policy-weight 1.0 --output ckpts/transformer

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use nnue_trainer_rs::transformer::{TransformerConfig, TransformerModel};
use nnue_trainer_rs::transformer_dataset::TokenDataset;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    tokens: PathBuf,

    #[arg(long, default_value_t = 0)]
    max_records: usize,

    #[arg(long, default_value_t = 50)]
    superbatches: usize,

    #[arg(long, default_value_t = 6104)]
    batches_per_sb: usize,

    #[arg(long, default_value_t = 1024)]
    batch: usize,

    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,

    /// stm 視点 cp を sigmoid target に変換するスケール
    #[arg(long, default_value_t = 600.0)]
    scale: f32,

    /// WDL lambda: result_target = wdl * result + (1-wdl) * sigmoid(score)
    #[arg(long, default_value_t = 0.5)]
    wdl: f32,

    /// policy CE 重み (0 で policy 学習無効)
    #[arg(long, default_value_t = 1.0)]
    policy_weight: f32,

    #[arg(long, default_value = "checkpoints/transformer")]
    save_dir: PathBuf,

    #[arg(long, default_value_t = 5)]
    save_every_sb: usize,

    #[arg(long, default_value = "cuda")]
    device: String,

    #[arg(long, default_value_t = 200)]
    log_iter: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = match args.device.as_str() {
        "cuda" => Device::new_cuda(0).unwrap_or_else(|e| {
            eprintln!("[train_transformer] CUDA init failed ({}), falling back to CPU", e);
            Device::Cpu
        }),
        _ => Device::Cpu,
    };
    eprintln!("[train_transformer] device: {:?}", device);
    eprintln!(
        "[train_transformer] batch={} sb={} batches/sb={} lr={} wdl={} policy_w={}",
        args.batch, args.superbatches, args.batches_per_sb, args.lr, args.wdl, args.policy_weight,
    );

    let max = if args.max_records == 0 {
        None
    } else {
        Some(args.max_records)
    };
    let dataset = TokenDataset::load_in_memory(&args.tokens, max)?;
    eprintln!("[train_transformer] dataset: {} records", dataset.total);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let cfg = TransformerConfig::nnue_b_default();
    let model = TransformerModel::new(cfg.clone(), vb)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.lr,
            weight_decay: args.weight_decay,
            ..Default::default()
        },
    )?;

    std::fs::create_dir_all(&args.save_dir)?;

    // RNG (xorshift) for batch sampling
    let mut state: u64 = args.seed.max(1);
    let next_idx = |s: &mut u64, n: usize| -> usize {
        let mut x = *s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *s = x;
        (x as usize) % n
    };

    let scale = args.scale as f64;
    let mut total_batches: u64 = 0;
    let mut indices = vec![0usize; args.batch];

    for sb in 1..=args.superbatches {
        let mut sb_loss_v = 0.0f64;
        let mut sb_loss_p = 0.0f64;
        let mut sb_count = 0u64;
        for _bi in 0..args.batches_per_sb {
            for i in 0..args.batch {
                indices[i] = next_idx(&mut state, dataset.total);
            }
            let batch = dataset.make_batch(&indices, &device)?;

            let (value_pred, policy_logits) = model.forward(&batch.tokens)?;

            // Value target: sigmoid(score/scale) と result を WDL λ で blend
            let score_sigmoid = batch.score.affine(1.0 / scale, 0.0)?.tanh()?;
            // result tensor は f32 [0, 1]、 score_sigmoid は f32 [-1, 1] → [0, 1] へ
            let score_sigmoid_01 = ((score_sigmoid + 1.0)? * 0.5)?;
            let target_value = (((&batch.result * args.wdl as f64)?)
                + (score_sigmoid_01 * (1.0 - args.wdl as f64))?)?;
            let value_pred_sigmoid = candle_nn::ops::sigmoid(&value_pred.squeeze(1)?)?;
            let loss_v = candle_nn::loss::mse(&value_pred_sigmoid, &target_value)?;

            // Policy CE (mask out -1 sentinel)
            // policy_idx: i64 [B]、 -1 は ignore
            // ここは簡略実装 (TODO: ignore_index 対応)
            let loss_p = if args.policy_weight > 0.0 {
                // Sentinel を 0 にクランプして CE を計算、 ただし -1 サンプルの寄与は手動マスク
                let pidx = batch.policy_idx.clone();
                let mask = pidx.ge(0i64)?.to_dtype(DType::F32)?;
                let safe_idx = pidx.maximum(&Tensor::zeros_like(&pidx)?)?;
                let log_prob = candle_nn::ops::log_softmax(&policy_logits, 1)?;
                let nll = log_prob.gather(&safe_idx.unsqueeze(1)?, 1)?.squeeze(1)?.neg()?;
                let masked = (nll * &mask)?;
                let n_valid = mask.sum_all()?.to_scalar::<f32>()?.max(1.0);
                masked.sum_all()?.affine(1.0 / n_valid as f64, 0.0)?
            } else {
                Tensor::zeros((), DType::F32, &device)?
            };

            let loss_v_val = loss_v.to_scalar::<f32>()? as f64;
            let loss_p_val = if args.policy_weight > 0.0 {
                loss_p.to_scalar::<f32>()? as f64
            } else {
                0.0
            };
            let total_loss = (loss_v + (loss_p * args.policy_weight as f64)?)?;
            opt.backward_step(&total_loss)?;

            sb_loss_v += loss_v_val;
            sb_loss_p += loss_p_val;
            sb_count += 1;
            total_batches += 1;
            if total_batches as usize % args.log_iter == 0 {
                eprintln!(
                    "[sb {sb}] batch {} | loss_v ~ {:.4} loss_p ~ {:.4}",
                    total_batches,
                    sb_loss_v / sb_count as f64,
                    sb_loss_p / sb_count as f64,
                );
            }
        }
        eprintln!(
            "[sb {sb}/{}] avg loss_v {:.4} loss_p {:.4} (n={})",
            args.superbatches,
            sb_loss_v / sb_count as f64,
            sb_loss_p / sb_count as f64,
            sb_count,
        );

        if sb % args.save_every_sb == 0 || sb == args.superbatches {
            let ckpt = args.save_dir.join(format!("transformer-sb{sb}.safetensors"));
            varmap.save(&ckpt)?;
            eprintln!("[train_transformer] saved ckpt: {}", ckpt.display());
        }
    }

    Ok(())
}
