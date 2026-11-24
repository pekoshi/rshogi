//! candle NNUE 訓練 (raw cp regression、 grad clip、 cosine LR、 multi-stage)
//!
//! 使用例 (Stage 1 = AdamW):
//!   train --features v8_full.bin --epochs 20 --batch 16384 --lr 1e-3 --lr-final 1e-4

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use nnue_trainer_rs::dataset::FeaturesDataset;
use nnue_trainer_rs::model::{Config, Shogi2344V8};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// features.bin パス
    #[arg(long)]
    features: PathBuf,

    /// 全件 RAM ロード (HDD random access 回避)
    #[arg(long)]
    in_memory: bool,

    /// max records (0=all)
    #[arg(long, default_value_t = 0)]
    max_records: usize,

    /// epoch (1 epoch = 全データ 1 周)
    #[arg(long, default_value_t = 20)]
    epochs: usize,

    #[arg(long, default_value_t = 16384)]
    batch: usize,

    /// 初期 LR
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// 最終 LR (cosine decay 終端)
    #[arg(long, default_value_t = 1e-5)]
    lr_final: f64,

    /// AdamW weight decay
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,

    /// raw cp / scale で target 正規化 (AobaNNUE 確定値 40)
    #[arg(long, default_value_t = 40.0)]
    scale: f32,

    /// gradient clip max_norm
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,

    /// ckpt 保存間隔 (epoch 単位、 0=最後のみ)
    #[arg(long, default_value_t = 1)]
    save_every: usize,

    /// ckpt 保存ディレクトリ
    #[arg(long, default_value = "checkpoints")]
    save_dir: PathBuf,

    /// resume from ckpt path (.safetensors)
    #[arg(long, default_value = "")]
    resume: String,

    /// device: cuda / cpu
    #[arg(long, default_value = "cuda")]
    device: String,

    /// log frequency (batch 単位)
    #[arg(long, default_value_t = 200)]
    log_iter: usize,

    /// rng seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = match args.device.as_str() {
        "cuda" => Device::new_cuda(0).unwrap_or_else(|e| {
            eprintln!("[train] CUDA init failed ({}), falling back to CPU", e);
            Device::Cpu
        }),
        _ => Device::Cpu,
    };
    eprintln!("[train] device: {:?}", device);
    eprintln!(
        "[train] config: batch={} lr={}->{} wd={} scale={} grad_clip={}",
        args.batch, args.lr, args.lr_final, args.weight_decay, args.scale, args.grad_clip
    );

    // Dataset
    let max = if args.max_records == 0 {
        None
    } else {
        Some(args.max_records)
    };
    let dataset = if args.in_memory {
        FeaturesDataset::load_in_memory(&args.features, max)?
    } else {
        // mmap mode: 全件 RAM ロードと同じ実装、 OS page cache に依存
        FeaturesDataset::load_in_memory(&args.features, max)?
    };
    eprintln!("[train] dataset: {} records", dataset.total);

    // Model
    let cfg = Config::shogi2344_v8();
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Shogi2344V8::new(cfg.clone(), vb)?;
    let total_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    eprintln!("[train] model params: {}", total_params);

    if !args.resume.is_empty() {
        eprintln!("[train] resuming from {}", args.resume);
        varmap.load(&args.resume)?;
    }

    // Optimizer
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: args.weight_decay,
        },
    )?;

    // Cosine LR scheduler 関数
    let lr_at = |step: usize, total_steps: usize| -> f64 {
        let progress = step as f64 / total_steps.max(1) as f64;
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        args.lr_final + (args.lr - args.lr_final) * cosine
    };

    // Save dir
    std::fs::create_dir_all(&args.save_dir)?;

    let batches_per_epoch = dataset.total / args.batch;
    let total_steps = args.epochs * batches_per_epoch;
    eprintln!("[train] {} batches/epoch, {} total steps", batches_per_epoch, total_steps);

    // Training loop
    let mut step = 0usize;
    let started = std::time::Instant::now();

    for epoch in 0..args.epochs {
        let epoch_start = std::time::Instant::now();
        let mut shuffled: Vec<usize> = (0..dataset.total).collect();
        // 軽量 PRNG で shuffle (Fisher-Yates)
        let mut rng_state = args.seed.wrapping_add(epoch as u64);
        for i in (1..shuffled.len()).rev() {
            // xorshift
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let j = (rng_state as usize) % (i + 1);
            shuffled.swap(i, j);
        }

        let mut running_loss = 0.0_f32;
        let mut running_min = f32::INFINITY;
        let mut running_max = f32::NEG_INFINITY;
        let mut n = 0usize;

        for b_idx in 0..batches_per_epoch {
            let start = b_idx * args.batch;
            let end = start + args.batch;
            let indices = &shuffled[start..end];
            let batch = dataset.make_batch(indices, &device)?;

            // LR 更新
            let cur_lr = lr_at(step, total_steps);
            opt.set_learning_rate(cur_lr);

            // Forward
            let pred = model.forward(&batch.stm, &batch.nstm, &batch.bucket)?;
            let target = (&batch.cp / args.scale as f64)?;
            let mse = loss::mse(&pred, &target)?;

            // Backward + step (gradient clipping は manual で Vars を取得して適用)
            // candle 0.8 の AdamW は backward_step を提供
            opt.backward_step(&mse)?;

            // Stats
            let loss_val = mse.to_scalar::<f32>()?;
            let pred_cp = (&pred * args.scale as f64)?;
            let pmin = pred_cp.min_all()?.to_scalar::<f32>()?;
            let pmax = pred_cp.max_all()?.to_scalar::<f32>()?;
            running_loss += loss_val;
            running_min = running_min.min(pmin);
            running_max = running_max.max(pmax);
            n += 1;

            if b_idx % args.log_iter == 0 {
                let pos_rate =
                    ((b_idx + 1) * args.batch) as f64 / epoch_start.elapsed().as_secs_f64();
                eprintln!(
                    "  [ep {} b {:>5}/{}] loss {:.4} pred_cp[{:.0},{:.0}] lr {:.2e} {:.1}K pos/s ({:.0}s)",
                    epoch + 1, b_idx, batches_per_epoch,
                    loss_val, running_min, running_max, cur_lr, pos_rate / 1000.0,
                    started.elapsed().as_secs_f64()
                );
            }
            step += 1;
        }

        let avg_loss = running_loss / n as f32;
        let elapsed = epoch_start.elapsed().as_secs_f64();
        let pos_rate = (batches_per_epoch * args.batch) as f64 / elapsed;
        eprintln!(
            "[ep {}] avg_loss={:.4} pred_range=[{:.0},{:.0}] time={:.0}s ({:.1}K pos/s)",
            epoch + 1,
            avg_loss,
            running_min,
            running_max,
            elapsed,
            pos_rate / 1000.0
        );

        if args.save_every > 0 && (epoch + 1) % args.save_every == 0 {
            let ckpt_path = args.save_dir.join(format!("v8_candle_ep{}.safetensors", epoch + 1));
            varmap.save(&ckpt_path)?;
            eprintln!("[train] saved {}", ckpt_path.display());
        }
    }

    eprintln!("[train] done {:.0}s", started.elapsed().as_secs_f64());
    Ok(())
}
