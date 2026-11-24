//! candle GPU forward+backward bench
//!
//! 目的: Bullet 並み速度 (250K pos/s) を candle で達成可能か検証

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use nnue_trainer_rs::model::{Config, Shogi2344V8};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or_else(|e| {
        eprintln!("[bench] CUDA init failed: {}, falling back to CPU", e);
        Device::Cpu
    });
    eprintln!("[bench] device: {:?}", device);

    let cfg = Config::shogi2344_v8();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Shogi2344V8::new(cfg.clone(), vb)?;
    let total_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    eprintln!("[bench] params: {}", total_params);

    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;

    for &b in &[8192, 16384, 32768] {
        // Random GPU tensors
        let stm_data: Vec<u32> =
            (0..(b * cfg.max_active)).map(|i| (i % cfg.num_features) as u32).collect();
        let nstm_data: Vec<u32> =
            (0..(b * cfg.max_active)).map(|i| ((i + 7) % cfg.num_features) as u32).collect();
        let bucket_data: Vec<u32> = (0..b).map(|i| (i % cfg.output_buckets) as u32).collect();
        let target_data: Vec<f32> = (0..b).map(|i| (i as f32 - (b as f32) / 2.0) / 100.0).collect();

        let stm = Tensor::from_vec(stm_data, (b, cfg.max_active), &device)?;
        let nstm = Tensor::from_vec(nstm_data, (b, cfg.max_active), &device)?;
        let bucket = Tensor::from_vec(bucket_data, b, &device)?;
        let target = Tensor::from_vec(target_data, b, &device)?;

        // warmup
        for _ in 0..5 {
            let pred = model.forward(&stm, &nstm, &bucket)?;
            let loss = loss::mse(&pred, &target)?;
            opt.backward_step(&loss)?;
        }
        device.synchronize()?;

        // bench
        let n_iter = 30;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iter {
            let pred = model.forward(&stm, &nstm, &bucket)?;
            let loss = loss::mse(&pred, &target)?;
            opt.backward_step(&loss)?;
        }
        device.synchronize()?;
        let elapsed = t0.elapsed().as_secs_f64();
        let pos = (n_iter * b) as f64;
        eprintln!(
            "  B={:>5}: {:>9.0} pos in {:>5.2}s ({:>4.0}K pos/s)",
            b,
            pos,
            elapsed,
            pos / elapsed / 1000.0
        );
    }

    Ok(())
}
