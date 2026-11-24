//! NNUE モデル (candle 実装) — shogi2344_v8 / NNUE-A,B 兼用 (Config で arch 切替)
//!
//! Padding 処理: `-1` は事前に `clamp(min=0)` してから embedding lookup、
//!   ft_weight の 0 行が padding 用になるよう、 学習時に行 0 を凍結する想定。
//! (より厳密な sentinel zero row 維持は将来追加)

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

#[derive(Clone, Debug)]
pub struct Config {
    pub num_features: usize,
    pub max_active: usize,
    pub l1_size: usize,
    pub l2_size: usize,
    pub l3_size: usize,
    pub output_buckets: usize,
}

impl Config {
    /// shogi2344_v8 (NNUE-A 標準): 2344 → 512 → 8 → 96 → 8 buckets
    pub fn shogi2344_v8() -> Self {
        Self {
            num_features: 2344,
            max_active: 40,
            l1_size: 512,
            l2_size: 8,
            l3_size: 96,
            output_buckets: 8,
        }
    }
}

pub struct Shogi2344V8 {
    pub cfg: Config,
    /// FT weight: [num_features, l1_size]
    pub ft_weight: Tensor,
    pub ft_bias: Tensor,
    pub l1: Linear,
    pub l2: Linear,
    pub l3: Linear,
    device: Device,
}

impl Shogi2344V8 {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let ft_weight = vb.get_with_hints(
            (cfg.num_features, cfg.l1_size),
            "ft_weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;
        let ft_bias = vb.get_with_hints(cfg.l1_size, "ft_bias", candle_nn::Init::Const(0.0))?;
        let l1 = linear(2 * cfg.l1_size, cfg.output_buckets * cfg.l2_size, vb.pp("l1"))?;
        let l2 = linear(cfg.l2_size * 2, cfg.output_buckets * cfg.l3_size, vb.pp("l2"))?;
        let l3 = linear(cfg.l3_size, cfg.output_buckets, vb.pp("l3"))?;

        Ok(Self {
            cfg,
            ft_weight,
            ft_bias,
            l1,
            l2,
            l3,
            device,
        })
    }

    /// stm_idx, nstm_idx: [B, MAX_ACTIVE] u32
    /// bucket: [B] u32
    /// Returns: [B] f32 (raw cp prediction、 推論側で × scale (40) で cp に変換)
    pub fn forward(&self, stm_idx: &Tensor, nstm_idx: &Tensor, bucket: &Tensor) -> Result<Tensor> {
        let b = stm_idx.dim(0)?;

        // Embedding lookup: [B, MAX_ACTIVE, L1]
        let stm_flat = stm_idx.flatten_all()?;
        let nstm_flat = nstm_idx.flatten_all()?;
        let stm_emb = self.ft_weight.index_select(&stm_flat, 0)?;
        let nstm_emb = self.ft_weight.index_select(&nstm_flat, 0)?;

        // Sum over active dim → [B, L1]
        let stm_l1 = stm_emb
            .reshape((b, self.cfg.max_active, self.cfg.l1_size))?
            .sum(1)?
            .broadcast_add(&self.ft_bias)?;
        let nstm_l1 = nstm_emb
            .reshape((b, self.cfg.max_active, self.cfg.l1_size))?
            .sum(1)?
            .broadcast_add(&self.ft_bias)?;

        // CReLU
        let stm_l1 = stm_l1.clamp(0.0_f32, 1.0_f32)?;
        let nstm_l1 = nstm_l1.clamp(0.0_f32, 1.0_f32)?;

        // Concat [B, 2*L1]
        let x = Tensor::cat(&[&stm_l1, &nstm_l1], 1)?;

        // L1 → [B, OUTPUT_BUCKETS * L2] → reshape → bucket select → [B, L2]
        let x = self.l1.forward(&x)?;
        let x = x.reshape((b, self.cfg.output_buckets, self.cfg.l2_size))?;
        let x = bucket_select_2d(&x, bucket)?;

        // squared trick + CReLU
        let x_sq = x.sqr()?;
        let x = Tensor::cat(&[&x, &x_sq], 1)?;
        let x = x.clamp(0.0_f32, 1.0_f32)?;

        // L2 → [B, OUTPUT_BUCKETS * L3] → bucket select → [B, L3]
        let x = self.l2.forward(&x)?;
        let x = x.reshape((b, self.cfg.output_buckets, self.cfg.l3_size))?;
        let x = bucket_select_2d(&x, bucket)?;
        let x = x.clamp(0.0_f32, 1.0_f32)?;

        // L3 → [B, OUTPUT_BUCKETS] → bucket select → [B]
        let x = self.l3.forward(&x)?;
        let x = bucket_select_1d(&x, bucket)?;

        Ok(x)
    }
}

/// [B, OUTPUT_BUCKETS, D] → [B, D] (bucket idx で行を選択)
fn bucket_select_2d(x: &Tensor, bucket: &Tensor) -> Result<Tensor> {
    let b = x.dim(0)?;
    let d = x.dim(2)?;
    let idx = bucket.unsqueeze(1)?.unsqueeze(2)?.broadcast_as((b, 1, d))?.contiguous()?;
    let g = x.gather(&idx, 1)?;
    Ok(g.squeeze(1)?)
}

/// [B, OUTPUT_BUCKETS] → [B]
fn bucket_select_1d(x: &Tensor, bucket: &Tensor) -> Result<Tensor> {
    let b = x.dim(0)?;
    let idx = bucket.unsqueeze(1)?.broadcast_as((b, 1))?.contiguous()?;
    let g = x.gather(&idx, 1)?;
    Ok(g.squeeze(1)?)
}

#[allow(dead_code)]
fn _device_check(d: &Device) -> &Device {
    d
}
