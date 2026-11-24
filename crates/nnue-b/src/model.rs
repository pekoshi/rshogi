//! NNUE-B Candle Transformer (encoder 6 layer + value/policy head)。
//!
//! 設計 (memory `p0_unified_results_20260518` 目標 7.4M):
//! - d_model = 256, n_layer = 8, n_heads = 8, d_ff = 1024
//! - input tokens: [`crate::input::NUM_TOKENS`] = 100 (CLS + 81 board + 14 hand + 4 meta)
//! - vocab_size: [`crate::input::VOCAB_SIZE`] = 512
//! - value head: CLS → Linear 1 (raw、 loss 側で tanh/sigmoid)
//! - policy head: CLS → Linear [`POLICY_SIZE`] = 2187 (dlshogi 互換)
//!
//! ## params 推定 (default config)
//!
//! ```text
//! tok_emb     : 512 × 256                = 131,072
//! pos_emb     : 100 × 256                = 25,600
//! 8 × block   :
//!   ln1 (γ,β) : 512
//!   q/k/v     : 3 × 256*256              = 196,608  (no bias)
//!   out_proj  : 256*256 + 256            = 65,792
//!   ln2 (γ,β) : 512
//!   fc1       : 256*1024 + 1024          = 263,168
//!   fc2       : 1024*256 + 256           = 262,400
//!   block 計  : 788,992
//! 8 layers    : 6,311,936
//! ln_out      : 512
//! value_head  : 257
//! policy_head : 256*2187 + 2187          = 562,059
//! ─────────────────────────────────────────
//! 合計         : ~ 7.0M params (Bullet v16 28.7M の 1/4)
//! ```
//!
//! GTX 1050 Ti 4 GB で batch 64-128 を想定 (mixed precision 必要かは要実測)。

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, Module,
    VarBuilder, VarMap,
};

use crate::input::{NUM_TOKENS, VOCAB_SIZE};

/// dlshogi `make_move_label` 互換 policy 出力次元。
pub const POLICY_SIZE: usize = 2187;

/// NNUE-B Transformer のハイパーパラメータ。
#[derive(Clone, Debug)]
pub struct NnueBConfig {
    pub vocab_size: usize,
    pub max_tokens: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub policy_size: usize,
    pub ln_eps: f64,
}

impl Default for NnueBConfig {
    fn default() -> Self {
        Self::nnue_b_default()
    }
}

impl NnueBConfig {
    /// 既定値 (memory `p0_unified_results_20260518` の 7.4M params 目標)。
    pub fn nnue_b_default() -> Self {
        Self {
            vocab_size: VOCAB_SIZE,
            max_tokens: NUM_TOKENS,
            d_model: 256,
            n_layers: 8,
            n_heads: 8,
            d_ff: 1024,
            policy_size: POLICY_SIZE,
            ln_eps: 1e-5,
        }
    }

    /// テスト用の極小設定 (params <50k、 CPU で forward を高速確認)。
    pub fn tiny_for_tests() -> Self {
        Self {
            vocab_size: VOCAB_SIZE,
            max_tokens: NUM_TOKENS,
            d_model: 32,
            n_layers: 2,
            n_heads: 4,
            d_ff: 64,
            policy_size: POLICY_SIZE,
            ln_eps: 1e-5,
        }
    }
}

struct MultiHeadAttention {
    n_heads: usize,
    d_head: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    scale: f64,
}

impl MultiHeadAttention {
    fn new(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let d_head = d_model / n_heads;
        Ok(Self {
            n_heads,
            d_head,
            q_proj: linear_no_bias(d_model, d_model, vb.pp("q"))?,
            k_proj: linear_no_bias(d_model, d_model, vb.pp("k"))?,
            v_proj: linear_no_bias(d_model, d_model, vb.pp("v"))?,
            out_proj: linear(d_model, d_model, vb.pp("out"))?,
            scale: (d_head as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let split = |t_in: &Tensor| -> Result<Tensor> {
            Ok(t_in.reshape((b, t, self.n_heads, self.d_head))?.transpose(1, 2)?.contiguous()?)
        };
        let q = split(&self.q_proj.forward(x)?)?;
        let k = split(&self.k_proj.forward(x)?)?;
        let v = split(&self.v_proj.forward(x)?)?;
        let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn = q.matmul(&kt)?;
        let attn = (attn / self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, self.n_heads * self.d_head))?;
        Ok(self.out_proj.forward(&out)?)
    }
}

struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(d_model, d_ff, vb.pp("fc1"))?,
            fc2: linear(d_ff, d_model, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = h.gelu()?;
        Ok(self.fc2.forward(&h)?)
    }
}

struct TransformerBlock {
    ln1: LayerNorm,
    attn: MultiHeadAttention,
    ln2: LayerNorm,
    ffn: FeedForward,
}

impl TransformerBlock {
    fn new(cfg: &NnueBConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(cfg.d_model, cfg.ln_eps, vb.pp("ln1"))?,
            attn: MultiHeadAttention::new(cfg.d_model, cfg.n_heads, vb.pp("attn"))?,
            ln2: layer_norm(cfg.d_model, cfg.ln_eps, vb.pp("ln2"))?,
            ffn: FeedForward::new(cfg.d_model, cfg.d_ff, vb.pp("ffn"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-LN Transformer
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h)?;
        let x = (x + h)?;
        let h = self.ln2.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        Ok((x + h)?)
    }
}

/// NNUE-B Transformer model 本体。
///
/// 推論用には `forward(tokens) -> (value, policy)` を呼ぶ。 訓練用には [`varmap`]
/// で AdamW 等の optimizer に渡す。
///
/// - `value`: `[B, 1]` raw (loss 側で sigmoid / tanh)
/// - `policy`: `[B, POLICY_SIZE]` logits
pub struct NnueBModel {
    pub cfg: NnueBConfig,
    pub device: Device,
    /// VarMap で重みの ownership を持つ。 ckpt save / load に使う。
    varmap: VarMap,
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_out: LayerNorm,
    value_head: Linear,
    policy_head: Linear,
}

impl NnueBModel {
    /// Random init で model を生成 (test / 訓練 fresh start 用)。
    pub fn new(cfg: NnueBConfig, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self::build(cfg, device, varmap, vb)
    }

    /// safetensors ckpt から load。
    pub fn load_safetensors(
        cfg: NnueBConfig,
        path: &std::path::Path,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Self::build(cfg, device, varmap, vb)?;
        model
            .varmap
            .load(path)
            .map_err(|e| anyhow::anyhow!("failed to load safetensors {:?}: {}", path, e))?;
        Ok(model)
    }

    /// 現在の重みを safetensors として保存。
    pub fn save_safetensors(&self, path: &std::path::Path) -> Result<()> {
        self.varmap
            .save(path)
            .map_err(|e| anyhow::anyhow!("failed to save safetensors {:?}: {}", path, e))?;
        Ok(())
    }

    /// 内部 VarMap への参照 (訓練 binary の AdamW init 用)。
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    fn build(cfg: NnueBConfig, device: Device, varmap: VarMap, vb: VarBuilder) -> Result<Self> {
        let tok_emb = embedding(cfg.vocab_size, cfg.d_model, vb.pp("tok_emb"))?;
        let pos_emb = embedding(cfg.max_tokens, cfg.d_model, vb.pp("pos_emb"))?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            blocks.push(TransformerBlock::new(&cfg, vb.pp(format!("blk{i}")))?);
        }
        let ln_out = layer_norm(cfg.d_model, cfg.ln_eps, vb.pp("ln_out"))?;
        let value_head = linear(cfg.d_model, 1, vb.pp("value_head"))?;
        let policy_head = linear(cfg.d_model, cfg.policy_size, vb.pp("policy_head"))?;
        Ok(Self {
            cfg,
            device,
            varmap,
            tok_emb,
            pos_emb,
            blocks,
            ln_out,
            value_head,
            policy_head,
        })
    }

    /// Forward pass。
    ///
    /// Args:
    /// - `tokens`: `[B, T]` i64、 `T = cfg.max_tokens`、 値域 `0..vocab_size`
    ///
    /// Returns:
    /// - `value`: `[B, 1]` f32
    /// - `policy_logits`: `[B, policy_size]` f32
    pub fn forward(&self, tokens: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t) = tokens.dims2()?;
        let tok = self.tok_emb.forward(tokens)?; // [B, T, d_model]
        let pos_ids = Tensor::arange(0i64, t as i64, tokens.device())?.unsqueeze(0)?;
        let pos = self.pos_emb.forward(&pos_ids)?; // [1, T, d_model]
        let mut h = (tok + pos.broadcast_as((b, t, self.cfg.d_model))?)?;
        for blk in self.blocks.iter() {
            h = blk.forward(&h)?;
        }
        let h = self.ln_out.forward(&h)?;
        // CUDA matmul は contiguous を要求する。 `.i((.., 0, ..))` は
        // stride 不連続な view を返すので contiguous 化してから head に流す。
        let cls = h.i((.., 0, ..))?.contiguous()?; // [B, d_model]
        let value = self.value_head.forward(&cls)?; // [B, 1]
        let policy = self.policy_head.forward(&cls)?; // [B, policy_size]
        Ok((value, policy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shape_default() -> Result<()> {
        let device = Device::Cpu;
        let cfg = NnueBConfig::tiny_for_tests();
        let model = NnueBModel::new(cfg.clone(), device.clone())?;
        let b = 2usize;
        let tokens = Tensor::zeros((b, cfg.max_tokens), DType::I64, &device)?;
        let (value, policy) = model.forward(&tokens)?;
        assert_eq!(value.dims2()?, (b, 1));
        assert_eq!(policy.dims2()?, (b, cfg.policy_size));
        Ok(())
    }

    #[test]
    fn save_load_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let cfg = NnueBConfig::tiny_for_tests();
        let model_a = NnueBModel::new(cfg.clone(), device.clone())?;

        let tmpdir = tempfile::tempdir()?;
        let ckpt = tmpdir.path().join("nnue_b_test.safetensors");
        model_a.save_safetensors(&ckpt)?;

        let model_b = NnueBModel::load_safetensors(cfg.clone(), &ckpt, device.clone())?;

        let tokens = Tensor::zeros((1, cfg.max_tokens), DType::I64, &device)?;
        let (va, pa) = model_a.forward(&tokens)?;
        let (vb, pb) = model_b.forward(&tokens)?;
        let va_v = va.flatten_all()?.to_vec1::<f32>()?;
        let vb_v = vb.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(va_v, vb_v, "value heads must match after load");
        let pa_v = pa.flatten_all()?.to_vec1::<f32>()?;
        let pb_v = pb.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(pa_v.len(), pb_v.len());
        // 量子化なしなので bit-exact 一致を期待
        for (a, b) in pa_v.iter().zip(pb_v.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "policy logits mismatch after load");
        }
        Ok(())
    }
}
