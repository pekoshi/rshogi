//! NNUE-B Candle Transformer model (PUCT 用 value + policy head)
//!
//! メモリ `nnue_b_transformer_plan_20260512.md` で確定した設計の Rust 実装。
//!
//! Architecture:
//! - Input tokens: CLS(1) + board(81) + hand(14) + king(2) = up to 98 tokens
//! - d_model=192, n_layers=6, n_heads=6, d_ff=768
//! - value head:  CLS → Linear 1 (tanh で [-1, 1])
//! - policy head: CLS → Linear 2187 (dlshogi-規格 logits)
//!
//! params 推定 ~5-8M (Bullet v16 28.7M の 1/4)。
//!
//! 入力 token は `[B, max_tokens]` の i64、 padding は `pad_id` で埋める想定。

use anyhow::Result;
use candle_core::{IndexOp, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, Module, VarBuilder,
};

/// NNUE-B Transformer のハイパーパラメータ。
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    /// 駒種 + 持ち駒種 + CLS + PAD の合計語彙数 (余裕を持って 48)
    pub vocab_size: usize,
    /// 位置埋め込み用の最大 token 数 (CLS + 81 + 14 + 2)
    pub max_tokens: usize,
    /// 隠れ次元
    pub d_model: usize,
    /// Transformer 層数
    pub n_layers: usize,
    /// Multi-head の head 数 (d_model % n_heads == 0 が必要)
    pub n_heads: usize,
    /// FFN 内部次元 (通常 4 × d_model)
    pub d_ff: usize,
    /// policy head 出力数 (dlshogi 規格 27 × 81)
    pub policy_size: usize,
    /// LayerNorm の epsilon
    pub ln_eps: f64,
}

impl TransformerConfig {
    /// メモ `nnue_b_transformer_plan_20260512` の既定値。
    pub fn nnue_b_default() -> Self {
        Self {
            vocab_size: 48,
            max_tokens: 98,
            d_model: 192,
            n_layers: 6,
            n_heads: 6,
            d_ff: 768,
            policy_size: 2187,
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
        let split_heads = |t_in: &Tensor| -> Result<Tensor> {
            Ok(t_in.reshape((b, t, self.n_heads, self.d_head))?.transpose(1, 2)?)
        };
        let q = split_heads(&self.q_proj.forward(x)?)?; // [b, h, t, d_head]
        let k = split_heads(&self.k_proj.forward(x)?)?;
        let v = split_heads(&self.v_proj.forward(x)?)?;

        // Scaled dot-product attention (no causal mask: encoder-only)
        let kt = k.transpose(D::Minus2, D::Minus1)?;
        let attn = q.matmul(&kt)?;
        let attn = (attn / self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [b, h, t, d_head]
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
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(cfg.d_model, cfg.ln_eps, vb.pp("ln1"))?,
            attn: MultiHeadAttention::new(cfg.d_model, cfg.n_heads, vb.pp("attn"))?,
            ln2: layer_norm(cfg.d_model, cfg.ln_eps, vb.pp("ln2"))?,
            ffn: FeedForward::new(cfg.d_model, cfg.d_ff, vb.pp("ffn"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm Transformer (Pre-LN)
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
/// `forward(tokens)` で `(value: [B, 1], policy: [B, policy_size])` を返す。
/// value は `tanh` 適用前の raw logit (loss 側で `tanh` 適用 or 直接 sigmoid_target)。
pub struct TransformerModel {
    pub cfg: TransformerConfig,
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_out: LayerNorm,
    value_head: Linear,
    policy_head: Linear,
}

impl TransformerModel {
    pub fn new(cfg: TransformerConfig, vb: VarBuilder) -> Result<Self> {
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
            tok_emb,
            pos_emb,
            blocks,
            ln_out,
            value_head,
            policy_head,
        })
    }

    /// Forward pass.
    ///
    /// Args:
    /// - `tokens`: `[B, T]` i64、 値域 `0..vocab_size`、 padding は 0 (PAD) 想定
    ///
    /// Returns: `(value, policy)`
    /// - `value`: `[B, 1]` f32 (raw value、 後段で tanh / sigmoid)
    /// - `policy`: `[B, policy_size]` f32 logits (softmax/CE は loss 側)
    pub fn forward(&self, tokens: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t) = tokens.dims2()?;
        let tok = self.tok_emb.forward(tokens)?; // [B, T, d_model]
                                                 // Position embedding: 0..T を broadcast
        let pos_ids = Tensor::arange(0i64, t as i64, tokens.device())?.unsqueeze(0)?;
        let pos = self.pos_emb.forward(&pos_ids)?; // [1, T, d_model]
        let mut h = (tok + pos.broadcast_as((b, t, self.cfg.d_model))?)?;
        for blk in self.blocks.iter() {
            h = blk.forward(&h)?;
        }
        let h = self.ln_out.forward(&h)?;
        // CLS token は idx 0 を採用 (token layout で先頭に置く前提)
        let cls = h.i((.., 0, ..))?; // [B, d_model]
        let value = self.value_head.forward(&cls)?; // [B, 1]
        let policy = self.policy_head.forward(&cls)?; // [B, policy_size]
        Ok((value, policy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn transformer_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = TransformerConfig::nnue_b_default();
        let model = TransformerModel::new(cfg.clone(), vb)?;
        let b = 2;
        let t = cfg.max_tokens;
        let tokens = Tensor::zeros((b, t), DType::I64, &device)?;
        let (value, policy) = model.forward(&tokens)?;
        assert_eq!(value.dims2()?, (b, 1));
        assert_eq!(policy.dims2()?, (b, cfg.policy_size));
        Ok(())
    }
}
