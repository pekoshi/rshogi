//! Runtime 内部で使う Transformer model 定義 (candle ベース)。
//!
//! `nnue-trainer-rs::transformer::TransformerModel` と同等の構造を、 `p2-transformer`
//! の token system (vocab=512, max_tokens=128, policy=2187) に合わせて持つ。
//! ckpt は safetensors を想定 (load 部分は `ModelHandle::load_safetensors`)。
//!
//! ## 注意
//!
//! - candle backend は `Device` (Cpu / Cuda) を切替えるだけで CPU / CUDA 両方
//!   で動く。 backend ごとに別 `ModelHandle` を保持する。
//! - 本 module は推論専用。 学習は `nnue-trainer-rs/bin/train_transformer.rs`
//!   を使う。

use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap, embedding, layer_norm, linear,
    linear_no_bias,
};

use p2_transformer::{MAX_TOKENS, POLICY_SIZE, VOCAB_SIZE};

/// Runtime 用 Transformer 設定。 token system の vocab / max_tokens に整合。
#[derive(Clone, Debug)]
pub struct RuntimeModelConfig {
    pub vocab_size: usize,
    pub max_tokens: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub policy_size: usize,
    pub ln_eps: f64,
}

impl RuntimeModelConfig {
    /// `p2-transformer` token system に合わせた既定値。
    pub fn p2_default() -> Self {
        Self {
            vocab_size: VOCAB_SIZE,
            max_tokens: MAX_TOKENS,
            d_model: 192,
            n_layers: 6,
            n_heads: 6,
            d_ff: 768,
            policy_size: POLICY_SIZE,
            ln_eps: 1e-5,
        }
    }

    /// Light (Matryoshka 浅) の構成。 ckpt が出来るまで未使用。
    pub fn p2_light() -> Self {
        let mut c = Self::p2_default();
        c.n_layers = 2;
        c
    }

    /// Medium の構成。
    pub fn p2_medium() -> Self {
        let mut c = Self::p2_default();
        c.n_layers = 4;
        c
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
        assert_eq!(d_model % n_heads, 0);
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
    fn new(cfg: &RuntimeModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(cfg.d_model, cfg.ln_eps, vb.pp("ln1"))?,
            attn: MultiHeadAttention::new(cfg.d_model, cfg.n_heads, vb.pp("attn"))?,
            ln2: layer_norm(cfg.d_model, cfg.ln_eps, vb.pp("ln2"))?,
            ffn: FeedForward::new(cfg.d_model, cfg.d_ff, vb.pp("ffn"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h)?;
        let x = (x + h)?;
        let h = self.ln2.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        Ok((x + h)?)
    }
}

/// AVX2 path 用に block 内 tensor 参照を 1 つにまとめた helper。
pub struct BlockTensors<'a> {
    pub ln1_gamma: &'a Tensor,
    pub ln1_beta: &'a Tensor,
    pub q_weight: &'a Tensor,
    pub k_weight: &'a Tensor,
    pub v_weight: &'a Tensor,
    pub out_weight: &'a Tensor,
    pub out_bias: &'a Tensor,
    pub ln2_gamma: &'a Tensor,
    pub ln2_beta: &'a Tensor,
    pub fc1_weight: &'a Tensor,
    pub fc1_bias: &'a Tensor,
    pub fc2_weight: &'a Tensor,
    pub fc2_bias: &'a Tensor,
}

/// Inference 用 model handle。 ckpt 未指定時は random init (test/bench 用)。
///
/// `forward(&Tensor) -> (value, policy)`:
/// - `value`: `[B, 1]` f32 raw (後段で tanh / sigmoid 適用)
/// - `policy`: `[B, POLICY_SIZE]` f32 logits
pub struct ModelHandle {
    pub cfg: RuntimeModelConfig,
    pub device: Device,
    /// `VarMap` を保持して embedding/weight の lifetime を保つ。
    _varmap: VarMap,
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_out: LayerNorm,
    value_head: Linear,
    policy_head: Linear,
}

impl ModelHandle {
    /// Random init で model を作成 (test/bench 用、 weight は VarMap の default 値)。
    pub fn random(cfg: RuntimeModelConfig, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self::build(cfg, device, varmap, vb)
    }

    /// safetensors ckpt から load。
    ///
    /// TODO: 実 ckpt の key prefix が定まったら `vb.pp(...)` の renaming map を追加。
    pub fn load_safetensors(
        cfg: RuntimeModelConfig,
        path: &std::path::Path,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        // safetensors を VarBuilder に load
        // VarMap::load は in-place で書き込む。 build 後に呼ぶ手順:
        //   1. random init で build
        //   2. varmap.load(path)
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut handle = Self::build(cfg, device, varmap, vb)?;
        handle
            ._varmap
            .load(path)
            .map_err(|e| anyhow::anyhow!("failed to load safetensors {:?}: {}", path, e))?;
        Ok(handle)
    }

    fn build(
        cfg: RuntimeModelConfig,
        device: Device,
        varmap: VarMap,
        vb: VarBuilder,
    ) -> Result<Self> {
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
            _varmap: varmap,
            tok_emb,
            pos_emb,
            blocks,
            ln_out,
            value_head,
            policy_head,
        })
    }

    /// AVX2 path 用: tok_emb tensor を `[VOCAB_SIZE, d_model]` で返す。
    pub fn tok_emb_tensor(&self) -> &Tensor {
        self.tok_emb.embeddings()
    }

    /// AVX2 path 用: pos_emb tensor を `[MAX_TOKENS, d_model]` で返す。
    pub fn pos_emb_tensor(&self) -> &Tensor {
        self.pos_emb.embeddings()
    }

    /// AVX2 path 用: ln_out (gamma, beta)。
    pub fn ln_out_tensors(&self) -> Result<(&Tensor, &Tensor)> {
        Ok((
            self.ln_out.weight(),
            self.ln_out.bias().ok_or_else(|| anyhow::anyhow!("ln_out has no bias"))?,
        ))
    }

    /// AVX2 path 用: value head (weight, bias)。
    pub fn value_head_tensors(&self) -> Result<(&Tensor, &Tensor)> {
        Ok((
            self.value_head.weight(),
            self.value_head
                .bias()
                .ok_or_else(|| anyhow::anyhow!("value_head has no bias"))?,
        ))
    }

    /// AVX2 path 用: policy head (weight, bias)。
    pub fn policy_head_tensors(&self) -> Result<(&Tensor, &Tensor)> {
        Ok((
            self.policy_head.weight(),
            self.policy_head
                .bias()
                .ok_or_else(|| anyhow::anyhow!("policy_head has no bias"))?,
        ))
    }

    /// AVX2 path 用: 指定 block の tensor 一式。
    pub fn block_tensors(&self, idx: usize) -> Result<BlockTensors<'_>> {
        let blk = self
            .blocks
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("block index out of range: {idx}"))?;
        Ok(BlockTensors {
            ln1_gamma: blk.ln1.weight(),
            ln1_beta: blk.ln1.bias().ok_or_else(|| anyhow::anyhow!("ln1 has no bias"))?,
            q_weight: blk.attn.q_proj.weight(),
            k_weight: blk.attn.k_proj.weight(),
            v_weight: blk.attn.v_proj.weight(),
            out_weight: blk.attn.out_proj.weight(),
            out_bias: blk
                .attn
                .out_proj
                .bias()
                .ok_or_else(|| anyhow::anyhow!("attn out_proj has no bias"))?,
            ln2_gamma: blk.ln2.weight(),
            ln2_beta: blk.ln2.bias().ok_or_else(|| anyhow::anyhow!("ln2 has no bias"))?,
            fc1_weight: blk.ffn.fc1.weight(),
            fc1_bias: blk.ffn.fc1.bias().ok_or_else(|| anyhow::anyhow!("fc1 has no bias"))?,
            fc2_weight: blk.ffn.fc2.weight(),
            fc2_bias: blk.ffn.fc2.bias().ok_or_else(|| anyhow::anyhow!("fc2 has no bias"))?,
        })
    }

    /// `tokens`: `[B, T]` i64 → `(value [B,1], policy [B, POLICY_SIZE])`。
    pub fn forward(&self, tokens: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t) = tokens.dims2()?;
        let tok = self.tok_emb.forward(tokens)?;
        let pos_ids = Tensor::arange(0i64, t as i64, tokens.device())?.unsqueeze(0)?;
        let pos = self.pos_emb.forward(&pos_ids)?;
        let mut h = (tok + pos.broadcast_as((b, t, self.cfg.d_model))?)?;
        for blk in self.blocks.iter() {
            h = blk.forward(&h)?;
        }
        let h = self.ln_out.forward(&h)?;
        let cls = h.i((.., 0, ..))?;
        let value = self.value_head.forward(&cls)?;
        let policy = self.policy_head.forward(&cls)?;
        Ok((value, policy))
    }
}
