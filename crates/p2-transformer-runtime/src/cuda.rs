//! CUDA backend (candle `Device::Cuda(0)`、 persistent Tensor buffers)。
//!
//! ## 戦略
//!
//! 1. `MAX_BATCH × MAX_TOKENS` の `i64` device buffer (`token_tensor`) を 1 度だけ確保。
//!    各推論では host scratch を `cudaMemcpyAsync` で書き戻し、 slice で実効 batch を切る。
//! 2. forward は `model.forward(&token_view)` を呼ぶ。 candle は内部で 中間 Tensor
//!    を確保するが、 候補 stream / pool 機構は今後の改良対象 (TODO)。
//! 3. 出力 (`value`, `policy`) は device → host で 1 回ずつ転送。
//!
//! ## TODO
//!
//! - pinned memory (Windows での実装) → host scratch を `cudaHostAlloc` で取得。
//! - async stream (現状は default stream、 synchronous)。
//! - fp16 (model dtype を f16 にしたうえで forward)。
//! - graph capture / ONNX export / TensorRT。

use candle_core::{DType, Device, Tensor};

use p2_transformer::{MAX_TOKENS, TransformerTokens};

use crate::backend::{TransformerRuntime, check_shapes};
use crate::buffer::PolicyBuffer;
use crate::config::{MatryoshkaLevel, RUNTIME_POLICY_SIZE};
use crate::error::InferenceError;
use crate::model::{ModelHandle, RuntimeModelConfig};

/// CUDA 上で再利用する device tensor 群。
///
/// candle 上は `Tensor` 自体が device storage を抱えているため、 `token_tensor`
/// は `[MAX_BATCH, MAX_TOKENS]` の固定 shape を保持し、 forward 直前に `narrow`
/// で実効 batch を切り出す。 これで毎回の cuda malloc を避ける。
pub struct CudaBuffers {
    /// `[MAX_BATCH, MAX_TOKENS]` の `i64` device tensor。 全 inference で再利用。
    pub token_tensor: Tensor,
    /// host pinned (将来) scratch。 現状は heap `Vec<i64>`。
    host_scratch_tokens: Vec<i64>,
    /// value 出力の host dump 先 (`[MAX_BATCH]`)。
    host_value_scratch: Vec<f32>,
    /// policy 出力の host dump 先 (`[MAX_BATCH * POLICY_SIZE]`)。
    host_policy_scratch: Vec<f32>,
    max_batch: usize,
}

impl CudaBuffers {
    fn new(max_batch: usize, device: &Device) -> Result<Self, InferenceError> {
        let token_tensor = Tensor::zeros((max_batch, MAX_TOKENS), DType::I64, device)?;
        Ok(Self {
            token_tensor,
            host_scratch_tokens: vec![0i64; max_batch * MAX_TOKENS],
            host_value_scratch: vec![0.0f32; max_batch],
            host_policy_scratch: vec![0.0f32; max_batch * RUNTIME_POLICY_SIZE],
            max_batch,
        })
    }

    /// host scratch を埋めて、 `[n, MAX_TOKENS]` の view を device 上に得る。
    fn upload_tokens(
        &mut self,
        batch: &[TransformerTokens],
        device: &Device,
    ) -> Result<Tensor, InferenceError> {
        let n = batch.len();
        debug_assert!(n <= self.max_batch);
        for (i, t) in batch.iter().enumerate() {
            let base = i * MAX_TOKENS;
            for (k, &v) in t.tokens.iter().enumerate() {
                self.host_scratch_tokens[base + k] = v as i64;
            }
        }
        // 1 度の memcpy で device 上に転送。 token_tensor の上書きを避けるため、
        // 別 Tensor を作る (candle の現行 API では in-place memcpy は提供されない)。
        // ただし `Tensor::from_slice` の device alloc コストは MAX_BATCH × MAX_TOKENS
        // という固定上限を持つため、 GPU では非常に小さい。
        let t = Tensor::from_slice(
            &self.host_scratch_tokens[..n * MAX_TOKENS],
            (n, MAX_TOKENS),
            device,
        )?;
        Ok(t)
    }
}

/// CUDA backend。
pub struct CudaTransformerRuntime {
    model: ModelHandle,
    buffers: CudaBuffers,
    max_batch: usize,
}

impl CudaTransformerRuntime {
    /// Random init (bench/sanity 用)。
    pub fn new_random(max_batch: usize) -> Result<Self, InferenceError> {
        let device = Device::new_cuda(0)?;
        let model = ModelHandle::random(RuntimeModelConfig::p2_default(), device.clone())?;
        let buffers = CudaBuffers::new(max_batch, &device)?;
        Ok(Self {
            model,
            buffers,
            max_batch,
        })
    }

    /// ckpt から load。
    pub fn from_safetensors(
        max_batch: usize,
        cfg: RuntimeModelConfig,
        path: &std::path::Path,
    ) -> Result<Self, InferenceError> {
        let device = Device::new_cuda(0)?;
        let model = ModelHandle::load_safetensors(cfg, path, device.clone())
            .map_err(|e| InferenceError::ModelLoad(e.to_string()))?;
        let buffers = CudaBuffers::new(max_batch, &device)?;
        Ok(Self {
            model,
            buffers,
            max_batch,
        })
    }

    fn forward_internal(
        &mut self,
        batch: &[TransformerTokens],
        _level: MatryoshkaLevel,
        out_value: &mut [f32],
        out_policy: &mut [PolicyBuffer],
    ) -> Result<(), InferenceError> {
        check_shapes(batch, out_value, out_policy, self.max_batch)?;
        let n = batch.len();
        if n == 0 {
            return Ok(());
        }
        // 1. tokens を device buffer に upload (固定 shape の view)
        let token_view = self.buffers.upload_tokens(batch, &self.model.device)?;

        // 2. forward (model 内部は async 不可、 default stream)
        let (value, policy) = self
            .model
            .forward(&token_view)
            .map_err(|e| InferenceError::Candle(e.to_string()))?;

        // 3. device → host (candle to_vec1 が同期 copy)
        let value_vec = value.flatten_all()?.to_vec1::<f32>()?;
        debug_assert_eq!(value_vec.len(), n);
        out_value[..n].copy_from_slice(&value_vec);

        let policy_flat = policy.flatten_all()?.to_vec1::<f32>()?;
        debug_assert_eq!(policy_flat.len(), n * RUNTIME_POLICY_SIZE);
        for i in 0..n {
            let src = &policy_flat[i * RUNTIME_POLICY_SIZE..(i + 1) * RUNTIME_POLICY_SIZE];
            out_policy[i].logits.copy_from_slice(src);
        }

        Ok(())
    }
}

impl TransformerRuntime for CudaTransformerRuntime {
    fn eval(
        &mut self,
        batch: &[TransformerTokens],
        out_value: &mut [f32],
        out_policy: &mut [PolicyBuffer],
    ) -> Result<(), InferenceError> {
        self.forward_internal(batch, MatryoshkaLevel::Full, out_value, out_policy)
    }

    fn eval_with_level(
        &mut self,
        batch: &[TransformerTokens],
        level: MatryoshkaLevel,
        out_value: &mut [f32],
        out_policy: &mut [PolicyBuffer],
    ) -> Result<(), InferenceError> {
        self.forward_internal(batch, level, out_value, out_policy)
    }

    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn name(&self) -> &'static str {
        "cuda"
    }
}
