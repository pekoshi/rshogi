//! CPU backend (candle CPU device、 単局面最速 + 小 batch 低 latency)。

use candle_core::{Device, Tensor};

use p2_transformer::{MAX_TOKENS, TransformerTokens};

use crate::backend::{TransformerRuntime, check_shapes};
use crate::buffer::PolicyBuffer;
use crate::config::{MatryoshkaLevel, RUNTIME_POLICY_SIZE};
use crate::error::InferenceError;
use crate::model::{ModelHandle, RuntimeModelConfig};

#[cfg(target_arch = "x86_64")]
use crate::avx2_forward::{AvxScratch, AvxWeights, forward_batch1_avx2};

/// CPU 用 token scratch (heap allocation を hot path から排除)。
///
/// max_batch × MAX_TOKENS の i64 を 1 本の `Vec` で連続確保。 推論ごとに
/// `tokens[i].tokens[k]` を u16 → i64 として詰め直す。
struct CpuScratch {
    /// `[max_batch * MAX_TOKENS]` の `i64` 連続バッファ。
    tokens_i64: Vec<i64>,
    max_batch: usize,
}

impl CpuScratch {
    fn new(max_batch: usize) -> Self {
        Self {
            tokens_i64: vec![0i64; max_batch * MAX_TOKENS],
            max_batch,
        }
    }

    /// `batch[0..n]` を `tokens_i64[0..n*MAX_TOKENS]` に詰める (alloc-free)。
    #[inline]
    fn pack(&mut self, batch: &[TransformerTokens]) {
        let n = batch.len();
        debug_assert!(n <= self.max_batch);
        for (i, t) in batch.iter().enumerate() {
            let base = i * MAX_TOKENS;
            for (k, &v) in t.tokens.iter().enumerate() {
                // SAFETY: base + k < n * MAX_TOKENS <= max_batch * MAX_TOKENS、
                // self.tokens_i64.len() と一致するため OOB なし。 unchecked_index
                // は使わず通常 indexing で良い (bounds check は最適化で消える)。
                self.tokens_i64[base + k] = v as i64;
            }
        }
    }
}

/// CPU backend。 candle `Device::Cpu` 上で `ModelHandle` を保持する。
///
/// AVX2 検出時は、 batch=1 path で手書き SIMD forward (`avx2_forward`) を使う。
/// それ以外は candle path にフォールバック。
pub struct CpuTransformerRuntime {
    model: ModelHandle,
    scratch: CpuScratch,
    max_batch: usize,
    /// AVX2 高速 path 用の重み + scratch (構築可能なら Some)。
    #[cfg(target_arch = "x86_64")]
    avx2: Option<Avx2Path>,
}

#[cfg(target_arch = "x86_64")]
struct Avx2Path {
    weights: AvxWeights,
    scratch: AvxScratch,
}

impl CpuTransformerRuntime {
    /// Random init (test/bench 用)。
    pub fn new_random(max_batch: usize) -> Result<Self, InferenceError> {
        let device = Device::Cpu;
        let model = ModelHandle::random(RuntimeModelConfig::p2_default(), device)?;
        Self::wrap(model, max_batch)
    }

    /// ckpt から load。
    pub fn from_safetensors(
        max_batch: usize,
        cfg: RuntimeModelConfig,
        path: &std::path::Path,
    ) -> Result<Self, InferenceError> {
        let device = Device::Cpu;
        let model = ModelHandle::load_safetensors(cfg, path, device)
            .map_err(|e| InferenceError::ModelLoad(e.to_string()))?;
        Self::wrap(model, max_batch)
    }

    /// ModelHandle を runtime に包む共通 helper。 AVX2 検出時は重みを展開する。
    fn wrap(model: ModelHandle, max_batch: usize) -> Result<Self, InferenceError> {
        #[cfg(target_arch = "x86_64")]
        let avx2 = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let weights = AvxWeights::from_model(&model)
                .map_err(|e| InferenceError::ModelLoad(e.to_string()))?;
            let scratch = AvxScratch::new(&model.cfg);
            Some(Avx2Path { weights, scratch })
        } else {
            None
        };

        Ok(Self {
            model,
            scratch: CpuScratch::new(max_batch),
            max_batch,
            #[cfg(target_arch = "x86_64")]
            avx2,
        })
    }

    /// 内部 forward (level は今 Full のみ反映)。
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

        // AVX2 高速 path: batch=1 のみ。
        #[cfg(target_arch = "x86_64")]
        if n == 1
            && let Some(avx2) = self.avx2.as_mut()
        {
            // SAFETY: Avx2Path は AVX2+FMA が検出された場合のみ Some に
            // なるため、 forward_batch1_avx2 が要求する target_feature を満たす。
            unsafe {
                forward_batch1_avx2(
                    &avx2.weights,
                    &mut avx2.scratch,
                    &batch[0],
                    &mut out_value[0],
                    &mut out_policy[0].logits,
                );
            }
            return Ok(());
        }

        // 1. tokens を contiguous i64 buffer に pack (allocation-free)
        self.scratch.pack(batch);

        // 2. candle Tensor を構築。 `Tensor::from_slice` は 1 度 alloc するが、
        //    CPU backend では device alloc コストは無視できる範囲。 将来必要
        //    なら `Tensor::new_from_storage` で scratch を共有する道もある。
        let token_tensor = Tensor::from_slice(
            &self.scratch.tokens_i64[..n * MAX_TOKENS],
            (n, MAX_TOKENS),
            &self.model.device,
        )?;

        // 3. forward
        let (value, policy) = self
            .model
            .forward(&token_tensor)
            .map_err(|e| InferenceError::Candle(e.to_string()))?;

        // 4. dump back (avoid re-alloc on every inference: use scratch slices)
        let value_vec = value.flatten_all()?.to_vec1::<f32>()?;
        debug_assert_eq!(value_vec.len(), n);
        out_value[..n].copy_from_slice(&value_vec);

        // policy: [n, POLICY_SIZE] を1度 to_vec1 (flatten) してから row-wise copy
        let policy_flat = policy.flatten_all()?.to_vec1::<f32>()?;
        debug_assert_eq!(policy_flat.len(), n * RUNTIME_POLICY_SIZE);
        for i in 0..n {
            let src = &policy_flat[i * RUNTIME_POLICY_SIZE..(i + 1) * RUNTIME_POLICY_SIZE];
            out_policy[i].logits.copy_from_slice(src);
        }

        Ok(())
    }
}

impl TransformerRuntime for CpuTransformerRuntime {
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
        // TODO: level に応じて Light/Medium model を切替える。 現状は Full のみ。
        self.forward_internal(batch, level, out_value, out_policy)
    }

    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

// CpuTransformerRuntime は `&mut self` 経由のみ public API を提供し、 内部に
// 共有可変状態を持たない。 ModelHandle は read-only 推論のみ実施するため、
// 別 thread から呼び出す場合は scheduler 側で worker thread に owner を移す。
// (Sync を main thread から呼び出す前提では不要。 trait は `Send + Sync` を要求するが
// 内部は Send で十分。 ただし candle Tensor / VarMap は Send + Sync を満たすため
// 自動派生で問題ない想定。)
