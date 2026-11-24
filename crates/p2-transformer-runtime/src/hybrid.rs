//! Hybrid routing wrapper。
//!
//! batch.len() / `EvalPriority` を見て CPU / GPU を切替える。 GPU backend が
//! 未 build (feature 無効) の場合は常に CPU にフォールバック。

use p2_transformer::TransformerTokens;

use crate::backend::TransformerRuntime;
use crate::buffer::PolicyBuffer;
use crate::config::{EvalPriority, HybridConfig, MatryoshkaLevel};
use crate::error::InferenceError;

/// CPU + (optional) GPU の両 backend を持ち、 size/priority で routing する。
///
/// `Box<dyn TransformerRuntime>` で保持するため、 backend の具象型に依存しない。
pub struct HybridTransformerRuntime {
    cpu: Box<dyn TransformerRuntime>,
    gpu: Option<Box<dyn TransformerRuntime>>,
    cfg: HybridConfig,
}

impl HybridTransformerRuntime {
    /// CPU のみ (GPU 未使用)。
    pub fn cpu_only(cpu: Box<dyn TransformerRuntime>, cfg: HybridConfig) -> Self {
        Self {
            cpu,
            gpu: None,
            cfg,
        }
    }

    /// CPU + GPU 構成。
    pub fn with_gpu(
        cpu: Box<dyn TransformerRuntime>,
        gpu: Box<dyn TransformerRuntime>,
        cfg: HybridConfig,
    ) -> Self {
        Self {
            cpu,
            gpu: Some(gpu),
            cfg,
        }
    }

    /// 設定参照。
    pub fn config(&self) -> &HybridConfig {
        &self.cfg
    }

    /// routing 判定 (公開: テスト / metrics 用)。
    ///
    /// 戻り値 `true` = GPU、 `false` = CPU。
    #[inline]
    pub fn should_use_gpu(&self, batch_len: usize, priority: EvalPriority) -> bool {
        if self.gpu.is_none() {
            return false;
        }
        match priority {
            EvalPriority::Urgent => false,
            EvalPriority::Bulk => true,
            EvalPriority::Normal => batch_len >= self.cfg.gpu_threshold,
        }
    }

    /// priority 指定付き評価。
    pub fn eval_with_priority(
        &mut self,
        batch: &[TransformerTokens],
        priority: EvalPriority,
        level: MatryoshkaLevel,
        out_value: &mut [f32],
        out_policy: &mut [PolicyBuffer],
    ) -> Result<(), InferenceError> {
        if self.should_use_gpu(batch.len(), priority) {
            // gpu.is_some() は should_use_gpu 内で確定
            let gpu = self.gpu.as_mut().expect("gpu present after should_use_gpu");
            gpu.eval_with_level(batch, level, out_value, out_policy)
        } else {
            self.cpu.eval_with_level(batch, level, out_value, out_policy)
        }
    }
}

impl TransformerRuntime for HybridTransformerRuntime {
    fn eval(
        &mut self,
        batch: &[TransformerTokens],
        out_value: &mut [f32],
        out_policy: &mut [PolicyBuffer],
    ) -> Result<(), InferenceError> {
        self.eval_with_priority(
            batch,
            EvalPriority::Normal,
            MatryoshkaLevel::Full,
            out_value,
            out_policy,
        )
    }

    fn eval_with_level(
        &mut self,
        batch: &[TransformerTokens],
        level: MatryoshkaLevel,
        out_value: &mut [f32],
        out_policy: &mut [PolicyBuffer],
    ) -> Result<(), InferenceError> {
        self.eval_with_priority(batch, EvalPriority::Normal, level, out_value, out_policy)
    }

    fn max_batch(&self) -> usize {
        // 安全側: CPU/GPU の min を返す (GPU 無ければ CPU)
        match &self.gpu {
            Some(gpu) => self.cpu.max_batch().min(gpu.max_batch()),
            None => self.cpu.max_batch(),
        }
    }

    fn name(&self) -> &'static str {
        "hybrid"
    }
}
