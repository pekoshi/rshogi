//! 訓練 loss / step / ckpt 保存ヘルパ。
//!
//! 実 dataset pipeline (PSV unpack → tokens) は別 binary 側で組み合わせる。
//! 本 module は「token tensor + score + result + policy_idx」が揃った後の
//! loss 計算と optimizer step だけを薄く包む。
//!
//! ## loss
//!
//! - value: `MSE(sigmoid(value_raw), target)` where
//!   `target = wdl * (result01) + (1 - wdl) * sigmoid(score / scale)`
//! - policy: `CE(log_softmax(policy_logits), policy_idx)`、 `policy_idx = -1`
//!   サンプルは loss 寄与から mask out
//!
//! score は cp 単位 (stm 視点)、 result は `{0.0, 0.5, 1.0}` (stm 視点) を想定。
//! 上流が Black 視点なら呼び出し前に flip しておくこと。

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// 1 batch 分の入出力 tensor。
pub struct TrainBatch {
    /// `[B, T]` i64
    pub tokens: Tensor,
    /// `[B]` f32、 stm 視点 cp
    pub score: Tensor,
    /// `[B]` f32、 stm 視点 result (0/0.5/1)
    pub result: Tensor,
    /// `[B]` i64、 -1 は mask 対象
    pub policy_idx: Tensor,
}

/// loss 計算結果。
pub struct LossOutputs {
    /// scalar f32
    pub value_loss: Tensor,
    /// scalar f32 (no policy 時は 0)
    pub policy_loss: Tensor,
    /// scalar f32: value_loss + policy_weight * policy_loss
    pub total: Tensor,
}

/// 訓練 loss を計算する。
///
/// - `value_raw`: `[B, 1]` モデル出力 (sigmoid 前)
/// - `policy_logits`: `[B, policy_size]` モデル出力 (softmax 前)
/// - `batch`: target データ
/// - `scale`: cp → sigmoid のスケール (例 600.0)
/// - `wdl`: result と sigmoid(score) を mix する係数 (0=score のみ、 1=result のみ)
/// - `policy_weight`: policy loss の重み (0 で policy 学習無効)
pub fn compute_loss(
    value_raw: &Tensor,
    policy_logits: &Tensor,
    batch: &TrainBatch,
    scale: f32,
    wdl: f32,
    policy_weight: f32,
    device: &Device,
) -> Result<LossOutputs> {
    // value target: wdl * result + (1 - wdl) * sigmoid(score / scale)
    let scale_f = scale as f64;
    let score_sigmoid = candle_nn::ops::sigmoid(&batch.score.affine(1.0 / scale_f, 0.0)?)?;
    let target_value = ((&batch.result * wdl as f64)? + (score_sigmoid * (1.0 - wdl as f64))?)?;
    let value_pred_sig = candle_nn::ops::sigmoid(&value_raw.squeeze(1)?)?;
    let value_loss = candle_nn::loss::mse(&value_pred_sig, &target_value)?;

    let policy_loss = if policy_weight > 0.0 {
        let mask = batch.policy_idx.ge(0i64)?.to_dtype(DType::F32)?;
        let safe_idx = batch.policy_idx.maximum(&Tensor::zeros_like(&batch.policy_idx)?)?;
        let log_prob = candle_nn::ops::log_softmax(policy_logits, 1)?;
        let nll = log_prob.gather(&safe_idx.unsqueeze(1)?, 1)?.squeeze(1)?.neg()?;
        let masked = (nll * &mask)?;
        let n_valid = mask.sum_all()?.to_scalar::<f32>()?.max(1.0);
        masked.sum_all()?.affine(1.0 / n_valid as f64, 0.0)?
    } else {
        Tensor::zeros((), DType::F32, device)?
    };

    let total = (&value_loss + (&policy_loss * policy_weight as f64)?)?;
    Ok(LossOutputs {
        value_loss,
        policy_loss,
        total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NnueBConfig, NnueBModel};

    #[test]
    fn loss_runs_on_random_inputs() -> Result<()> {
        let device = Device::Cpu;
        let cfg = NnueBConfig::tiny_for_tests();
        let model = NnueBModel::new(cfg.clone(), device.clone())?;
        let b = 4usize;
        let tokens = Tensor::zeros((b, cfg.max_tokens), DType::I64, &device)?;
        let score = Tensor::from_vec(vec![100.0f32, -200.0, 0.0, 300.0], b, &device)?;
        let result = Tensor::from_vec(vec![1.0f32, 0.0, 0.5, 1.0], b, &device)?;
        let policy = Tensor::from_vec(vec![10i64, 20, -1, 5], b, &device)?;
        let batch = TrainBatch {
            tokens: tokens.clone(),
            score,
            result,
            policy_idx: policy,
        };
        let (v, p) = model.forward(&tokens)?;
        let out = compute_loss(&v, &p, &batch, 600.0, 0.5, 1.0, &device)?;
        // scalar shape
        assert_eq!(out.total.dims().len(), 0);
        Ok(())
    }
}
