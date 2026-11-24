//! Value-safe 学習ユーティリティ
//!
//! - Huber loss (sigmoid 出力空間)
//! - Per-head LR multiplier (gradient scaling, AdamW state は単一)
//! - Global gradient clipping (L2 norm)
//! - Early stop tracker (val_value_loss 監視)

use p2_nnue::types::{
    P2_INPUT_DIM, P2_L1_DIM, P2_L2_DIM, P2_OUTLOOK_DIM, P2_OUTPUT_DIM, P2_POLICY_DIM,
};

use super::forward_f32::sigmoid;

// ----------------------------------------------------------------------
// Parameter group offsets (TOTAL_PARAMS に対する flat vector の境界)
//
// 順序: L1w, L1b, L2w, L2b, | L3w, L3b, | PolicyW, PolicyB, | OutlookW, OutlookB
// shared:  [0, SHARED_END)
// value:   [SHARED_END, VALUE_END)
// policy:  [VALUE_END, POLICY_END)
// outlook: [POLICY_END, OUTLOOK_END)
// ----------------------------------------------------------------------

pub const SHARED_END: usize =
    P2_L1_DIM * P2_INPUT_DIM + P2_L1_DIM + P2_L2_DIM * P2_L1_DIM + P2_L2_DIM;
pub const VALUE_END: usize = SHARED_END + P2_OUTPUT_DIM * P2_L2_DIM + P2_OUTPUT_DIM;
pub const POLICY_END: usize = VALUE_END + P2_POLICY_DIM * P2_L2_DIM + P2_POLICY_DIM;
pub const OUTLOOK_END: usize = POLICY_END + P2_OUTLOOK_DIM * P2_L2_DIM + P2_OUTLOOK_DIM;

// ----------------------------------------------------------------------
// Huber loss (sigmoid output space)
//
// loss = 0.5 * err^2          if |err| <= delta
//        delta * (|err| - 0.5*delta)  otherwise
// where err = sigmoid(logit) - target
//
// d_loss/d_logit = clamp(err, -delta, delta) * sigmoid * (1 - sigmoid)
// ----------------------------------------------------------------------

/// Huber loss (sigmoid 出力空間)
pub fn huber_loss_sigmoid(logit: f32, target: f32, delta: f32) -> f32 {
    let s = sigmoid(logit);
    let err = s - target;
    let abs_err = err.abs();
    if abs_err <= delta {
        0.5 * err * err
    } else {
        delta * (abs_err - 0.5 * delta)
    }
}

/// d_huber/d_logit (sigmoid 出力空間)
pub fn d_huber_sigmoid_d_logit(logit: f32, target: f32, delta: f32) -> f32 {
    let s = sigmoid(logit);
    let err = s - target;
    let clamped = err.clamp(-delta, delta);
    // d_sigmoid/d_logit = s(1-s)
    clamped * s * (1.0 - s)
}

/// 損失種別
#[derive(Debug, Clone, Copy)]
pub enum ValueLoss {
    /// BCE-with-logits (既存挙動)
    Bce,
    /// Huber loss (sigmoid 空間)
    Huber { delta: f32 },
}

impl ValueLoss {
    /// 順伝播 loss 計算
    pub fn forward(self, logit: f32, target: f32) -> f32 {
        match self {
            ValueLoss::Bce => super::backward::bce_with_logits_loss(logit, target),
            ValueLoss::Huber { delta } => huber_loss_sigmoid(logit, target, delta),
        }
    }

    /// d_loss/d_logit
    pub fn d_logit(self, logit: f32, target: f32) -> f32 {
        match self {
            ValueLoss::Bce => sigmoid(logit) - target,
            ValueLoss::Huber { delta } => d_huber_sigmoid_d_logit(logit, target, delta),
        }
    }
}

// ----------------------------------------------------------------------
// Per-head LR multiplier
//
// AdamW state は単一なので、勾配を group ごとに乗じることで等価な効果を得る。
// (multiplier 0.5 → effective LR 半分)
// ----------------------------------------------------------------------

/// Per-head LR multiplier 設定
#[derive(Debug, Clone, Copy)]
pub struct LrMultipliers {
    pub shared: f32,
    pub value_head: f32,
    pub policy_head: f32,
    pub outlook_head: f32,
}

impl LrMultipliers {
    pub fn uniform() -> Self {
        Self {
            shared: 1.0,
            value_head: 1.0,
            policy_head: 1.0,
            outlook_head: 1.0,
        }
    }

    /// 平坦化された勾配ベクトルに per-head 係数を適用する。
    pub fn apply(&self, grad_flat: &mut [f32]) {
        assert_eq!(grad_flat.len(), OUTLOOK_END, "grad length mismatch");
        if (self.shared - 1.0).abs() > f32::EPSILON {
            for g in &mut grad_flat[0..SHARED_END] {
                *g *= self.shared;
            }
        }
        if (self.value_head - 1.0).abs() > f32::EPSILON {
            for g in &mut grad_flat[SHARED_END..VALUE_END] {
                *g *= self.value_head;
            }
        }
        if (self.policy_head - 1.0).abs() > f32::EPSILON {
            for g in &mut grad_flat[VALUE_END..POLICY_END] {
                *g *= self.policy_head;
            }
        }
        if (self.outlook_head - 1.0).abs() > f32::EPSILON {
            for g in &mut grad_flat[POLICY_END..OUTLOOK_END] {
                *g *= self.outlook_head;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Global gradient clipping (L2 norm)
// ----------------------------------------------------------------------

/// 勾配ベクトルの L2 norm を計算
pub fn grad_l2_norm(grad_flat: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for &g in grad_flat {
        acc += (g as f64) * (g as f64);
    }
    acc.sqrt() as f32
}

/// L2 norm が `max_norm` を超える場合に勾配をスケールダウン。
/// 戻り値は (適用前の norm, スケール後の norm)。
pub fn clip_grad_norm(grad_flat: &mut [f32], max_norm: f32) -> (f32, f32) {
    let norm = grad_l2_norm(grad_flat);
    if norm > max_norm && max_norm > 0.0 {
        let scale = max_norm / norm;
        for g in grad_flat.iter_mut() {
            *g *= scale;
        }
        (norm, max_norm)
    } else {
        (norm, norm)
    }
}

// ----------------------------------------------------------------------
// Group-wise grad statistics (logging)
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct GradStats {
    pub shared_norm: f32,
    pub value_norm: f32,
    pub policy_norm: f32,
    pub outlook_norm: f32,
    pub total_norm: f32,
}

impl GradStats {
    pub fn from_flat(grad_flat: &[f32]) -> Self {
        let shared_norm = grad_l2_norm(&grad_flat[0..SHARED_END]);
        let value_norm = grad_l2_norm(&grad_flat[SHARED_END..VALUE_END]);
        let policy_norm = grad_l2_norm(&grad_flat[VALUE_END..POLICY_END]);
        let outlook_norm = grad_l2_norm(&grad_flat[POLICY_END..OUTLOOK_END]);
        let total_norm = (shared_norm * shared_norm
            + value_norm * value_norm
            + policy_norm * policy_norm
            + outlook_norm * outlook_norm)
            .sqrt();
        Self {
            shared_norm,
            value_norm,
            policy_norm,
            outlook_norm,
            total_norm,
        }
    }
}

// ----------------------------------------------------------------------
// Early stop tracker
// ----------------------------------------------------------------------

/// val_value_loss を監視して連続悪化を検知する。
pub struct EarlyStop {
    pub patience: u32,
    pub best: f32,
    pub bad_count: u32,
    pub regress_max: f32,
}

impl EarlyStop {
    pub fn new(patience: u32, regress_max: f32) -> Self {
        Self {
            patience,
            best: f32::INFINITY,
            bad_count: 0,
            regress_max,
        }
    }

    /// 1 epoch 終了時に呼び出す。
    /// 戻り値は (improved: 過去最良を更新したか, should_stop: 停止すべきか)。
    pub fn step(&mut self, val_value_loss: f32) -> (bool, bool) {
        let improved = val_value_loss < self.best - 1e-6;
        if improved {
            self.best = val_value_loss;
            self.bad_count = 0;
        } else {
            // 相対悪化 > regress_max の場合のみ count
            let regress = val_value_loss - self.best;
            if regress > self.regress_max {
                self.bad_count += 1;
            }
        }
        let should_stop = self.bad_count >= self.patience;
        (improved, should_stop)
    }
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huber_loss_zero_at_target() {
        // logit -> sigmoid = 0.5、 target = 0.5 → loss = 0
        let loss = huber_loss_sigmoid(0.0, 0.5, 0.1);
        assert!(loss.abs() < 1e-6, "expected near zero, got {loss}");
    }

    #[test]
    fn test_huber_loss_within_delta() {
        // small error: quadratic
        let logit = 0.0;
        let target = 0.55;
        let delta = 0.1;
        let s = sigmoid(logit); // = 0.5
        let err = s - target; // = -0.05
        let expected = 0.5 * err * err;
        let loss = huber_loss_sigmoid(logit, target, delta);
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_beyond_delta() {
        let logit = 0.0;
        let target = 1.0;
        let delta = 0.1;
        // err = 0.5 - 1.0 = -0.5、|err|=0.5 > delta
        let expected = delta * (0.5 - 0.5 * delta);
        let loss = huber_loss_sigmoid(logit, target, delta);
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm() {
        let mut grad = vec![3.0f32, 4.0]; // norm = 5
        let (before, after) = clip_grad_norm(&mut grad, 1.0);
        assert!((before - 5.0).abs() < 1e-5);
        assert!((after - 1.0).abs() < 1e-5);
        // 元の方向を保ったままスケールされている
        assert!((grad[0] - 0.6).abs() < 1e-5);
        assert!((grad[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_clip_grad_norm_no_change() {
        let mut grad = vec![0.3f32, 0.4]; // norm = 0.5 < 1.0
        let (before, after) = clip_grad_norm(&mut grad, 1.0);
        assert!((before - 0.5).abs() < 1e-5);
        assert!((after - 0.5).abs() < 1e-5);
        assert!((grad[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_offsets_consistency() {
        // P2_INPUT_DIM=9728, L1=512, L2=256, OUT=1, POLICY=2187, OUTLOOK=1
        assert_eq!(SHARED_END, 512 * P2_INPUT_DIM + 512 + 256 * 512 + 256);
        assert_eq!(VALUE_END, SHARED_END + 256 + 1);
        assert_eq!(POLICY_END, VALUE_END + 2187 * 256 + 2187);
        assert_eq!(OUTLOOK_END, POLICY_END + 256 + 1);
    }

    #[test]
    fn test_early_stop_basic() {
        let mut es = EarlyStop::new(3, 0.0);
        let (imp, stop) = es.step(0.7);
        assert!(imp);
        assert!(!stop);
        let (imp, stop) = es.step(0.71); // regress 0.01 > 0.0 → bad++
        assert!(!imp);
        assert!(!stop);
        let (imp, stop) = es.step(0.72);
        assert!(!imp);
        assert!(!stop); // bad=2
        let (_, stop) = es.step(0.73);
        assert!(stop); // bad=3, patience=3
    }

    #[test]
    fn test_early_stop_recovers() {
        let mut es = EarlyStop::new(3, 0.0);
        es.step(0.7);
        es.step(0.71); // bad=1
        es.step(0.69); // improved, bad=0
        let (_, stop) = es.step(0.7);
        assert!(!stop); // bad=1, not stop
    }

    #[test]
    fn test_lr_multipliers_apply() {
        let lr = LrMultipliers {
            shared: 0.5,
            value_head: 0.25,
            policy_head: 1.0,
            outlook_head: 0.75,
        };
        let mut grad = vec![1.0f32; OUTLOOK_END];
        lr.apply(&mut grad);
        assert!((grad[0] - 0.5).abs() < 1e-6);
        assert!((grad[SHARED_END - 1] - 0.5).abs() < 1e-6);
        assert!((grad[SHARED_END] - 0.25).abs() < 1e-6);
        assert!((grad[VALUE_END - 1] - 0.25).abs() < 1e-6);
        assert!((grad[VALUE_END] - 1.0).abs() < 1e-6);
        assert!((grad[POLICY_END - 1] - 1.0).abs() < 1e-6);
        assert!((grad[POLICY_END] - 0.75).abs() < 1e-6);
        assert!((grad[OUTLOOK_END - 1] - 0.75).abs() < 1e-6);
    }
}
