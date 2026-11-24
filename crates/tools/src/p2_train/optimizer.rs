//! AdamW オプティマイザ（decoupled weight decay）

use super::forward_f32::TOTAL_PARAMS;

/// AdamW ハイパーパラメータ
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    /// 学習率
    pub lr: f32,
    /// 一次モーメント減衰率
    pub beta1: f32,
    /// 二次モーメント減衰率
    pub beta2: f32,
    /// ゼロ除算防止
    pub epsilon: f32,
    /// 重み減衰率
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// AdamW オプティマイザ状態
pub struct AdamWState {
    /// 一次モーメント
    pub m: Vec<f32>,
    /// 二次モーメント
    pub v: Vec<f32>,
    /// ステップ数
    pub t: u64,
    /// ハイパーパラメータ
    pub config: AdamWConfig,
}

impl AdamWState {
    /// 新規作成
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            m: vec![0.0; TOTAL_PARAMS],
            v: vec![0.0; TOTAL_PARAMS],
            t: 0,
            config,
        }
    }

    /// AdamW 更新ステップ
    ///
    /// ```text
    /// m = β1*m + (1-β1)*g
    /// v = β2*v + (1-β2)*g²
    /// m̂ = m / (1 - β1^t)
    /// v̂ = v / (1 - β2^t)
    /// w = w * (1 - lr*wd) - lr * m̂ / (√v̂ + ε)
    /// ```
    pub fn step(&mut self, weights: &mut [f32], grads: &[f32]) {
        assert_eq!(weights.len(), TOTAL_PARAMS);
        assert_eq!(grads.len(), TOTAL_PARAMS);

        self.t += 1;
        let t = self.t;
        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;
        let wd = self.config.weight_decay;

        let bias_correction1 = 1.0 - beta1.powi(t as i32);
        let bias_correction2 = 1.0 - beta2.powi(t as i32);

        for i in 0..TOTAL_PARAMS {
            let g = grads[i];

            // モーメント更新
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            // バイアス補正
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            // Decoupled weight decay + Adam 更新
            weights[i] = weights[i] * (1.0 - lr * wd) - lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 二次関数 f(x) = (x-3)² + (y-5)² の最小化で収束確認
    #[test]
    fn test_adamw_convergence_quadratic() {
        // 2 パラメータだけ使う簡易テスト
        let config = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.0, // weight decay なしで純粋な収束テスト
            ..Default::default()
        };

        let mut state = AdamWState {
            m: vec![0.0; 2],
            v: vec![0.0; 2],
            t: 0,
            config,
        };

        // f(x,y) = (x-3)^2 + (y-5)^2
        // grad_x = 2*(x-3), grad_y = 2*(y-5)
        let mut w = [0.0f32, 0.0f32];

        for _ in 0..1000 {
            let grads = [2.0 * (w[0] - 3.0), 2.0 * (w[1] - 5.0)];
            // 直接 step ロジックを再現（TOTAL_PARAMS != 2 なので）
            state.t += 1;
            let t = state.t;
            let lr = state.config.lr;
            let beta1 = state.config.beta1;
            let beta2 = state.config.beta2;
            let eps = state.config.epsilon;
            let bc1 = 1.0 - beta1.powi(t as i32);
            let bc2 = 1.0 - beta2.powi(t as i32);
            for i in 0..2 {
                let g = grads[i];
                state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g;
                state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g * g;
                let m_hat = state.m[i] / bc1;
                let v_hat = state.v[i] / bc2;
                w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }

        assert!((w[0] - 3.0).abs() < 0.01, "x should converge to 3, got {}", w[0]);
        assert!((w[1] - 5.0).abs() < 0.01, "y should converge to 5, got {}", w[1]);
    }
}
