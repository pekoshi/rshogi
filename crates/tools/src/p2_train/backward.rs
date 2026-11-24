//! 逆伝播 + BCEWithLogits loss + masked policy CE loss
//!
//! f32 順伝播の中間値（`ForwardState`）を使い、全パラメータの勾配を計算する。

use p2_nnue::types::{
    P2_INPUT_DIM, P2_L1_DIM, P2_L2_DIM, P2_OUTLOOK_DIM, P2_OUTPUT_DIM, P2_POLICY_DIM,
    WEIGHT_SCALE_BITS,
};

use super::forward_f32::{F32Weights, ForwardState, TOTAL_PARAMS};

/// ClippedReLU の上限（整数推論と同じ: >> 6 後 clamp [0, 127]）
const RELU_MAX_PRE: f32 = 127.0 * (1 << WEIGHT_SCALE_BITS) as f32;

/// 勾配（F32Weights と同じレイアウト）
pub struct Gradients {
    pub l1_weights: Vec<f32>,
    pub l1_biases: Vec<f32>,
    pub l2_weights: Vec<f32>,
    pub l2_biases: Vec<f32>,
    pub l3_weights: Vec<f32>,
    pub l3_biases: Vec<f32>,
    pub policy_weights: Vec<f32>,
    pub policy_biases: Vec<f32>,
    pub outlook_weights: Vec<f32>,
    pub outlook_biases: Vec<f32>,
}

impl Gradients {
    /// ゼロ初期化
    pub fn zeros() -> Self {
        Self {
            l1_weights: vec![0.0; P2_L1_DIM * P2_INPUT_DIM],
            l1_biases: vec![0.0; P2_L1_DIM],
            l2_weights: vec![0.0; P2_L2_DIM * P2_L1_DIM],
            l2_biases: vec![0.0; P2_L2_DIM],
            l3_weights: vec![0.0; P2_OUTPUT_DIM * P2_L2_DIM],
            l3_biases: vec![0.0; P2_OUTPUT_DIM],
            policy_weights: vec![0.0; P2_POLICY_DIM * P2_L2_DIM],
            policy_biases: vec![0.0; P2_POLICY_DIM],
            outlook_weights: vec![0.0; P2_OUTLOOK_DIM * P2_L2_DIM],
            outlook_biases: vec![0.0; P2_OUTLOOK_DIM],
        }
    }

    /// フラットな f32 ベクトルに展開（AdamW 用）
    pub fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(TOTAL_PARAMS);
        flat.extend_from_slice(&self.l1_weights);
        flat.extend_from_slice(&self.l1_biases);
        flat.extend_from_slice(&self.l2_weights);
        flat.extend_from_slice(&self.l2_biases);
        flat.extend_from_slice(&self.l3_weights);
        flat.extend_from_slice(&self.l3_biases);
        flat.extend_from_slice(&self.policy_weights);
        flat.extend_from_slice(&self.policy_biases);
        flat.extend_from_slice(&self.outlook_weights);
        flat.extend_from_slice(&self.outlook_biases);
        flat
    }

    /// 別の勾配を累積加算
    pub fn accumulate(&mut self, other: &Gradients) {
        add_slice(&mut self.l1_weights, &other.l1_weights);
        add_slice(&mut self.l1_biases, &other.l1_biases);
        add_slice(&mut self.l2_weights, &other.l2_weights);
        add_slice(&mut self.l2_biases, &other.l2_biases);
        add_slice(&mut self.l3_weights, &other.l3_weights);
        add_slice(&mut self.l3_biases, &other.l3_biases);
        add_slice(&mut self.policy_weights, &other.policy_weights);
        add_slice(&mut self.policy_biases, &other.policy_biases);
        add_slice(&mut self.outlook_weights, &other.outlook_weights);
        add_slice(&mut self.outlook_biases, &other.outlook_biases);
    }
}

fn add_slice(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

/// BCEWithLogits loss（数値安定版）
///
/// `loss = max(logit, 0) - logit * target + ln(1 + exp(-|logit|))`
#[inline]
pub fn bce_with_logits_loss(logit: f32, target: f32) -> f32 {
    logit.max(0.0) - logit * target + (1.0 + (-logit.abs()).exp()).ln()
}

/// 合法手マスク付き softmax cross-entropy loss
///
/// 合法手 `legal_labels` の logits だけで softmax をとり、target_label の負対数尤度を返す。
/// 数値安定化のため log-sum-exp を使用する。
pub fn masked_softmax_ce_loss(
    policy_logits: &[f32],
    target_label: usize,
    legal_labels: &[u32],
) -> f32 {
    if legal_labels.is_empty() {
        return 0.0;
    }
    let mut max_logit = f32::NEG_INFINITY;
    for &lbl in legal_labels {
        let v = policy_logits[lbl as usize];
        if v > max_logit {
            max_logit = v;
        }
    }
    let mut sum_exp = 0.0f32;
    for &lbl in legal_labels {
        sum_exp += (policy_logits[lbl as usize] - max_logit).exp();
    }
    let target_logit = policy_logits[target_label];
    let log_sum_exp = sum_exp.ln() + max_logit;
    log_sum_exp - target_logit
}

/// ClippedReLU の勾配
///
/// `d_out/d_in = 1/RELU_SCALE if 0 < pre < RELU_MAX_PRE, else 0`
#[inline]
fn clipped_relu_grad(pre: f32) -> f32 {
    let scale = (1 << WEIGHT_SCALE_BITS) as f32;
    if pre > 0.0 && pre < RELU_MAX_PRE {
        1.0 / scale
    } else {
        0.0
    }
}

/// 1 サンプルの逆伝播 (triple head: value + policy + outlook)
///
/// `d_value_logit` / `d_outlook_logit` は外部計算済みの d_loss/d_logit (BCE/Huber 任意)。
/// `lambda_policy` は policy 損失への重み係数（0.0 だと policy 学習無効）。
/// `legal_labels` が空、もしくは `target_label` が legal に含まれない場合、
/// policy 勾配は 0 として扱う。
pub fn backward(
    weights: &F32Weights,
    state: &ForwardState,
    input: &[f32; P2_INPUT_DIM],
    d_value_logit: f32,
    d_outlook_logit: f32,
    target_label: usize,
    legal_labels: &[u32],
    lambda_policy: f32,
) -> Gradients {
    let mut grad = Gradients::zeros();

    let d_logit = d_value_logit;

    // --- Value head: Layer 3 (256 → 1) ---
    grad.l3_biases[0] = d_logit;
    for (gw, &l2p) in grad.l3_weights.iter_mut().zip(state.l2_post.iter()) {
        *gw = d_logit * l2p;
    }
    // d_value/d_l2_post[i] = d_logit * l3_weights[i]
    let mut d_l2_post: Vec<f32> = weights.l3_weights.iter().map(|&w| d_logit * w).collect();

    // --- Outlook head: 256 → 1 (BCE/Huber は呼び出し側) ---
    grad.outlook_biases[0] = d_outlook_logit;
    for (gw, &l2p) in grad.outlook_weights.iter_mut().zip(state.l2_post.iter()) {
        *gw = d_outlook_logit * l2p;
    }
    // d_l2_post への outlook 経路寄与を加算
    for (i, &w) in weights.outlook_weights.iter().enumerate() {
        d_l2_post[i] += d_outlook_logit * w;
    }

    // --- Policy head: masked softmax CE 勾配 ---
    // d_policy_logit[j] = lambda * (probs_masked[j] - 1[j == target_label])
    // 非合法ラベルへの勾配は 0 とする。
    let policy_active = lambda_policy > 0.0
        && !legal_labels.is_empty()
        && legal_labels.iter().any(|&l| l as usize == target_label);

    if policy_active {
        // legal 内で max を取って数値安定化
        let mut max_logit = f32::NEG_INFINITY;
        for &lbl in legal_labels {
            let v = state.policy_logits[lbl as usize];
            if v > max_logit {
                max_logit = v;
            }
        }
        let mut sum_exp = 0.0f32;
        // exp_logits を一時バッファに格納
        let mut exp_logits: Vec<(usize, f32)> = Vec::with_capacity(legal_labels.len());
        for &lbl in legal_labels {
            let e = (state.policy_logits[lbl as usize] - max_logit).exp();
            sum_exp += e;
            exp_logits.push((lbl as usize, e));
        }
        let inv_sum = 1.0 / sum_exp;

        // d_policy_logit を計算しつつ policy_weights/biases の勾配を蓄積
        for (idx, e) in &exp_logits {
            let prob = e * inv_sum;
            let onehot = if *idx == target_label { 1.0 } else { 0.0 };
            let d_pl = lambda_policy * (prob - onehot);
            grad.policy_biases[*idx] = d_pl;
            let row = idx * P2_L2_DIM;
            for k in 0..P2_L2_DIM {
                grad.policy_weights[row + k] = d_pl * state.l2_post[k];
            }
            // d_l2_post への寄与を加算
            for k in 0..P2_L2_DIM {
                d_l2_post[k] += d_pl * weights.policy_weights[row + k];
            }
        }
    }

    // --- ClippedReLU L2 ---
    let d_l2_pre: Vec<f32> = d_l2_post
        .iter()
        .zip(state.l2_pre.iter())
        .map(|(&d, &pre)| d * clipped_relu_grad(pre))
        .collect();

    // --- Layer 2 (512 → 256) ---
    for (j, &d) in d_l2_pre.iter().enumerate() {
        grad.l2_biases[j] = d;
        let row_offset = j * P2_L1_DIM;
        for (i, &l1p) in state.l1_post.iter().enumerate() {
            grad.l2_weights[row_offset + i] = d * l1p;
        }
    }
    // d_loss/d_l1_post
    let mut d_l1_post = vec![0.0f32; P2_L1_DIM];
    for (j, &d) in d_l2_pre.iter().enumerate() {
        let row_offset = j * P2_L1_DIM;
        for (i, dl1) in d_l1_post.iter_mut().enumerate() {
            *dl1 += d * weights.l2_weights[row_offset + i];
        }
    }

    // --- ClippedReLU L1 ---
    let d_l1_pre: Vec<f32> = d_l1_post
        .iter()
        .zip(state.l1_pre.iter())
        .map(|(&d, &pre)| d * clipped_relu_grad(pre))
        .collect();

    // --- Layer 1 (2048 → 512) ---
    for (j, &d) in d_l1_pre.iter().enumerate() {
        grad.l1_biases[j] = d;
        let row_offset = j * P2_INPUT_DIM;
        for (i, &inp) in input.iter().enumerate() {
            grad.l1_weights[row_offset + i] = d * inp;
        }
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    // 2026-10-08: sigmoid の import が無く `cargo test -p tools --lib` が
    //   コンパイルできなかった (本件とは無関係の既存の壊れ。 1 行の import 追加のみ)。
    use crate::p2_train::forward_f32::{F32Weights, forward, sigmoid};

    /// 数値微分との一致検証
    #[test]
    fn test_backward_numerical_gradient() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);

        // 小さな重みで初期化（勾配が消失しない範囲）
        let weights = F32Weights::he_init(&mut rng);

        // スパースなバイナリ入力（実際の NNUE 入力に近い）
        let mut input = [0.0f32; P2_INPUT_DIM];
        input[0] = 1.0;
        input[100] = 1.0;
        input[500] = 1.0;
        input[1000] = 1.0;

        let target = 0.7f32;

        let state = forward(&weights, &input);
        // BCE 勾配を呼び出し側で計算（既存挙動の確認）
        let d_logit = sigmoid(state.logit) - target;
        let grad = backward(&weights, &state, &input, d_logit, 0.0, 0, &[], 0.0);

        let eps = 1e-4;

        // L3 bias の数値微分チェック
        {
            let mut w_plus = weights.to_flat();
            let mut w_minus = weights.to_flat();
            let idx = P2_L1_DIM * P2_INPUT_DIM
                + P2_L1_DIM
                + P2_L2_DIM * P2_L1_DIM
                + P2_L2_DIM
                + P2_OUTPUT_DIM * P2_L2_DIM; // L3 bias offset
            w_plus[idx] += eps;
            w_minus[idx] -= eps;

            let wp = F32Weights::from_flat(&w_plus);
            let wm = F32Weights::from_flat(&w_minus);
            let sp = forward(&wp, &input);
            let sm = forward(&wm, &input);
            let loss_p = bce_with_logits_loss(sp.logit, target);
            let loss_m = bce_with_logits_loss(sm.logit, target);
            let numerical = (loss_p - loss_m) / (2.0 * eps);
            let analytical = grad.l3_biases[0];

            assert!(
                (numerical - analytical).abs() < 1e-2,
                "L3 bias grad mismatch: numerical={numerical}, analytical={analytical}"
            );
        }

        // L3 weight[0] の数値微分チェック
        {
            let mut w_plus = weights.to_flat();
            let mut w_minus = weights.to_flat();
            let idx = P2_L1_DIM * P2_INPUT_DIM + P2_L1_DIM + P2_L2_DIM * P2_L1_DIM + P2_L2_DIM; // L3 weight offset
            w_plus[idx] += eps;
            w_minus[idx] -= eps;

            let wp = F32Weights::from_flat(&w_plus);
            let wm = F32Weights::from_flat(&w_minus);
            let sp = forward(&wp, &input);
            let sm = forward(&wm, &input);
            let loss_p = bce_with_logits_loss(sp.logit, target);
            let loss_m = bce_with_logits_loss(sm.logit, target);
            let numerical = (loss_p - loss_m) / (2.0 * eps);
            let analytical = grad.l3_weights[0];

            assert!(
                (numerical - analytical).abs() < 1e-2,
                "L3 weight[0] grad mismatch: numerical={numerical}, analytical={analytical}"
            );
        }

        // L2 bias[0] の数値微分チェック
        {
            let mut w_plus = weights.to_flat();
            let mut w_minus = weights.to_flat();
            let idx = P2_L1_DIM * P2_INPUT_DIM + P2_L1_DIM + P2_L2_DIM * P2_L1_DIM;
            w_plus[idx] += eps;
            w_minus[idx] -= eps;

            let wp = F32Weights::from_flat(&w_plus);
            let wm = F32Weights::from_flat(&w_minus);
            let sp = forward(&wp, &input);
            let sm = forward(&wm, &input);
            let loss_p = bce_with_logits_loss(sp.logit, target);
            let loss_m = bce_with_logits_loss(sm.logit, target);
            let numerical = (loss_p - loss_m) / (2.0 * eps);
            let analytical = grad.l2_biases[0];

            assert!(
                (numerical - analytical).abs() < 1e-2,
                "L2 bias[0] grad mismatch: numerical={numerical}, analytical={analytical}"
            );
        }
    }

    #[test]
    fn test_bce_with_logits_known_values() {
        // logit=0, target=0.5 → loss = ln(2) ≈ 0.6931
        let loss = bce_with_logits_loss(0.0, 0.5);
        assert!((loss - 2.0f32.ln()).abs() < 1e-5);

        // logit=0, target=1.0 → loss = ln(2) ≈ 0.6931
        let loss = bce_with_logits_loss(0.0, 1.0);
        assert!((loss - 2.0f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_gradients_accumulate() {
        let mut g1 = Gradients::zeros();
        g1.l3_biases[0] = 1.0;
        g1.l1_biases[0] = 2.0;

        let mut g2 = Gradients::zeros();
        g2.l3_biases[0] = 3.0;
        g2.l1_biases[0] = 4.0;

        g1.accumulate(&g2);
        assert_eq!(g1.l3_biases[0], 4.0);
        assert_eq!(g1.l1_biases[0], 6.0);
    }
}
