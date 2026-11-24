//! Policy-Value forward pass (f32 参照実装)
//!
//! HalfKAv2_hm active indices を入力に取り、 value + policy_logits を返す。

use super::config::{
    EVAL_SCALE, INPUT_DIM, L1_SIZE, L2_SIZE, L3_SIZE, NormalisationConfig, POLICY_DIMS, RMS_EPS,
};
use super::loader::PolicyValueWeights;

pub struct PolicyValueNet {
    pub w: PolicyValueWeights,
}

pub struct PolicyValueOutput {
    /// value_raw: NN 生出力 (sigmoid 前、 logit スケール、 範囲 ~[-5, 5])
    pub value_raw: f32,
    /// value_cp: 補助情報 (sigmoid(value_raw) * EVAL_SCALE)
    pub value_cp: f32,
    /// policy logits (生スコア、 推論側で softmax)
    pub policy_logits: Vec<f32>,
}

#[inline]
fn screlu_f32(x: f32) -> f32 {
    let c = x.clamp(0.0, 1.0);
    c * c
}

#[inline]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn leaky_relu_f32(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.01 * x }
}

fn rms_norm_inplace(x: &mut [f32], eps: f32) {
    let n = x.len() as f32;
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let mean_sq = sum_sq / n;
    let rms_inv = 1.0 / (mean_sq + eps).sqrt();
    for v in x.iter_mut() {
        *v *= rms_inv;
    }
}

/// row-major matrix-vector: out[i] = bias[i] + Σ_j W[i*in_dim + j] * x[j]
fn matvec(out: &mut [f32], bias: &[f32], w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) {
    assert_eq!(out.len(), out_dim);
    assert_eq!(bias.len(), out_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(x.len(), in_dim);
    for i in 0..out_dim {
        let mut s = bias[i];
        let row_off = i * in_dim;
        for j in 0..in_dim {
            s += w[row_off + j] * x[j];
        }
        out[i] = s;
    }
}

impl PolicyValueNet {
    pub fn new(w: PolicyValueWeights) -> Self {
        Self { w }
    }

    /// HalfKAv2_hm active indices から FT 出力を計算 (スパース sum + bias)
    fn ft_forward(&self, active_indices: &[u32]) -> Vec<f32> {
        let mut acc = self.w.l0b.clone();
        for &idx in active_indices {
            let idx = idx as usize;
            for i in 0..L1_SIZE {
                acc[i] += self.w.l0w[i * INPUT_DIM + idx];
            }
        }
        acc
    }

    pub fn forward(&self, stm_active: &[u32], nstm_active: &[u32]) -> PolicyValueOutput {
        // FT 両 perspective
        let stm_acc = self.ft_forward(stm_active);
        let nstm_acc = self.ft_forward(nstm_active);

        if std::env::var("PV_CPU_DUMP").is_ok() {
            eprintln!("[CPU_DUMP] stm_acc[0..5]={:?}", &stm_acc[0..5]);
            eprintln!(
                "[CPU_DUMP] stm_acc sum={:.4} max={:.4} min={:.4}",
                stm_acc.iter().sum::<f32>(),
                stm_acc.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                stm_acc.iter().cloned().fold(f32::INFINITY, f32::min)
            );
        }

        // SCReLU
        let mut stm_h: Vec<f32> = stm_acc.iter().map(|&x| screlu_f32(x)).collect();
        let mut nstm_h: Vec<f32> = nstm_acc.iter().map(|&x| screlu_f32(x)).collect();

        // RMSNorm (CompactRms variant のみ)
        if self.w.cfg.normalisation == NormalisationConfig::RmsNormPerPerspective {
            rms_norm_inplace(&mut stm_h, RMS_EPS);
            rms_norm_inplace(&mut nstm_h, RMS_EPS);
        }

        // concat (stm + nstm)
        let mut concat = Vec::with_capacity(2 * L1_SIZE);
        concat.extend_from_slice(&stm_h);
        concat.extend_from_slice(&nstm_h);

        if std::env::var("PV_CPU_DUMP").is_ok() {
            eprintln!("[CPU_DUMP] concat sum={:.4}", concat.iter().sum::<f32>());
        }

        // value head (v15/v16: LeakyReLU、 dying SCReLU 回避)
        let mut h1 = vec![0.0; L2_SIZE];
        matvec(&mut h1, &self.w.l1b, &self.w.l1w, &concat, L2_SIZE, 2 * L1_SIZE);
        let h1_act: Vec<f32> = h1.iter().map(|&x| leaky_relu_f32(x)).collect();
        if std::env::var("PV_CPU_DUMP").is_ok() {
            eprintln!("[CPU_DUMP] h1 (raw)[0..5]={:?}", &h1[0..5]);
            eprintln!("[CPU_DUMP] h1_act sum={:.4}", h1_act.iter().sum::<f32>());
        }
        let mut h2 = vec![0.0; L3_SIZE];
        matvec(&mut h2, &self.w.l2b, &self.w.l2w, &h1_act, L3_SIZE, L2_SIZE);
        let h2_act: Vec<f32> = h2.iter().map(|&x| leaky_relu_f32(x)).collect();
        if std::env::var("PV_CPU_DUMP").is_ok() {
            eprintln!(
                "[CPU_DUMP] h2_act sum={:.4} max={:.4}",
                h2_act.iter().sum::<f32>(),
                h2_act.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
        let mut value_out = vec![0.0; 1];
        matvec(&mut value_out, &self.w.v_outb, &self.w.v_outw, &h2_act, 1, L3_SIZE);
        let value_raw = value_out[0];
        let value_cp = (1.0 / (1.0 + (-value_raw).exp())) * EVAL_SCALE;
        // debug
        if std::env::var("PV_DEBUG").is_ok() {
            let h2_sum: f32 = h2_act.iter().sum();
            let h2_max = h2_act.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let vw_sum: f32 = self.w.v_outw.iter().sum();
            let vw_max = self.w.v_outw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let vw_min = self.w.v_outw.iter().cloned().fold(f32::INFINITY, f32::min);
            eprintln!(
                "[PV_DEBUG] value_raw={:.6} value_out={:.6} v_outb={:.6} h2_sum={:.3} h2_max={:.3} v_outw[sum/max/min]={:.3}/{:.3}/{:.3}",
                value_raw, value_out[0], self.w.v_outb[0], h2_sum, h2_max, vw_sum, vw_max, vw_min
            );
        }

        // SwiGLU policy head
        let ph = self.w.cfg.policy_hidden;
        let mut gate_pre = vec![0.0; ph];
        matvec(&mut gate_pre, &self.w.pol_w1b, &self.w.pol_w1w, &concat, ph, 2 * L1_SIZE);
        let mut value_pre = vec![0.0; ph];
        matvec(&mut value_pre, &self.w.pol_w3b, &self.w.pol_w3w, &concat, ph, 2 * L1_SIZE);
        let swiglu: Vec<f32> =
            gate_pre.iter().zip(value_pre.iter()).map(|(&g, &v)| silu_f32(g) * v).collect();

        let mut policy_logits = vec![0.0; POLICY_DIMS];
        matvec(&mut policy_logits, &self.w.pol_ob, &self.w.pol_ow, &swiglu, POLICY_DIMS, ph);

        PolicyValueOutput {
            value_raw,
            value_cp,
            policy_logits,
        }
    }
}
