//! XL v3 用 CPU 評価器 (f32 baseline、 量子化なし).
//!
//! P3 の責務: PyTorch model.forward と数値が一致する Rust f32 推論を提供する。
//! P4 で AVX2 INT8 SIMD 化、 P5 で CUDA に展開。
//!
//! Forward 構造 (model.py P2NnueHybrid と一致):
//!   1. input 9728 → effective 9639 → reshape (119, 9, 9)
//!   2. Conv 8 層 (5x5 + 3x3 x 7) + GroupNorm(1) + ClippedReLU(1.0) ループ
//!   3. flatten (conv_ch × 9 × 9)
//!   4. fc1 + SCReLU
//!   5. phase 別 LayerStacks: fc2[s] + SCReLU → fc3[s] + SCReLU
//!   6. head_ln (LayerNorm)
//!   7. value_head / outlook_head: 線形 → i32 value
//!   8. policy MLP 3 段: fc1' + LN + SCReLU → fc2' + LN + SCReLU → fc3'
//!
//! 注意: PyTorch の `nn.GroupNorm(1, C)` は 1 sample 内全 (C, H, W) を 1 group で正規化する.

use std::sync::Arc;

use crate::network_v3::P2NetworkV3;
use crate::types_v3::{V3_EFFECTIVE_DIM, V3_INPUT_CHANNELS, V3_VALUE_SCALE};

/// XL v3 forward の出力
pub struct P2OutputV3 {
    /// 評価値 (i32 量子化、 V3_VALUE_SCALE 倍されている。 ÷ V3_VALUE_SCALE で [-1.0, 1.0] 勝率)
    pub value: i32,
    /// 大局観 (i32 量子化、 上同)
    pub outlook: i32,
    /// policy logits (f32、 softmax 前)。 caller で legal mask + softmax して使う。
    pub policy_logits: Vec<f32>,
}

/// XL v3 用 CPU 評価器
pub struct CpuEvaluatorV3 {
    pub network: Arc<P2NetworkV3>,
}

impl CpuEvaluatorV3 {
    pub fn new(network: Arc<P2NetworkV3>) -> Self {
        Self { network }
    }

    /// Forward 推論
    ///
    /// Args:
    ///   feats: 9728 dim binary feature (i8、 0 or 1)
    ///   phase_idx: LayerStack index (0..n_stacks)
    pub fn forward(&self, feats: &[i8], phase_idx: usize) -> P2OutputV3 {
        let net = &self.network;
        let cfg = net.config;
        let conv_ch = cfg.conv_ch as usize;
        let fc1 = cfg.fc1 as usize;
        let fc2 = cfg.fc2 as usize;
        let fc3 = cfg.fc3 as usize;
        let mlp = cfg.policy_mlp as usize;
        let policy_dim = cfg.policy_dim as usize;
        let n_conv = cfg.n_conv_layers as usize;

        debug_assert_eq!(feats.len(), 9728);
        debug_assert!(phase_idx < cfg.n_stacks as usize);

        // ── 1. input → (119, 9, 9) f32 ──
        let mut x = vec![0.0f32; V3_INPUT_CHANNELS * 9 * 9];
        for (i, &b) in feats.iter().take(V3_EFFECTIVE_DIM).enumerate() {
            x[i] = b as f32;
        }
        let mut cur_ch = V3_INPUT_CHANNELS;

        // ── 2. Conv 部 (n_conv 層) ──
        let mut x_buf = x;
        for li in 0..n_conv {
            let kernel = if li == 0 { 5usize } else { 3 };
            let pad = kernel / 2;
            let in_ch = cur_ch;
            let out_ch = conv_ch;

            let conv_out = conv2d_f32(
                &x_buf,
                in_ch,
                9,
                9,
                kernel,
                pad,
                &net.conv_weights[li],
                &net.conv_biases[li],
                out_ch,
            );
            // GroupNorm(1, C): 1 sample 内全 (C, H, W) を平均/分散で正規化
            let normed = group_norm_1grp(
                &conv_out,
                out_ch,
                9,
                9,
                &net.conv_ln_weights[li],
                &net.conv_ln_biases[li],
                1e-5,
            );
            // ClippedReLU(1.0) = clamp(x, 0, 1)
            let clipped: Vec<f32> = normed.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
            x_buf = clipped;
            cur_ch = out_ch;
        }

        // ── 3. flatten ──
        debug_assert_eq!(x_buf.len(), conv_ch * 9 * 9);

        // ── 4. fc1 + SCReLU ──
        let h1 = linear_f32(&x_buf, &net.fc1_weight, &net.fc1_bias, fc1);
        let h1 = scrlu(&h1);

        // ── 5. LayerStack (phase 別) ──
        let h2 = linear_f32(
            &h1,
            &net.fc2_stack_weights[phase_idx],
            &net.fc2_stack_biases[phase_idx],
            fc2,
        );
        let h2 = scrlu(&h2);
        let h3 = linear_f32(
            &h2,
            &net.fc3_stack_weights[phase_idx],
            &net.fc3_stack_biases[phase_idx],
            fc3,
        );
        let h3 = scrlu(&h3);

        // ── 6. head_ln ──
        let h_ln = layer_norm(&h3, &net.head_ln_weight, &net.head_ln_bias, 1e-5);

        // ── 7. value / outlook ──
        let v_logit = dot(&net.value_head_weight, &h_ln) + net.value_head_bias[0];
        let o_logit = dot(&net.outlook_head_weight, &h_ln) + net.outlook_head_bias[0];
        // sigmoid → cp 換算は caller 側で行う。 ここでは sigmoid を取らず value_scale 倍にする。
        let v_winrate = sigmoid(v_logit);
        let o_winrate = sigmoid(o_logit);
        // [0, 1] sigmoid → [-1, 1] に変換 (NNUE 規約に合わせる)
        let value_signed = (v_winrate * 2.0 - 1.0).clamp(-1.0, 1.0);
        let outlook_signed = (o_winrate * 2.0 - 1.0).clamp(-1.0, 1.0);
        let value_i32 = (value_signed * V3_VALUE_SCALE as f32) as i32;
        let outlook_i32 = (outlook_signed * V3_VALUE_SCALE as f32) as i32;

        // ── 8. policy 3 段 MLP ──
        let p1 = linear_f32(&h_ln, &net.policy_fc1_weight, &net.policy_fc1_bias, mlp);
        let p1 = layer_norm(&p1, &net.policy_ln1_weight, &net.policy_ln1_bias, 1e-5);
        let p1 = scrlu(&p1);
        // Dropout 推論時 identity
        let p2 = linear_f32(&p1, &net.policy_fc2_weight, &net.policy_fc2_bias, mlp);
        let p2 = layer_norm(&p2, &net.policy_ln2_weight, &net.policy_ln2_bias, 1e-5);
        let p2 = scrlu(&p2);
        let policy_logits =
            linear_f32(&p2, &net.policy_fc3_weight, &net.policy_fc3_bias, policy_dim);

        P2OutputV3 {
            value: value_i32,
            outlook: outlook_i32,
            policy_logits,
        }
    }
}

// ────────────────────────────────────────────────────────────────
// 演算プリミティブ (naive、 f32)
// ────────────────────────────────────────────────────────────────

/// 2D Convolution (NCHW、 padding="same", stride=1)
///
/// weights: (Cout, Cin, K, K) row-major
/// 入力: (Cin, H, W) row-major
/// 出力: (Cout, H, W) row-major
fn conv2d_f32(
    x: &[f32],
    cin: usize,
    h: usize,
    w: usize,
    kernel: usize,
    pad: usize,
    weights: &[f32],
    biases: &[f32],
    cout: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), cin * h * w);
    debug_assert_eq!(weights.len(), cout * cin * kernel * kernel);
    debug_assert_eq!(biases.len(), cout);
    let mut out = vec![0.0f32; cout * h * w];
    for oc in 0..cout {
        let bias = biases[oc];
        for oy in 0..h {
            for ox in 0..w {
                let mut sum = bias;
                for ic in 0..cin {
                    for ky in 0..kernel {
                        let iy = oy as isize + ky as isize - pad as isize;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..kernel {
                            let ix = ox as isize + kx as isize - pad as isize;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            let w_idx = ((oc * cin + ic) * kernel + ky) * kernel + kx;
                            let x_idx = (ic * h + iy as usize) * w + ix as usize;
                            sum += weights[w_idx] * x[x_idx];
                        }
                    }
                }
                out[(oc * h + oy) * w + ox] = sum;
            }
        }
    }
    out
}

/// GroupNorm(1, C): 1 sample 内全 (C, H, W) で平均/分散を計算し正規化、 channel ごとに weight/bias 適用
fn group_norm_1grp(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    weight: &[f32], // (C,)
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    let n = c * h * w;
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std_inv = 1.0 / (var + eps).sqrt();
    let mut out = vec![0.0f32; n];
    for ci in 0..c {
        let wc = weight[ci];
        let bc = bias[ci];
        for i in 0..(h * w) {
            let idx = ci * h * w + i;
            out[idx] = (x[idx] - mean) * std_inv * wc + bc;
        }
    }
    out
}

/// LayerNorm (1D、 入力次元 = D)
fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len() as f32;
    let mean = x.iter().sum::<f32>() / d;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d;
    let std_inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .map(|((&v, &w), &b)| (v - mean) * std_inv * w + b)
        .collect()
}

/// Linear (out, in) row-major. y = W x + b
fn linear_f32(x: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(bias.len(), out_dim);
    let mut out = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut sum = bias[o];
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        for i in 0..in_dim {
            sum += row[i] * x[i];
        }
        out[o] = sum;
    }
    out
}

/// SCReLU: y = clamp(x, 0, 1)^2
fn scrlu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let c = v.clamp(0.0, 1.0);
            c * c
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_naive() {
        let x = vec![1.0f32, 2.0, 3.0];
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2x3 = identity-ish
        let b = vec![10.0, 20.0];
        let y = linear_f32(&x, &w, &b, 2);
        assert_eq!(y, vec![11.0, 22.0]);
    }

    #[test]
    fn scrlu_basic() {
        let x = vec![-1.0, 0.0, 0.5, 1.0, 2.0];
        let y = scrlu(&x);
        assert_eq!(y, vec![0.0, 0.0, 0.25, 1.0, 1.0]);
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let b = vec![0.0; 4];
        let y = layer_norm(&x, &w, &b, 1e-5);
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        let var: f32 = y.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((mean).abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn conv_identity() {
        // 1x1 kernel + identity weight
        let cin = 1;
        let cout = 1;
        let h = 3;
        let w = 3;
        let kernel = 1;
        let pad = 0;
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weights = vec![1.0f32]; // (Cout, Cin, K, K) = (1, 1, 1, 1)
        let biases = vec![0.0f32];
        let y = conv2d_f32(&x, cin, h, w, kernel, pad, &weights, &biases, cout);
        assert_eq!(y, x);
    }
}
