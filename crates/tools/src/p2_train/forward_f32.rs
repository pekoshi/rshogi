//! F32 順伝播（学習用）
//!
//! 整数推論と同じスケールで f32 計算を行う。
//! 中間値を `ForwardState` に保持し、逆伝播に渡す。

use p2_nnue::network::P2Network;
use p2_nnue::types::{
    P2_INPUT_DIM, P2_L1_DIM, P2_L2_DIM, P2_OUTLOOK_DIM, P2_OUTPUT_DIM, P2_POLICY_DIM,
    WEIGHT_SCALE_BITS,
};

/// 全層の f32 重み（triple head: value + policy + outlook）
pub struct F32Weights {
    /// L1 weights [L1_DIM × INPUT_DIM] 行優先
    pub l1_weights: Vec<f32>,
    /// L1 biases [L1_DIM]
    pub l1_biases: Vec<f32>,
    /// L2 weights [L2_DIM × L1_DIM] 行優先
    pub l2_weights: Vec<f32>,
    /// L2 biases [L2_DIM]
    pub l2_biases: Vec<f32>,
    /// L3 weights [OUTPUT_DIM × L2_DIM] 行優先
    pub l3_weights: Vec<f32>,
    /// L3 biases [OUTPUT_DIM]
    pub l3_biases: Vec<f32>,
    /// Policy head weights [POLICY_DIM × L2_DIM] 行優先
    pub policy_weights: Vec<f32>,
    /// Policy head biases [POLICY_DIM]
    pub policy_biases: Vec<f32>,
    /// Outlook head weights [OUTLOOK_DIM × L2_DIM] 行優先
    pub outlook_weights: Vec<f32>,
    /// Outlook head biases [OUTLOOK_DIM]
    pub outlook_biases: Vec<f32>,
}

/// 学習対象パラメータ総数（triple head: value + policy + outlook）
pub const TOTAL_PARAMS: usize = P2_L1_DIM * P2_INPUT_DIM
    + P2_L1_DIM
    + P2_L2_DIM * P2_L1_DIM
    + P2_L2_DIM
    + P2_OUTPUT_DIM * P2_L2_DIM
    + P2_OUTPUT_DIM
    + P2_POLICY_DIM * P2_L2_DIM
    + P2_POLICY_DIM
    + P2_OUTLOOK_DIM * P2_L2_DIM
    + P2_OUTLOOK_DIM;

/// ClippedReLU の上限（整数推論と同じ: >> 6 後 clamp [0, 127]）
const RELU_MAX: f32 = 127.0;

/// ClippedReLU のスケールファクタ (1 << WEIGHT_SCALE_BITS)
const RELU_SCALE: f32 = (1 << WEIGHT_SCALE_BITS) as f32;

impl F32Weights {
    /// P2Network の量子化重みから f32 重みを構築
    pub fn from_network(net: &P2Network) -> Self {
        let flat = net.weights_to_f32();
        Self::from_flat(&flat)
    }

    /// フラットな f32 ベクトルから構築
    ///
    /// 順序: L1w, L1b, L2w, L2b, L3w, L3b, PolicyW, PolicyB, OutlookW, OutlookB
    pub fn from_flat(flat: &[f32]) -> Self {
        assert_eq!(flat.len(), TOTAL_PARAMS);
        let mut pos = 0;

        let l1_w = P2_L1_DIM * P2_INPUT_DIM;
        let l1_weights = flat[pos..pos + l1_w].to_vec();
        pos += l1_w;
        let l1_biases = flat[pos..pos + P2_L1_DIM].to_vec();
        pos += P2_L1_DIM;

        let l2_w = P2_L2_DIM * P2_L1_DIM;
        let l2_weights = flat[pos..pos + l2_w].to_vec();
        pos += l2_w;
        let l2_biases = flat[pos..pos + P2_L2_DIM].to_vec();
        pos += P2_L2_DIM;

        let l3_w = P2_OUTPUT_DIM * P2_L2_DIM;
        let l3_weights = flat[pos..pos + l3_w].to_vec();
        pos += l3_w;
        let l3_biases = flat[pos..pos + P2_OUTPUT_DIM].to_vec();
        pos += P2_OUTPUT_DIM;

        let pol_w = P2_POLICY_DIM * P2_L2_DIM;
        let policy_weights = flat[pos..pos + pol_w].to_vec();
        pos += pol_w;
        let policy_biases = flat[pos..pos + P2_POLICY_DIM].to_vec();
        pos += P2_POLICY_DIM;

        let out_w = P2_OUTLOOK_DIM * P2_L2_DIM;
        let outlook_weights = flat[pos..pos + out_w].to_vec();
        pos += out_w;
        let outlook_biases = flat[pos..pos + P2_OUTLOOK_DIM].to_vec();

        Self {
            l1_weights,
            l1_biases,
            l2_weights,
            l2_biases,
            l3_weights,
            l3_biases,
            policy_weights,
            policy_biases,
            outlook_weights,
            outlook_biases,
        }
    }

    /// フラットな f32 ベクトルに展開
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

    /// He 初期化（量子化スケール RELU_SCALE=64 を考慮し、std を 64 倍する）
    ///
    /// forward の ClippedReLU が `clamp(x/64, 0, 127)` であるため、
    /// 通常の He init では l1_pre が小さすぎて signal が潰れて学習が進まない。
    /// 量子化推論側は i8/i16 weight (±127 / ±32767) を想定するので、
    /// 学習 f32 でも同等の絶対値スケールに init する。
    pub fn he_init(rng: &mut impl rand::Rng) -> Self {
        /// Box-Muller 変換で標準正規乱数を生成
        fn randn(rng: &mut impl rand::Rng) -> f64 {
            let u1: f64 = rng.random_range(1e-10f64..1.0);
            let u2: f64 = rng.random_range(0.0f64..std::f64::consts::TAU);
            (-2.0 * u1.ln()).sqrt() * u2.cos()
        }

        // ClippedReLU の量子化 scale (=64) と forward chain の増幅を考慮。
        // × 64 だと初期 logit が飽和、× 1 だと信号潰れるため、× 16 を選ぶ。
        // 試走で初期 logit std ≈ 0.3、value loss ≈ ln(2) からスタート可能。
        const INIT_SCALE: f64 = 16.0;

        let mut he_layer = |fan_in: usize, count: usize| -> Vec<f32> {
            let std = (2.0 / fan_in as f64).sqrt() * INIT_SCALE;
            (0..count).map(|_| (randn(rng) * std) as f32).collect()
        };

        Self {
            l1_weights: he_layer(P2_INPUT_DIM, P2_L1_DIM * P2_INPUT_DIM),
            l1_biases: vec![0.0; P2_L1_DIM],
            l2_weights: he_layer(P2_L1_DIM, P2_L2_DIM * P2_L1_DIM),
            l2_biases: vec![0.0; P2_L2_DIM],
            l3_weights: he_layer(P2_L2_DIM, P2_OUTPUT_DIM * P2_L2_DIM),
            l3_biases: vec![0.0; P2_OUTPUT_DIM],
            policy_weights: he_layer(P2_L2_DIM, P2_POLICY_DIM * P2_L2_DIM),
            policy_biases: vec![0.0; P2_POLICY_DIM],
            outlook_weights: he_layer(P2_L2_DIM, P2_OUTLOOK_DIM * P2_L2_DIM),
            outlook_biases: vec![0.0; P2_OUTLOOK_DIM],
        }
    }
}

/// 順伝播の中間値（逆伝播用）
pub struct ForwardState {
    /// L1 ReLU 前 [L1_DIM]
    pub l1_pre: Vec<f32>,
    /// L1 ReLU 後 [L1_DIM]
    pub l1_post: Vec<f32>,
    /// L2 ReLU 前 [L2_DIM]
    pub l2_pre: Vec<f32>,
    /// L2 ReLU 後 [L2_DIM]
    pub l2_post: Vec<f32>,
    /// Value head 出力（logit）
    pub logit: f32,
    /// Policy head 出力（生 logits, [POLICY_DIM]）
    pub policy_logits: Vec<f32>,
    /// Outlook head 出力（logit、 sigmoid 適用は loss 側）
    pub outlook_logit: f32,
}

/// ClippedReLU (f32): clamp(x / RELU_SCALE, 0, RELU_MAX)
#[inline]
fn clipped_relu(x: f32) -> f32 {
    (x / RELU_SCALE).clamp(0.0, RELU_MAX)
}

/// f32 順伝播
///
/// 整数推論と同スケール:
/// - L1: out = Σ(w * input) + bias
/// - ReLU: clamp(out / 64, 0, 127)
/// - L2, L3: 同様
/// - 最終出力: logit（sigmoid は loss 関数側で適用）
pub fn forward(weights: &F32Weights, input: &[f32; P2_INPUT_DIM]) -> ForwardState {
    // Layer 1: affine + ClippedReLU
    let mut l1_pre = vec![0.0f32; P2_L1_DIM];
    for (j, pre) in l1_pre.iter_mut().enumerate() {
        let mut acc = weights.l1_biases[j];
        let row = &weights.l1_weights[j * P2_INPUT_DIM..(j + 1) * P2_INPUT_DIM];
        for (w, &x) in row.iter().zip(input.iter()) {
            acc += w * x;
        }
        *pre = acc;
    }
    let l1_post: Vec<f32> = l1_pre.iter().map(|&x| clipped_relu(x)).collect();

    // Layer 2: affine + ClippedReLU
    let mut l2_pre = vec![0.0f32; P2_L2_DIM];
    for (j, pre) in l2_pre.iter_mut().enumerate() {
        let mut acc = weights.l2_biases[j];
        let row = &weights.l2_weights[j * P2_L1_DIM..(j + 1) * P2_L1_DIM];
        for (w, &x) in row.iter().zip(l1_post.iter()) {
            acc += w * x;
        }
        *pre = acc;
    }
    let l2_post: Vec<f32> = l2_pre.iter().map(|&x| clipped_relu(x)).collect();

    // Layer 3: affine (no activation) — value head
    let mut logit = weights.l3_biases[0];
    for (&w, &x) in weights.l3_weights.iter().zip(l2_post.iter()) {
        logit += w * x;
    }

    // Policy head: affine (no activation, no scaling) [POLICY_DIM × L2_DIM]
    let mut policy_logits = vec![0.0f32; P2_POLICY_DIM];
    for (j, pl) in policy_logits.iter_mut().enumerate() {
        let mut acc = weights.policy_biases[j];
        let row = &weights.policy_weights[j * P2_L2_DIM..(j + 1) * P2_L2_DIM];
        for (w, &x) in row.iter().zip(l2_post.iter()) {
            acc += w * x;
        }
        *pl = acc;
    }

    // Outlook head: affine (no activation), 1 出力
    let mut outlook_logit = weights.outlook_biases[0];
    for (&w, &x) in weights.outlook_weights.iter().zip(l2_post.iter()) {
        outlook_logit += w * x;
    }

    ForwardState {
        l1_pre,
        l1_post,
        l2_pre,
        l2_post,
        logit,
        policy_logits,
        outlook_logit,
    }
}

/// sigmoid 関数
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipped_relu_values() {
        assert_eq!(clipped_relu(-100.0), 0.0);
        assert_eq!(clipped_relu(0.0), 0.0);
        assert_eq!(clipped_relu(64.0), 1.0);
        assert_eq!(clipped_relu(127.0 * 64.0), 127.0);
        assert_eq!(clipped_relu(200.0 * 64.0), 127.0);
    }

    #[test]
    fn test_forward_zero_weights() {
        let weights = F32Weights::from_flat(&vec![0.0; TOTAL_PARAMS]);
        let input = [0.0f32; P2_INPUT_DIM];
        let state = forward(&weights, &input);
        assert_eq!(state.logit, 0.0);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_he_init_nonzero() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let w = F32Weights::he_init(&mut rng);
        // 重みがゼロでないことを確認
        assert!(w.l1_weights.iter().any(|&x| x != 0.0));
        assert!(w.l2_weights.iter().any(|&x| x != 0.0));
        assert!(w.l3_weights.iter().any(|&x| x != 0.0));
        // バイアスはゼロ
        assert!(w.l1_biases.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_flat_roundtrip() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let w = F32Weights::he_init(&mut rng);
        let flat = w.to_flat();
        let w2 = F32Weights::from_flat(&flat);
        assert_eq!(w.l1_weights, w2.l1_weights);
        assert_eq!(w.l3_biases, w2.l3_biases);
    }
}
