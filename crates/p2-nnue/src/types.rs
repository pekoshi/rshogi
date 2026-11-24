//! P2 NNUE ネットワークの型定義と定数

/// 評価値スケール (1<<15)
pub const VALUE_SCALE: i32 = 1 << 15;

/// 入力次元数 (V3: dlshogi 互換 features 9639 + 89 padding = 9728、 32 byte alignment)
pub const P2_INPUT_DIM: usize = 9728;
/// Layer 1 出力次元数
pub const P2_L1_DIM: usize = 512;
/// Layer 2 出力次元数
pub const P2_L2_DIM: usize = 256;
/// 最終出力次元数
pub const P2_OUTPUT_DIM: usize = 1;

/// Policy head 出力次元数（dlshogi 形式: 27 方向 × 81 マス）
pub const P2_POLICY_DIM: usize = 2187;

/// Outlook (大局観) head 出力次元数 (1 = scalar)
pub const P2_OUTLOOK_DIM: usize = 1;

/// ClippedReLU のシフト量
pub const WEIGHT_SCALE_BITS: u32 = 6;

/// ネットワーク幅パラメータ（学習側と推論側で共有）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelConfig {
    pub input_dim: usize,
    pub l1_dim: usize,
    pub l2_dim: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_dim: P2_INPUT_DIM,
            l1_dim: P2_L1_DIM,
            l2_dim: P2_L2_DIM,
        }
    }
}

/// P2 NNUE の出力
pub struct P2NnueOutput {
    /// 生の評価値 (i32)
    pub value: i32,
    /// Policy hint: L2 ReLU 出力（共有表現 256D）
    pub policy_hint: Option<P2PolicyHint>,
    /// v1 では常に None（型予約のみ）
    pub uncertainty: Option<f32>,
}

/// Policy hint: L2 ReLU 出力（共有表現 256D）のコピー
///
/// 追加レイヤーなし。学習側で per-legal-move prior を教師として使う。
pub struct P2PolicyHint {
    /// 共有表現ベクトル（L2 ReLU 出力、u8 [0,127]）
    pub shared_repr: [u8; P2_L2_DIM],
}

/// Policy logits: dlshogi 形式 (27 × 81 = 2187) の生スコア。
/// 推論側で softmax + legal_moves でマスクして使用する。
pub struct P2PolicyLogits {
    pub logits: Box<[i32; P2_POLICY_DIM]>,
}
