//! P2 NNUE V3 型定義: XL v3 (Conv 8 層 + Layer Stacks + 3 段 policy MLP).
//!
//! V2 (`types.rs`) との関係:
//! - V2 は 3 層 FC NNUE (L1=512, L2=256) で、 既存の独自 SIMD/CUDA カーネルを使用。
//! - V3 (本ファイル) は Conv + 大 trunk + 深 policy MLP の Hybrid モデル。
//! - V3 の重みは独自バイナリ `.p2nn v3` (magic "P2NN" + version=3) でロードする。
//! - 量子化形式は段階的に拡張: v1 = f32、 v2 = INT8 SIMD、 v3 = INT8 + GPU。

/// magic bytes for `.p2nn v3` files
pub const V3_MAGIC: &[u8; 4] = b"P2NN";
/// supported version (`.p2nn` 内 u32 version field)
pub const V3_VERSION: u32 = 3;

/// 入力次元 (V2 と共通: dlshogi 互換 features 9639 + 89 padding = 9728、 32 byte alignment)
pub const V3_INPUT_DIM: usize = 9728;
/// 実 features 次元 (padding 抜き、 reshape 用)
pub const V3_EFFECTIVE_DIM: usize = 9639;
/// 入力チャンネル (features1 62ch + features2 57ch)
pub const V3_INPUT_CHANNELS: usize = 119;
/// Policy 出力次元 (dlshogi 27 方向 × 81 マス)
pub const V3_POLICY_DIM: usize = 2187;
/// 評価値スケール (V2 と共通)
pub const V3_VALUE_SCALE: i32 = 1 << 15;

/// 量子化フォーマット (`.p2nn v3` ヘッダー内)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFmt {
    F32 = 0,
    Int8 = 1,
    Mixed = 2,
}

impl QuantFmt {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::Int8),
            2 => Some(Self::Mixed),
            _ => None,
        }
    }
}

/// XL v3 model 構造設定
///
/// `.p2nn v3` ヘッダーから読み込まれ、 各層の次元を保持する。
#[derive(Debug, Clone, Copy)]
pub struct ModelConfigV3 {
    pub n_conv_layers: u8,
    pub conv_ch: u16,
    pub fc1: u32,
    pub fc2: u32,
    pub fc3: u32,
    pub n_stacks: u8,
    pub policy_mlp: u32,
    pub policy_dim: u32,
    pub quant_fmt: QuantFmt,
}

impl ModelConfigV3 {
    /// XL v3 (180M) のデフォルト構成
    pub const fn xl_v3_180m() -> Self {
        Self {
            n_conv_layers: 8,
            conv_ch: 256,
            fc1: 5376,
            fc2: 2048,
            fc3: 1024,
            n_stacks: 4,
            policy_mlp: 2048,
            policy_dim: V3_POLICY_DIM as u32,
            quant_fmt: QuantFmt::F32,
        }
    }

    /// 平面化した Conv 出力次元 (conv_ch × 9 × 9)
    pub fn flat_dim(&self) -> usize {
        self.conv_ch as usize * 9 * 9
    }

    /// パラメータ数の概算 (検証用)
    pub fn estimated_params(&self) -> u64 {
        let n = self.n_conv_layers as u64;
        let c = self.conv_ch as u64;
        let mut total: u64 = 0;
        // Conv1 (5x5): 119*c*25 + c
        total += V3_INPUT_CHANNELS as u64 * c * 25 + c;
        // Conv2..n (3x3)
        if n > 1 {
            total += (n - 1) * (c * c * 9 + c);
        }
        // GroupNorm
        total += n * (c + c);
        // fc1
        total += self.flat_dim() as u64 * self.fc1 as u64 + self.fc1 as u64;
        // fc2/fc3 stacks
        let ns = self.n_stacks as u64;
        total += ns * (self.fc1 as u64 * self.fc2 as u64 + self.fc2 as u64);
        total += ns * (self.fc2 as u64 * self.fc3 as u64 + self.fc3 as u64);
        // head_ln
        total += self.fc3 as u64 + self.fc3 as u64;
        // value/outlook
        total += 2 * (self.fc3 as u64 + 1);
        // policy MLP 3 段 + LN x 2
        total += self.fc3 as u64 * self.policy_mlp as u64 + self.policy_mlp as u64;
        total += self.policy_mlp as u64 + self.policy_mlp as u64; // ln1
        total += self.policy_mlp as u64 * self.policy_mlp as u64 + self.policy_mlp as u64;
        total += self.policy_mlp as u64 + self.policy_mlp as u64; // ln2
        total += self.policy_mlp as u64 * self.policy_dim as u64 + self.policy_dim as u64;
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xl_v3_180m_params_about_180m() {
        let cfg = ModelConfigV3::xl_v3_180m();
        let n = cfg.estimated_params();
        // Python smoke では 179,608,717 が報告されているため近似一致を期待
        let expected = 179_608_717_u64;
        let diff = n.abs_diff(expected);
        assert!(
            diff * 1000 < expected, // 0.1% 以内
            "params mismatch: rust={}, python={}, diff={}",
            n,
            expected,
            diff,
        );
    }
}
