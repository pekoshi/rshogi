//! Policy-Value NNUE 共通 config

/// 入力 dim (HalfKAv2_hm = 73,305)
pub const INPUT_DIM: usize = 73_305;
/// FT 隠れ次元 (L1)
pub const L1_SIZE: usize = 2048;
/// value head L2
pub const L2_SIZE: usize = 32;
/// value head L3
pub const L3_SIZE: usize = 96;
/// policy logits 次元 (27 方向 × 81 マス)
pub const POLICY_DIMS: usize = 2187;
/// 推論 eval scale (訓練側 SCALE = 600.0)
pub const EVAL_SCALE: f32 = 600.0;
/// RMSNorm epsilon
pub const RMS_EPS: f32 = 1e-6;

/// 入力 normalisation 設定 (post-SCReLU、 perspective 別 RMSNorm)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalisationConfig {
    /// 正規化なし
    None,
    /// post-SCReLU で各 perspective を RMSNorm して concat
    RmsNormPerPerspective,
}

/// arch バリエーション
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyValueVariant {
    /// POLICY_HIDDEN=512、 LeakyReLU value (v16)
    Tiny,
    /// POLICY_HIDDEN=1376、 RMSNorm なし (v11)
    Compact,
    /// POLICY_HIDDEN=1376、 post-SCReLU RMSNorm あり (v12)
    CompactRms,
    /// POLICY_HIDDEN=2048、 RMSNorm なし (v13)
    Wide,
}

impl PolicyValueVariant {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "tiny" => Some(Self::Tiny),
            "compact" => Some(Self::Compact),
            "compact_rms" | "compactrms" => Some(Self::CompactRms),
            "wide" => Some(Self::Wide),
            _ => None,
        }
    }

    pub fn config(self) -> PolicyValueConfig {
        match self {
            Self::Tiny => PolicyValueConfig {
                policy_hidden: 512,
                normalisation: NormalisationConfig::None,
            },
            Self::Compact => PolicyValueConfig {
                policy_hidden: 1376,
                normalisation: NormalisationConfig::None,
            },
            Self::CompactRms => PolicyValueConfig {
                policy_hidden: 1376,
                normalisation: NormalisationConfig::RmsNormPerPerspective,
            },
            Self::Wide => PolicyValueConfig {
                policy_hidden: 2048,
                normalisation: NormalisationConfig::None,
            },
        }
    }
}

/// 推論時 config
#[derive(Debug, Clone, Copy)]
pub struct PolicyValueConfig {
    /// SwiGLU policy hidden 次元
    pub policy_hidden: usize,
    pub normalisation: NormalisationConfig,
}

/// raw.bin の f32 数 (load 時の検証用)
pub fn expected_raw_bin_floats(cfg: &PolicyValueConfig) -> usize {
    let ph = cfg.policy_hidden;
    let value_part =
        // l0w + l0b
        L1_SIZE * INPUT_DIM + L1_SIZE
        // l1w + l1b
        + L2_SIZE * 2 * L1_SIZE + L2_SIZE
        // l2w + l2b
        + L3_SIZE * L2_SIZE + L3_SIZE
        // v_outw + v_outb
        + L3_SIZE + 1;
    let policy_part =
        // pol_w1w + pol_w1b
        ph * 2 * L1_SIZE + ph
        // pol_w3w + pol_w3b
        + ph * 2 * L1_SIZE + ph
        // pol_ow + pol_ob
        + POLICY_DIMS * ph + POLICY_DIMS;
    value_part + policy_part
}

/// raw.bin float 数から variant を自動推定 (USI option 省略時のフォールバック)
pub fn infer_variant_from_size(n_floats: usize) -> Option<PolicyValueVariant> {
    [
        PolicyValueVariant::Tiny,
        PolicyValueVariant::Compact,
        PolicyValueVariant::CompactRms,
        PolicyValueVariant::Wide,
    ]
    .into_iter()
    .find(|&v| expected_raw_bin_floats(&v.config()) == n_floats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_expected_floats() {
        // Compact / CompactRms は raw.bin 上同サイズ (RMSNorm は param なし版)
        assert_eq!(expected_raw_bin_floats(&PolicyValueVariant::Compact.config()), 164_551_500);
        assert_eq!(expected_raw_bin_floats(&PolicyValueVariant::CompactRms.config()), 164_551_500);
    }

    #[test]
    fn wide_expected_floats() {
        let n = expected_raw_bin_floats(&PolicyValueVariant::Wide.config());
        assert!(n > 164_551_500);
        assert!(n < 200_000_000);
    }

    #[test]
    fn variant_parse() {
        assert_eq!(PolicyValueVariant::parse("compact"), Some(PolicyValueVariant::Compact));
        assert_eq!(PolicyValueVariant::parse("compact_rms"), Some(PolicyValueVariant::CompactRms));
        assert_eq!(PolicyValueVariant::parse("Wide"), Some(PolicyValueVariant::Wide));
        assert_eq!(PolicyValueVariant::parse("unknown"), None);
    }
}
