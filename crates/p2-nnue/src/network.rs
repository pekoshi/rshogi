//! P2 NNUE ネットワーク構造体と順伝播
//!
//! 3 層ネットワーク: `2048 → 512 → 256 → 1`
//! Policy head: `256 → 2187` (dlshogi 形式、共有表現 L2 から分岐)

use std::io::{self, Read, Write};

use crate::align::Aligned;
use crate::features::P2NnueInput;
use crate::io::P2NnueHeader;
use crate::layers::{AffineTransform, ClippedReLU, InputAffineTransform};
use crate::types::{
    P2_INPUT_DIM, P2_L1_DIM, P2_L2_DIM, P2_OUTLOOK_DIM, P2_OUTPUT_DIM, P2_POLICY_DIM, P2NnueOutput,
    P2PolicyHint,
};

/// P2 NNUE ネットワーク
pub struct P2Network {
    pub(crate) l1: InputAffineTransform<P2_INPUT_DIM, P2_L1_DIM>,
    pub(crate) l2: AffineTransform<P2_L1_DIM, P2_L2_DIM>,
    pub(crate) l3: AffineTransform<P2_L2_DIM, P2_OUTPUT_DIM>,
    /// Policy head (256 → 2187)。L2 ReLU 出力から共有表現を分岐させて計算する。
    pub(crate) policy_head: AffineTransform<P2_L2_DIM, P2_POLICY_DIM>,
    /// Outlook (大局観) head (256 → 1)。 game_result 由来の長期勝率予測を出す。
    pub(crate) outlook_head: AffineTransform<P2_L2_DIM, P2_OUTLOOK_DIM>,
}

impl P2Network {
    /// バイナリストリームからネットワークを読み込み
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let header = P2NnueHeader::read(reader)?;
        header.validate()?;

        let l1 = InputAffineTransform::read(reader)?;
        let l2 = AffineTransform::read(reader)?;
        let l3 = AffineTransform::read(reader)?;
        let policy_head = AffineTransform::read(reader)?;
        let outlook_head = AffineTransform::read(reader)?;

        Ok(Self {
            l1,
            l2,
            l3,
            policy_head,
            outlook_head,
        })
    }

    /// 全ゼロ重みで初期化
    pub fn new_zeroed() -> Self {
        Self {
            l1: InputAffineTransform::new_zeroed(),
            l2: AffineTransform::new_zeroed(),
            l3: AffineTransform::new_zeroed(),
            policy_head: AffineTransform::new_zeroed(),
            outlook_head: AffineTransform::new_zeroed(),
        }
    }

    /// 順伝播（value + policy hint）。ヒープ割り当て禁止。全バッファ Aligned<> でスタック配置。
    pub fn forward(&self, input: &P2NnueInput) -> P2NnueOutput {
        // Layer 1: 2048 → 512
        let mut l1_out = Aligned([0i32; P2_L1_DIM]);
        self.l1.propagate(&input.piece_features, &mut l1_out.0);

        let mut l1_relu = Aligned([0u8; P2_L1_DIM]);
        ClippedReLU::propagate(&l1_out.0, &mut l1_relu.0);

        // Layer 2: 512 → 256
        let mut l2_out = Aligned([0i32; P2_L2_DIM]);
        self.l2.propagate(&l1_relu.0, &mut l2_out.0);

        let mut l2_relu = Aligned([0u8; P2_L2_DIM]);
        ClippedReLU::propagate(&l2_out.0, &mut l2_relu.0);

        // Policy hint: L2 ReLU 出力をコピー
        let policy_hint = P2PolicyHint {
            shared_repr: l2_relu.0,
        };

        // Layer 3: 256 → 1 (value head)
        let mut output = [0i32; P2_OUTPUT_DIM];
        self.l3.propagate(&l2_relu.0, &mut output);

        P2NnueOutput {
            value: output[0],
            policy_hint: Some(policy_hint),
            uncertainty: None,
        }
    }

    /// .p2nn バイナリ書き出し
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let header = P2NnueHeader::default_valid();
        writer.write_all(&header.to_bytes())?;
        self.l1.write(writer)?;
        self.l2.write(writer)?;
        self.l3.write(writer)?;
        self.policy_head.write(writer)?;
        self.outlook_head.write(writer)?;
        Ok(())
    }

    /// 全重みを f32 ベクトルに展開（論理行優先順序）
    ///
    /// 順序: L1w, L1b, L2w, L2b, L3w, L3b, PolicyW, PolicyB, OutlookW, OutlookB
    /// L1 weights は i16 → f32、L2/L3/Policy/Outlook weights は i8 → f32、biases は i32 → f32。
    pub fn weights_to_f32(&self) -> Vec<f32> {
        let l1_w_count = P2_L1_DIM * P2_INPUT_DIM;
        let l2_w_count = P2_L2_DIM * P2_L1_DIM;
        let l3_w_count = P2_OUTPUT_DIM * P2_L2_DIM;
        let pol_w_count = P2_POLICY_DIM * P2_L2_DIM;
        let out_w_count = P2_OUTLOOK_DIM * P2_L2_DIM;
        let total = l1_w_count
            + P2_L1_DIM
            + l2_w_count
            + P2_L2_DIM
            + l3_w_count
            + P2_OUTPUT_DIM
            + pol_w_count
            + P2_POLICY_DIM
            + out_w_count
            + P2_OUTLOOK_DIM;
        let mut flat = Vec::with_capacity(total);

        // L1 weights [L1_DIM][INPUT_DIM] (i16 → f32)
        let l1_padded = InputAffineTransform::<P2_INPUT_DIM, P2_L1_DIM>::PADDED_IN;
        for j in 0..P2_L1_DIM {
            for i in 0..P2_INPUT_DIM {
                flat.push(self.l1.weights[j * l1_padded + i] as f32);
            }
        }
        // L1 biases (i32 → f32)
        for j in 0..P2_L1_DIM {
            flat.push(self.l1.biases[j] as f32);
        }

        // L2 weights [L2_DIM][L1_DIM] — スクランブル逆変換して論理順
        Self::affine_weights_to_f32::<P2_L1_DIM, P2_L2_DIM>(&self.l2, &mut flat);
        // L2 biases
        for j in 0..P2_L2_DIM {
            flat.push(self.l2.biases[j] as f32);
        }

        // L3 weights [OUTPUT_DIM][L2_DIM]
        Self::affine_weights_to_f32::<P2_L2_DIM, P2_OUTPUT_DIM>(&self.l3, &mut flat);
        // L3 biases
        for j in 0..P2_OUTPUT_DIM {
            flat.push(self.l3.biases[j] as f32);
        }

        // Policy head weights [POLICY_DIM][L2_DIM]
        Self::affine_weights_to_f32::<P2_L2_DIM, P2_POLICY_DIM>(&self.policy_head, &mut flat);
        // Policy head biases
        for j in 0..P2_POLICY_DIM {
            flat.push(self.policy_head.biases[j] as f32);
        }

        // Outlook head weights [OUTLOOK_DIM][L2_DIM]
        Self::affine_weights_to_f32::<P2_L2_DIM, P2_OUTLOOK_DIM>(&self.outlook_head, &mut flat);
        // Outlook head biases
        for j in 0..P2_OUTLOOK_DIM {
            flat.push(self.outlook_head.biases[j] as f32);
        }

        flat
    }

    /// AffineTransform の重みを論理順で f32 に展開（スクランブル逆変換含む）
    fn affine_weights_to_f32<const IN: usize, const OUT: usize>(
        layer: &AffineTransform<IN, OUT>,
        flat: &mut Vec<f32>,
    ) {
        let padded_in = AffineTransform::<IN, OUT>::PADDED_IN;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if AffineTransform::<IN, OUT>::should_use_scrambled_weights() {
                for j in 0..OUT {
                    for i in 0..IN {
                        let linear_idx = j * padded_in + i;
                        let scrambled_idx =
                            AffineTransform::<IN, OUT>::get_weight_index_scrambled(linear_idx);
                        flat.push(layer.weights[scrambled_idx] as f32);
                    }
                }
                return;
            }
        }

        for j in 0..OUT {
            for i in 0..IN {
                flat.push(layer.weights[j * padded_in + i] as f32);
            }
        }
    }

    /// f32 ベクトルから量子化して重みを設定
    ///
    /// 順序は `weights_to_f32()` と同一: L1w, L1b, L2w, L2b, L3w, L3b, PolicyW, PolicyB, OutlookW, OutlookB
    /// L1 weights: `round(v).clamp(-32768, 32767)` → i16
    /// L2/L3/Policy/Outlook weights: `round(v).clamp(-128, 127)` → i8
    /// Biases: `round(v).clamp(i32::MIN, i32::MAX)` → i32
    pub fn set_weights_from_f32(&mut self, flat: &[f32]) {
        let l1_w_count = P2_L1_DIM * P2_INPUT_DIM;
        let l2_w_count = P2_L2_DIM * P2_L1_DIM;
        let l3_w_count = P2_OUTPUT_DIM * P2_L2_DIM;
        let pol_w_count = P2_POLICY_DIM * P2_L2_DIM;
        let out_w_count = P2_OUTLOOK_DIM * P2_L2_DIM;
        let expected = l1_w_count
            + P2_L1_DIM
            + l2_w_count
            + P2_L2_DIM
            + l3_w_count
            + P2_OUTPUT_DIM
            + pol_w_count
            + P2_POLICY_DIM
            + out_w_count
            + P2_OUTLOOK_DIM;
        assert_eq!(flat.len(), expected, "flat weight vector length mismatch");

        let mut pos = 0;

        // L1 weights (f32 → i16)
        let l1_padded = InputAffineTransform::<P2_INPUT_DIM, P2_L1_DIM>::PADDED_IN;
        for j in 0..P2_L1_DIM {
            for i in 0..P2_INPUT_DIM {
                self.l1.weights[j * l1_padded + i] =
                    flat[pos].round().clamp(-32768.0, 32767.0) as i16;
                pos += 1;
            }
            // パディング部分はゼロクリア
            for i in P2_INPUT_DIM..l1_padded {
                self.l1.weights[j * l1_padded + i] = 0;
            }
        }
        // L1 biases (f32 → i32)
        for j in 0..P2_L1_DIM {
            self.l1.biases[j] = flat[pos].round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            pos += 1;
        }

        // L2 weights (f32 → i8, スクランブル変換)
        Self::set_affine_weights_from_f32::<P2_L1_DIM, P2_L2_DIM>(
            &mut self.l2,
            &flat[pos..pos + l2_w_count],
        );
        pos += l2_w_count;
        // L2 biases
        for j in 0..P2_L2_DIM {
            self.l2.biases[j] = flat[pos].round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            pos += 1;
        }

        // L3 weights
        Self::set_affine_weights_from_f32::<P2_L2_DIM, P2_OUTPUT_DIM>(
            &mut self.l3,
            &flat[pos..pos + l3_w_count],
        );
        pos += l3_w_count;
        // L3 biases
        for j in 0..P2_OUTPUT_DIM {
            self.l3.biases[j] = flat[pos].round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            pos += 1;
        }

        // Policy weights
        Self::set_affine_weights_from_f32::<P2_L2_DIM, P2_POLICY_DIM>(
            &mut self.policy_head,
            &flat[pos..pos + pol_w_count],
        );
        pos += pol_w_count;
        // Policy biases
        for j in 0..P2_POLICY_DIM {
            self.policy_head.biases[j] =
                flat[pos].round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            pos += 1;
        }

        // Outlook weights
        Self::set_affine_weights_from_f32::<P2_L2_DIM, P2_OUTLOOK_DIM>(
            &mut self.outlook_head,
            &flat[pos..pos + out_w_count],
        );
        pos += out_w_count;
        // Outlook biases
        for j in 0..P2_OUTLOOK_DIM {
            self.outlook_head.biases[j] =
                flat[pos].round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            pos += 1;
        }
    }

    /// AffineTransform に f32 から量子化して重みを設定（スクランブル変換含む）
    fn set_affine_weights_from_f32<const IN: usize, const OUT: usize>(
        layer: &mut AffineTransform<IN, OUT>,
        flat: &[f32],
    ) {
        let padded_in = AffineTransform::<IN, OUT>::PADDED_IN;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if AffineTransform::<IN, OUT>::should_use_scrambled_weights() {
                for j in 0..OUT {
                    for i in 0..IN {
                        let linear_idx = j * padded_in + i;
                        let scrambled_idx =
                            AffineTransform::<IN, OUT>::get_weight_index_scrambled(linear_idx);
                        layer.weights[scrambled_idx] =
                            flat[j * IN + i].round().clamp(-128.0, 127.0) as i8;
                    }
                    // パディング部分はゼロクリア（スクランブル形式ではインデックスが分散するのでスキップ）
                }
                return;
            }
        }

        for j in 0..OUT {
            for i in 0..IN {
                layer.weights[j * padded_in + i] =
                    flat[j * IN + i].round().clamp(-128.0, 127.0) as i8;
            }
            for i in IN..padded_in {
                layer.weights[j * padded_in + i] = 0;
            }
        }
    }

    /// value のみを返す順伝播（policy hint 不要な場合の高速パス）
    pub fn forward_value_only(&self, input: &P2NnueInput) -> i32 {
        // Layer 1: 2048 → 512
        let mut l1_out = Aligned([0i32; P2_L1_DIM]);
        self.l1.propagate(&input.piece_features, &mut l1_out.0);

        let mut l1_relu = Aligned([0u8; P2_L1_DIM]);
        ClippedReLU::propagate(&l1_out.0, &mut l1_relu.0);

        // Layer 2: 512 → 256
        let mut l2_out = Aligned([0i32; P2_L2_DIM]);
        self.l2.propagate(&l1_relu.0, &mut l2_out.0);

        let mut l2_relu = Aligned([0u8; P2_L2_DIM]);
        ClippedReLU::propagate(&l2_out.0, &mut l2_relu.0);

        // Layer 3: 256 → 1 (value head)
        let mut output = [0i32; P2_OUTPUT_DIM];
        self.l3.propagate(&l2_relu.0, &mut output);
        output[0]
    }
}
