//! XL v3 ネットワーク構造体 + `.p2nn v3` ローダー (f32 baseline).
//!
//! 既存 V2 の `network.rs` は触らず、 XL v3 専用に並列実装。
//! ロードまでが本ファイルの責務。 forward 推論は `evaluator/cpu_v3.rs` で実装する。

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use crate::types_v3::{ModelConfigV3, QuantFmt, V3_INPUT_CHANNELS, V3_MAGIC, V3_VERSION};

/// `.p2nn v3` のヘッダー (4 + 4 + 36 + padding = 64 bytes)
const HEADER_BYTES: usize = 64;

/// f32 重み (Vec で動的サイズ、 V3 は層ごとの形状が config から決まる)
pub struct P2NetworkV3 {
    pub config: ModelConfigV3,
    /// Conv 層の重み (n_conv_layers 個)、 各 (Cout, Cin, kH, kW) を flat 化した row-major
    pub conv_weights: Vec<Vec<f32>>,
    pub conv_biases: Vec<Vec<f32>>,
    /// Conv 後の GroupNorm (1 group = LayerNorm 相当)、 weight/bias は per-channel
    pub conv_ln_weights: Vec<Vec<f32>>,
    pub conv_ln_biases: Vec<Vec<f32>>,
    /// fc1: (fc1, flat_dim) row-major
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    /// fc2 stacks: n_stacks 個、 各 (fc2, fc1)
    pub fc2_stack_weights: Vec<Vec<f32>>,
    pub fc2_stack_biases: Vec<Vec<f32>>,
    /// fc3 stacks: n_stacks 個、 各 (fc3, fc2)
    pub fc3_stack_weights: Vec<Vec<f32>>,
    pub fc3_stack_biases: Vec<Vec<f32>>,
    /// head 共通 LayerNorm (fc3,)
    pub head_ln_weight: Vec<f32>,
    pub head_ln_bias: Vec<f32>,
    /// value head: (1, fc3)
    pub value_head_weight: Vec<f32>,
    pub value_head_bias: Vec<f32>,
    /// outlook head: (1, fc3)
    pub outlook_head_weight: Vec<f32>,
    pub outlook_head_bias: Vec<f32>,
    /// policy 専用 3 段 MLP
    pub policy_fc1_weight: Vec<f32>, // (mlp, fc3)
    pub policy_fc1_bias: Vec<f32>,
    pub policy_ln1_weight: Vec<f32>,
    pub policy_ln1_bias: Vec<f32>,
    pub policy_fc2_weight: Vec<f32>, // (mlp, mlp)
    pub policy_fc2_bias: Vec<f32>,
    pub policy_ln2_weight: Vec<f32>,
    pub policy_ln2_bias: Vec<f32>,
    pub policy_fc3_weight: Vec<f32>, // (policy_dim, mlp)
    pub policy_fc3_bias: Vec<f32>,
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        let mut a = [0u8; 4];
        a.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(a));
    }
    Ok(out)
}

impl P2NetworkV3 {
    /// `.p2nn v3` ファイルを読み込む
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);
        Self::load(&mut r)
    }

    pub fn load<R: Read>(r: &mut R) -> io::Result<Self> {
        // magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != V3_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid magic {:?} (expected {:?})", magic, V3_MAGIC),
            ));
        }
        let version = read_u32(r)?;
        if version != V3_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version {} (expected {})", version, V3_VERSION),
            ));
        }
        // header
        let n_conv_layers = read_u8(r)?;
        let conv_ch = read_u16(r)?;
        let _input_ch = read_u16(r)?;
        let _input_dim = read_u32(r)?;
        let _effective_dim = read_u32(r)?;
        let fc1 = read_u32(r)?;
        let fc2 = read_u32(r)?;
        let fc3 = read_u32(r)?;
        let n_stacks = read_u8(r)?;
        let policy_mlp = read_u32(r)?;
        let policy_dim = read_u32(r)?;
        let quant_fmt_u8 = read_u8(r)?;
        let quant_fmt = QuantFmt::from_u8(quant_fmt_u8).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown quant_fmt {}", quant_fmt_u8),
            )
        })?;

        // padding to 64 bytes
        let header_used = 4 + 4 + 1 + 2 + 2 + 4 + 4 + 4 + 4 + 4 + 1 + 4 + 4 + 1;
        let pad = HEADER_BYTES - header_used;
        let mut padding = vec![0u8; pad];
        r.read_exact(&mut padding)?;

        let config = ModelConfigV3 {
            n_conv_layers,
            conv_ch,
            fc1,
            fc2,
            fc3,
            n_stacks,
            policy_mlp,
            policy_dim,
            quant_fmt,
        };

        if quant_fmt != QuantFmt::F32 {
            // 量子化版は P4 で対応予定
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("quant_fmt {:?} not yet supported (only F32)", quant_fmt),
            ));
        }

        // Conv 部
        let mut conv_weights = Vec::with_capacity(n_conv_layers as usize);
        let mut conv_biases = Vec::with_capacity(n_conv_layers as usize);
        for i in 0..n_conv_layers {
            let cin = if i == 0 {
                V3_INPUT_CHANNELS
            } else {
                conv_ch as usize
            };
            let kernel = if i == 0 { 5 * 5 } else { 3 * 3 };
            let n_w = conv_ch as usize * cin * kernel;
            conv_weights.push(read_f32_vec(r, n_w)?);
            conv_biases.push(read_f32_vec(r, conv_ch as usize)?);
        }
        let mut conv_ln_weights = Vec::with_capacity(n_conv_layers as usize);
        let mut conv_ln_biases = Vec::with_capacity(n_conv_layers as usize);
        for _ in 0..n_conv_layers {
            conv_ln_weights.push(read_f32_vec(r, conv_ch as usize)?);
            conv_ln_biases.push(read_f32_vec(r, conv_ch as usize)?);
        }

        // Trunk
        let flat_dim = config.flat_dim();
        let fc1_weight = read_f32_vec(r, fc1 as usize * flat_dim)?;
        let fc1_bias = read_f32_vec(r, fc1 as usize)?;

        let mut fc2_stack_weights = Vec::with_capacity(n_stacks as usize);
        let mut fc2_stack_biases = Vec::with_capacity(n_stacks as usize);
        for _ in 0..n_stacks {
            fc2_stack_weights.push(read_f32_vec(r, fc2 as usize * fc1 as usize)?);
            fc2_stack_biases.push(read_f32_vec(r, fc2 as usize)?);
        }
        let mut fc3_stack_weights = Vec::with_capacity(n_stacks as usize);
        let mut fc3_stack_biases = Vec::with_capacity(n_stacks as usize);
        for _ in 0..n_stacks {
            fc3_stack_weights.push(read_f32_vec(r, fc3 as usize * fc2 as usize)?);
            fc3_stack_biases.push(read_f32_vec(r, fc3 as usize)?);
        }

        let head_ln_weight = read_f32_vec(r, fc3 as usize)?;
        let head_ln_bias = read_f32_vec(r, fc3 as usize)?;
        let value_head_weight = read_f32_vec(r, fc3 as usize)?;
        let value_head_bias = read_f32_vec(r, 1)?;
        let outlook_head_weight = read_f32_vec(r, fc3 as usize)?;
        let outlook_head_bias = read_f32_vec(r, 1)?;

        let policy_fc1_weight = read_f32_vec(r, policy_mlp as usize * fc3 as usize)?;
        let policy_fc1_bias = read_f32_vec(r, policy_mlp as usize)?;
        let policy_ln1_weight = read_f32_vec(r, policy_mlp as usize)?;
        let policy_ln1_bias = read_f32_vec(r, policy_mlp as usize)?;
        let policy_fc2_weight = read_f32_vec(r, policy_mlp as usize * policy_mlp as usize)?;
        let policy_fc2_bias = read_f32_vec(r, policy_mlp as usize)?;
        let policy_ln2_weight = read_f32_vec(r, policy_mlp as usize)?;
        let policy_ln2_bias = read_f32_vec(r, policy_mlp as usize)?;
        let policy_fc3_weight = read_f32_vec(r, policy_dim as usize * policy_mlp as usize)?;
        let policy_fc3_bias = read_f32_vec(r, policy_dim as usize)?;

        Ok(P2NetworkV3 {
            config,
            conv_weights,
            conv_biases,
            conv_ln_weights,
            conv_ln_biases,
            fc1_weight,
            fc1_bias,
            fc2_stack_weights,
            fc2_stack_biases,
            fc3_stack_weights,
            fc3_stack_biases,
            head_ln_weight,
            head_ln_bias,
            value_head_weight,
            value_head_bias,
            outlook_head_weight,
            outlook_head_bias,
            policy_fc1_weight,
            policy_fc1_bias,
            policy_ln1_weight,
            policy_ln1_bias,
            policy_fc2_weight,
            policy_fc2_bias,
            policy_ln2_weight,
            policy_ln2_bias,
            policy_fc3_weight,
            policy_fc3_bias,
        })
    }
}
