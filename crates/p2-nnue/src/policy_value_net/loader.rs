//! Policy-Value NNUE raw.bin (f32 column-major) loader

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use super::config::{
    INPUT_DIM, L1_SIZE, L2_SIZE, L3_SIZE, POLICY_DIMS, PolicyValueConfig, expected_raw_bin_floats,
};

/// 全 weight (f32、 row-major (out, in) で保持)
pub struct PolicyValueWeights {
    pub cfg: PolicyValueConfig,
    /// FT: (L1, INPUT_DIM)
    pub l0w: Vec<f32>,
    pub l0b: Vec<f32>,
    /// L1: (L2, 2*L1)
    pub l1w: Vec<f32>,
    pub l1b: Vec<f32>,
    /// L2: (L3, L2)
    pub l2w: Vec<f32>,
    pub l2b: Vec<f32>,
    /// value out: (1, L3)
    pub v_outw: Vec<f32>,
    pub v_outb: Vec<f32>,
    /// SwiGLU policy W1: (PH, 2*L1)
    pub pol_w1w: Vec<f32>,
    pub pol_w1b: Vec<f32>,
    /// SwiGLU policy W3: (PH, 2*L1)
    pub pol_w3w: Vec<f32>,
    pub pol_w3b: Vec<f32>,
    /// SwiGLU output: (POLICY_DIMS, PH)
    pub pol_ow: Vec<f32>,
    pub pol_ob: Vec<f32>,
}

pub fn load_raw_bin<P: AsRef<Path>>(
    path: P,
    cfg: PolicyValueConfig,
) -> io::Result<PolicyValueWeights> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "raw.bin size not multiple of 4"));
    }
    let n_floats = bytes.len() / 4;
    let expected = expected_raw_bin_floats(&cfg);
    if n_floats != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("raw.bin float count mismatch: got {}, expected {}", n_floats, expected),
        ));
    }

    let mut all = Vec::with_capacity(n_floats);
    for chunk in bytes.chunks_exact(4) {
        let mut a = [0u8; 4];
        a.copy_from_slice(chunk);
        all.push(f32::from_le_bytes(a));
    }

    let ph = cfg.policy_hidden;
    let mut off = 0usize;
    let take_col_major = |out: usize, inp: usize, off: &mut usize| -> Vec<f32> {
        // column-major (in, out) → row-major (out, in) へ転置
        let total = out * inp;
        let raw = &all[*off..*off + total];
        *off += total;
        let mut transposed = vec![0.0f32; total];
        for i in 0..inp {
            for o in 0..out {
                transposed[o * inp + i] = raw[i * out + o];
            }
        }
        transposed
    };
    let take_bias = |n: usize, off: &mut usize| -> Vec<f32> {
        let v = all[*off..*off + n].to_vec();
        *off += n;
        v
    };

    let l0w = take_col_major(L1_SIZE, INPUT_DIM, &mut off);
    let l0b = take_bias(L1_SIZE, &mut off);
    let l1w = take_col_major(L2_SIZE, 2 * L1_SIZE, &mut off);
    let l1b = take_bias(L2_SIZE, &mut off);
    let l2w = take_col_major(L3_SIZE, L2_SIZE, &mut off);
    let l2b = take_bias(L3_SIZE, &mut off);
    let v_outw = take_col_major(1, L3_SIZE, &mut off);
    let v_outb = take_bias(1, &mut off);
    let pol_w1w = take_col_major(ph, 2 * L1_SIZE, &mut off);
    let pol_w1b = take_bias(ph, &mut off);
    let pol_w3w = take_col_major(ph, 2 * L1_SIZE, &mut off);
    let pol_w3b = take_bias(ph, &mut off);
    let pol_ow = take_col_major(POLICY_DIMS, ph, &mut off);
    let pol_ob = take_bias(POLICY_DIMS, &mut off);

    assert_eq!(off, n_floats, "weight load offset mismatch");

    Ok(PolicyValueWeights {
        cfg,
        l0w,
        l0b,
        l1w,
        l1b,
        l2w,
        l2b,
        v_outw,
        v_outb,
        pol_w1w,
        pol_w1b,
        pol_w3w,
        pol_w3b,
        pol_ow,
        pol_ob,
    })
}
