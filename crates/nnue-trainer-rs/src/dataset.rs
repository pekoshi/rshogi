//! features.bin reader (memmap 風 in-memory 読込)
//!
//! Record (164 byte/record、 既定):
//!   stm_idx:  [i16; 40]    -1 = pad
//!   nstm_idx: [i16; 40]
//!   cp:       i16
//!   bucket:   u8
//!   result:   u8
//!
//! Record (168 byte/record、 `with_policy=true` で出力された features_p.bin):
//!   既定 164 byte + policy_idx u16 LE (offset 164-165) + reserved u16 LE (offset 166-167)
//!   policy_idx は dlshogi 規格 0..2186、 0xFFFF = no policy (sentinel)。

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// 既定 record サイズ (policy なし)
pub const RECORD_SIZE: usize = 164;
/// policy 拡張版 record サイズ (164 base + 2B policy_idx + 2B reserved)
pub const RECORD_SIZE_WITH_POLICY: usize = 168;
pub const MAX_ACTIVE: usize = 40;
/// policy_idx の sentinel (no policy)
pub const POLICY_SENTINEL: u16 = 0xFFFF;

pub struct FeaturesDataset {
    /// 全件 RAM ロード (in_memory mode、 random access 高速化)
    raw: Vec<u8>,
    pub total: usize,
    pub record_size: usize,
    pub with_policy: bool,
}

impl FeaturesDataset {
    /// 既定 (164 byte/record、 policy なし) で読込。
    pub fn load_in_memory(path: &Path, max_records: Option<usize>) -> Result<Self> {
        Self::load_in_memory_with_policy(path, max_records, false)
    }

    /// `with_policy=true` で 168 byte/record として読込。 features_p.bin 用。
    pub fn load_in_memory_with_policy(
        path: &Path,
        max_records: Option<usize>,
        with_policy: bool,
    ) -> Result<Self> {
        let record_size = if with_policy {
            RECORD_SIZE_WITH_POLICY
        } else {
            RECORD_SIZE
        };
        let mut f = File::open(path)?;
        let size = f.metadata()?.len() as usize;
        let mut total = size / record_size;
        if let Some(max) = max_records {
            total = total.min(max);
        }
        let nbytes = total * record_size;
        let mut raw = vec![0u8; nbytes];
        f.read_exact(&mut raw)?;
        eprintln!(
            "[dataset] loaded {} records ({} MB, record_size={}, with_policy={})",
            total,
            nbytes / 1024 / 1024,
            record_size,
            with_policy,
        );
        Ok(Self {
            raw,
            total,
            record_size,
            with_policy,
        })
    }

    /// indices で指定された record の batch を構築 → GPU tensor へ転送
    pub fn make_batch(&self, indices: &[usize], device: &Device) -> Result<Batch> {
        let b = indices.len();
        let mut stm = vec![0i64; b * MAX_ACTIVE];
        let mut nstm = vec![0i64; b * MAX_ACTIVE];
        let mut cp = vec![0f32; b];
        let mut bucket = vec![0i64; b];
        let mut result = vec![0f32; b];
        // policy_idx: -1 で sentinel、 0..2186 が valid。 loss 計算側で -1 を mask する想定。
        let mut policy = if self.with_policy {
            vec![-1i64; b]
        } else {
            Vec::new()
        };

        for (i, &idx) in indices.iter().enumerate() {
            let off = idx * self.record_size;
            // stm_idx: 40 × i16 (offset 0..80)
            for k in 0..MAX_ACTIVE {
                let lo = self.raw[off + k * 2] as u16;
                let hi = self.raw[off + k * 2 + 1] as u16;
                stm[i * MAX_ACTIVE + k] = ((hi << 8) | lo) as i16 as i64;
            }
            // nstm_idx: 40 × i16 (offset 80..160)
            for k in 0..MAX_ACTIVE {
                let lo = self.raw[off + 80 + k * 2] as u16;
                let hi = self.raw[off + 80 + k * 2 + 1] as u16;
                nstm[i * MAX_ACTIVE + k] = ((hi << 8) | lo) as i16 as i64;
            }
            // cp: i16 (offset 160..162)
            let cp_lo = self.raw[off + 160] as u16;
            let cp_hi = self.raw[off + 161] as u16;
            cp[i] = (((cp_hi << 8) | cp_lo) as i16) as f32;
            // bucket: u8 (offset 162)
            bucket[i] = self.raw[off + 162] as i64;
            // result: u8 (offset 163)
            result[i] = self.raw[off + 163] as f32 / 2.0;

            if self.with_policy {
                // policy_idx: u16 LE (offset 164-165)。 0xFFFF sentinel は -1 に正規化。
                let p_lo = self.raw[off + 164] as u16;
                let p_hi = self.raw[off + 165] as u16;
                let p_u16 = (p_hi << 8) | p_lo;
                policy[i] = if p_u16 == POLICY_SENTINEL {
                    -1
                } else {
                    p_u16 as i64
                };
            }
        }

        let policy_tensor = if self.with_policy {
            Some(Tensor::from_vec(policy, b, device)?)
        } else {
            None
        };

        Ok(Batch {
            stm: Tensor::from_vec(stm, (b, MAX_ACTIVE), device)?,
            nstm: Tensor::from_vec(nstm, (b, MAX_ACTIVE), device)?,
            cp: Tensor::from_vec(cp, b, device)?,
            bucket: Tensor::from_vec(bucket, b, device)?,
            result: Tensor::from_vec(result, b, device)?,
            policy_idx: policy_tensor,
        })
    }
}

pub struct Batch {
    pub stm: Tensor,
    pub nstm: Tensor,
    pub cp: Tensor,
    pub bucket: Tensor,
    pub result: Tensor,
    /// `with_policy=true` でロードされたときのみ Some。
    /// 形状 `[B]`、 値域 -1 (sentinel) または 0..2186 (dlshogi policy idx)。
    pub policy_idx: Option<Tensor>,
}
