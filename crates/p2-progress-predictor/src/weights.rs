//! 重み serialization (trainer feature 専用)
//!
//! `ProgressWeights` ↔ `Vec<f32>` ↔ binary file の往復を担う。
//!
//! ファイル形式 (`progress.bin`):
//! - magic: ASCII "P2PG" (4 byte)
//! - version: u32 LE
//! - feature_dim: u32 LE (=32 を期待)
//! - hidden1: u32 LE (=32)
//! - hidden2: u32 LE (=16)
//! - output:  u32 LE (=1)
//! - weights: f32 LE で w1, b1, w2, b2, w3, b3 を順に flatten

use crate::features::FEATURE_DIM;
use crate::model::{ProgressWeights, HIDDEN1, HIDDEN2, OUTPUT};
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"P2PG";
const VERSION: u32 = 1;

/// Total f32 count (w1+b1+w2+b2+w3+b3)
pub const fn weight_count() -> usize {
    HIDDEN1 * FEATURE_DIM + HIDDEN1 + HIDDEN2 * HIDDEN1 + HIDDEN2 + OUTPUT * HIDDEN2 + OUTPUT
}

/// `ProgressWeights` を flat Vec に変換する。
pub fn weights_to_vec(w: &ProgressWeights) -> Vec<f32> {
    let mut out = Vec::with_capacity(weight_count());
    for row in &w.w1 {
        out.extend_from_slice(row);
    }
    out.extend_from_slice(&w.b1);
    for row in &w.w2 {
        out.extend_from_slice(row);
    }
    out.extend_from_slice(&w.b2);
    for row in &w.w3 {
        out.extend_from_slice(row);
    }
    out.extend_from_slice(&w.b3);
    out
}

/// flat Vec から `ProgressWeights` を組み立てる。 length check 付き。
pub fn weights_from_slice(data: &[f32]) -> Result<ProgressWeights> {
    let expected = weight_count();
    if data.len() != expected {
        bail!("weight length mismatch: got {}, expected {}", data.len(), expected);
    }
    let mut w = ProgressWeights::zeros();
    let mut off = 0;
    for j in 0..HIDDEN1 {
        w.w1[j].copy_from_slice(&data[off..off + FEATURE_DIM]);
        off += FEATURE_DIM;
    }
    w.b1.copy_from_slice(&data[off..off + HIDDEN1]);
    off += HIDDEN1;
    for j in 0..HIDDEN2 {
        w.w2[j].copy_from_slice(&data[off..off + HIDDEN1]);
        off += HIDDEN1;
    }
    w.b2.copy_from_slice(&data[off..off + HIDDEN2]);
    off += HIDDEN2;
    for j in 0..OUTPUT {
        w.w3[j].copy_from_slice(&data[off..off + HIDDEN2]);
        off += HIDDEN2;
    }
    w.b3.copy_from_slice(&data[off..off + OUTPUT]);
    off += OUTPUT;
    debug_assert_eq!(off, expected);
    Ok(w)
}

/// 重みを binary file に保存。
pub fn save(path: &Path, w: &ProgressWeights) -> Result<()> {
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&(FEATURE_DIM as u32).to_le_bytes())?;
    f.write_all(&(HIDDEN1 as u32).to_le_bytes())?;
    f.write_all(&(HIDDEN2 as u32).to_le_bytes())?;
    f.write_all(&(OUTPUT as u32).to_le_bytes())?;
    let vec = weights_to_vec(w);
    for v in &vec {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// 重みを binary file から読み込む。
pub fn load(path: &Path) -> Result<ProgressWeights> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("invalid magic in {}", path.display());
    }
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    let ver = u32::from_le_bytes(buf);
    if ver != VERSION {
        bail!("unsupported version {}", ver);
    }
    f.read_exact(&mut buf)?;
    let fdim = u32::from_le_bytes(buf) as usize;
    f.read_exact(&mut buf)?;
    let h1 = u32::from_le_bytes(buf) as usize;
    f.read_exact(&mut buf)?;
    let h2 = u32::from_le_bytes(buf) as usize;
    f.read_exact(&mut buf)?;
    let o = u32::from_le_bytes(buf) as usize;
    if fdim != FEATURE_DIM || h1 != HIDDEN1 || h2 != HIDDEN2 || o != OUTPUT {
        bail!(
            "shape mismatch: file=({},{},{},{}) expected=({},{},{},{})",
            fdim,
            h1,
            h2,
            o,
            FEATURE_DIM,
            HIDDEN1,
            HIDDEN2,
            OUTPUT,
        );
    }
    let mut vec = vec![0.0f32; weight_count()];
    for v in vec.iter_mut() {
        f.read_exact(&mut buf)?;
        *v = f32::from_le_bytes(buf);
    }
    weights_from_slice(&vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_vec_default() {
        let w = ProgressWeights::default_baseline();
        let v = weights_to_vec(&w);
        assert_eq!(v.len(), weight_count());
        let w2 = weights_from_slice(&v).unwrap();
        assert_eq!(w.b3[0], w2.b3[0]);
    }

    #[test]
    fn weight_count_matches_expected() {
        // 32*32 + 32 + 16*32 + 16 + 1*16 + 1 = 1024 + 32 + 512 + 16 + 16 + 1 = 1601
        assert_eq!(weight_count(), 1601);
    }
}
