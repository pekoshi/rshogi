//! V2 重み serialization (trainer feature 専用)
//!
//! `WeightsV2` ↔ `Vec<f32>` ↔ binary file の往復を担う。
//!
//! ファイル形式 (`progress_v2.bin`):
//! - magic: ASCII "P2PV2" (5 byte) → V1 (`P2PG`) と区別
//! - version: u32 LE (=1)
//! - feature_dim: u32 LE (=48)
//! - hidden1: u32 LE (=64)
//! - hidden2: u32 LE (=32)
//! - hidden3: u32 LE (=16)
//! - output:  u32 LE (=1)
//! - weights: f32 LE で w1, b1, w2, b2, w3, b3, w4, b4 を順に flatten
//!
//! 全 entry little-endian。

use crate::features_v2::FEATURE_DIM_V2;
use crate::model_v2::{WeightsV2, HIDDEN1_V2, HIDDEN2_V2, HIDDEN3_V2, OUTPUT_V2};
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 5] = b"P2PV2";
const VERSION: u32 = 1;

/// Total f32 count (w1+b1+w2+b2+w3+b3+w4+b4)
pub const fn weight_count_v2() -> usize {
    HIDDEN1_V2 * FEATURE_DIM_V2
        + HIDDEN1_V2
        + HIDDEN2_V2 * HIDDEN1_V2
        + HIDDEN2_V2
        + HIDDEN3_V2 * HIDDEN2_V2
        + HIDDEN3_V2
        + OUTPUT_V2 * HIDDEN3_V2
        + OUTPUT_V2
}

/// `WeightsV2` を flat Vec に変換する。
pub fn weights_to_vec_v2(w: &WeightsV2) -> Vec<f32> {
    let mut out = Vec::with_capacity(weight_count_v2());
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
    for row in &w.w4 {
        out.extend_from_slice(row);
    }
    out.extend_from_slice(&w.b4);
    out
}

/// flat Vec から `WeightsV2` を組み立てる。 length check 付き。
pub fn weights_from_slice_v2(data: &[f32]) -> Result<Box<WeightsV2>> {
    let expected = weight_count_v2();
    if data.len() != expected {
        bail!("weight length mismatch: got {}, expected {}", data.len(), expected);
    }
    let mut w = WeightsV2::zeros();
    let mut off = 0;
    for j in 0..HIDDEN1_V2 {
        w.w1[j].copy_from_slice(&data[off..off + FEATURE_DIM_V2]);
        off += FEATURE_DIM_V2;
    }
    w.b1.copy_from_slice(&data[off..off + HIDDEN1_V2]);
    off += HIDDEN1_V2;
    for j in 0..HIDDEN2_V2 {
        w.w2[j].copy_from_slice(&data[off..off + HIDDEN1_V2]);
        off += HIDDEN1_V2;
    }
    w.b2.copy_from_slice(&data[off..off + HIDDEN2_V2]);
    off += HIDDEN2_V2;
    for j in 0..HIDDEN3_V2 {
        w.w3[j].copy_from_slice(&data[off..off + HIDDEN2_V2]);
        off += HIDDEN2_V2;
    }
    w.b3.copy_from_slice(&data[off..off + HIDDEN3_V2]);
    off += HIDDEN3_V2;
    for j in 0..OUTPUT_V2 {
        w.w4[j].copy_from_slice(&data[off..off + HIDDEN3_V2]);
        off += HIDDEN3_V2;
    }
    w.b4.copy_from_slice(&data[off..off + OUTPUT_V2]);
    off += OUTPUT_V2;
    debug_assert_eq!(off, expected);
    Ok(w)
}

/// 重みを binary file に保存。
pub fn save_v2(path: &Path, w: &WeightsV2) -> Result<()> {
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&(FEATURE_DIM_V2 as u32).to_le_bytes())?;
    f.write_all(&(HIDDEN1_V2 as u32).to_le_bytes())?;
    f.write_all(&(HIDDEN2_V2 as u32).to_le_bytes())?;
    f.write_all(&(HIDDEN3_V2 as u32).to_le_bytes())?;
    f.write_all(&(OUTPUT_V2 as u32).to_le_bytes())?;
    let vec = weights_to_vec_v2(w);
    for v in &vec {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// 重みを binary file から読み込む。
pub fn load_v2(path: &Path) -> Result<Box<WeightsV2>> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut magic = [0u8; 5];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("invalid V2 magic in {}: {:?}", path.display(), magic);
    }
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    let ver = u32::from_le_bytes(buf);
    if ver != VERSION {
        bail!("unsupported V2 version {}", ver);
    }
    f.read_exact(&mut buf)?;
    let fdim = u32::from_le_bytes(buf) as usize;
    f.read_exact(&mut buf)?;
    let h1 = u32::from_le_bytes(buf) as usize;
    f.read_exact(&mut buf)?;
    let h2 = u32::from_le_bytes(buf) as usize;
    f.read_exact(&mut buf)?;
    let h3 = u32::from_le_bytes(buf) as usize;
    f.read_exact(&mut buf)?;
    let o = u32::from_le_bytes(buf) as usize;
    if fdim != FEATURE_DIM_V2
        || h1 != HIDDEN1_V2
        || h2 != HIDDEN2_V2
        || h3 != HIDDEN3_V2
        || o != OUTPUT_V2
    {
        bail!(
            "V2 shape mismatch: file=({},{},{},{},{}) expected=({},{},{},{},{})",
            fdim,
            h1,
            h2,
            h3,
            o,
            FEATURE_DIM_V2,
            HIDDEN1_V2,
            HIDDEN2_V2,
            HIDDEN3_V2,
            OUTPUT_V2,
        );
    }
    let mut vec = vec![0.0f32; weight_count_v2()];
    for v in vec.iter_mut() {
        f.read_exact(&mut buf)?;
        *v = f32::from_le_bytes(buf);
    }
    weights_from_slice_v2(&vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_v2::WeightsV2;

    #[test]
    fn weight_count_matches_5761() {
        // 48*64 + 64 + 64*32 + 32 + 32*16 + 16 + 16*1 + 1 = 5761
        assert_eq!(weight_count_v2(), 5761);
    }

    #[test]
    fn roundtrip_vec_zeros() {
        let w = WeightsV2::zeros();
        let v = weights_to_vec_v2(&w);
        assert_eq!(v.len(), weight_count_v2());
        let w2 = weights_from_slice_v2(&v).unwrap();
        // sample 同値 check
        assert_eq!(w.b4[0], w2.b4[0]);
        assert_eq!(w.w1[0][0], w2.w1[0][0]);
    }

    #[test]
    fn roundtrip_file_random() {
        use crate::model_v2::WeightsV2;
        // 適当な non-zero 重みを作って save/load 一致確認
        let mut w = WeightsV2::zeros();
        let mut s: u32 = 0xABCDEF01;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (((s >> 8) & 0xffff) as f32 / 65536.0) - 0.5
        };
        for row in w.w1.iter_mut() {
            for v in row.iter_mut() {
                *v = next();
            }
        }
        for b in w.b1.iter_mut() {
            *b = next();
        }
        w.b4[0] = next();
        w.w4[0][3] = 0.1234;

        let tmp = std::env::temp_dir().join("p2pv2_roundtrip_test.bin");
        save_v2(&tmp, &w).unwrap();

        // ファイルサイズ確認: header (5 + 4*6 = 29 byte) + 5761*4 = 29 + 23044 = 23073 byte
        let meta = std::fs::metadata(&tmp).unwrap();
        assert_eq!(meta.len(), 29 + (weight_count_v2() as u64) * 4);

        let w2 = load_v2(&tmp).unwrap();
        for i in 0..HIDDEN1_V2 {
            for j in 0..FEATURE_DIM_V2 {
                assert!((w.w1[i][j] - w2.w1[i][j]).abs() < 1e-6);
            }
        }
        assert!((w.b4[0] - w2.b4[0]).abs() < 1e-6);
        assert!((w.w4[0][3] - 0.1234).abs() < 1e-6);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_v1_magic_is_rejected() {
        // V1 の P2PG magic を持つ file は reject される
        let tmp = std::env::temp_dir().join("p2pv2_v1magic_test.bin");
        {
            let mut f = File::create(&tmp).unwrap();
            // V1 magic 4 byte + padding (V2 は 5 byte magic を読むので "P2PG" + 1 byte 別文字)
            f.write_all(b"P2PG").unwrap();
            f.write_all(b"X").unwrap();
            // 残り header 適当
            f.write_all(&[0u8; 256]).unwrap();
        }
        let r = load_v2(&tmp);
        assert!(r.is_err(), "V1 magic was accepted by V2 loader");
        let _ = std::fs::remove_file(&tmp);
    }
}
