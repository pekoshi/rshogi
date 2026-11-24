//! チェックポイント保存・読み込み
//!
//! 重み + AdamW 状態をバイナリで永続化する。
//!
//! ```text
//! [magic 4B][version 4B]
//! [epoch 4B][step 8B][train_loss 4B][val_loss 4B][lr 4B]
//! [num_params 4B]
//! [f32 × N weights][f32 × N m][f32 × N v][t 8B]
//! ```

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::forward_f32::TOTAL_PARAMS;
use super::optimizer::{AdamWConfig, AdamWState};

/// チェックポイントマジック "TRCK"
const CHECKPOINT_MAGIC: u32 = 0x4B435254;
/// チェックポイントバージョン
const CHECKPOINT_VERSION: u32 = 1;

/// チェックポイントメタデータ
pub struct CheckpointMeta {
    pub epoch: u32,
    pub step: u64,
    pub train_loss: f32,
    pub val_loss: f32,
    pub lr: f32,
}

/// チェックポイントを保存
pub fn save_checkpoint(
    path: &Path,
    meta: &CheckpointMeta,
    weights: &[f32],
    optimizer: &AdamWState,
) -> io::Result<()> {
    assert_eq!(weights.len(), TOTAL_PARAMS);

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // ヘッダ
    w.write_all(&CHECKPOINT_MAGIC.to_le_bytes())?;
    w.write_all(&CHECKPOINT_VERSION.to_le_bytes())?;

    // メタデータ
    w.write_all(&meta.epoch.to_le_bytes())?;
    w.write_all(&meta.step.to_le_bytes())?;
    w.write_all(&meta.train_loss.to_le_bytes())?;
    w.write_all(&meta.val_loss.to_le_bytes())?;
    w.write_all(&meta.lr.to_le_bytes())?;

    // パラメータ数
    w.write_all(&(TOTAL_PARAMS as u32).to_le_bytes())?;

    // 重み
    for &val in weights {
        w.write_all(&val.to_le_bytes())?;
    }

    // optimizer m
    for &val in &optimizer.m {
        w.write_all(&val.to_le_bytes())?;
    }

    // optimizer v
    for &val in &optimizer.v {
        w.write_all(&val.to_le_bytes())?;
    }

    // optimizer t
    w.write_all(&optimizer.t.to_le_bytes())?;

    w.flush()?;
    Ok(())
}

/// チェックポイントを読み込み
pub fn load_checkpoint(
    path: &Path,
    adam_config: AdamWConfig,
) -> io::Result<(CheckpointMeta, Vec<f32>, AdamWState)> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);

    let read_u32 = |r: &mut BufReader<File>| -> io::Result<u32> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    };
    let read_u64 = |r: &mut BufReader<File>| -> io::Result<u64> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    };
    let read_f32 = |r: &mut BufReader<File>| -> io::Result<f32> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    };

    // ヘッダ検証
    let magic = read_u32(&mut r)?;
    if magic != CHECKPOINT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("不正なチェックポイントマジック: 0x{magic:08X}"),
        ));
    }
    let version = read_u32(&mut r)?;
    if version != CHECKPOINT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("未対応バージョン: {version}"),
        ));
    }

    // メタデータ
    let epoch = read_u32(&mut r)?;
    let step = read_u64(&mut r)?;
    let train_loss = read_f32(&mut r)?;
    let val_loss = read_f32(&mut r)?;
    let lr = read_f32(&mut r)?;

    let num_params = read_u32(&mut r)? as usize;
    if num_params != TOTAL_PARAMS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("パラメータ数不一致: expected {TOTAL_PARAMS}, got {num_params}"),
        ));
    }

    // 重み
    let mut weights = Vec::with_capacity(TOTAL_PARAMS);
    for _ in 0..TOTAL_PARAMS {
        weights.push(read_f32(&mut r)?);
    }

    // optimizer m
    let mut m = Vec::with_capacity(TOTAL_PARAMS);
    for _ in 0..TOTAL_PARAMS {
        m.push(read_f32(&mut r)?);
    }

    // optimizer v
    let mut v = Vec::with_capacity(TOTAL_PARAMS);
    for _ in 0..TOTAL_PARAMS {
        v.push(read_f32(&mut r)?);
    }

    // optimizer t
    let t = read_u64(&mut r)?;

    let meta = CheckpointMeta {
        epoch,
        step,
        train_loss,
        val_loss,
        lr,
    };

    let state = AdamWState {
        m,
        v,
        t,
        config: adam_config,
    };

    Ok((meta, weights, state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.ckpt");

        let meta = CheckpointMeta {
            epoch: 3,
            step: 12345,
            train_loss: 0.5,
            val_loss: 0.55,
            lr: 0.001,
        };
        let weights = vec![1.0f32; TOTAL_PARAMS];
        let mut state = AdamWState::new(AdamWConfig::default());
        state.t = 100;
        state.m[0] = 0.1;
        state.v[0] = 0.01;

        save_checkpoint(&path, &meta, &weights, &state).unwrap();
        let (meta2, weights2, state2) = load_checkpoint(&path, AdamWConfig::default()).unwrap();

        assert_eq!(meta2.epoch, 3);
        assert_eq!(meta2.step, 12345);
        assert!((meta2.train_loss - 0.5).abs() < 1e-6);
        assert_eq!(weights2.len(), TOTAL_PARAMS);
        assert_eq!(weights2[0], 1.0);
        assert_eq!(state2.t, 100);
        assert!((state2.m[0] - 0.1).abs() < 1e-6);
        assert!((state2.v[0] - 0.01).abs() < 1e-6);
    }
}
