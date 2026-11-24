//! Full Threat index を使う scalar residual head。
//!
//! 重みは `[feature][bucket]` とし、1特徴の8 bucketを連続配置する。
//! accumulator stack entryには2視点 x 8 bucketのi32だけを保持する。

use std::{
    io::{self, Read},
    sync::atomic::{AtomicI32, Ordering},
};

use crate::position::Position;
use crate::types::Color;

use super::accumulator::{DirtyPiece, IndexList};
use super::threat_features::{self, MAX_CHANGED_THREAT_FEATURES, THREAT_DIMENSIONS};

/// ckpt末尾sectionのmagic。
pub const SCALAR_THREAT_MAGIC: &[u8; 8] = b"P2STHR\0\0";
/// section format version。
pub const SCALAR_THREAT_VERSION: u32 = 1;
/// 進行度bucket数。
pub const NUM_SCALAR_THREAT_BUCKETS: usize = 8;
/// i16重み総数。
pub const SCALAR_THREAT_WEIGHT_COUNT: usize = THREAT_DIMENSIONS * NUM_SCALAR_THREAT_BUCKETS;
/// 学習時の量子化値を評価へ加える既定の除数。
pub const SCALAR_THREAT_OUTPUT_DIVISOR_DEFAULT: i32 = 8;
/// USI optionで上書き可能な評価寄与の除数。
pub static SCALAR_THREAT_OUTPUT_DIVISOR: AtomicI32 =
    AtomicI32::new(SCALAR_THREAT_OUTPUT_DIVISOR_DEFAULT);

/// Scalar Threatの評価寄与除数を更新する。
pub fn set_output_divisor(divisor: i32) {
    SCALAR_THREAT_OUTPUT_DIVISOR.store(divisor.max(1), Ordering::Relaxed);
}

/// 現在のScalar Threat評価寄与除数を返す。
#[inline]
pub fn output_divisor() -> i32 {
    SCALAR_THREAT_OUTPUT_DIVISOR.load(Ordering::Relaxed)
}

/// feature-majorの量子化済み重み。
pub struct ScalarThreatWeights {
    values: Box<[i16]>,
}

impl ScalarThreatWeights {
    /// magic消費後のsection headerと重みを読む。
    pub fn read_after_magic<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != SCALAR_THREAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "ScalarThreat version {version} unsupported (expected {SCALAR_THREAT_VERSION})"
                ),
            ));
        }
        reader.read_exact(&mut buf4)?;
        let count = u32::from_le_bytes(buf4) as usize;
        if count != SCALAR_THREAT_WEIGHT_COUNT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ScalarThreat count mismatch: expected {SCALAR_THREAT_WEIGHT_COUNT}, got {count}"
                ),
            ));
        }

        let mut values = vec![0i16; count].into_boxed_slice();
        let mut buf2 = [0u8; 2];
        for value in &mut values {
            reader.read_exact(&mut buf2)?;
            *value = i16::from_le_bytes(buf2);
        }
        Ok(Self { values })
    }

    #[inline]
    fn row(&self, feature: usize) -> &[i16] {
        debug_assert!(feature < THREAT_DIMENSIONS);
        let start = feature * NUM_SCALAR_THREAT_BUCKETS;
        &self.values[start..start + NUM_SCALAR_THREAT_BUCKETS]
    }
}

#[inline]
fn add_row(acc: &mut [i32; NUM_SCALAR_THREAT_BUCKETS], row: &[i16]) {
    for bucket in 0..NUM_SCALAR_THREAT_BUCKETS {
        acc[bucket] += i32::from(row[bucket]);
    }
}

#[inline]
fn sub_row(acc: &mut [i32; NUM_SCALAR_THREAT_BUCKETS], row: &[i16]) {
    for bucket in 0..NUM_SCALAR_THREAT_BUCKETS {
        acc[bucket] -= i32::from(row[bucket]);
    }
}

/// 現局面から1視点分をfull refreshする。
pub fn refresh_perspective(
    pos: &Position,
    weights: &ScalarThreatWeights,
    perspective: Color,
    out: &mut [i32; NUM_SCALAR_THREAT_BUCKETS],
) {
    out.fill(0);
    let king_sq = pos.king_square(perspective);
    threat_features::for_each_active_threat_index(pos, perspective, king_sq, |idx| {
        add_row(out, weights.row(idx));
    });
}

/// 直前局面のcacheから1視点分を差分更新する。
pub fn update_perspective(
    pos: &Position,
    dirty_piece: &DirtyPiece,
    weights: &ScalarThreatWeights,
    perspective: Color,
    previous: &[i32; NUM_SCALAR_THREAT_BUCKETS],
    out: &mut [i32; NUM_SCALAR_THREAT_BUCKETS],
) {
    let king_sq = pos.king_square(perspective);
    if threat_features::needs_threat_refresh(dirty_piece, king_sq, perspective) {
        refresh_perspective(pos, weights, perspective, out);
        return;
    }

    *out = *previous;
    let mut removed = IndexList::<MAX_CHANGED_THREAT_FEATURES>::new();
    let mut added = IndexList::<MAX_CHANGED_THREAT_FEATURES>::new();
    if !threat_features::append_changed_threat_indices(
        pos,
        dirty_piece,
        perspective,
        king_sq,
        &mut removed,
        &mut added,
    ) {
        refresh_perspective(pos, weights, perspective, out);
        return;
    }
    for idx in removed.iter() {
        sub_row(out, weights.row(idx));
    }
    for idx in added.iter() {
        add_row(out, weights.row(idx));
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn section_header_and_zero_weights_are_read() {
        let mut bytes = Vec::with_capacity(8 + SCALAR_THREAT_WEIGHT_COUNT * 2);
        bytes.extend_from_slice(&SCALAR_THREAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(SCALAR_THREAT_WEIGHT_COUNT as u32).to_le_bytes());
        bytes.resize(bytes.len() + SCALAR_THREAT_WEIGHT_COUNT * 2, 0);
        let weights = ScalarThreatWeights::read_after_magic(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(weights.values.len(), SCALAR_THREAT_WEIGHT_COUNT);
        assert!(weights.values.iter().all(|&value| value == 0));
    }

    #[test]
    fn feature_rows_are_bucket_contiguous() {
        let mut values = vec![0i16; SCALAR_THREAT_WEIGHT_COUNT];
        let feature = THREAT_DIMENSIONS - 1;
        for bucket in 0..NUM_SCALAR_THREAT_BUCKETS {
            values[feature * NUM_SCALAR_THREAT_BUCKETS + bucket] = bucket as i16;
        }
        let weights = ScalarThreatWeights {
            values: values.into_boxed_slice(),
        };
        assert_eq!(weights.row(feature), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn output_divisor_is_clamped_to_one() {
        let before = output_divisor();
        set_output_divisor(24);
        assert_eq!(output_divisor(), 24);
        set_output_divisor(0);
        assert_eq!(output_divisor(), 1);
        set_output_divisor(before);
    }
}
