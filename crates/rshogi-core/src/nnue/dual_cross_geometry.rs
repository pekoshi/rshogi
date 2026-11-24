//! Dual Crossへ攻撃方向・距離を加えるゼロ平均残差head。

use std::io::{self, Read};

use crate::bitboard::Bitboard;
use crate::nnue::dual_delta::MoveDelta;
use crate::position::Position;
use crate::types::{Color, Square};

use super::dual_cross::{
    MAX_ACTIVE_DUAL_CROSS_FEATURES, build_affected_sq_set, oriented_sq_idx, piece_attacks_from,
    piece_type_to_cross_idx,
};

pub const MAGIC: &[u8; 8] = b"P2DCG1\0\0";
pub const VERSION: u32 = 1;
pub const NUM_BUCKETS: usize = 8;
pub const NUM_PIECE_TYPES: usize = 14;
pub const NUM_DIRECTIONS: usize = 8;
pub const NUM_DISTANCE_CLASSES: usize = 4;
pub const NUM_GEOMETRY_CLASSES: usize = NUM_DIRECTIONS * NUM_DISTANCE_CLASSES;
pub const NUM_PAIR_TYPES: usize = NUM_PIECE_TYPES * NUM_PIECE_TYPES;
pub const DIM_PER_PERSPECTIVE: usize = NUM_PAIR_TYPES * NUM_GEOMETRY_CLASSES;
pub const WEIGHT_COUNT: usize = 2 * DIM_PER_PERSPECTIVE * NUM_BUCKETS;

/// runtime layout: `[perspective][pair×geometry][bucket]`。
pub struct DualCrossGeometryWeights {
    values: Box<[i16]>,
}

impl DualCrossGeometryWeights {
    pub fn read_after_magic<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("DualCrossGeometry version {version} unsupported"),
            ));
        }
        reader.read_exact(&mut buf4)?;
        let count = u32::from_le_bytes(buf4) as usize;
        if count != WEIGHT_COUNT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("DualCrossGeometry count mismatch: expected {WEIGHT_COUNT}, got {count}"),
            ));
        }

        let mut disk = vec![0i16; count];
        let mut buf2 = [0u8; 2];
        for value in &mut disk {
            reader.read_exact(&mut buf2)?;
            *value = i16::from_le_bytes(buf2);
        }
        let mut values = vec![0i16; count];
        for perspective in 0..2 {
            for bucket in 0..NUM_BUCKETS {
                for index in 0..DIM_PER_PERSPECTIVE {
                    let src = ((perspective * NUM_BUCKETS + bucket) * DIM_PER_PERSPECTIVE) + index;
                    let dst = ((perspective * DIM_PER_PERSPECTIVE + index) * NUM_BUCKETS) + bucket;
                    values[dst] = disk[src];
                }
            }
        }
        Ok(Self {
            values: values.into_boxed_slice(),
        })
    }

    #[inline]
    fn row(&self, perspective: Color, index: u16) -> &[i16] {
        let start = (perspective.index() * DIM_PER_PERSPECTIVE + index as usize) * NUM_BUCKETS;
        &self.values[start..start + NUM_BUCKETS]
    }
}

#[inline]
fn geometry_class(own_sq: Square, target_sq: Square, perspective: Color) -> usize {
    let own = oriented_sq_idx(own_sq, perspective);
    let target = oriented_sq_idx(target_sq, perspective);
    let own_file = (own / 9) as i32;
    let own_rank = (own % 9) as i32;
    let target_file = (target / 9) as i32;
    let target_rank = (target % 9) as i32;
    let df = (target_file - own_file).signum();
    let dr = (target_rank - own_rank).signum();
    let direction = match (df, dr) {
        (-1, -1) => 0,
        (0, -1) => 1,
        (1, -1) => 2,
        (-1, 0) => 3,
        (1, 0) => 4,
        (-1, 1) => 5,
        (0, 1) => 6,
        (1, 1) => 7,
        _ => unreachable!("attacker and target cannot occupy the same square"),
    };
    let distance = (target_file - own_file).abs().max((target_rank - own_rank).abs());
    let distance_class = match distance {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        _ => 3,
    };
    direction * NUM_DISTANCE_CLASSES + distance_class
}

#[inline]
fn active_index(
    own_sq: Square,
    target_sq: Square,
    own_piece: crate::types::Piece,
    target_piece: crate::types::Piece,
    perspective: Color,
) -> u16 {
    let pair = piece_type_to_cross_idx(own_piece.piece_type()) * NUM_PIECE_TYPES
        + piece_type_to_cross_idx(target_piece.piece_type());
    (pair * NUM_GEOMETRY_CLASSES + geometry_class(own_sq, target_sq, perspective)) as u16
}

fn collect(
    pos: &Position,
    perspective: Color,
    source_filter: Option<Bitboard>,
    indices: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    source_squares: &mut [u8; MAX_ACTIVE_DUAL_CROSS_FEATURES],
) -> usize {
    let occupied = pos.occupied();
    let enemy = pos.pieces_c(!perspective);
    let mut sources = pos.pieces_c(perspective);
    if let Some(filter) = source_filter {
        sources &= filter;
    }
    let mut len = 0;
    'outer: for own_sq in sources.iter() {
        let own_piece = pos.piece_on(own_sq);
        let targets = piece_attacks_from(own_piece, own_sq, occupied) & enemy;
        for target_sq in targets.iter() {
            if len == MAX_ACTIVE_DUAL_CROSS_FEATURES {
                break 'outer;
            }
            indices[len] =
                active_index(own_sq, target_sq, own_piece, pos.piece_on(target_sq), perspective);
            source_squares[len] = own_sq.raw();
            len += 1;
        }
    }
    len
}

#[inline]
fn add_row(acc: &mut [i32; NUM_BUCKETS], row: &[i16]) {
    for (sum, &weight) in acc.iter_mut().zip(row) {
        *sum += i32::from(weight);
    }
}

#[inline]
fn sub_row(acc: &mut [i32; NUM_BUCKETS], row: &[i16]) {
    for (sum, &weight) in acc.iter_mut().zip(row) {
        *sum -= i32::from(weight);
    }
}

pub fn refresh(
    pos: &Position,
    weights: &DualCrossGeometryWeights,
    perspective: Color,
    indices: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    source_squares: &mut [u8; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    len_out: &mut u16,
    accumulator: &mut [i32; NUM_BUCKETS],
) {
    accumulator.fill(0);
    let len = collect(pos, perspective, None, indices, source_squares);
    *len_out = len as u16;
    for &index in &indices[..len] {
        add_row(accumulator, weights.row(perspective, index));
    }
}

fn multiset_diff(
    old: &mut [u16],
    new: &mut [u16],
    removed: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    added: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
) -> (usize, usize) {
    old.sort_unstable();
    new.sort_unstable();
    let (mut i, mut j, mut r, mut a) = (0, 0, 0, 0);
    while i < old.len() || j < new.len() {
        if i < old.len() && (j == new.len() || old[i] < new[j]) {
            removed[r] = old[i];
            r += 1;
            i += 1;
        } else if j < new.len() && (i == old.len() || new[j] < old[i]) {
            added[a] = new[j];
            a += 1;
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    (r, a)
}

#[allow(clippy::too_many_arguments)]
pub fn update(
    pos: &Position,
    delta: &MoveDelta,
    weights: &DualCrossGeometryWeights,
    perspective: Color,
    previous_indices: &[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    previous_source_squares: &[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    previous_len: u16,
    current_indices: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    current_source_squares: &mut [u8; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    current_len: &mut u16,
    accumulator: &mut [i32; NUM_BUCKETS],
) {
    if delta.king_moved_any {
        refresh(
            pos,
            weights,
            perspective,
            current_indices,
            current_source_squares,
            current_len,
            accumulator,
        );
        return;
    }

    let affected = build_affected_sq_set(delta);
    let mut old_affected = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let mut old_len = 0;
    let mut kept_len = 0;
    for i in 0..previous_len as usize {
        let sq = Square::from_u8(previous_source_squares[i]).expect("cached source square");
        if affected.contains(sq) {
            old_affected[old_len] = previous_indices[i];
            old_len += 1;
        } else {
            current_indices[kept_len] = previous_indices[i];
            current_source_squares[kept_len] = previous_source_squares[i];
            kept_len += 1;
        }
    }

    let mut new_affected = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let mut new_sources = [0u8; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let new_len = collect(pos, perspective, Some(affected), &mut new_affected, &mut new_sources);
    // multiset_diff は引数を in-place sort するため、cache 本体に書き戻す
    // new_affected / new_sources の index↔source_square 対応を壊さないよう、
    // diff 計算には new_affected の使い捨てコピーだけを渡す。
    // old_affected は対の source 配列を持たず破棄されるので in-place sort で良い。
    let mut new_for_diff = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    new_for_diff[..new_len].copy_from_slice(&new_affected[..new_len]);
    let mut removed = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let mut added = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let (removed_len, added_len) = multiset_diff(
        &mut old_affected[..old_len],
        &mut new_for_diff[..new_len],
        &mut removed,
        &mut added,
    );
    for &index in &removed[..removed_len] {
        sub_row(accumulator, weights.row(perspective, index));
    }
    for &index in &added[..added_len] {
        add_row(accumulator, weights.row(perspective, index));
    }

    let copy_len = new_len.min(MAX_ACTIVE_DUAL_CROSS_FEATURES - kept_len);
    current_indices[kept_len..kept_len + copy_len].copy_from_slice(&new_affected[..copy_len]);
    current_source_squares[kept_len..kept_len + copy_len].copy_from_slice(&new_sources[..copy_len]);
    *current_len = (kept_len + copy_len) as u16;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_are_small_and_u16_addressable() {
        assert_eq!(DIM_PER_PERSPECTIVE, 6_272);
        assert_eq!(WEIGHT_COUNT, 100_352);
        assert!(DIM_PER_PERSPECTIVE <= u16::MAX as usize);
    }
}
