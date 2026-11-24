//! Position → HalfKAv2_hm active indices アダプタ
//!
//! rshogi-core の HalfKA_hm feature encoding を使い、 PolicyValueNet が必要とする
//! u32 active indices (stm/nstm) を抽出する。

use rshogi_core::nnue::features::{Feature, HalfKA_hm};
use rshogi_core::nnue::{IndexList, MAX_ACTIVE_FEATURES};
use rshogi_core::position::Position;

/// Position から両 perspective の active indices を取得
///
/// # Returns
/// `(stm_indices, nstm_indices)` の組。 各 Vec は最大 MAX_ACTIVE_FEATURES (=54) 要素。
pub fn position_to_active_indices(pos: &Position) -> (Vec<u32>, Vec<u32>) {
    let stm = pos.side_to_move();
    let nstm = !stm;

    let mut stm_active: IndexList<MAX_ACTIVE_FEATURES> = IndexList::new();
    let mut nstm_active: IndexList<MAX_ACTIVE_FEATURES> = IndexList::new();

    HalfKA_hm::append_active_indices(pos, stm, &mut stm_active);
    HalfKA_hm::append_active_indices(pos, nstm, &mut nstm_active);

    (index_list_to_vec(&stm_active), index_list_to_vec(&nstm_active))
}

fn index_list_to_vec<const N: usize>(list: &IndexList<N>) -> Vec<u32> {
    let mut out = Vec::with_capacity(list.len());
    for idx in list.iter() {
        out.push(idx as u32);
    }
    out
}
