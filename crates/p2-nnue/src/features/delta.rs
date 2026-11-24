//! V3 では差分更新を無効化 (毎手 fresh 抽出)
//!
//! dlshogi 互換 features は王の移動・利きが盤面全体に影響するため、
//! incremental 更新コストが大きい。 V3 ではフル再計算を採用。
//! 互換性のために関数は残すが no-op として動く。

use rshogi_core::position::Position;
use rshogi_core::types::{Color, Move, Piece};

use super::extract::extract_features;
use super::feature_types::{P2FeatureDelta, P2NnueInput};

/// V3: 差分計算は no-op (recompute_full=true を返す)。
/// 呼び出し側は `apply_delta` で fresh 抽出する設計。
pub fn compute_delta(
    _pos: &Position,
    _mv: Move,
    _captured: Piece,
    _stm_before: Color,
    _prev_input: &P2NnueInput,
) -> P2FeatureDelta {
    P2FeatureDelta {
        recompute_full: true,
    }
}

/// V3: delta 適用は fresh 抽出に置き換える。
pub fn apply_delta(pos: &Position, _delta: &P2FeatureDelta, input: &mut P2NnueInput) {
    let fresh = extract_features(pos);
    *input = fresh;
}
