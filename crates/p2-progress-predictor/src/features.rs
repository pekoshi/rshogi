//! Position → 32 次元 feature vector (allocation 0)
//!
//! 構成 (32 dims):
//! ```text
//!  0..10  : 既存 EndgameFeatures (material_phase, king_dist, mobility,
//!           attack_pressure, eval_magnitude, tt_cutoff(=0), check_freq(=0),
//!           mcts_check_bias(=0), engagement, promotion_progress)
//! 10..23  : board piece counts (Pawn..Dragon、 13 種、 両手番合算 / 正規化)
//! 23..29  : hand counts (Pawn..Rook、 6 種、 両手番合算 / 正規化)
//! 29      : side_to_move (黒=1.0, 白=0.0)
//! 30      : in_check (1/0)
//! 31      : game_ply / 256.0 clamped
//! ```
//!
//! 全 entry は概ね [0, 1] 範囲に正規化済。 negative がありうるのは
//! 既存 EndgameFeatures.attack_pressure の方向のみで、 元実装で
//! `clamp(0,1)` 済。

use p2_search::phase::{compute_features as compute_endgame_features, EndgameFeatures};
use rshogi_core::position::Position;
use rshogi_core::types::{Color, PieceType};

/// feature ベクトルの次元数 (推論モデル input dim と一致)
pub const FEATURE_DIM: usize = 32;

/// 駒種ごとの正規化定数 (盤上最大枚数の概算、 平手初期 + 取り尽くし安全係数)
const NORM_BOARD: [f32; 13] = [
    18.0, // Pawn
    4.0,  // Lance
    4.0,  // Knight
    4.0,  // Silver
    2.0,  // Bishop
    2.0,  // Rook
    4.0,  // Gold
    6.0,  // ProPawn (成駒上限の概算)
    4.0,  // ProLance
    4.0,  // ProKnight
    4.0,  // ProSilver
    2.0,  // Horse
    2.0,  // Dragon
];

/// 手駒 正規化定数 (両手番合算上限の概算)
const NORM_HAND: [f32; 6] = [
    18.0, // Pawn
    4.0,  // Lance
    4.0,  // Knight
    4.0,  // Silver
    2.0,  // Bishop
    2.0,  // Rook
];

const BOARD_PIECE_TYPES: [PieceType; 13] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Gold,
    PieceType::ProPawn,
    PieceType::ProLance,
    PieceType::ProKnight,
    PieceType::ProSilver,
    PieceType::Horse,
    PieceType::Dragon,
];

const HAND_PIECE_TYPES: [PieceType; 6] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Bishop,
    PieceType::Rook,
];

/// 32 次元 feature をスタック上で保持。
///
/// `Copy` を実装することでホットパス上での所有権移動 / clone を伴わない受け渡しを保証。
#[derive(Debug, Clone, Copy)]
pub struct ProgressFeatures {
    pub data: [f32; FEATURE_DIM],
}

impl Default for ProgressFeatures {
    fn default() -> Self {
        Self {
            data: [0.0; FEATURE_DIM],
        }
    }
}

impl ProgressFeatures {
    /// raw EndgameFeatures (10 dim) からフルベクトルに展開する。
    ///
    /// piece counts / side / ply 部分は別途 `fill_position` で埋める想定。
    pub fn from_endgame(ef: &EndgameFeatures) -> Self {
        let mut out = Self::default();
        out.data[0] = ef.material_phase;
        out.data[1] = ef.king_distance;
        out.data[2] = ef.mobility;
        out.data[3] = ef.attack_pressure;
        out.data[4] = ef.eval_magnitude;
        out.data[5] = ef.tt_cutoff_rate;
        out.data[6] = ef.check_frequency;
        out.data[7] = ef.mcts_check_bias;
        out.data[8] = ef.engagement;
        out.data[9] = ef.promotion_progress;
        out
    }

    /// 推論モデルに渡す slice 参照。
    #[inline]
    pub fn as_slice(&self) -> &[f32; FEATURE_DIM] {
        &self.data
    }
}

/// Position から完全な 32 次元特徴量を抽出。 allocation 0。
///
/// `eval_cp` は (10 次元目の) eval_magnitude 計算に使用。 評価値が
/// 取れない呼び出し元 (EndgameDetector など) は 0 を渡す。
pub fn compute_features(pos: &Position, eval_cp: i32) -> ProgressFeatures {
    let ef = compute_endgame_features(pos, eval_cp);
    let mut feats = ProgressFeatures::from_endgame(&ef);

    // [10..23) board piece counts (両手番合算 / 正規化)
    for (i, &pt) in BOARD_PIECE_TYPES.iter().enumerate() {
        let n = pos.pieces(Color::Black, pt).count() + pos.pieces(Color::White, pt).count();
        let v = n as f32 / NORM_BOARD[i];
        feats.data[10 + i] = v.clamp(0.0, 1.0);
    }

    // [23..29) hand counts
    let black_hand = pos.hand(Color::Black);
    let white_hand = pos.hand(Color::White);
    for (i, &pt) in HAND_PIECE_TYPES.iter().enumerate() {
        let n = black_hand.count(pt) + white_hand.count(pt);
        let v = n as f32 / NORM_HAND[i];
        feats.data[23 + i] = v.clamp(0.0, 1.0);
    }

    // [29] side_to_move (黒=1.0, 白=0.0)
    feats.data[29] = if pos.side_to_move() == Color::Black {
        1.0
    } else {
        0.0
    };

    // [30] in_check
    feats.data[30] = if pos.in_check() { 1.0 } else { 0.0 };

    // [31] game_ply / 256.0 clamped
    feats.data[31] = (pos.game_ply() as f32 / 256.0).clamp(0.0, 1.0);

    feats
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::position::Position;

    /// startpos の特徴量が想定範囲に収まることを確認する smoke test。
    #[test]
    fn startpos_features_in_range() {
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let f = compute_features(&pos, 0);
        for (i, v) in f.data.iter().enumerate() {
            assert!((-0.01..=1.01).contains(v), "feature[{}] = {} out of [0,1]", i, v,);
        }
        // startpos では engagement / promotion がほぼ 0
        assert!(f.data[8].abs() < 0.05, "engagement should be ~0 at startpos");
        assert!(f.data[9].abs() < 0.05, "promotion_progress should be ~0 at startpos");
        // 黒手番
        assert_eq!(f.data[29], 1.0);
        // 王手なし
        assert_eq!(f.data[30], 0.0);
    }

    #[test]
    fn feature_dim_is_32() {
        assert_eq!(FEATURE_DIM, 32);
    }
}
