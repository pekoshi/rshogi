//! P2 特徴量の補助 helper (V3: 旧 index 系は廃止)
//!
//! V3 では features1/features2 のレイアウトが変わったため、
//! 旧 board_piece_index/king_square_index/hand_bucket_index は撤去。
//! game_phase / king_bucket のみ別系メタ情報として残す。

use rshogi_core::position::Position;
use rshogi_core::types::Color;

/// 玉バケット (0..14): STM 視点の玉位置を 15 バケットに分類
#[inline]
pub fn compute_king_bucket(pos: &Position) -> u8 {
    let stm = pos.side_to_move();
    let king_sq = pos.king_square(stm);
    let sq = match stm {
        Color::White => king_sq.inverse(),
        _ => king_sq,
    };
    let file = sq.file().index();
    let rank = sq.rank().index();
    let mirror_file = file.min(8 - file);
    let rank_group = rank / 3;
    (mirror_file * 3 + rank_group) as u8
}

/// ゲームフェーズ (0=endgame, 255=opening)
#[inline]
pub fn compute_game_phase(pos: &Position) -> u8 {
    let total = pos.occupied().count();
    let non_king = total.saturating_sub(2);
    ((non_king * 255 / 38).min(255)) as u8
}
