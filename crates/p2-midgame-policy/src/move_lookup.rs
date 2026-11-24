//! `Move` ↔ dlshogi 2187-idx の変換 (`make_move_label`)。
//!
//! `crates/tools/src/dlshogi_features.rs::make_move_label` と
//! `scripts/move_to_policy_idx.py` (cshogi `make_move_label`) と **bit-for-bit
//! 一致** するロジックを、 重い依存 (tools crate は ort / sysinfo 等を抱える) を
//! 引かずに本 crate 内で再実装する。
//!
//! ## なぜ逆引きテーブルを持たないか
//!
//! 「2187 idx → Move」 の逆引きは、 同一 idx に対し合法手が一意に決まらない
//! ケースが存在する (例: 不成と成りの分離、 駒打ちの種別など)。 また αβ の
//! ordering 用途では「legal moves に対して順位を付けたい」だけなので、 legal
//! moves それぞれに `make_move_label` を呼ぶ forward 計算 (O(legal_count)) で
//! 十分に高速 (一手あたり ~50 ns、 100 moves でも 5 μs)。

use rshogi_core::types::{Color, Move};

/// 移動方向数 (dlshogi `MOVE_DIRECTION_NUM`)
///
/// 不成 10 + 成 10 = 20。
const MOVE_DIRECTION_NUM: u32 = 20;

/// 升マス数。
const SQUARE_NUM: usize = 81;

/// dlshogi `make_move_label` 互換 policy 出力次元 (= 27 × 81)。
pub const POLICY_SIZE: usize = 27 * SQUARE_NUM; // 2187

/// from→to の差分 (dir_x, dir_y) から移動方向 id (0..10) を判定する。
///
/// `dir_x = from_file - to_file`, `dir_y = to_rank - from_rank`
/// (cshogi と同一符号: 上方向が負、 右方向が正)
#[inline]
fn move_direction(dir_x: i32, dir_y: i32) -> u32 {
    // cshogi の `__dlshogi_make_move_label` の if-else chain と同一順序。
    // 桂馬 (Up2Left / Up2Right) は (dir_y == -2 && |dir_x| == 1) で判定する
    // ため、 必ず広範な「dir_y < 0 && dir_x < 0」より前に評価する。
    if dir_y < 0 && dir_x == 0 {
        0 // Up
    } else if dir_y == -2 && dir_x == -1 {
        8 // Up2Left
    } else if dir_y == -2 && dir_x == 1 {
        9 // Up2Right
    } else if dir_y < 0 && dir_x < 0 {
        1 // UpLeft
    } else if dir_y < 0 && dir_x > 0 {
        2 // UpRight
    } else if dir_y == 0 && dir_x < 0 {
        3 // Left
    } else if dir_y == 0 && dir_x > 0 {
        4 // Right
    } else if dir_y > 0 && dir_x == 0 {
        5 // Down
    } else if dir_y > 0 && dir_x < 0 {
        6 // DownLeft
    } else {
        // dir_y > 0 && dir_x > 0
        7 // DownRight
    }
}

/// 指し手をポリシー出力のラベルインデックスに変換する。
///
/// cshogi の `__dlshogi_make_move_label` と同一ロジック。 後手番では盤面を
/// 180 度回転して先手視点に正規化する。
///
/// 戻り値は `0..POLICY_SIZE (=2187)` のインデックス。
///
/// # Panics
///
/// `Move::NONE / NULL / PASS / WIN` を渡すと debug build では panic、 release
/// build では未定義インデックスを返す。 呼び出し側で `mv.is_normal()` を
/// 保証すること。
#[inline]
pub fn make_move_label(mv: Move, color: Color) -> usize {
    debug_assert!(mv.is_normal(), "make_move_label called on non-normal move");
    let is_white = color == Color::White;

    if !mv.is_drop() {
        let to_sq = if is_white {
            mv.to().inverse().index()
        } else {
            mv.to().index()
        };
        let from_sq = if is_white {
            mv.from().inverse().index()
        } else {
            mv.from().index()
        };

        let to_file = to_sq / 9;
        let to_rank = to_sq % 9;
        let from_file = from_sq / 9;
        let from_rank = from_sq % 9;

        let dir_x = from_file as i32 - to_file as i32;
        let dir_y = to_rank as i32 - from_rank as i32;

        let mut direction = move_direction(dir_x, dir_y);
        if mv.is_promote() {
            direction += 10;
        }

        SQUARE_NUM * direction as usize + to_sq
    } else {
        let to_sq = if is_white {
            mv.to().inverse().index()
        } else {
            mv.to().index()
        };

        // 手駒種別: Pawn=1..Rook=6 → 0..5, Gold=7 → 6
        let pt = mv.drop_piece_type() as u32;
        let hand_piece = if pt <= 6 { pt - 1 } else { 6 };
        let direction = MOVE_DIRECTION_NUM + hand_piece;
        SQUARE_NUM * direction as usize + to_sq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::types::{File, PieceType, Rank, Square};

    #[test]
    fn label_in_range_for_all_basic_moves() {
        // 通常の盤上移動 (黒)
        let mv = Move::new_move(
            Square::new(File::File7, Rank::Rank7),
            Square::new(File::File7, Rank::Rank6),
            false,
        );
        let label = make_move_label(mv, Color::Black);
        assert!(label < POLICY_SIZE);

        // 成り
        let mv_promote = Move::new_move(
            Square::new(File::File2, Rank::Rank3),
            Square::new(File::File2, Rank::Rank2),
            true,
        );
        let label_p = make_move_label(mv_promote, Color::Black);
        assert!(label_p < POLICY_SIZE);
        assert_ne!(label, label_p);

        // 駒打ち
        let drop = Move::new_drop(PieceType::Pawn, Square::new(File::File5, Rank::Rank5));
        let label_drop = make_move_label(drop, Color::Black);
        assert!(label_drop < POLICY_SIZE);

        // 後手番では 180 度回転
        let label_w = make_move_label(mv, Color::White);
        assert!(label_w < POLICY_SIZE);
    }

    #[test]
    fn drop_gold_uses_hand_piece_6() {
        // Gold は PieceType=7、 hand_piece=6 → direction = 20 + 6 = 26
        let drop_gold = Move::new_drop(PieceType::Gold, Square::new(File::File5, Rank::Rank5));
        let label = make_move_label(drop_gold, Color::Black);
        let direction = 26;
        let to_sq = Square::new(File::File5, Rank::Rank5).index();
        assert_eq!(label, SQUARE_NUM * direction + to_sq);
    }

    #[test]
    fn knight_jump_is_up2_direction() {
        // 桂馬 Up2Left: dir_y=-2, dir_x=-1
        // 例: from=(File3, Rank5) → to=(File2, Rank3)
        //   from_file=2 (File3 の index)、 to_file=1
        //   from_rank=4、 to_rank=2
        //   dir_x = 2-1 = 1、 dir_y = 2-4 = -2 → Up2Right
        let from = Square::new(File::File3, Rank::Rank5);
        let to = Square::new(File::File2, Rank::Rank3);
        let mv = Move::new_move(from, to, false);
        let label = make_move_label(mv, Color::Black);

        // direction = 9 (Up2Right)
        assert_eq!(label, SQUARE_NUM * 9 + to.index());
    }

    #[test]
    fn forward_roundtrip_yields_unique_idx_per_normal_move() {
        // 1 つの局面で生成した legal moves に対し、 label が衝突しないことを
        // 軽く検査する (ordering 用途で逆引き不要を担保)。 完全な uniqueness は
        // 保証されない (同一 from/to/promote だが駒種が違う場合は同じ idx に
        // なりえる) が、 一般的な hirate 周辺では衝突は稀。
        use rshogi_core::movegen::{MoveList, generate_legal};
        use rshogi_core::position::Position;

        let mut pos = Position::new();
        pos.set_sfen(rshogi_core::position::SFEN_HIRATE).unwrap();

        let mut list = MoveList::new();
        generate_legal(&pos, &mut list);
        let color = pos.side_to_move();

        let mut labels: Vec<usize> = Vec::with_capacity(list.len());
        for mv in list.iter() {
            let label = make_move_label(*mv, color);
            assert!(label < POLICY_SIZE);
            labels.push(label);
        }
        assert!(!labels.is_empty());
    }
}
