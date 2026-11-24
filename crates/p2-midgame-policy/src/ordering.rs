//! 合法手リストに policy score を付与して並び替える core。
//!
//! ## 混合方式 (採用)
//!
//! ```text
//!   final_score = α * policy_logit + history_score + (killer ? K : 0) + capture_bonus
//! ```
//!
//! ただし本クレートが扱う範囲は **policy 側だけ**。 history / killer /
//! capture_bonus は呼び出し側 (rshogi-core MovePicker) が持つため、 本クレートは
//! `(Move, policy_logit)` の pair を返すに留める。 αβ 側で `α * policy_logit`
//! を加算した最終スコアでソートする。
//!
//! ## 採用しなかった案
//!
//! - **Top-K 最優先 (policy gating)**: policy の top-K を history/killer/SEE の
//!   前に強制挿入する方式。 ckpt が overfitting している場合に致命傷になりやすく、
//!   default OFF で慎重に検証する本タスクの方針に合わない。 selfplay で root
//!   prior を実証した後で第 2 段階として導入する。
//! - **逆引きテーブル (idx → Move)**: ordering 用途では legal moves が手元に
//!   あるため、 forward `make_move_label` の O(N) で十分。 idx は理論上、
//!   不成/成り違いで衝突するケースがあり逆引きの一意性も保証できない。

use rshogi_core::types::{Color, Move};

use crate::move_lookup::{POLICY_SIZE, make_move_label};

/// 1 手 + その policy logit。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredMove {
    pub mv: Move,
    /// raw logit (softmax 前)。 呼び出し側で α を掛けて history と混合する。
    pub logit: f32,
}

/// 合法手最大数を上回らない固定容量。 [`MAX_MOVES`] = 600 を保持しても 600 *
/// 12 = 7200 byte と CPU L1 に十分収まる。
///
/// [`MAX_MOVES`]: rshogi_core::movegen::MAX_MOVES
pub const MAX_SCORED_MOVES: usize = rshogi_core::movegen::MAX_MOVES;

/// `Vec` を使わずに最大 [`MAX_SCORED_MOVES`] 件の [`ScoredMove`] を持つ固定長
/// 配列。 ホットパスでの heap allocation を回避する。
pub struct PolicyScoreList {
    moves: [ScoredMove; MAX_SCORED_MOVES],
    len: usize,
}

impl PolicyScoreList {
    /// 空のリストを作成。
    pub fn new() -> Self {
        Self {
            moves: [ScoredMove {
                mv: Move::NONE,
                logit: 0.0,
            }; MAX_SCORED_MOVES],
            len: 0,
        }
    }

    /// 長さ。
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 空か。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// スライスとして取得 (有効範囲のみ)。
    #[inline]
    pub fn as_slice(&self) -> &[ScoredMove] {
        &self.moves[..self.len]
    }

    /// 可変スライスとして取得 (有効範囲のみ)。
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [ScoredMove] {
        &mut self.moves[..self.len]
    }

    /// 末尾に push (容量を超えたら無視)。
    #[inline]
    pub fn push(&mut self, scored: ScoredMove) {
        if self.len < MAX_SCORED_MOVES {
            self.moves[self.len] = scored;
            self.len += 1;
        }
    }

    /// 長さを 0 にリセット (内部 buffer は再利用)。
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// `logit` 降順で安定ソート (top-K 抽出用、 安定性は同 logit 間の入力順保持に必要)。
    pub fn sort_by_logit_desc(&mut self) {
        // 安定 sort: 同 logit を持つ手の生成順を維持し、 NPS への影響を最小化する。
        // (unstable sort は標準的に高速だが、 動作再現性のためここでは安定 sort を採る)。
        self.moves[..self.len]
            .sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(std::cmp::Ordering::Equal));
    }
}

impl Default for PolicyScoreList {
    fn default() -> Self {
        Self::new()
    }
}

/// 合法手列に対し policy logit を割り当てる core 関数 (sort なし)。
///
/// - `moves`: 合法手列 (`rshogi-core` movegen 出力など)
/// - `color`: `pos.side_to_move()` (`make_move_label` の正規化用)
/// - `logits`: 推論結果 (`POLICY_SIZE = 2187`)
/// - `out`: 出力 (clear して再充填する)
///
/// 各手の logit は `logits[make_move_label(mv, color)]`。 非通常手
/// (`mv.is_normal()` でない) はスキップせず logit=0 として残す (呼び出し側
/// で必要なら filter する)。
pub fn score_moves(moves: &[Move], color: Color, logits: &[f32], out: &mut PolicyScoreList) {
    debug_assert_eq!(logits.len(), POLICY_SIZE);
    out.clear();
    for &mv in moves {
        let logit = if mv.is_normal() {
            let idx = make_move_label(mv, color);
            // idx は make_move_label 仕様上 0..POLICY_SIZE。 debug_assert で
            // 保護し、 release では bounds check で潜在 panic を避けるため
            // 安全な indexing にしておく。
            debug_assert!(idx < POLICY_SIZE);
            *logits.get(idx).unwrap_or(&0.0)
        } else {
            0.0
        };
        out.push(ScoredMove { mv, logit });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::movegen::{MoveList, generate_legal};
    use rshogi_core::position::{Position, SFEN_HIRATE};

    #[test]
    fn score_moves_attaches_logits_to_each_move() {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();

        let mut list = MoveList::new();
        generate_legal(&pos, &mut list);
        let color = pos.side_to_move();

        // logits[idx]=idx*0.001 で生成 → ordering が単調になるか確認
        let mut logits = vec![0f32; POLICY_SIZE];
        for (i, v) in logits.iter_mut().enumerate() {
            *v = i as f32 * 0.001;
        }

        let mut scored = PolicyScoreList::new();
        // legal moves を Move slice にする
        let moves: Vec<Move> = list.iter().copied().collect();
        score_moves(&moves, color, &logits, &mut scored);
        assert_eq!(scored.len(), moves.len());

        // 各 logit が正しい idx に対応するか
        for sm in scored.as_slice() {
            let idx = make_move_label(sm.mv, color);
            assert!((sm.logit - idx as f32 * 0.001).abs() < 1e-5);
        }
    }

    #[test]
    fn sort_by_logit_desc_orders_correctly() {
        use rshogi_core::types::{File, Rank, Square};
        let mut list = PolicyScoreList::new();
        // 3 手、 logit 0.1 / 0.3 / 0.2
        let m1 = Move::new_move(
            Square::new(File::File1, Rank::Rank1),
            Square::new(File::File1, Rank::Rank2),
            false,
        );
        let m2 = Move::new_move(
            Square::new(File::File2, Rank::Rank1),
            Square::new(File::File2, Rank::Rank2),
            false,
        );
        let m3 = Move::new_move(
            Square::new(File::File3, Rank::Rank1),
            Square::new(File::File3, Rank::Rank2),
            false,
        );
        list.push(ScoredMove { mv: m1, logit: 0.1 });
        list.push(ScoredMove { mv: m2, logit: 0.3 });
        list.push(ScoredMove { mv: m3, logit: 0.2 });
        list.sort_by_logit_desc();
        let slice = list.as_slice();
        assert_eq!(slice[0].mv, m2);
        assert_eq!(slice[1].mv, m3);
        assert_eq!(slice[2].mv, m1);
    }

    #[test]
    fn top_k_returns_highest_logits() {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        let mut list = MoveList::new();
        generate_legal(&pos, &mut list);
        let color = pos.side_to_move();

        // 特定の手 (7g7f) に高 logit を割り当て、 top-1 が当該手になるか確認
        use rshogi_core::types::{File, Rank, Square};
        let target = Move::new_move(
            Square::new(File::File7, Rank::Rank7),
            Square::new(File::File7, Rank::Rank6),
            false,
        );
        let mut logits = vec![0f32; POLICY_SIZE];
        logits[make_move_label(target, color)] = 100.0;

        let moves: Vec<Move> = list.iter().copied().collect();
        let mut scored = PolicyScoreList::new();
        score_moves(&moves, color, &logits, &mut scored);
        scored.sort_by_logit_desc();
        // generate_legal は Move に piece info を含めるため、 Move 全体での
        // 等値判定は失敗する。 from/to/promote を抽出して比較する (raw 16bit)。
        assert_eq!(scored.as_slice()[0].mv.raw(), target.raw());
    }
}
