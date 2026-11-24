//! BasicDfpnEngine — dlshogi 流 df-pn 実装
//!
//! ## アルゴリズム概要
//!
//! - OR ノード (攻め手): 王手のみ生成
//!   - `pn(n) = min(child.pn)`、 `dn(n) = sum(child.dn)`
//!   - 最小 pn の子を展開、 threshold backup で `thpn_child = min(thpn, second_best_pn + 1)`
//! - AND ノード (受け手): 全合法手 (応手) 生成
//!   - `pn(n) = sum(child.pn)`、 `dn(n) = min(child.dn)`
//!   - 最小 dn の子を展開、 threshold backup 対称
//!
//! ## dlshogi 流の最適化
//!
//! 1. **mate1ply ショートカット**: OR-node entry 直前に `mate_1ply` check → 浅い詰みを即 pn=0
//! 2. **TT に手駒情報** (`tt::DfpnTtEntry`): hash_high + hand exact match で proof piece
//! 3. **threshold backup**: 単純な `+1` ではなく second-best との差分を使用
//! 4. **infinity = 100_000_000** (dlshogi style、 saturating で overflow 回避)
//!
//! ## TODO (将来拡張)
//!
//! - 完全な proof piece dominance check (現状 exact hand match のみ)
//! - path-based hash (GHI 問題対策)
//! - mate_3ply 統合 (探索木縮小)

use rshogi_core::mate::mate_1ply;
use rshogi_core::movegen::{ExtMoveBuffer, GenType, MoveList, generate_legal, generate_with_type};
use rshogi_core::position::Position;
use rshogi_core::types::{Move, RepetitionState};

use super::tt::DfpnTranspositionTable;
use super::{DfpnEngine, DfpnLimits, DfpnResult};

/// dlshogi 流の pn/dn 上限値
const INF: u32 = 100_000_000;

/// 1 ノードで生成しうる手の上限 (将棋の最大合法手数は 593 程度だが余裕を持って 256 + α)
const MAX_MOVES: usize = 600;

/// BasicDfpnEngine
pub struct BasicDfpnEngine {
    tt: DfpnTranspositionTable,
    node_count: u32,
}

/// 子ノード探索結果バッファ
struct ChildSlots {
    moves: [Move; MAX_MOVES],
    pns: [u32; MAX_MOVES],
    dns: [u32; MAX_MOVES],
    count: usize,
}

impl ChildSlots {
    fn new() -> Self {
        Self {
            moves: [Move::NONE; MAX_MOVES],
            pns: [1; MAX_MOVES],
            dns: [1; MAX_MOVES],
            count: 0,
        }
    }
}

/// mate_3ply: AND-node 位置 (王手を受ける側) で「全合法応手 → 各 1-ply で詰み」を判定。
///
/// 真なら現局面は 3 ply 詰みの第 2 手目 (1 ply 受け + 1 ply 攻め = 残 2 ply で詰む完了)。
/// 親 OR-node の child としては実質詰み (pn=0)。
///
/// 注意: pos は AND-node 位置で in_check 中 (受け手番) であることを想定。
/// in_check でなければ詰将棋の論理が成立しないため None。
fn check_mate_3ply(pos: &mut Position) -> bool {
    if !pos.in_check() {
        return false;
    }
    let mut ml = MoveList::new();
    generate_legal(pos, &mut ml);
    if ml.is_empty() {
        // 受け方なし = 既に詰み (この case は mate_1ply で捕捉済のはずだが念のため)
        return true;
    }
    for i in 0..ml.len() {
        let mv = ml.at(i);
        let gives_check = pos.gives_check(mv);
        let _ = pos.do_move(mv, gives_check);
        let mate_one = mate_1ply(pos);
        pos.undo_move(mv);
        if mate_one.is_none() {
            return false; // この evasion で詰まない = 3 ply 詰みではない
        }
    }
    true // 全 evasion で 1-ply 詰む = 3 ply で詰み完成
}

impl BasicDfpnEngine {
    /// 指定 TT サイズで作成
    pub fn new(tt_size_mb: usize) -> Self {
        Self {
            tt: DfpnTranspositionTable::new(tt_size_mb),
            node_count: 0,
        }
    }

    /// OR/AND を 1 つにまとめた inner search。 OR_NODE で挙動を切り替える。
    fn inner_search<const OR_NODE: bool>(
        &mut self,
        pos: &mut Position,
        pn_limit: u32,
        dn_limit: u32,
        depth: u16,
        limits: &DfpnLimits,
    ) {
        if self.should_stop(limits) {
            return;
        }

        let key = pos.key();
        let hand = pos.hand(pos.side_to_move());

        // (1) Proof piece dominance check
        // - OR-node: より少ない手駒で詰み証明済の entry があれば、 現在も詰み (best move 流用)
        // - AND-node: より多い手駒でも不詰証明済の entry があれば、 現在も不詰
        if OR_NODE && let Some(dom) = self.tt.probe_proof_dominance(key, hand) {
            let bm = dom.best_move;
            self.tt.store(key, hand, 0, INF, bm);
            return;
        }
        if !OR_NODE && let Some(_dom) = self.tt.probe_disproof_dominance(key, hand) {
            self.tt.store(key, hand, INF, 0, Move::NONE);
            return;
        }

        // (2) 千日手 check (path-based hash の簡易代替)
        // - Win: 連続王手の千日手で先手勝ち → 詰み
        // - Lose / Draw: 不詰
        // - Superior / Inferior: 探索継続 (None と同様)
        match pos.repetition_state(depth as i32) {
            RepetitionState::Win => {
                self.tt.store(key, hand, 0, INF, Move::NONE);
                return;
            }
            RepetitionState::Lose | RepetitionState::Draw => {
                self.tt.store(key, hand, INF, 0, Move::NONE);
                return;
            }
            _ => {}
        }

        // TT check: 既に解決済 (pn=0 or dn=0) なら return、 進行中の値で limit 超えなら return
        if let Some(entry) = self.tt.probe(key, hand) {
            if entry.pn == 0 || entry.dn == 0 {
                return;
            }
            if entry.pn >= pn_limit || entry.dn >= dn_limit {
                return;
            }
        }

        // 深さ制限: child 未展開で return
        if depth >= limits.max_depth {
            self.tt.store(key, hand, 1, 1, Move::NONE);
            return;
        }

        // mate1ply ショートカット (OR-node のみ、 王手中でない)
        if OR_NODE
            && !pos.in_check()
            && let Some(mv) = mate_1ply(pos)
        {
            self.tt.store(key, hand, 0, INF, mv);
            return;
        }

        // 王手中の OR-node は動けない (受け手番ではない攻め手)
        if OR_NODE && pos.in_check() {
            self.tt.store(key, hand, INF, 0, Move::NONE);
            return;
        }

        // 子手生成
        let mut children = ChildSlots::new();
        if OR_NODE {
            // 王手手のみ
            let mut buf = ExtMoveBuffer::new();
            let n = generate_with_type(pos, GenType::ChecksAll, &mut buf, None);
            for i in 0..n {
                let mv = buf.get(i).mv;
                if pos.is_legal(mv) && children.count < MAX_MOVES {
                    children.moves[children.count] = mv;
                    children.count += 1;
                }
            }
        } else {
            // 全合法手 (応手)
            let mut ml = MoveList::new();
            generate_legal(pos, &mut ml);
            for i in 0..ml.len() {
                if children.count >= MAX_MOVES {
                    break;
                }
                children.moves[children.count] = ml.at(i);
                children.count += 1;
            }
        }

        // 子なし
        if children.count == 0 {
            if OR_NODE {
                // 王手手なし → 不詰
                self.tt.store(key, hand, INF, 0, Move::NONE);
            } else {
                // 応手なし → 詰み
                self.tt.store(key, hand, 0, INF, Move::NONE);
            }
            return;
        }

        // 反復深化ループ
        loop {
            if self.should_stop(limits) {
                return;
            }

            // 各子の pn/dn を取得 (TT 経由)
            for i in 0..children.count {
                let mv = children.moves[i];
                let gives_check = pos.gives_check(mv);
                let _ = pos.do_move(mv, gives_check);
                let child_key = pos.key();
                let child_hand = pos.hand(pos.side_to_move());
                if let Some(entry) = self.tt.probe(child_key, child_hand) {
                    children.pns[i] = entry.pn;
                    children.dns[i] = entry.dn;
                } else if OR_NODE && check_mate_3ply(pos) {
                    // (3) mate_3ply ショートカット
                    // OR-node child は AND-node。 全 evasion で 1-ply 詰みなら、
                    // この child は実質 3 ply 詰みの中間 AND-node = pn=0 (詰み済)
                    children.pns[i] = 0;
                    children.dns[i] = INF;
                    self.tt.store(child_key, child_hand, 0, INF, Move::NONE);
                } else {
                    children.pns[i] = 1;
                    children.dns[i] = 1;
                }
                pos.undo_move(mv);
            }

            // 集約
            let mut best_idx = 0usize;
            let mut second_idx = 0usize;
            let agg_pn: u32;
            let agg_dn: u32;
            if OR_NODE {
                // pn = min(child.pn)、 dn = sum(child.dn)
                let mut min_pn = INF;
                let mut second_pn = INF;
                let mut sum_dn: u32 = 0;
                for i in 0..children.count {
                    let p = children.pns[i];
                    if p < min_pn {
                        second_pn = min_pn;
                        second_idx = best_idx;
                        min_pn = p;
                        best_idx = i;
                    } else if p < second_pn {
                        second_pn = p;
                        second_idx = i;
                    }
                    sum_dn = sum_dn.saturating_add(children.dns[i]);
                }
                agg_pn = min_pn;
                agg_dn = sum_dn.min(INF);
                let _ = second_idx; // for debug
                // 終了判定
                if min_pn == 0 {
                    self.tt.store(key, hand, 0, INF, children.moves[best_idx]);
                    return;
                }
                if sum_dn == 0 {
                    self.tt.store(key, hand, INF, 0, children.moves[best_idx]);
                    return;
                }
                if min_pn >= pn_limit || agg_dn >= dn_limit {
                    self.tt.store(key, hand, agg_pn, agg_dn, children.moves[best_idx]);
                    return;
                }
                // 子の threshold (threshold backup、 dlshogi style)
                let child_pn_limit = pn_limit.min(second_pn.saturating_add(1));
                let child_dn_limit =
                    dn_limit.saturating_sub(sum_dn).saturating_add(children.dns[best_idx]);
                // 子展開
                let mv = children.moves[best_idx];
                let gives_check = pos.gives_check(mv);
                let _ = pos.do_move(mv, gives_check);
                self.inner_search::<false>(pos, child_pn_limit, child_dn_limit, depth + 1, limits);
                pos.undo_move(mv);
                // store 暫定値 (loop で再度更新される)
                self.tt.store(key, hand, agg_pn, agg_dn, children.moves[best_idx]);
            } else {
                // AND: pn = sum(child.pn)、 dn = min(child.dn)
                let mut sum_pn: u32 = 0;
                let mut min_dn = INF;
                let mut second_dn = INF;
                for i in 0..children.count {
                    let d = children.dns[i];
                    if d < min_dn {
                        second_dn = min_dn;
                        second_idx = best_idx;
                        min_dn = d;
                        best_idx = i;
                    } else if d < second_dn {
                        second_dn = d;
                        second_idx = i;
                    }
                    sum_pn = sum_pn.saturating_add(children.pns[i]);
                }
                let _ = second_idx;
                agg_pn = sum_pn.min(INF);
                agg_dn = min_dn;
                if min_dn == 0 {
                    self.tt.store(key, hand, INF, 0, Move::NONE);
                    return;
                }
                if sum_pn == 0 {
                    self.tt.store(key, hand, 0, INF, Move::NONE);
                    return;
                }
                if agg_pn >= pn_limit || min_dn >= dn_limit {
                    self.tt.store(key, hand, agg_pn, agg_dn, Move::NONE);
                    return;
                }
                let child_pn_limit =
                    pn_limit.saturating_sub(sum_pn).saturating_add(children.pns[best_idx]);
                let child_dn_limit = dn_limit.min(second_dn.saturating_add(1));
                let mv = children.moves[best_idx];
                let gives_check = pos.gives_check(mv);
                let _ = pos.do_move(mv, gives_check);
                self.inner_search::<true>(pos, child_pn_limit, child_dn_limit, depth + 1, limits);
                pos.undo_move(mv);
                self.tt.store(key, hand, agg_pn, agg_dn, Move::NONE);
            }
        }
    }

    /// 停止条件チェック
    #[inline]
    fn should_stop(&mut self, limits: &DfpnLimits) -> bool {
        self.node_count = self.node_count.saturating_add(1);
        self.node_count > limits.max_nodes
    }
}

impl DfpnEngine for BasicDfpnEngine {
    fn search(&mut self, pos: &mut Position, limits: &DfpnLimits) -> DfpnResult {
        self.node_count = 0;
        self.tt.new_search();

        // ルートは OR-node (攻め番)
        self.inner_search::<true>(pos, INF, INF, 0, limits);

        let key = pos.key();
        let hand = pos.hand(pos.side_to_move());
        if let Some(entry) = self.tt.probe(key, hand) {
            if entry.pn == 0 {
                return DfpnResult::ProvenMate {
                    best_move: entry.best_move,
                    proof_ply: 0,
                    nodes: self.node_count,
                };
            }
            if entry.dn == 0 {
                return DfpnResult::ProvenNoMate {
                    nodes: self.node_count,
                };
            }
        }

        let reason = if self.node_count >= limits.max_nodes {
            super::DfpnUnknownReason::NodeLimit
        } else {
            super::DfpnUnknownReason::DepthLimit
        };
        DfpnResult::Unknown {
            nodes: self.node_count,
            reason,
        }
    }

    fn clear_tt(&mut self) {
        self.tt.clear();
    }
}
