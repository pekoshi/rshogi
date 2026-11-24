//! `mate_dfpn` — alpha_beta 内から呼び出す軽量 df-pn (= 詰将棋専用) エンジン
//!
//! ## 経緯
//!
//! - 以前は `p2-search::dfpn::basic::BasicDfpnEngine` が dfpn 実装
//!   (= MateSolverThread 経由で別 thread から呼ばれていた)。
//! - rshogi-core → p2-search への依存は循環依存のため、 alpha_beta 内 (= search 内)
//!   から呼ぶには rshogi-core 内に同等品を持つ必要がある。
//! - 本 module は p2-search の `BasicDfpnEngine` を **rshogi-core 内に duplicate** し、
//!   alpha_beta 内呼び出し用に小さな entry point (`MateDfpnEngine::probe`) を提供する。
//!
//! ## アルゴリズム
//!
//! dlshogi 流の OR/AND df-pn:
//! - OR ノード (= 攻め手): 王手のみ生成、 `pn = min(child.pn)`、 `dn = sum(child.dn)`
//! - AND ノード (= 受け手): 全合法手生成、 `pn = sum(child.pn)`、 `dn = min(child.dn)`
//! - mate_1ply ショートカット + proof piece dominance + threshold backup
//!
//! ## alpha_beta 内呼び出し condition (YO 風)
//!
//! `alpha_beta.rs` の Step 12 ProbCut 直後、 moves_loop 直前で:
//!
//! ```text
//! !in_check && excluded_move.is_none() && pv_node
//!     && depth >= 8 && pos.game_ply() >= 25
//! ```
//!
//! を満たす場合のみ `MateDfpnEngine::probe(pos, budget_nodes, max_depth)` を呼ぶ。
//! budget_nodes は depth に応じて 1000 * depth 程度を default とする。

use crate::mate::mate_1ply;
use crate::movegen::{ExtMoveBuffer, GenType, MoveList, generate_legal, generate_with_type};
use crate::position::Position;
use crate::types::{Hand, Move, RepetitionState};

/// dlshogi 流の pn/dn 上限値
const INF: u32 = 100_000_000;

/// 1 ノードで生成しうる手の上限 (= 将棋の最大合法手数は 593、 余裕を持って 600)
const MAX_MOVES: usize = 600;

// =============================================================================
// 専用 TT (= dlshogi 流 cluster + proof piece dominance)
// =============================================================================

/// df-pn 専用 TT entry
#[derive(Clone, Copy)]
struct DfpnTtEntry {
    hash_high: u32,
    hand: Hand,
    pn: u32,
    dn: u32,
    best_move: Move,
    generation: u8,
    _pad: [u8; 3],
}

impl DfpnTtEntry {
    const EMPTY: Self = Self {
        hash_high: 0,
        hand: Hand::EMPTY,
        pn: 1,
        dn: 1,
        best_move: Move::NONE,
        generation: 0,
        _pad: [0; 3],
    };

    #[inline]
    const fn is_empty(&self) -> bool {
        self.pn == 1 && self.dn == 1 && self.hash_high == 0
    }
}

/// 1 cluster あたりの entry 数
const CLUSTER_SIZE: usize = 4;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct DfpnCluster {
    entries: [DfpnTtEntry; CLUSTER_SIZE],
}

impl DfpnCluster {
    const EMPTY: Self = Self {
        entries: [DfpnTtEntry::EMPTY; CLUSTER_SIZE],
    };
}

/// df-pn 専用 TT
struct DfpnTranspositionTable {
    clusters: Box<[DfpnCluster]>,
    mask: usize,
    generation: u8,
}

impl DfpnTranspositionTable {
    fn new(size_mb: usize) -> Self {
        let cluster_size = std::mem::size_of::<DfpnCluster>();
        let num_clusters = ((size_mb * 1024 * 1024) / cluster_size).next_power_of_two().max(16);
        let clusters = vec![DfpnCluster::EMPTY; num_clusters].into_boxed_slice();
        Self {
            clusters,
            mask: num_clusters - 1,
            generation: 0,
        }
    }

    fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.generation = 1;
        }
    }

    fn probe(&self, key: u64, hand: Hand) -> Option<&DfpnTtEntry> {
        let hash_high = (key >> 32) as u32;
        let cluster_idx = (key as usize) & self.mask;
        let cluster = &self.clusters[cluster_idx];
        cluster
            .entries
            .iter()
            .find(|&entry| entry.hash_high == hash_high && entry.hand == hand)
    }

    fn probe_proof_dominance(&self, key: u64, hand: Hand) -> Option<&DfpnTtEntry> {
        let hash_high = (key >> 32) as u32;
        let cluster_idx = (key as usize) & self.mask;
        let cluster = &self.clusters[cluster_idx];
        cluster.entries.iter().find(|&entry| {
            entry.hash_high == hash_high && entry.pn == 0 && hand.is_superior_or_equal(entry.hand)
        })
    }

    fn probe_disproof_dominance(&self, key: u64, hand: Hand) -> Option<&DfpnTtEntry> {
        let hash_high = (key >> 32) as u32;
        let cluster_idx = (key as usize) & self.mask;
        let cluster = &self.clusters[cluster_idx];
        cluster.entries.iter().find(|&entry| {
            entry.hash_high == hash_high && entry.dn == 0 && entry.hand.is_superior_or_equal(hand)
        })
    }

    fn store(&mut self, key: u64, hand: Hand, pn: u32, dn: u32, best_move: Move) {
        let hash_high = (key >> 32) as u32;
        let cluster_idx = (key as usize) & self.mask;
        let generation = self.generation;
        let cluster = &mut self.clusters[cluster_idx];

        for entry in cluster.entries.iter_mut() {
            if entry.hash_high == hash_high && entry.hand == hand {
                entry.pn = pn;
                entry.dn = dn;
                entry.best_move = best_move;
                entry.generation = generation;
                return;
            }
        }

        let mut victim_idx = 0;
        let mut victim_score = i32::MAX;
        for (i, entry) in cluster.entries.iter().enumerate() {
            if entry.is_empty() {
                victim_idx = i;
                break;
            }
            let age = generation.wrapping_sub(entry.generation) as i32;
            let preserve = if entry.pn == 0 || entry.dn == 0 {
                -100
            } else {
                0
            };
            let score = -(age + preserve);
            if score < victim_score {
                victim_score = score;
                victim_idx = i;
            }
        }
        cluster.entries[victim_idx] = DfpnTtEntry {
            hash_high,
            hand,
            pn,
            dn,
            best_move,
            generation,
            _pad: [0; 3],
        };
    }
}

// =============================================================================
// 子ノードバッファ
// =============================================================================

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

/// mate_3ply ショートカット (= AND-node 位置で 全 evasion → 1-ply 詰みなら詰み確定)
fn check_mate_3ply(pos: &mut Position) -> bool {
    if !pos.in_check() {
        return false;
    }
    let mut ml = MoveList::new();
    generate_legal(pos, &mut ml);
    if ml.is_empty() {
        return true;
    }
    for i in 0..ml.len() {
        let mv = ml.at(i);
        let gives_check = pos.gives_check(mv);
        let _ = pos.do_move(mv, gives_check);
        let mate_one = mate_1ply(pos);
        pos.undo_move(mv);
        if mate_one.is_none() {
            return false;
        }
    }
    true
}

// =============================================================================
// df-pn engine 本体
// =============================================================================

/// alpha_beta 内呼び出し用 df-pn engine
pub struct MateDfpnEngine {
    tt: DfpnTranspositionTable,
    node_count: u32,
    max_nodes: u32,
    max_depth: u16,
}

/// df-pn 探索結果
#[derive(Debug, Clone, Copy)]
pub enum MateDfpnResult {
    /// 詰み証明 (= best_move 付き)
    Mate { best_move: Move },
    /// 不詰証明
    NoMate,
    /// node/depth budget 切れ
    Unknown,
}

impl MateDfpnEngine {
    /// 指定 TT サイズ (MB) で作成
    pub fn new(tt_size_mb: usize) -> Self {
        Self {
            tt: DfpnTranspositionTable::new(tt_size_mb),
            node_count: 0,
            max_nodes: 0,
            max_depth: 0,
        }
    }

    /// 現在局面 (= OR-node、 攻め番) から詰みを探索する。
    ///
    /// - `max_nodes`: df-pn が消費する最大 node 数 (= budget)
    /// - `max_depth`: 探索の最大深さ
    pub fn probe(&mut self, pos: &mut Position, max_nodes: u32, max_depth: u16) -> MateDfpnResult {
        self.node_count = 0;
        self.max_nodes = max_nodes;
        self.max_depth = max_depth;
        self.tt.new_search();

        self.inner_search::<true>(pos, INF, INF, 0);

        let key = pos.key();
        let hand = pos.hand(pos.side_to_move());
        if let Some(entry) = self.tt.probe(key, hand) {
            if entry.pn == 0 {
                return MateDfpnResult::Mate {
                    best_move: entry.best_move,
                };
            }
            if entry.dn == 0 {
                return MateDfpnResult::NoMate;
            }
        }
        MateDfpnResult::Unknown
    }

    #[inline]
    fn should_stop(&mut self) -> bool {
        self.node_count = self.node_count.saturating_add(1);
        self.node_count > self.max_nodes
    }

    fn inner_search<const OR_NODE: bool>(
        &mut self,
        pos: &mut Position,
        pn_limit: u32,
        dn_limit: u32,
        depth: u16,
    ) {
        if self.should_stop() {
            return;
        }

        let key = pos.key();
        let hand = pos.hand(pos.side_to_move());

        // Proof piece dominance
        if OR_NODE && let Some(dom) = self.tt.probe_proof_dominance(key, hand) {
            let bm = dom.best_move;
            self.tt.store(key, hand, 0, INF, bm);
            return;
        }
        if !OR_NODE && self.tt.probe_disproof_dominance(key, hand).is_some() {
            self.tt.store(key, hand, INF, 0, Move::NONE);
            return;
        }

        // 千日手 check
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

        // TT check
        if let Some(entry) = self.tt.probe(key, hand) {
            if entry.pn == 0 || entry.dn == 0 {
                return;
            }
            if entry.pn >= pn_limit || entry.dn >= dn_limit {
                return;
            }
        }

        if depth >= self.max_depth {
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

        // 王手中の OR-node は動けない
        if OR_NODE && pos.in_check() {
            self.tt.store(key, hand, INF, 0, Move::NONE);
            return;
        }

        // 子手生成
        let mut children = ChildSlots::new();
        if OR_NODE {
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

        if children.count == 0 {
            if OR_NODE {
                self.tt.store(key, hand, INF, 0, Move::NONE);
            } else {
                self.tt.store(key, hand, 0, INF, Move::NONE);
            }
            return;
        }

        // 反復深化ループ
        loop {
            if self.should_stop() {
                return;
            }

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
                    children.pns[i] = 0;
                    children.dns[i] = INF;
                    self.tt.store(child_key, child_hand, 0, INF, Move::NONE);
                } else {
                    children.pns[i] = 1;
                    children.dns[i] = 1;
                }
                pos.undo_move(mv);
            }

            let mut best_idx = 0usize;
            let agg_pn: u32;
            let agg_dn: u32;
            if OR_NODE {
                let mut min_pn = INF;
                let mut second_pn = INF;
                let mut sum_dn: u32 = 0;
                for i in 0..children.count {
                    let p = children.pns[i];
                    if p < min_pn {
                        second_pn = min_pn;
                        min_pn = p;
                        best_idx = i;
                    } else if p < second_pn {
                        second_pn = p;
                    }
                    sum_dn = sum_dn.saturating_add(children.dns[i]);
                }
                agg_pn = min_pn;
                agg_dn = sum_dn.min(INF);
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
                let child_pn_limit = pn_limit.min(second_pn.saturating_add(1));
                let child_dn_limit =
                    dn_limit.saturating_sub(sum_dn).saturating_add(children.dns[best_idx]);
                let mv = children.moves[best_idx];
                let gives_check = pos.gives_check(mv);
                let _ = pos.do_move(mv, gives_check);
                self.inner_search::<false>(pos, child_pn_limit, child_dn_limit, depth + 1);
                pos.undo_move(mv);
                self.tt.store(key, hand, agg_pn, agg_dn, children.moves[best_idx]);
            } else {
                let mut sum_pn: u32 = 0;
                let mut min_dn = INF;
                let mut second_dn = INF;
                for i in 0..children.count {
                    let d = children.dns[i];
                    if d < min_dn {
                        second_dn = min_dn;
                        min_dn = d;
                        best_idx = i;
                    } else if d < second_dn {
                        second_dn = d;
                    }
                    sum_pn = sum_pn.saturating_add(children.pns[i]);
                }
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
                self.inner_search::<true>(pos, child_pn_limit, child_dn_limit, depth + 1);
                pos.undo_move(mv);
                self.tt.store(key, hand, agg_pn, agg_dn, Move::NONE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn engine_constructable() {
        // 1 MB TT で作れる
        let _e = MateDfpnEngine::new(1);
    }

    #[test]
    fn hirate_returns_unknown_or_nomate() {
        let mut pos = Position::new();
        pos.set_sfen(crate::position::SFEN_HIRATE).unwrap();
        let mut e = MateDfpnEngine::new(1);
        let r = e.probe(&mut pos, 5_000, 8);
        // 平手初期局面は詰みではない (= NoMate or Unknown)
        if let MateDfpnResult::Mate { .. } = r {
            panic!("hirate should not be mate")
        }
    }

    #[test]
    fn simple_mate_detected() {
        // mate_1ply test_drop_mate_gold_corner と同じ局面 (= 1 手詰み、
        // dfpn でも当然詰みとして返る)
        let mut pos = Position::new();
        pos.set_sfen("7Pk/6R2/9/9/9/9/9/9/4K4 b G 1").unwrap();
        let mut e = MateDfpnEngine::new(1);
        let r = e.probe(&mut pos, 50_000, 15);
        match r {
            MateDfpnResult::Mate { best_move } => assert!(best_move != Move::NONE),
            other => panic!("expected mate, got {:?}", other),
        }
    }
}
