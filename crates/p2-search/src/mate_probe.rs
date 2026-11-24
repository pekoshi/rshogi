//! 詰み探索プローブ (1-7 ply)
//!
//! MCTS ノード評価時に短手数の詰みを検出し、
//! 正確な詰み値を返す。SearchMode によるゲート付き。

use rshogi_core::movegen::{ExtMoveBuffer, GenType, MoveList, generate_legal, generate_with_type};
use rshogi_core::position::Position;
use rshogi_core::types::Move;

use crate::phase::SearchMode;
use crate::tactical_cache::TacticalCache;

/// 詰み探索の制限パラメータ
#[derive(Debug, Clone)]
pub struct MateProbeLimits {
    /// 最大探索 ply
    pub max_ply: u8,
    /// 最大ノード数
    pub max_nodes: u32,
    /// ルートノードのみで実行するか
    pub root_only: bool,
    /// 最多訪問ノードのみで実行するか
    pub top_visit_only: bool,
}

impl Default for MateProbeLimits {
    fn default() -> Self {
        Self {
            max_ply: 3,
            max_nodes: 10_000,
            root_only: false,
            top_visit_only: false,
        }
    }
}

/// 詰み探索結果
#[derive(Debug, Clone)]
pub enum MateResult {
    /// 詰みなし
    NoMate,
    /// 詰み発見
    MateFound {
        best_move: Move,
        ply: u8,
        nodes: u32,
    },
    /// 探索打ち切り（制限超過）
    Unknown { reason: MateUnknownReason },
}

/// 探索打ち切り理由
#[derive(Debug, Clone, Copy)]
pub enum MateUnknownReason {
    NodeLimit,
    DepthLimit,
}

/// 1手詰め探索（既存 mate_1ply ラッパー）
pub fn mate_probe_1ply(pos: &mut Position) -> MateResult {
    match rshogi_core::mate::mate_1ply(pos) {
        Some(mv) => MateResult::MateFound {
            best_move: mv,
            ply: 1,
            nodes: 1,
        },
        None => MateResult::NoMate,
    }
}

/// 3手詰め探索
///
/// 王手手を生成し、各王手について相手の全応手で mate_1ply を確認。
/// 全応手が1手詰めなら3手詰め。
pub fn mate_probe_3ply(pos: &mut Position, max_nodes: u32) -> MateResult {
    if pos.in_check() {
        return MateResult::NoMate;
    }

    let mut nodes: u32 = 0;

    // 王手手を生成
    let mut check_buf = ExtMoveBuffer::new();
    let check_count = generate_with_type(pos, GenType::ChecksAll, &mut check_buf, None);

    for i in 0..check_count {
        let mv = check_buf.get(i).mv;

        // 合法性チェック
        if !pos.is_legal(mv) {
            continue;
        }

        nodes += 1;
        if nodes > max_nodes {
            return MateResult::Unknown {
                reason: MateUnknownReason::NodeLimit,
            };
        }

        let gives_check = true; // 王手手なので必ず check
        let _ = pos.do_move(mv, gives_check);

        // 相手の全応手で mate_1ply を確認
        let mut all_mate = true;
        let mut evasion_moves = MoveList::new();
        generate_legal(pos, &mut evasion_moves);

        if evasion_moves.is_empty() {
            // 応手なし = 既に詰み（1手詰め）
            pos.undo_move(mv);
            return MateResult::MateFound {
                best_move: mv,
                ply: 1,
                nodes,
            };
        }

        for j in 0..evasion_moves.len() {
            let resp = evasion_moves.at(j);
            nodes += 1;
            if nodes > max_nodes {
                pos.undo_move(mv);
                return MateResult::Unknown {
                    reason: MateUnknownReason::NodeLimit,
                };
            }

            let resp_check = pos.gives_check(resp);
            let _ = pos.do_move(resp, resp_check);

            // この局面で mate_1ply があるか
            let mate = rshogi_core::mate::mate_1ply(pos);
            pos.undo_move(resp);

            if mate.is_none() {
                all_mate = false;
                break;
            }
        }

        pos.undo_move(mv);

        if all_mate {
            return MateResult::MateFound {
                best_move: mv,
                ply: 3,
                nodes,
            };
        }
    }

    MateResult::NoMate
}

/// 再帰的詰み探索（5-7 ply 用）
///
/// 攻め手（OR ノード）: 王手手のいずれかで詰めば詰み
/// 受け手（AND ノード）: 全応手で詰まなければ不詰
fn mate_search_recursive(
    pos: &mut Position,
    remaining_ply: u8,
    nodes: &mut u32,
    max_nodes: u32,
    is_attacker: bool,
) -> Option<Move> {
    if *nodes > max_nodes {
        return None;
    }

    if remaining_ply == 0 {
        return None;
    }

    if is_attacker {
        // OR ノード: 王手手を生成して各手を試す
        if pos.in_check() {
            return None;
        }

        // 1手詰めチェック
        if remaining_ply >= 1
            && let Some(mv) = rshogi_core::mate::mate_1ply(pos)
        {
            return Some(mv);
        }

        if remaining_ply < 3 {
            return None;
        }

        let mut check_buf = ExtMoveBuffer::new();
        let check_count = generate_with_type(pos, GenType::ChecksAll, &mut check_buf, None);

        for i in 0..check_count {
            let mv = check_buf.get(i).mv;
            if !pos.is_legal(mv) {
                continue;
            }

            *nodes += 1;
            if *nodes > max_nodes {
                return None;
            }

            let _ = pos.do_move(mv, true);

            // 受け手（AND ノード）を再帰
            let defended = mate_search_recursive(pos, remaining_ply - 1, nodes, max_nodes, false);
            pos.undo_move(mv);

            if defended.is_some() {
                return Some(mv);
            }
        }

        None
    } else {
        // AND ノード: 全応手で詰みを回避できるか
        let mut evasion_moves = MoveList::new();
        generate_legal(pos, &mut evasion_moves);

        if evasion_moves.is_empty() {
            // 応手なし = 詰み
            return Some(Move::NONE); // sentinel: 詰み確認
        }

        for j in 0..evasion_moves.len() {
            let resp = evasion_moves.at(j);
            *nodes += 1;
            if *nodes > max_nodes {
                return None;
            }

            let resp_check = pos.gives_check(resp);
            let _ = pos.do_move(resp, resp_check);

            let result = mate_search_recursive(pos, remaining_ply - 1, nodes, max_nodes, true);
            pos.undo_move(resp);

            result?;
        }

        // 全応手で詰み → 詰み確認
        Some(Move::NONE)
    }
}

/// 5手詰め探索
pub fn mate_probe_5ply(pos: &mut Position, max_nodes: u32) -> MateResult {
    if pos.in_check() {
        return MateResult::NoMate;
    }

    let mut nodes: u32 = 0;
    match mate_search_recursive(pos, 5, &mut nodes, max_nodes, true) {
        Some(mv) if mv != Move::NONE => MateResult::MateFound {
            best_move: mv,
            ply: 5,
            nodes,
        },
        Some(_) => MateResult::NoMate, // sentinel case
        None => {
            if nodes > max_nodes {
                MateResult::Unknown {
                    reason: MateUnknownReason::NodeLimit,
                }
            } else {
                MateResult::NoMate
            }
        }
    }
}

/// 7手詰め探索
pub fn mate_probe_7ply(pos: &mut Position, max_nodes: u32) -> MateResult {
    if pos.in_check() {
        return MateResult::NoMate;
    }

    let mut nodes: u32 = 0;
    match mate_search_recursive(pos, 7, &mut nodes, max_nodes, true) {
        Some(mv) if mv != Move::NONE => MateResult::MateFound {
            best_move: mv,
            ply: 7,
            nodes,
        },
        Some(_) => MateResult::NoMate,
        None => {
            if nodes > max_nodes {
                MateResult::Unknown {
                    reason: MateUnknownReason::NodeLimit,
                }
            } else {
                MateResult::NoMate
            }
        }
    }
}

/// 統合プローブ関数
///
/// SearchMode に応じて適切な深さの詰み探索を実行する。
/// - Level 1 (1ply): 常時、root のみ
/// - Level 2 (3ply): Endgame 以上、root/top_visit
/// - Level 3 (5ply): MateThreat 以上、top_visit のみ
/// - Level 4 (7ply): ForcedMate のみ（デフォルト OFF）
pub fn mate_probe(
    pos: &mut Position,
    mode: SearchMode,
    cache: Option<&mut TacticalCache>,
    is_root: bool,
    is_top_visit: bool,
) -> Option<f32> {
    // キャッシュチェック
    if let Some(cache) = cache.as_ref() {
        let key = pos.key();
        if let Some(entry) = cache.probe(key) {
            if entry.kind == TacticalCache::KIND_MATE {
                // 詰み発見済み → 1.0（相手視点で-1.0 は backup 側で処理）
                return Some(1.0);
            } else if entry.kind == TacticalCache::KIND_NO_MATE {
                return None;
            }
        }
    }

    // Level 1: 1手詰め（常時、root のみ）
    if is_root && let MateResult::MateFound { .. } = mate_probe_1ply(pos) {
        if let Some(cache) = cache {
            cache.store(pos.key(), TacticalCache::KIND_MATE, 1, Move::NONE);
        }
        return Some(1.0);
    }

    // Level 2: 3手詰め（Endgame 以上、root/top_visit）
    if mode >= SearchMode::Endgame
        && (is_root || is_top_visit)
        && let MateResult::MateFound { best_move, .. } = mate_probe_3ply(pos, 1000)
    {
        if let Some(cache) = cache {
            cache.store(pos.key(), TacticalCache::KIND_MATE, 3, best_move);
        }
        return Some(1.0);
    }

    // Level 3: 5手詰め（MateThreat 以上、top_visit のみ）
    if mode >= SearchMode::MateThreat
        && is_top_visit
        && let MateResult::MateFound { best_move, .. } = mate_probe_5ply(pos, 5000)
    {
        if let Some(cache) = cache {
            cache.store(pos.key(), TacticalCache::KIND_MATE, 5, best_move);
        }
        return Some(1.0);
    }

    // Level 4: 7手詰め（ForcedMate のみ、デフォルト OFF）
    if mode == SearchMode::ForcedMate
        && is_top_visit
        && let MateResult::MateFound { best_move, .. } = mate_probe_7ply(pos, 10_000)
    {
        if let Some(cache) = cache {
            cache.store(pos.key(), TacticalCache::KIND_MATE, 7, best_move);
        }
        return Some(1.0);
    }

    None
}
