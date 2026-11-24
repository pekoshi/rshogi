//! df-pn (Depth-First Proof Number) 探索エンジン
//!
//! 詰将棋の証明/反証に特化した探索アルゴリズム。
//! `SearchMode == ForcedMate` かつ `dfpn_enabled == true` の場合のみ起動。

pub mod basic;
pub mod tt;

use rshogi_core::position::Position;
use rshogi_core::types::Move;

pub use basic::BasicDfpnEngine;
pub use tt::DfpnTranspositionTable;

/// df-pn 探索の制限パラメータ
#[derive(Debug, Clone)]
pub struct DfpnLimits {
    /// 最大ノード数
    pub max_nodes: u32,
    /// 最大探索深さ
    pub max_depth: u16,
    /// TT サイズ (MB)
    pub tt_size_mb: usize,
}

impl Default for DfpnLimits {
    fn default() -> Self {
        Self {
            max_nodes: 50_000,
            max_depth: 15,
            tt_size_mb: 4,
        }
    }
}

/// df-pn 探索結果
#[derive(Debug, Clone)]
pub enum DfpnResult {
    /// 詰み証明
    ProvenMate {
        best_move: Move,
        proof_ply: u16,
        nodes: u32,
    },
    /// 不詰証明
    ProvenNoMate { nodes: u32 },
    /// 探索打ち切り
    Unknown {
        nodes: u32,
        reason: DfpnUnknownReason,
    },
}

/// 探索打ち切り理由
#[derive(Debug, Clone, Copy)]
pub enum DfpnUnknownReason {
    NodeLimit,
    DepthLimit,
}

/// df-pn エンジン trait
pub trait DfpnEngine {
    /// 詰み探索を実行
    fn search(&mut self, pos: &mut Position, limits: &DfpnLimits) -> DfpnResult;
    /// TT をクリア
    fn clear_tt(&mut self);
}
