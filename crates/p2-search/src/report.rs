//! 探索レポート（統計情報）

use crate::phase::{EndgameFeatures, GamePhase, SearchMode};

/// 探索レポート
#[derive(Debug, Clone)]
pub struct SearchReport {
    /// 現在のフェーズ
    pub phase: GamePhase,
    /// 探索モード
    pub search_mode: SearchMode,
    /// 終盤スコア [0,1]
    pub endgame_score: f32,
    /// 終盤特徴量
    pub features: EndgameFeatures,
    /// 詰み探索統計
    pub mate_probes: MateProbeStats,
    /// df-pn 統計
    pub dfpn_stats: DfpnStats,
    /// バッチ評価統計
    pub batch_stats: BatchStats,
    /// キャッシュ統計
    pub cache_stats: CacheStats,
}

/// 詰み探索統計
#[derive(Debug, Clone, Default)]
pub struct MateProbeStats {
    pub probes_1ply: u32,
    pub found_1ply: u32,
    pub probes_3ply: u32,
    pub found_3ply: u32,
    pub probes_5ply: u32,
    pub found_5ply: u32,
    pub probes_7ply: u32,
    pub found_7ply: u32,
}

/// df-pn 統計
#[derive(Debug, Clone, Default)]
pub struct DfpnStats {
    pub invocations: u32,
    pub proven_mates: u32,
    pub proven_no_mates: u32,
    pub unknowns: u32,
    pub total_nodes: u64,
}

/// バッチ評価統計
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub total_batches: u32,
    pub total_evals: u64,
    pub avg_batch_size: f32,
}

/// キャッシュ統計
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u32,
    pub misses: u32,
    pub stores: u32,
}

impl SearchReport {
    /// 空のレポートを作成
    pub fn empty() -> Self {
        Self {
            phase: GamePhase::Opening,
            search_mode: SearchMode::Normal,
            endgame_score: 0.0,
            features: EndgameFeatures {
                material_phase: 0.0,
                king_distance: 0.0,
                mobility: 0.0,
                attack_pressure: 0.0,
                eval_magnitude: 0.0,
                tt_cutoff_rate: 0.0,
                check_frequency: 0.0,
                mcts_check_bias: 0.0,
                engagement: 0.0,
                promotion_progress: 0.0,
            },
            mate_probes: MateProbeStats::default(),
            dfpn_stats: DfpnStats::default(),
            batch_stats: BatchStats::default(),
            cache_stats: CacheStats::default(),
        }
    }
}
