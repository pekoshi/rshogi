/// PUCT-MCTS 探索パラメータ
#[derive(Debug, Clone)]
pub struct PuctConfig {
    /// 探索定数 (PUCT の c_puct)
    pub c_puct: f32,
    /// 最大子ノード数
    pub top_k: usize,
    /// ノード総数上限
    pub arena_capacity: u32,
    /// ルートノイズ Dirichlet α
    pub dirichlet_alpha: f32,
    /// ノイズ混合率 ε
    pub dirichlet_epsilon: f32,
    /// Virtual loss 値
    pub virtual_loss: f32,

    /// FPU reduction (dlshogi 流): 未訪問子の Q を parent_q - c_fpu * sqrt(visited_nnrate) で推定
    /// 0.0 で従来動作 (未訪問 Q=0)
    pub c_fpu: f32,
    /// FPU reduction (root): root では通常 0.0 (full exploration)
    pub c_fpu_root: f32,
    /// バッチサイズ
    pub batch_size: u16,
    /// バッチ評価を使用するか
    pub use_batching: bool,
    /// MateProbe を有効にするか
    pub mate_probe_enabled: bool,
    /// df-pn を有効にするか
    pub dfpn_enabled: bool,
    /// df-pn 置換表サイズ (MB)
    pub dfpn_tt_mb: usize,
    /// TacticalCache エントリ数
    pub tactical_cache_entries: usize,

    // ── Gumbel AlphaZero (Danihelka et al, ICLR 2022) ──
    /// Gumbel AZ を使う (false なら通常 PUCT + visit-count argmax)
    pub use_gumbel: bool,
    /// 初期候補数 m (sequential halving)。 root 子の上位 m を候補とする。 通常 16
    pub gumbel_m_initial: usize,
    /// sigma transform の c_visit (論文推奨 50.0)
    pub gumbel_c_visit: f32,
    /// sigma transform の c_scale (論文推奨 1.0)
    pub gumbel_c_scale: f32,

    // ── MCTS NN eval cache (transposition で同一局面の再評価を回避) ──
    /// NN 評価キャッシュ有効化 (false なら毎回 NN forward)
    pub use_eval_cache: bool,
    /// キャッシュエントリ数 (power-of-2 に切り上げ)。 65536 ≈ 512 KB、 1M で 8 MB
    pub eval_cache_entries: usize,

    // ── Tactical PUCT Stage 3: risk-aware backup ──
    /// risk-aware backup 有効化 (Q を Q_avg - λ·variance に補正)
    pub enable_risk_backup: bool,
    /// variance penalty 係数 (推奨 0.1〜0.3、 大きいほど分散の大きい手を回避)
    pub risk_variance_weight: f32,

    // ── Tactical PUCT Stage 2: tactical bonus + risk penalty ──
    /// tactical PUCT 有効化 (false なら通常 PUCT score)
    pub enable_tactical_puct: bool,
    /// tactical 各種重み (Default で標準値)
    pub tactical_check_bonus: f32,
    pub tactical_capture_bonus: f32,
    pub tactical_promotion_bonus: f32,
    pub tactical_drop_bonus: f32,
    pub tactical_king_attack_bonus: f32,
    pub tactical_see_bad_penalty: f32,

    // ── Tactical PUCT Stage 1: phase-aware ──
    /// phase-aware MCTS 有効化 (false なら通常 PUCT)
    pub phase_aware_enabled: bool,
    /// 終盤で MCTS を skip して αβ 単独へ (true で skip、 SearchController の routing と整合)
    pub skip_mcts_in_endgame: bool,
    /// phase 別 c_puct (Opening は policy prior 強め、 Late Middle は Q 重視で小さく)
    pub c_puct_opening: f32,
    pub c_puct_middlegame: f32,
    pub c_puct_late_middlegame: f32,
    /// phase 別 top_k (Opening 多、 Late Middle 少)
    pub opening_topk: usize,
    pub middlegame_topk: usize,
    pub late_middlegame_topk: usize,
    /// phase 別 αβ correction weight (hybrid.rs ab_weight を phase で上書き)
    pub ab_weight_opening: f32,
    pub ab_weight_middlegame: f32,
    pub ab_weight_late_middlegame: f32,

    // ── 中盤後半 MCTS+αβ ハイブリッド (crate::hybrid::HybridConfig と同期) ──
    /// ハイブリッド検証 ON/OFF (中盤後半フェーズでのみ発動)
    pub use_hybrid: bool,
    /// TopK 候補数
    pub hybrid_topk: usize,
    /// 1 候補あたりの αβ probe 時間 (ms)
    pub hybrid_probe_ms: u64,
    /// hybrid 詳細ログ
    pub hybrid_log: bool,
    /// MCTS value 補正 ON/OFF
    pub hybrid_value_correction: bool,
    /// 補正率 (0.0 = 補正なし、 1.0 = αβ 完全採用)
    pub hybrid_correction_rate: f32,
    /// blunder guard ON/OFF
    pub hybrid_blunder_guard: bool,
    /// good move rescue ON/OFF
    pub hybrid_good_move_rescue: bool,
    /// final_score の MCTS 重み
    pub hybrid_mcts_weight: f32,
    /// final_score の αβ 重み
    pub hybrid_ab_weight: f32,
    /// final_score の policy 重み (将来拡張、 default 0.05)
    pub hybrid_policy_weight: f32,
    /// final_score の visit 重み (将来拡張、 default 0.05)
    pub hybrid_visit_weight: f32,
}

impl Default for PuctConfig {
    fn default() -> Self {
        Self {
            c_puct: 2.5,
            top_k: 32,
            arena_capacity: 1_048_576,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            virtual_loss: 3.0,
            c_fpu: 0.27,     // dlshogi default
            c_fpu_root: 0.0, // dlshogi: root は full exploration
            batch_size: 64,
            use_batching: true,
            mate_probe_enabled: true,
            dfpn_enabled: false,
            dfpn_tt_mb: 4,
            tactical_cache_entries: 65536,

            // Tactical PUCT Stage 3: risk-aware backup (default 無効)
            enable_risk_backup: false,
            risk_variance_weight: 0.20,

            // Tactical PUCT Stage 2: tactical bonus / risk penalty (default 無効)
            enable_tactical_puct: false,
            tactical_check_bonus: 0.10,
            tactical_capture_bonus: 0.06,
            tactical_promotion_bonus: 0.05,
            tactical_drop_bonus: 0.03,
            tactical_king_attack_bonus: 0.05,
            tactical_see_bad_penalty: 0.20,

            // Tactical PUCT Stage 1: phase-aware (default 無効)
            phase_aware_enabled: false,
            skip_mcts_in_endgame: true, // SearchController の routing と整合
            c_puct_opening: 3.0,
            c_puct_middlegame: 2.5, // 既存 c_puct と同じ default
            c_puct_late_middlegame: 2.0,
            opening_topk: 24,
            middlegame_topk: 16,
            late_middlegame_topk: 8,
            ab_weight_opening: 0.15,
            ab_weight_middlegame: 0.35,
            ab_weight_late_middlegame: 0.70,

            // Gumbel AZ デフォルト無効 (段階移行のため、 検証後に true 切替)
            use_gumbel: false,
            gumbel_m_initial: 16,
            gumbel_c_visit: 50.0,
            gumbel_c_scale: 1.0,

            // MCTS eval cache: デフォルト無効 (NPS bench 後に true 切替予定)
            // 65536 entries ≈ 512 KB、 NN forward を ~30-40% スキップ可能
            use_eval_cache: false,
            eval_cache_entries: 65_536,

            // ハイブリッド: デフォルト無効 (検証後に true 切替)
            use_hybrid: false,
            hybrid_topk: 6,
            hybrid_probe_ms: 20,
            hybrid_log: false,
            hybrid_value_correction: true,
            hybrid_correction_rate: 0.35,
            hybrid_blunder_guard: true,
            hybrid_good_move_rescue: true,
            hybrid_mcts_weight: 0.55,
            hybrid_ab_weight: 0.35,
            hybrid_policy_weight: 0.05,
            hybrid_visit_weight: 0.05,
        }
    }
}

impl PuctConfig {
    /// PuctConfig から TacticalWeights を抽出
    pub fn to_tactical_weights(&self) -> crate::tactical::TacticalWeights {
        crate::tactical::TacticalWeights {
            check_bonus: self.tactical_check_bonus,
            capture_bonus: self.tactical_capture_bonus,
            promotion_bonus: self.tactical_promotion_bonus,
            drop_bonus: self.tactical_drop_bonus,
            king_attack_bonus: self.tactical_king_attack_bonus,
            see_bad_penalty: self.tactical_see_bad_penalty,
        }
    }

    /// phase + endgame_score から c_puct を返す。 phase_aware_enabled=false なら base c_puct
    pub fn c_puct_for_phase(
        &self,
        phase: Option<crate::phase::GamePhase>,
        endgame_score: f32,
    ) -> f32 {
        if !self.phase_aware_enabled {
            return self.c_puct;
        }
        match phase {
            Some(crate::phase::GamePhase::Opening) => self.c_puct_opening,
            Some(crate::phase::GamePhase::Middlegame) => {
                if endgame_score >= crate::hybrid::MIDDLE_LATE_SCORE_LO {
                    self.c_puct_late_middlegame
                } else {
                    self.c_puct_middlegame
                }
            }
            Some(crate::phase::GamePhase::Endgame) | None => self.c_puct,
        }
    }

    /// phase + endgame_score から top_k を返す。 phase_aware_enabled=false なら base top_k
    pub fn top_k_for_phase(
        &self,
        phase: Option<crate::phase::GamePhase>,
        endgame_score: f32,
    ) -> usize {
        if !self.phase_aware_enabled {
            return self.top_k;
        }
        match phase {
            Some(crate::phase::GamePhase::Opening) => self.opening_topk,
            Some(crate::phase::GamePhase::Middlegame) => {
                if endgame_score >= crate::hybrid::MIDDLE_LATE_SCORE_LO {
                    self.late_middlegame_topk
                } else {
                    self.middlegame_topk
                }
            }
            Some(crate::phase::GamePhase::Endgame) | None => self.top_k,
        }
    }

    /// phase + endgame_score から αβ correction weight を返す
    pub fn ab_weight_for_phase(
        &self,
        phase: Option<crate::phase::GamePhase>,
        endgame_score: f32,
    ) -> f32 {
        if !self.phase_aware_enabled {
            return self.hybrid_ab_weight;
        }
        match phase {
            Some(crate::phase::GamePhase::Opening) => self.ab_weight_opening,
            Some(crate::phase::GamePhase::Middlegame) => {
                if endgame_score >= crate::hybrid::MIDDLE_LATE_SCORE_LO {
                    self.ab_weight_late_middlegame
                } else {
                    self.ab_weight_middlegame
                }
            }
            Some(crate::phase::GamePhase::Endgame) | None => self.hybrid_ab_weight,
        }
    }

    /// PuctConfig から HybridConfig を抽出
    pub fn to_hybrid_config(&self) -> crate::hybrid::HybridConfig {
        crate::hybrid::HybridConfig {
            enabled: self.use_hybrid,
            topk: self.hybrid_topk,
            probe_ms: self.hybrid_probe_ms,
            log_enabled: self.hybrid_log,
            correction_enabled: self.hybrid_value_correction,
            correction_rate: self.hybrid_correction_rate,
            blunder_guard: self.hybrid_blunder_guard,
            good_move_rescue: self.hybrid_good_move_rescue,
            mcts_weight: self.hybrid_mcts_weight,
            ab_weight: self.hybrid_ab_weight,
            policy_weight: self.hybrid_policy_weight,
            visit_weight: self.hybrid_visit_weight,
        }
    }
}
