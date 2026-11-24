use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use p2_nnue::evaluator::NnueEvaluator;
use p2_nnue::features::extract_features;
use p2_nnue::queue::{EvalRequestQueue, EvalResultQueue};
use p2_nnue::types::VALUE_SCALE;
use rand_xoshiro::Xoshiro256PlusPlus;
use rshogi_core::movegen::{MoveList, generate_legal};
use rshogi_core::position::Position;
use rshogi_core::types::Move;

use crate::batch_eval::BatchCollector;
use crate::eval_cache::MctsEvalCache;
use crate::puct::{PuctBackend, PuctLimits, PuctResult};

use super::config::PuctConfig;
use super::eval_bridge::EvalBridge;
use super::tree::MctsTree;

/// PUCT-MCTS 探索エンジン
pub struct PuctEngine {
    tree: MctsTree,
    eval_bridge: EvalBridge,
    stop_flag: Arc<AtomicBool>,
    rng: Xoshiro256PlusPlus,
    config: PuctConfig,
    /// NN 評価キャッシュ (use_eval_cache=true 時のみ Some)
    eval_cache: Option<MctsEvalCache>,
}

impl PuctEngine {
    /// Direct モードで PuctEngine を作成
    pub fn new_direct(config: PuctConfig, evaluator: Arc<dyn NnueEvaluator + Send + Sync>) -> Self {
        Self::new(config, EvalBridge::Direct(evaluator))
    }

    /// Policy-Value モード (bullet NNUE: HalfKAv2_hm + SwiGLU) で PuctEngine を作成
    ///
    /// - batch flush は P2NnueInput 前提なので off
    /// - Gumbel halving 強制有効 (少 leaf でも MCTS 質高、 dlshogi 系標準)
    pub fn new_policy_value(
        mut config: PuctConfig,
        evaluator: std::sync::Arc<p2_nnue::policy_value_net::PolicyValueEvaluator>,
    ) -> Self {
        config.use_batching = false;
        config.use_gumbel = true;
        Self::new(config, EvalBridge::PolicyValue(evaluator))
    }

    /// Policy-Value GPU モード (cudarc、 feature `pv-cuda`)
    #[cfg(feature = "pv-cuda")]
    pub fn new_policy_value_gpu(
        mut config: PuctConfig,
        evaluator: std::sync::Arc<p2_nnue::policy_value_net::PolicyValueEvaluatorGpu>,
    ) -> Self {
        config.use_batching = false;
        config.use_gumbel = true;
        Self::new(config, EvalBridge::PolicyValueGpu(evaluator))
    }

    /// Queue モードで PuctEngine を作成
    pub fn new_with_queue(
        config: PuctConfig,
        evaluator: Arc<dyn NnueEvaluator + Send + Sync>,
        request_queue: Arc<EvalRequestQueue>,
        result_queue: Arc<EvalResultQueue>,
    ) -> Self {
        Self::new(
            config,
            EvalBridge::Queued {
                evaluator,
                request_queue,
                result_queue,
            },
        )
    }

    /// EvalBridge を直接指定して PuctEngine を作成
    pub fn new(config: PuctConfig, eval_bridge: EvalBridge) -> Self {
        use rand::SeedableRng;
        let eval_cache = if config.use_eval_cache {
            Some(MctsEvalCache::new(config.eval_cache_entries))
        } else {
            None
        };
        Self {
            tree: MctsTree::new(config.clone()),
            config,
            eval_bridge,
            stop_flag: Arc::new(AtomicBool::new(false)),
            rng: Xoshiro256PlusPlus::seed_from_u64(0xDEAD_BEEF),
            eval_cache,
        }
    }

    /// eval cache の読み取り参照 (テスト・統計用)
    pub fn eval_cache(&self) -> Option<&MctsEvalCache> {
        self.eval_cache.as_ref()
    }

    /// stop フラグの共有ハンドルを取得（テスト・外部制御用）
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop_flag)
    }

    /// evaluator への参照を取得
    pub fn evaluator(&self) -> &(dyn NnueEvaluator + Send + Sync) {
        self.eval_bridge.evaluator()
    }

    /// NNUE 生評価値を [-1.0, 1.0] の勝率に変換
    fn value_to_winrate(raw: i32) -> f32 {
        (raw as f32 / VALUE_SCALE as f32).clamp(-1.0, 1.0)
    }

    /// バッチ結果を backup + (cache 有効なら) eval cache に store
    fn flush_and_backup_with_cache(
        tree: &mut MctsTree,
        batch: &mut BatchCollector,
        evaluator: &(dyn NnueEvaluator + Send + Sync),
        cache: Option<&mut MctsEvalCache>,
    ) {
        let results = batch.flush(evaluator);
        if let Some(cache) = cache {
            for result in &results {
                tree.remove_virtual_loss(&result.path);
                tree.backup(&result.path, result.value);
                if result.pos_key != 0 {
                    cache.store(result.pos_key, result.raw_value);
                }
            }
        } else {
            for result in &results {
                tree.remove_virtual_loss(&result.path);
                tree.backup(&result.path, result.value);
            }
        }
    }
}

impl PuctBackend for PuctEngine {
    fn search(&mut self, pos: &Position, limits: &PuctLimits) -> PuctResult {
        self.stop_flag.store(false, Ordering::SeqCst);

        // SearchMode を取得（後続 Step で mate_probe / dfpn / policy 補正に使用）
        let _search_mode = limits.search_mode;

        // Tactical PUCT Stage 1: phase-aware MCTS skip in Endgame
        // SearchController は Endgame で αβ に routing するが、 MCTS が直接呼ばれた
        // 場合の保険としてここでも skip 判定。 valid bestmove を返さないと
        // 上位 USI が困るので、 first legal move を返す (αβ に切替推奨を log)
        if self.config.phase_aware_enabled
            && self.config.skip_mcts_in_endgame
            && matches!(limits.game_phase, Some(crate::phase::GamePhase::Endgame))
        {
            let mut root_moves = MoveList::new();
            generate_legal(pos, &mut root_moves);
            let first = if root_moves.is_empty() {
                Move::NONE
            } else {
                root_moves.at(0)
            };
            return PuctResult {
                bestmove: first,
                value: 0.0,
                nodes: 0,
                pv: vec![],
            };
        }

        // phase-aware パラメータを取得
        let phase_c_puct = self.config.c_puct_for_phase(limits.game_phase, limits.endgame_score);
        let phase_top_k = self.config.top_k_for_phase(limits.game_phase, limits.endgame_score);

        // ルート初期化
        let root = self.tree.init_root();

        // 作業用 Position
        let mut work_pos = pos.clone();

        // ルート展開
        let mut root_moves = MoveList::new();
        generate_legal(&work_pos, &mut root_moves);

        if root_moves.is_empty() {
            return PuctResult {
                bestmove: Move::NONE,
                value: -1.0,
                nodes: 0,
                pv: vec![],
            };
        }

        // Tactical PUCT Stage 2: tactical_score をキャッシュしながら expand
        let tactical_weights = if self.config.enable_tactical_puct {
            Some(self.config.to_tactical_weights())
        } else {
            None
        };
        self.tree.expand_with_topk_and_tactical(
            root,
            &root_moves,
            phase_top_k,
            Some(pos),
            tactical_weights.as_ref(),
        );

        // ★ Layer 3: Opponent mate check at root (頓死回避)
        // PolicyValue モードで自動発動。 各 root child について「相手番 mate_probe_1ply」
        // を実行、 詰まれる手 (= 頓死手) の prior を 0.001 倍に減点。
        // memory: [[mate_engine_midgame_integration_20260520]]
        let is_pv_mode =
            matches!(self.eval_bridge, crate::mcts::eval_bridge::EvalBridge::PolicyValue(_));
        #[cfg(feature = "pv-cuda")]
        let is_pv_mode = is_pv_mode
            || matches!(self.eval_bridge, crate::mcts::eval_bridge::EvalBridge::PolicyValueGpu(_));
        if is_pv_mode {
            use crate::mate_probe::{MateResult, mate_probe_1ply};
            let root_node = self.tree.arena.get(root);
            let first_child = root_node.first_child;
            let n_children = root_node.num_children as u32;
            let mut blunder_count = 0u32;
            for i in 0..n_children {
                let child_idx = first_child + i;
                let mv = self.tree.arena.get(child_idx).mv;
                if mv == Move::NONE {
                    continue;
                }
                let gives_check = work_pos.gives_check(mv);
                work_pos.do_move(mv, gives_check);
                let opponent_mate = mate_probe_1ply(&mut work_pos);
                work_pos.undo_move(mv);
                if matches!(opponent_mate, MateResult::MateFound { .. }) {
                    // 相手から 1 手詰めされる手 = 頓死、 大幅減点
                    let child = self.tree.arena.get_mut(child_idx);
                    child.prior *= 0.001;
                    blunder_count += 1;
                }
            }
            if blunder_count > 0 {
                eprintln!(
                    "info string [Layer3] {} blunder move(s) detected and penalized",
                    blunder_count
                );
            }
        }

        // Dirichlet noise をルートに適用 (Gumbel AZ では Dirichlet 不要なので skip)
        if !self.config.use_gumbel {
            self.tree.add_dirichlet_noise(root, &mut self.rng);
        } else {
            // Gumbel AZ: ルート子に Gumbel(0,1) noise をサンプル + sequential halving 初期化
            self.tree.sample_root_gumbel(&mut self.rng);
            // total_budget は max_nodes 優先、 無ければ arena_capacity を上限とする
            let total_budget = if limits.max_nodes > 0 {
                limits.max_nodes.min(self.config.arena_capacity as u64) as u32
            } else {
                self.config.arena_capacity
            };
            self.tree.init_gumbel_halving(total_budget);
        }

        let mut nodes: u64 = 0;
        let start = std::time::Instant::now();

        // バッチ評価コレクター
        let use_batching = self.config.use_batching;
        let mut batch = BatchCollector::new(self.config.batch_size);

        // メインループ
        loop {
            // 停止判定
            if self.stop_flag.load(Ordering::Relaxed) {
                break;
            }
            if limits.max_nodes > 0 && nodes >= limits.max_nodes {
                break;
            }
            if limits.max_time_ms > 0 && start.elapsed().as_millis() as u64 >= limits.max_time_ms {
                break;
            }

            // arena 満杯チェック
            if self.tree.arena.len() >= self.tree.arena.capacity() {
                break;
            }

            // 1. 選択 (Gumbel AZ なら root のみ Gumbel + sequential halving、 非 root は PUCT)
            let path = if self.config.use_gumbel {
                self.tree.select_with_gumbel_root(root)
            } else {
                self.tree.select_with_options(
                    root,
                    phase_c_puct,
                    self.config.enable_tactical_puct,
                    self.config.enable_risk_backup,
                    self.config.risk_variance_weight,
                )
            };

            // 2. Virtual loss 追加
            self.tree.apply_virtual_loss(&path);

            // 3. path に沿って局面を進める（root は skip）
            for i in 1..path.len() {
                let mv = self.tree.arena.get(path.get(i)).mv;
                let gives_check = work_pos.gives_check(mv);
                let _ = work_pos.do_move(mv, gives_check);
            }

            // 4. 葉ノードの評価
            let leaf_idx = path.leaf();
            let leaf = self.tree.arena.get(leaf_idx);

            if leaf.is_terminal {
                // 終端: 合法手なし = 手番側の敗北
                // 即時 backup
                for i in (1..path.len()).rev() {
                    let mv = self.tree.arena.get(path.get(i)).mv;
                    work_pos.undo_move(mv);
                }
                self.tree.remove_virtual_loss(&path);
                self.tree.backup(&path, -1.0);
                nodes += 1;
                if self.config.use_gumbel {
                    let _ = self.tree.record_gumbel_visit_and_maybe_halve();
                }
                continue;
            }

            let needs_eval = if leaf.first_child == super::node::MctsNode::INVALID {
                // 未展開 → 展開
                let mut leaf_moves = MoveList::new();
                generate_legal(&work_pos, &mut leaf_moves);

                if leaf_moves.is_empty() {
                    self.tree.arena.get_mut(leaf_idx).is_terminal = true;
                    // 即時 backup
                    for i in (1..path.len()).rev() {
                        let mv = self.tree.arena.get(path.get(i)).mv;
                        work_pos.undo_move(mv);
                    }
                    self.tree.remove_virtual_loss(&path);
                    self.tree.backup(&path, -1.0);
                    nodes += 1;
                    if self.config.use_gumbel {
                        let _ = self.tree.record_gumbel_visit_and_maybe_halve();
                    }
                    continue;
                }

                self.tree.expand_with_topk_and_tactical(
                    leaf_idx,
                    &leaf_moves,
                    phase_top_k,
                    Some(&work_pos),
                    tactical_weights.as_ref(),
                );
                true
            } else {
                // 既に展開済み
                true
            };

            if needs_eval {
                // eval cache 早期 hit 判定 (use_eval_cache=true 時のみ)
                let cached_value: Option<f32> = if let Some(cache) = self.eval_cache.as_mut() {
                    cache.probe(work_pos.key()).map(Self::value_to_winrate)
                } else {
                    None
                };

                if let Some(value) = cached_value {
                    // cache hit: NN 評価をスキップして即 backup
                    for i in (1..path.len()).rev() {
                        let mv = self.tree.arena.get(path.get(i)).mv;
                        work_pos.undo_move(mv);
                    }
                    self.tree.remove_virtual_loss(&path);
                    self.tree.backup(&path, value);
                } else if use_batching {
                    // バッチモード: 特徴量抽出 → submit → undo
                    let input = extract_features(&work_pos);
                    let pos_key = work_pos.key(); // cache 用に保持
                    for i in (1..path.len()).rev() {
                        let mv = self.tree.arena.get(path.get(i)).mv;
                        work_pos.undo_move(mv);
                    }
                    batch.submit_with_key(&path, &input, pos_key);

                    // フラッシュ判定
                    if batch.should_flush() {
                        let evaluator = self.eval_bridge.evaluator();
                        Self::flush_and_backup_with_cache(
                            &mut self.tree,
                            &mut batch,
                            evaluator,
                            self.eval_cache.as_mut(),
                        );
                    }
                } else {
                    // ★ Mate-at-leaves: NN eval 前に短手数詰み検出 (1ply、 高速)
                    // 詰みあり: NN eval skip、 即時詰み値返却で rollout 短縮
                    let mate_value: Option<f32> = if self.config.mate_probe_enabled {
                        match crate::mate_probe::mate_probe_1ply(&mut work_pos) {
                            crate::mate_probe::MateResult::MateFound { .. } => {
                                // 自分の手番で 1 手詰みを発見 = 自分勝ち
                                Some(1.0)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };

                    let value = if let Some(mv) = mate_value {
                        mv
                    } else {
                        let input = extract_features(&work_pos);
                        let pos_key = work_pos.key();
                        let raw =
                            self.eval_bridge.evaluate_leaf(leaf_idx, &input, &work_pos, nodes);
                        let v = Self::value_to_winrate(raw);
                        if let Some(cache) = self.eval_cache.as_mut() {
                            cache.store(pos_key, raw);
                        }
                        v
                    };

                    for i in (1..path.len()).rev() {
                        let mv = self.tree.arena.get(path.get(i)).mv;
                        work_pos.undo_move(mv);
                    }

                    self.tree.remove_virtual_loss(&path);
                    self.tree.backup(&path, value);
                }
            }

            nodes += 1;
            if self.config.use_gumbel {
                let _ = self.tree.record_gumbel_visit_and_maybe_halve();
            }
        }

        // ループ後: 残りバッチフラッシュ
        if batch.pending_count() > 0 {
            let evaluator = self.eval_bridge.evaluator();
            Self::flush_and_backup_with_cache(
                &mut self.tree,
                &mut batch,
                evaluator,
                self.eval_cache.as_mut(),
            );
        }

        // 最良手 (Gumbel AZ なら improved policy で argmax、 標準なら最多訪問)
        let best_idx = if self.config.use_gumbel {
            self.tree.best_child_gumbel(root)
        } else {
            self.tree.best_child(root)
        };
        let mut bestmove = self.tree.arena.get(best_idx).mv;

        // 中盤後半 hybrid 検証: 中盤後半で発動条件を満たす場合、 αβ probe で再選定
        if self.config.use_hybrid {
            let phase = limits.game_phase.unwrap_or(crate::phase::GamePhase::Middlegame);
            let topk = self.tree.extract_topk_for_hybrid(root, self.config.hybrid_topk);
            let mut hcfg = self.config.to_hybrid_config();
            // Stage 1: phase-aware なら ab_weight を phase 別値で上書き
            if self.config.phase_aware_enabled {
                let aw = self.config.ab_weight_for_phase(limits.game_phase, limits.endgame_score);
                hcfg.ab_weight = aw;
                hcfg.mcts_weight = (1.0 - aw - 0.10).max(0.10);
            }
            if let Some((hybrid_best, _cands)) =
                crate::hybrid::run_hybrid_selection(&hcfg, pos, phase, limits.endgame_score, &topk)
            {
                bestmove = hybrid_best;
            }
        }

        // 勝率: ルートの Q 値
        let root_node = self.tree.arena.get(root);
        let root_value = if root_node.visit_count > 0 {
            root_node.value_sum / root_node.visit_count as f32
        } else {
            0.0
        };

        let pv = self.tree.extract_pv();

        PuctResult {
            bestmove,
            value: root_value,
            nodes,
            pv,
        }
    }

    fn stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }
}
