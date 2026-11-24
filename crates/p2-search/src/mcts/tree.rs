use rshogi_core::movegen::MoveList;
use rshogi_core::types::Move;

use super::config::PuctConfig;
use super::node::{MctsNode, NodeArena};

/// PUCT 選択パス（スタック配列、ヒープ割り当てなし）
pub struct SelectionPath {
    indices: [u32; 256],
    len: usize,
}

impl Default for SelectionPath {
    fn default() -> Self {
        Self {
            indices: [0; 256],
            len: 0,
        }
    }
}

impl SelectionPath {
    /// 空のパスを作成
    pub fn new() -> Self {
        Self::default()
    }

    /// ノード index をパスに追加
    pub fn push(&mut self, idx: u32) {
        self.indices[self.len] = idx;
        self.len += 1;
    }

    /// パスの長さ
    pub fn len(&self) -> usize {
        self.len
    }

    /// パスが空か
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// i 番目のノード index を取得
    pub fn get(&self, i: usize) -> u32 {
        self.indices[i]
    }

    /// 葉ノード（最後の要素）の index を取得
    pub fn leaf(&self) -> u32 {
        self.indices[self.len - 1]
    }
}

/// MCTS 探索木
pub struct MctsTree {
    pub arena: NodeArena,
    pub root: u32,
    config: PuctConfig,
    /// Gumbel AZ: ルート子ノードに紐づく Gumbel(0,1) noise (search 開始時に 1 度サンプル)
    /// indexing: 0..root.num_children
    gumbel_noise: Vec<f32>,
    /// Gumbel AZ Phase 2: sequential halving 状態 (use_gumbel=true 時のみ Some)
    gumbel_state: Option<super::gumbel::SequentialHalving>,
}

impl MctsTree {
    /// 新しい探索木を作成
    pub fn new(config: PuctConfig) -> Self {
        let arena = NodeArena::new(config.arena_capacity);
        Self {
            arena,
            root: MctsNode::INVALID,
            config,
            gumbel_noise: Vec::with_capacity(64),
            gumbel_state: None,
        }
    }

    /// ルートノードを確保
    pub fn init_root(&mut self) -> u32 {
        self.arena.clear();
        let idx = self.arena.alloc().expect("arena が空なのに alloc 失敗");
        let node = self.arena.get_mut(idx);
        node.mv = Move::NONE;
        node.parent = MctsNode::INVALID;
        self.root = idx;
        self.gumbel_noise.clear();
        self.gumbel_state = None;
        idx
    }

    /// PUCT スコアで葉ノードまで降りる (base c_puct 使用)
    pub fn select(&self, root: u32) -> SelectionPath {
        self.select_with_c_puct(root, self.config.c_puct)
    }

    /// PUCT スコアで葉ノードまで降りる (caller が c_puct を override、 phase-aware 用)
    pub fn select_with_c_puct(&self, root: u32, c_puct: f32) -> SelectionPath {
        self.select_with_c_puct_and_tactical(root, c_puct, false)
    }

    /// PUCT + tactical_score で葉ノードまで降りる (Tactical PUCT Stage 2 用)
    pub fn select_with_c_puct_and_tactical(
        &self,
        root: u32,
        c_puct: f32,
        use_tactical: bool,
    ) -> SelectionPath {
        self.select_with_options(root, c_puct, use_tactical, false, 0.0)
    }

    /// Stage 3: q_risk を含む完全 select
    pub fn select_with_options(
        &self,
        root: u32,
        c_puct: f32,
        use_tactical: bool,
        use_risk: bool,
        risk_lambda: f32,
    ) -> SelectionPath {
        let mut path = SelectionPath::new();
        let mut current = root;
        path.push(current);

        loop {
            let node = self.arena.get(current);
            if node.first_child == MctsNode::INVALID || node.num_children == 0 {
                break;
            }

            let best_child =
                self.puct_select_child_full(current, c_puct, use_tactical, use_risk, risk_lambda);
            path.push(best_child);
            current = best_child;
        }

        path
    }

    /// 合法手から子ノードを生成（top_k 制限、prior = 1/N 均等）
    ///
    /// 展開した子ノード数を返す。
    pub fn expand(&mut self, node_idx: u32, moves: &MoveList) -> u16 {
        self.expand_with_topk(node_idx, moves, self.config.top_k)
    }

    /// expand 時に各子ノードへ tactical_score (= bonus - penalty) をキャッシュ
    ///
    /// `tactical_weights` が None なら全子の tactical_score=0 (旧挙動と等価)。
    pub fn expand_with_topk_and_tactical(
        &mut self,
        node_idx: u32,
        moves: &MoveList,
        top_k: usize,
        pos: Option<&rshogi_core::position::Position>,
        tactical_weights: Option<&crate::tactical::TacticalWeights>,
    ) -> u16 {
        let written = self.expand_with_topk(node_idx, moves, top_k);
        if written == 0 {
            return 0;
        }
        if let (Some(pos), Some(weights)) = (pos, tactical_weights) {
            let first = self.arena.get(node_idx).first_child;
            for i in 0..written as u32 {
                let child_idx = first + i;
                let mv = self.arena.get(child_idx).mv;
                let delta = crate::tactical::tactical_delta(pos, mv, weights, None);
                self.arena.get_mut(child_idx).tactical_score = delta;
            }
        }
        written
    }

    /// top_k を caller が override できる version (phase-aware MCTS 用)
    pub fn expand_with_topk(&mut self, node_idx: u32, moves: &MoveList, top_k: usize) -> u16 {
        let n = moves.len().min(top_k) as u16;
        if n == 0 {
            self.arena.get_mut(node_idx).is_terminal = true;
            return 0;
        }

        let first = match self.arena.alloc_batch(n) {
            Some(idx) => idx,
            None => return 0, // arena 満杯
        };

        let prior = 1.0 / n as f32;

        for i in 0..n {
            let child_idx = first + i as u32;
            let child = self.arena.get_mut(child_idx);
            child.mv = moves.at(i as usize);
            child.parent = node_idx;
            child.prior = prior;
        }

        let node = self.arena.get_mut(node_idx);
        node.first_child = first;
        node.num_children = n;

        n
    }

    /// 葉→根へ value を逆伝播（手番反転で符号反転）
    ///
    /// Stage 3: 各ノードの value_min/max/sq_sum も更新する (variance 計算用)
    pub fn backup(&mut self, path: &SelectionPath, leaf_value: f32) {
        let mut value = leaf_value;
        let mut i = path.len();
        while i > 0 {
            i -= 1;
            let idx = path.get(i);
            let node = self.arena.get_mut(idx);
            node.visit_count += 1;
            node.value_sum += value;
            node.value_sq_sum += value * value;
            if value < node.value_min {
                node.value_min = value;
            }
            if value > node.value_max {
                node.value_max = value;
            }
            value = -value;
        }
    }

    /// 選択パスに virtual loss を追加
    pub fn apply_virtual_loss(&mut self, path: &SelectionPath) {
        let vl = self.config.virtual_loss as u32;
        for i in 0..path.len() {
            let idx = path.get(i);
            self.arena.get_mut(idx).virtual_loss_count += vl;
        }
    }

    /// 選択パスから virtual loss を除去
    pub fn remove_virtual_loss(&mut self, path: &SelectionPath) {
        let vl = self.config.virtual_loss as u32;
        for i in 0..path.len() {
            let idx = path.get(i);
            let node = self.arena.get_mut(idx);
            node.virtual_loss_count = node.virtual_loss_count.saturating_sub(vl);
        }
    }

    /// ルート子ノードの prior に Dirichlet ノイズを混合
    pub fn add_dirichlet_noise(&mut self, node_idx: u32, rng: &mut impl rand::Rng) {
        let node = self.arena.get(node_idx);
        let n = node.num_children as usize;
        if n == 0 {
            return;
        }
        let first = node.first_child;

        let mut noise = [0.0f32; 256];
        super::noise::dirichlet_noise(rng, self.config.dirichlet_alpha, &mut noise[..n]);

        let eps = self.config.dirichlet_epsilon;
        for (i, &eta) in noise[..n].iter().enumerate() {
            let child = self.arena.get_mut(first + i as u32);
            child.prior = (1.0 - eps) * child.prior + eps * eta;
        }
    }

    /// root 子から TopK 候補を抽出 (visits → mcts_value → policy 順)
    ///
    /// hybrid モジュールに渡すための入力。
    /// 戻り値: visits 多い順 (同 visits なら mcts_value 大、 さらに同なら policy 大) で K 個まで。
    pub fn extract_topk_for_hybrid(
        &self,
        node_idx: u32,
        k: usize,
    ) -> Vec<crate::hybrid::McTsCandidateInput> {
        let node = self.arena.get(node_idx);
        let first = node.first_child;
        let n = node.num_children as usize;
        if n == 0 {
            return Vec::new();
        }

        let mut all: Vec<crate::hybrid::McTsCandidateInput> = (0..n)
            .map(|i| {
                let child = self.arena.get(first + i as u32);
                let visits = child.visit_count;
                let mcts_value = if visits > 0 {
                    child.value_sum / visits as f32
                } else {
                    0.0
                };
                crate::hybrid::McTsCandidateInput {
                    mv: child.mv,
                    mcts_value_raw: mcts_value,
                    policy: child.prior,
                    visits,
                }
            })
            .collect();

        all.sort_by(|a, b| {
            b.visits.cmp(&a.visits).then_with(|| {
                b.mcts_value_raw
                    .partial_cmp(&a.mcts_value_raw)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        b.policy.partial_cmp(&a.policy).unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
        });
        all.truncate(k);
        all
    }

    /// 最多訪問子ノードの index を返す
    pub fn best_child(&self, node_idx: u32) -> u32 {
        let node = self.arena.get(node_idx);
        let first = node.first_child;
        let n = node.num_children;

        let mut best_idx = first;
        let mut best_visits = 0u32;

        for i in 0..n as u32 {
            let child = self.arena.get(first + i);
            if child.visit_count > best_visits {
                best_visits = child.visit_count;
                best_idx = first + i;
            }
        }

        best_idx
    }

    /// ルート子の Gumbel noise を 1 度だけサンプリング
    ///
    /// Gumbel AZ では search 中の root action 選択と最終 bestmove 選択で同じ
    /// noise を再利用する (Gumbel-max trick の正しさのため)。
    pub fn sample_root_gumbel(&mut self, rng: &mut impl rand::Rng) {
        let root_node = self.arena.get(self.root);
        let n = root_node.num_children as usize;
        self.gumbel_noise.clear();
        self.gumbel_noise.resize(n, 0.0);
        super::gumbel::fill_gumbel_noise(rng, &mut self.gumbel_noise);
    }

    /// Gumbel + completed_q による最終 bestmove 選択
    ///
    /// `bestmove = argmax_a (logit(a) + g(a) + sigma(completed_q(a)))`
    ///
    /// `logit(a) = log(prior(a))` (prior が NN policy なら NN policy の log)。
    /// 訪問されていない子は v_pi(s) で completed_q を補完。
    pub fn best_child_gumbel(&self, node_idx: u32) -> u32 {
        let node = self.arena.get(node_idx);
        let first = node.first_child;
        let n = node.num_children as usize;
        if n == 0 {
            return MctsNode::INVALID;
        }

        // 子 metrics を一時バッファに展開
        let mut priors = vec![0.0f32; n];
        let mut visits = vec![0u32; n];
        let mut q_values = vec![0.0f32; n];
        let mut logits = vec![0.0f32; n];
        let eps = 1e-9f32;
        for i in 0..n {
            let child = self.arena.get(first + i as u32);
            let p = child.prior.max(eps);
            priors[i] = p;
            logits[i] = p.ln();
            visits[i] = child.visit_count;
            q_values[i] = if child.visit_count > 0 {
                child.value_sum / child.visit_count as f32
            } else {
                0.0
            };
        }

        // gumbel noise: search 中にサンプルしてあるはず。 無ければ 0 で fallback
        let zero_noise: Vec<f32>;
        let gumbel = if self.gumbel_noise.len() == n {
            self.gumbel_noise.as_slice()
        } else {
            zero_noise = vec![0.0f32; n];
            zero_noise.as_slice()
        };

        super::gumbel::select_best_action_with_gumbel(
            &logits,
            gumbel,
            &visits,
            &q_values,
            &priors,
            self.config.gumbel_c_visit,
            self.config.gumbel_c_scale,
        ) as u32
            + first
    }

    // ────────────────────────────────────────────────────────────
    // Gumbel AZ Phase 2: in-search root selection + sequential halving
    // ────────────────────────────────────────────────────────────

    /// Sequential halving 状態を初期化 (search 開始時に呼ぶ、 sample_root_gumbel の後)
    pub fn init_gumbel_halving(&mut self, total_budget: u32) {
        let root_node = self.arena.get(self.root);
        let n_children = root_node.num_children as usize;
        let m = self.config.gumbel_m_initial.min(n_children).max(1);
        let mut sh = super::gumbel::SequentialHalving::new(m, total_budget);

        // m_initial < n_children の場合: 上位 m を logit + Gumbel で選定
        if m < n_children {
            let first = root_node.first_child;
            let eps = 1e-9f32;
            let mut scored: Vec<(usize, f32)> = (0..n_children)
                .map(|i| {
                    let child = self.arena.get(first + i as u32);
                    let p = child.prior.max(eps);
                    let g = self.gumbel_noise.get(i).copied().unwrap_or(0.0);
                    (i, p.ln() + g)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<usize> = scored.into_iter().take(m).map(|(i, _)| i).collect();
            sh.set_initial_candidates(top);
        }

        self.gumbel_state = Some(sh);
    }

    /// Sequential halving の現候補で、 訪問数最少の root 子を選ぶ
    /// (round-robin: 候補内で visit が偏らないように均等配分)
    fn pick_root_child_gumbel(&self, root_node_idx: u32) -> u32 {
        let root = self.arena.get(root_node_idx);
        let first = root.first_child;
        let state = self
            .gumbel_state
            .as_ref()
            .expect("gumbel_state must be Some when use_gumbel=true");
        let candidates = state.candidates();
        debug_assert!(!candidates.is_empty());

        let mut best_idx = candidates[0];
        let mut best_visits = u32::MAX;
        for &i in candidates {
            let child = self.arena.get(first + i as u32);
            let total = child.visit_count + child.virtual_loss_count;
            if total < best_visits {
                best_visits = total;
                best_idx = i;
            }
        }
        first + best_idx as u32
    }

    /// Gumbel root + 通常 PUCT (非 root) で葉まで降りる
    pub fn select_with_gumbel_root(&self, root: u32) -> SelectionPath {
        let mut path = SelectionPath::new();
        let mut current = root;
        path.push(current);

        let root_node = self.arena.get(root);
        if root_node.first_child == MctsNode::INVALID || root_node.num_children == 0 {
            return path;
        }

        // Root: Gumbel candidates の中から最少 visit を選ぶ
        let chosen = self.pick_root_child_gumbel(root);
        path.push(chosen);
        current = chosen;

        // 非 root: 通常 PUCT
        loop {
            let node = self.arena.get(current);
            if node.first_child == MctsNode::INVALID || node.num_children == 0 {
                break;
            }
            let best_child = self.puct_select_child(current);
            path.push(best_child);
            current = best_child;
        }

        path
    }

    /// 1 visit 進めて、 phase 終了なら halve を実行する。 全 phase 完了で true を返す。
    pub fn record_gumbel_visit_and_maybe_halve(&mut self) -> bool {
        let phase_done = match self.gumbel_state.as_mut() {
            Some(state) => state.record_visit(),
            None => return true,
        };
        if !phase_done {
            return false;
        }

        // phase 終了 → 現候補を再評価して上位半分を残す
        let root_node = self.arena.get(self.root);
        let first = root_node.first_child;
        let n = root_node.num_children as usize;
        let eps = 1e-9f32;

        let mut priors = vec![0.0f32; n];
        let mut visits = vec![0u32; n];
        let mut q_values = vec![0.0f32; n];
        let mut logits = vec![0.0f32; n];
        for i in 0..n {
            let child = self.arena.get(first + i as u32);
            let p = child.prior.max(eps);
            priors[i] = p;
            logits[i] = p.ln();
            visits[i] = child.visit_count;
            q_values[i] = if child.visit_count > 0 {
                child.value_sum / child.visit_count as f32
            } else {
                0.0
            };
        }
        let v_pi = super::gumbel::mixed_value(&priors, &visits, &q_values);
        let mut cq = vec![0.0f32; n];
        super::gumbel::completed_q(&visits, &q_values, v_pi, &mut cq);
        let max_visits = visits.iter().copied().max().unwrap_or(0);

        let state = self.gumbel_state.as_mut().unwrap();
        let mut scored: Vec<(usize, f32)> = state
            .candidates()
            .iter()
            .map(|&i| {
                let g = self.gumbel_noise.get(i).copied().unwrap_or(0.0);
                let s = logits[i]
                    + g
                    + super::gumbel::sigma_transform(
                        cq[i],
                        max_visits,
                        self.config.gumbel_c_visit,
                        self.config.gumbel_c_scale,
                    );
                (i, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let sorted: Vec<usize> = scored.into_iter().map(|(i, _)| i).collect();
        state.halve(sorted);

        state.is_done()
    }

    /// gumbel_state への immutable 参照 (テスト用)
    pub fn gumbel_state(&self) -> Option<&super::gumbel::SequentialHalving> {
        self.gumbel_state.as_ref()
    }

    /// 最多訪問チェーンを PV として抽出
    pub fn extract_pv(&self) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut current = self.root;

        loop {
            let node = self.arena.get(current);
            if node.num_children == 0 || node.first_child == MctsNode::INVALID {
                break;
            }

            let best = self.best_child(current);
            let best_node = self.arena.get(best);
            if best_node.visit_count == 0 {
                break;
            }
            pv.push(best_node.mv);
            current = best;
        }

        pv
    }

    /// PUCT スコアで最良の子ノードを選択 (base c_puct 使用)
    fn puct_select_child(&self, parent_idx: u32) -> u32 {
        self.puct_select_child_with_c(parent_idx, self.config.c_puct)
    }

    /// 完全版: PUCT + tactical + risk-aware Q (q_risk)
    fn puct_select_child_full(
        &self,
        parent_idx: u32,
        c_puct: f32,
        use_tactical: bool,
        use_risk: bool,
        risk_lambda: f32,
    ) -> u32 {
        let parent = self.arena.get(parent_idx);
        let first = parent.first_child;
        let n = parent.num_children;
        let sqrt_n = (parent.visit_count as f32).sqrt();
        let is_root = parent_idx == 0;

        // FPU reduction (dlshogi 流): 未訪問子の Q を parent_q - c_fpu * sqrt(visited_nnrate) で推定
        let c_fpu = if is_root {
            self.config.c_fpu_root
        } else {
            self.config.c_fpu
        };
        let (fpu_q, _) = if c_fpu > 0.0 {
            let mut visited_nnrate = 0.0f32;
            let mut parent_sum_win = 0.0f32;
            let mut parent_sum_visits: u32 = 0;
            for i in 0..n as u32 {
                let child = self.arena.get(first + i);
                if child.visit_count > 0 {
                    visited_nnrate += child.prior;
                    parent_sum_win += child.value_sum;
                    parent_sum_visits += child.visit_count;
                }
            }
            let parent_q = if parent_sum_visits > 0 {
                parent_sum_win / parent_sum_visits as f32
            } else {
                0.0
            };
            let fpu_reduction = c_fpu * visited_nnrate.sqrt();
            ((parent_q - fpu_reduction).max(-1.0), visited_nnrate)
        } else {
            (0.0f32, 0.0f32)
        };

        let mut best_idx = first;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..n as u32 {
            let child_idx = first + i;
            let child = self.arena.get(child_idx);
            let total = child.visit_count + child.virtual_loss_count;
            let q = if total == 0 {
                fpu_q
            } else if use_risk {
                child.q_risk(risk_lambda)
            } else {
                child.value_sum / total as f32
            };
            let u = c_puct * child.prior * sqrt_n / (1 + total) as f32;
            let mut score = q + u;
            if use_tactical {
                score += child.tactical_score;
            }

            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }

        best_idx
    }

    /// PUCT スコアで最良の子ノードを選択 (c_puct caller 指定、 phase-aware 用)
    fn puct_select_child_with_c(&self, parent_idx: u32, c_puct: f32) -> u32 {
        let parent = self.arena.get(parent_idx);
        let first = parent.first_child;
        let n = parent.num_children;
        let sqrt_n = (parent.visit_count as f32).sqrt();
        let is_root = parent_idx == 0;

        // FPU reduction (dlshogi 流): puct_select_child_full と同等
        let c_fpu = if is_root {
            self.config.c_fpu_root
        } else {
            self.config.c_fpu
        };
        let fpu_q = if c_fpu > 0.0 {
            let mut visited_nnrate = 0.0f32;
            let mut parent_sum_win = 0.0f32;
            let mut parent_sum_visits: u32 = 0;
            for i in 0..n as u32 {
                let child = self.arena.get(first + i);
                if child.visit_count > 0 {
                    visited_nnrate += child.prior;
                    parent_sum_win += child.value_sum;
                    parent_sum_visits += child.visit_count;
                }
            }
            let parent_q = if parent_sum_visits > 0 {
                parent_sum_win / parent_sum_visits as f32
            } else {
                0.0
            };
            (parent_q - c_fpu * visited_nnrate.sqrt()).max(-1.0)
        } else {
            0.0
        };

        let mut best_idx = first;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..n as u32 {
            let child_idx = first + i;
            let child = self.arena.get(child_idx);
            let total = child.visit_count + child.virtual_loss_count;
            let q = if total == 0 {
                fpu_q
            } else {
                child.value_sum / total as f32
            };
            let u = c_puct * child.prior * sqrt_n / (1 + total) as f32;
            let score = q + u;

            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }

        best_idx
    }
}
