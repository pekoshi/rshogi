//! Gumbel AlphaZero (Danihelka et al, ICLR 2022) の math primitives.
//!
//! 参考: "Policy improvement by planning with Gumbel"
//! https://openreview.net/pdf?id=bERaNdoegnO
//!
//! 主要な要素:
//! - Gumbel(0, 1) noise sampling
//! - sigma transform: monotonically increasing function of completed_q
//! - completed_q: 訪問済みノードは Q、 未訪問ノードは v_pi(s) で補完
//! - improved policy: pi'(a) = softmax(logit(a) + sigma(completed_q(a)))
//! - sequential halving phase manager
//!
//! root 選択 (探索中):
//!   action = argmax(logit + g + sigma(completed_q))
//!   ただし sequential halving で候補を絞り込む
//!
//! 最終 best move (探索後):
//!   bestmove = argmax(improved_logits(a))
//!
//! 論文の主張: +30-100 ELO、 sims を 1/4 に減らしても同強度

/// Gumbel(0, 1) を 1 個サンプリング
///
/// `g = -log(-log(u))` where `u ~ Uniform(0, 1)`
#[inline]
pub fn sample_gumbel(u: f32) -> f32 {
    // u は (0, 1) 開区間に clamp (0 と 1 で log が爆発するため)
    let eps = 1e-9;
    let u = u.clamp(eps, 1.0 - eps);
    -(-u.ln()).ln()
}

/// 配列に Gumbel noise を一括充填
pub fn fill_gumbel_noise(rng: &mut impl rand::Rng, out: &mut [f32]) {
    for slot in out.iter_mut() {
        let u: f32 = rng.random();
        *slot = sample_gumbel(u);
    }
}

/// sigma transform: completed_q を logit 空間に持ち込む単調増加関数
///
/// `sigma(q) = (c_visit + max_visits) * c_scale * q`
///
/// max_visits は同階層の子ノードの最大訪問数 (sequential halving の current 候補内)。
/// c_visit, c_scale は Danihelka 論文の推奨値: c_visit = 50.0, c_scale = 1.0。
#[inline]
pub fn sigma_transform(q: f32, max_visits: u32, c_visit: f32, c_scale: f32) -> f32 {
    (c_visit + max_visits as f32) * c_scale * q
}

/// 親ノードの mixed value v_pi(s) を計算
///
/// 訪問済みの子ノードについて、 prior 重み付き平均で v_pi を求める。
/// 未訪問の completed_q を補完する用途。
///
/// `v_pi(s) = (sum_{a visited} pi(a) * q(a)) / (sum_{a visited} pi(a))`
pub fn mixed_value(priors: &[f32], visits: &[u32], q_values: &[f32]) -> f32 {
    debug_assert_eq!(priors.len(), visits.len());
    debug_assert_eq!(priors.len(), q_values.len());

    let mut weight_sum = 0.0f32;
    let mut weighted_q = 0.0f32;
    for i in 0..priors.len() {
        if visits[i] > 0 {
            weight_sum += priors[i];
            weighted_q += priors[i] * q_values[i];
        }
    }
    if weight_sum > 0.0 {
        weighted_q / weight_sum
    } else {
        0.0
    }
}

/// completed_q を計算
///
/// `completed_q(a) = q(a)        if a が訪問済み`
/// `                = v_pi(s)    if 未訪問`
pub fn completed_q(visits: &[u32], q_values: &[f32], v_pi: f32, out: &mut [f32]) {
    debug_assert_eq!(visits.len(), q_values.len());
    debug_assert_eq!(visits.len(), out.len());
    for i in 0..visits.len() {
        out[i] = if visits[i] > 0 { q_values[i] } else { v_pi };
    }
}

/// improved logits を計算
///
/// `pi'(a) ∝ exp(logit(a) + sigma(completed_q(a)))`
///
/// 出力は softmax 前の logit。 root のlogit + Gumbel noise + sigma(completed_q) として
/// argmax を取れば improved policy のサンプル相当 (Gumbel-max trick)。
pub fn improved_logits(
    base_logits: &[f32],
    completed_q_values: &[f32],
    max_visits: u32,
    c_visit: f32,
    c_scale: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(base_logits.len(), completed_q_values.len());
    debug_assert_eq!(base_logits.len(), out.len());
    for i in 0..base_logits.len() {
        out[i] =
            base_logits[i] + sigma_transform(completed_q_values[i], max_visits, c_visit, c_scale);
    }
}

/// Gumbel + 完備 Q による action selection
///
/// `action = argmax_a (logit(a) + g(a) + sigma(completed_q(a)))`
///
/// 探索中の root action 選択、 および最終 bestmove 選択 (g=0 でも良い、 論文準拠) で使用。
/// `candidates`: 現在の候補 index リスト (sequential halving で絞られた集合)。
pub fn gumbel_argmax(
    base_logits: &[f32],
    gumbel_noise: &[f32],
    completed_q_values: &[f32],
    candidates: &[usize],
    max_visits: u32,
    c_visit: f32,
    c_scale: f32,
) -> usize {
    debug_assert!(!candidates.is_empty());
    let mut best_idx = candidates[0];
    let mut best_score = f32::NEG_INFINITY;
    for &i in candidates {
        let score = base_logits[i]
            + gumbel_noise[i]
            + sigma_transform(completed_q_values[i], max_visits, c_visit, c_scale);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

/// 探索後の最終 bestmove 選択 (Gumbel noise も含めた improved policy の argmax)
///
/// gumbel noise はルート展開時にサンプルした値を使用。
pub fn select_best_action_with_gumbel(
    base_logits: &[f32],
    gumbel_noise: &[f32],
    visits: &[u32],
    q_values: &[f32],
    priors: &[f32],
    c_visit: f32,
    c_scale: f32,
) -> usize {
    let n = base_logits.len();
    debug_assert_eq!(gumbel_noise.len(), n);
    debug_assert_eq!(visits.len(), n);
    debug_assert_eq!(q_values.len(), n);
    debug_assert_eq!(priors.len(), n);

    let v_pi = mixed_value(priors, visits, q_values);
    let mut cq = vec![0.0f32; n];
    completed_q(visits, q_values, v_pi, &mut cq);

    let max_visits = visits.iter().copied().max().unwrap_or(0);
    let candidates: Vec<usize> = (0..n).collect();
    gumbel_argmax(base_logits, gumbel_noise, &cq, &candidates, max_visits, c_visit, c_scale)
}

/// Sequential halving phase 管理
///
/// 候補 m 個から始め、 各 phase で m を半分にしていく。
/// 各 phase の visit budget は `n_total / log2(m_initial)`。
/// 候補が 1 個になるまで継続。
pub struct SequentialHalving {
    /// 初期候補数 m
    initial_m: usize,
    /// 現 phase の候補 indices (m 要素 → m/2 → m/4 → ...)
    candidates: Vec<usize>,
    /// 各 phase の visit budget
    phase_budget: u32,
    /// 現 phase の visit カウンタ
    phase_visits: u32,
    /// log2(m_initial)、 phase 数
    log2_m: usize,
    /// 現在の phase index (0..log2_m)
    current_phase: usize,
}

impl SequentialHalving {
    /// 新規作成
    ///
    /// `m_initial`: 初期候補数 (通常 16 や num_children のうち小さい方)
    /// `total_budget`: 探索総 visit 数
    pub fn new(m_initial: usize, total_budget: u32) -> Self {
        let m = m_initial.max(1);
        let log2_m = (m as f64).log2().ceil() as usize;
        let log2_m = log2_m.max(1);
        let phase_budget = total_budget / log2_m as u32;
        let phase_budget = phase_budget.max(1);

        Self {
            initial_m: m,
            candidates: (0..m).collect(),
            phase_budget,
            phase_visits: 0,
            log2_m,
            current_phase: 0,
        }
    }

    /// 候補に対する初期 indices を user 側 set
    pub fn set_initial_candidates(&mut self, indices: Vec<usize>) {
        self.candidates = indices;
    }

    /// 現在の候補集合
    pub fn candidates(&self) -> &[usize] {
        &self.candidates
    }

    /// 各候補に割り当てる per-action visit budget (現 phase)
    ///
    /// 例: m=16, budget=64 → 各候補 4 visits → halving → m=8 → ...
    pub fn per_action_budget(&self) -> u32 {
        if self.candidates.is_empty() {
            return 0;
        }
        (self.phase_budget / self.candidates.len() as u32).max(1)
    }

    /// 1 visit 進める。 phase が終わったら true (= halving のタイミング)
    pub fn record_visit(&mut self) -> bool {
        self.phase_visits += 1;
        self.phase_visits >= self.phase_budget
    }

    /// 現候補のうち improved logit でソートして上位半分に絞る
    pub fn halve(&mut self, sorted_indices_by_score: Vec<usize>) {
        let new_len = self.candidates.len().div_ceil(2);
        self.candidates = sorted_indices_by_score.into_iter().take(new_len).collect();
        self.phase_visits = 0;
        self.current_phase += 1;
    }

    /// 全 phase 完了 (候補 1 個に絞られた)
    pub fn is_done(&self) -> bool {
        self.candidates.len() <= 1 || self.current_phase >= self.log2_m
    }

    pub fn current_phase(&self) -> usize {
        self.current_phase
    }

    pub fn initial_m(&self) -> usize {
        self.initial_m
    }
}

// ────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn gumbel_distribution_sanity() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut samples = vec![0.0f32; 10_000];
        fill_gumbel_noise(&mut rng, &mut samples);
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        // Gumbel(0,1) の theoretical mean ≈ Euler–Mascheroni constant ≈ 0.5772
        assert!((mean - 0.5772).abs() < 0.05, "mean={}", mean);
    }

    #[test]
    fn sigma_monotonic() {
        let max_v = 100u32;
        let s_lo = sigma_transform(-1.0, max_v, 50.0, 1.0);
        let s_mid = sigma_transform(0.0, max_v, 50.0, 1.0);
        let s_hi = sigma_transform(1.0, max_v, 50.0, 1.0);
        assert!(s_lo < s_mid);
        assert!(s_mid < s_hi);
        assert_eq!(s_mid, 0.0);
    }

    #[test]
    fn mixed_value_with_visits() {
        let priors = vec![0.5, 0.3, 0.2];
        let visits = vec![10u32, 5, 0];
        let q = vec![0.8, -0.4, 0.0];
        let v_pi = mixed_value(&priors, &visits, &q);
        // 訪問済み: a=0,1
        // weight_sum = 0.5 + 0.3 = 0.8
        // weighted_q = 0.5*0.8 + 0.3*(-0.4) = 0.40 - 0.12 = 0.28
        // v_pi = 0.28 / 0.8 = 0.35
        assert!((v_pi - 0.35).abs() < 1e-5, "v_pi={}", v_pi);
    }

    #[test]
    fn completed_q_unvisited_uses_v_pi() {
        let visits = vec![5u32, 0, 3];
        let q = vec![0.8, 0.0, -0.5];
        let v_pi = 0.2;
        let mut cq = vec![0.0f32; 3];
        completed_q(&visits, &q, v_pi, &mut cq);
        assert_eq!(cq[0], 0.8);
        assert_eq!(cq[1], 0.2); // 未訪問 → v_pi
        assert_eq!(cq[2], -0.5);
    }

    #[test]
    fn gumbel_argmax_picks_best() {
        // 3 候補。 logit + sigma(completed_q) の合計が大きい方が選ばれる
        let logits = vec![0.0f32, 0.0, 0.0];
        let gumbel = vec![0.0f32, 0.0, 0.0];
        let cq = vec![-1.0f32, 0.5, -0.2];
        let candidates = vec![0, 1, 2];
        let chosen = gumbel_argmax(&logits, &gumbel, &cq, &candidates, 10, 50.0, 1.0);
        assert_eq!(chosen, 1); // cq[1] が最大
    }

    #[test]
    fn select_best_action_with_gumbel_smoke() {
        let logits = vec![0.0f32, 0.5, -0.2];
        let gumbel = vec![0.0f32, 0.0, 0.0];
        let visits = vec![10u32, 5, 0];
        let q_values = vec![0.3, 0.6, 0.0];
        let priors = vec![0.4, 0.3, 0.3];
        let chosen = select_best_action_with_gumbel(
            &logits, &gumbel, &visits, &q_values, &priors, 50.0, 1.0,
        );
        // sigma が支配的 (max_visits=10、 50+10=60)
        // a=0: 0 + 0 + 60*1*0.3 = 18
        // a=1: 0.5 + 0 + 60*1*0.6 = 36.5
        // a=2: -0.2 + 0 + 60*1*v_pi (未訪問)、 v_pi = (0.4*0.3+0.3*0.6)/(0.4+0.3) ≈ 0.43
        //       → -0.2 + 60*0.43 = 25.6
        assert_eq!(chosen, 1);
    }

    #[test]
    fn sequential_halving_phases() {
        let mut sh = SequentialHalving::new(8, 64);
        // log2(8) = 3 phases
        assert_eq!(sh.candidates().len(), 8);
        assert!(!sh.is_done());

        // halving 1 回目: 8 → 4
        sh.halve((0..8).collect());
        assert_eq!(sh.candidates().len(), 4);
        assert!(!sh.is_done());

        // 2 回目: 4 → 2
        sh.halve((0..4).collect());
        assert_eq!(sh.candidates().len(), 2);

        // 3 回目: 2 → 1
        sh.halve((0..2).collect());
        assert_eq!(sh.candidates().len(), 1);
        assert!(sh.is_done());
    }

    #[test]
    fn sequential_halving_per_action_budget() {
        // 16 候補、 total 256: log2(16)=4 phases、 phase_budget=64
        // m=16 → 各候補 4 visits、 m=8 → 8 visits、 m=4 → 16 visits、 m=2 → 32 visits
        let sh = SequentialHalving::new(16, 256);
        assert_eq!(sh.per_action_budget(), 4);
    }
}
