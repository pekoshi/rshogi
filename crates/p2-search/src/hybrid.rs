//! 中盤後半 MCTS TopK → rshogi αβ プローブ ハイブリッド選択
//!
//! 中盤後半 (既存 phase 判定で middle late 相当) で MCTS が出した上位 TopK
//! 候補に対して rshogi αβ で短時間プローブし、 戦術的見落としを補正する。
//!
//! # 動作モード
//!
//! 1. MCTS 探索完了後、 root 子から TopK (visits → mcts_value → policy 順) 抽出
//! 2. 各候補手を指した局面で rshogi αβ を `probe_ms` ms 探索 (centipawn 取得)
//! 3. αβ 値で MCTS value を補正:
//!    - 通常: corrected = mcts_norm + correction_rate * (ab_norm - mcts_norm)
//!    - blunder guard: MCTS が高評価 + αβ が悪手判定 → 強制下方
//!    - good move rescue: MCTS が低評価 + αβ が好手判定 → 上方修正
//! 4. final_score = w_mcts·corrected + w_ab·ab_norm + w_policy·policy + w_visit·visit
//! 5. final_score 最大の手を選ぶ
//!
//! # 既存資源の再利用 (新規 Phase enum 作らない)
//!
//! - 中盤後半判定: `phase::GamePhase::Middlegame` + `compute_endgame_score(features)`
//!   で score ∈ [`ENDGAME_EXIT_THRESHOLD` (0.45), `ENDGAME_ENTER_THRESHOLD` (0.65)) 域
//! - αβ: `rshogi_core::search::Search` を `movetime` 制限付きで呼ぶ
//!
//! # 設定
//!
//! `mcts::PuctConfig::hybrid` に集約。 default は無効 (use_hybrid=false)。

use rshogi_core::position::Position;
use rshogi_core::search::{LimitsType, Search, SearchInfo};
use rshogi_core::types::{Move, Value};

use crate::phase::GamePhase;

/// 中盤後半判定の endgame_score 範囲: [LO, HI)
pub const MIDDLE_LATE_SCORE_LO: f32 = 0.45;
pub const MIDDLE_LATE_SCORE_HI: f32 = 0.65;

/// αβ score の clamp 上限 / 下限 (cp 空間)
pub const AB_CP_CLAMP: i32 = 2000;

/// αβ score の tanh 正規化 scale
pub const AB_CP_SCALE: f32 = 600.0;

/// ハイブリッド設定 (PuctConfig からコピーされる subset)
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// 全体 ON/OFF
    pub enabled: bool,
    /// TopK 候補数
    pub topk: usize,
    /// 1 候補あたりの αβ probe 時間 (ms)
    pub probe_ms: u64,
    /// 詳細ログ出力
    pub log_enabled: bool,

    // ── value 補正 ──
    /// MCTS value 補正を行うか
    pub correction_enabled: bool,
    /// 補正率 (0.0 = 補正なし、 1.0 = αβ 完全採用)
    pub correction_rate: f32,
    /// blunder guard (MCTS 高評価 × αβ 悪手判定 → 強制下方)
    pub blunder_guard: bool,
    /// good move rescue (MCTS 低評価 × αβ 好手 → 上方)
    pub good_move_rescue: bool,

    // ── final_score 重み ──
    pub mcts_weight: f32,
    pub ab_weight: f32,
    pub policy_weight: f32,
    pub visit_weight: f32,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            topk: 6,
            probe_ms: 20,
            log_enabled: true,
            correction_enabled: true,
            correction_rate: 0.35,
            blunder_guard: true,
            good_move_rescue: true,
            mcts_weight: 0.55,
            ab_weight: 0.35,
            policy_weight: 0.05,
            visit_weight: 0.05,
        }
    }
}

/// 補正適用フラグ (ログ用)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionFlag {
    Normal,
    BlunderGuard,
    BlunderGuardSevere,
    GoodMoveRescue,
}

/// MCTS root 子 1 手分の TopK 入力データ
#[derive(Debug, Clone, Copy)]
pub struct McTsCandidateInput {
    pub mv: Move,
    pub mcts_value_raw: f32, // MCTS Q (-1.0..=1.0、 root 子の Q 値)
    pub policy: f32,
    pub visits: u32,
}

/// ハイブリッド評価後の候補
#[derive(Debug, Clone)]
pub struct HybridCandidate {
    pub mv: Move,
    pub mcts_value_raw: f32,
    pub mcts_value_norm: f32, // [-1.0, 1.0]
    pub policy: f32,
    pub visits: u32,
    pub ab_score_cp: Option<i32>,
    pub ab_norm: Option<f32>,
    pub corrected_mcts_value: f32,
    pub final_score: f32,
    pub flag: CorrectionFlag,
}

/// MCTS value を [-1, 1] に正規化
///
/// MCTS root 子の Q 値はバックアップ時に符号反転され、 leaf の value (winrate
/// [-1, 1]) を最終的に保持する。 そのまま clamp して返す。
#[inline]
pub fn normalize_mcts_value(v: f32) -> f32 {
    v.clamp(-1.0, 1.0)
}

/// αβ centipawn を [-1, 1] へ tanh 正規化
#[inline]
pub fn normalize_ab_score_cp(score_cp: i32) -> f32 {
    let clamped = score_cp.clamp(-AB_CP_CLAMP, AB_CP_CLAMP);
    (clamped as f32 / AB_CP_SCALE).tanh()
}

/// 1 つの MCTS value に αβ value で補正をかける
///
/// 戻り値: (corrected_value, applied_flag)
pub fn correct_mcts_value(
    mcts_norm: f32,
    ab_norm: f32,
    config: &HybridConfig,
) -> (f32, CorrectionFlag) {
    if !config.correction_enabled {
        return (mcts_norm.clamp(-1.0, 1.0), CorrectionFlag::Normal);
    }

    let delta = ab_norm - mcts_norm;
    let mut corrected = (mcts_norm + config.correction_rate * delta).clamp(-1.0, 1.0);
    let mut flag = CorrectionFlag::Normal;

    // ── blunder guard: MCTS 高評価 × αβ 悪手 ──
    if config.blunder_guard {
        if mcts_norm > 0.50 && ab_norm < -0.50 {
            corrected = corrected.min(-0.30);
            flag = CorrectionFlag::BlunderGuardSevere;
        } else if mcts_norm > 0.30 && ab_norm < -0.30 {
            corrected = corrected.min(-0.10);
            flag = CorrectionFlag::BlunderGuard;
        }
    }

    // ── good move rescue: MCTS 低評価 × αβ 好手 ──
    if config.good_move_rescue && mcts_norm < 0.00 && ab_norm > 0.40 {
        corrected = corrected.max(0.15);
        // blunder guard と排他: rescue が後勝ち
        if flag == CorrectionFlag::Normal {
            flag = CorrectionFlag::GoodMoveRescue;
        }
    }

    (corrected.clamp(-1.0, 1.0), flag)
}

/// 中盤後半判定 (既存 phase API のみで判断、 新規 enum 作らない)
///
/// `phase` が Middlegame、 かつ `endgame_score` が transition zone
/// [`MIDDLE_LATE_SCORE_LO`, `MIDDLE_LATE_SCORE_HI`) のとき true。
pub fn is_middle_late(phase: GamePhase, endgame_score: f32) -> bool {
    matches!(phase, GamePhase::Middlegame)
        && (MIDDLE_LATE_SCORE_LO..MIDDLE_LATE_SCORE_HI).contains(&endgame_score)
}

/// 候補手を `pos` 上で指した局面で rshogi αβ を `time_ms` 探索し、
/// **手番側 (= 元 pos の手番側)** から見た cp スコアを返す。
///
/// 内部処理:
/// 1. pos.clone()、 do_move(mv, gives_check)
/// 2. movetime=time_ms で Search::go
/// 3. result.score を cp にして、 「指した側の視点」 = pos の手番視点に符号反転
///
/// mate scores は `to_cp()` で大きな値を返すため、 呼び出し側で必要に応じ clamp する。
pub fn ab_probe_after_move(pos: &Position, mv: Move, time_ms: u64) -> i32 {
    let mut work = pos.clone();
    let gives_check = work.gives_check(mv);
    work.do_move(mv, gives_check);

    let mut search = Search::new(8); // 16MB だと per-probe で重い、 8MB tt
    let mut limits = LimitsType::default();
    limits.movetime = time_ms.min(i64::MAX as u64) as i64;

    let result = search.go(&mut work, limits, None::<&mut dyn FnMut(&SearchInfo)>);
    let cp = result.score.to_cp();
    // 指した後の局面では opponent 手番。 score は opponent 視点なので反転
    // mate score (Value::INFINITE 含む) も整数 i32 範囲で扱える
    -saturating_clamp_cp(cp)
}

/// cp の i32 → safe range (mate は ±30000 で固定) clamp
#[inline]
fn saturating_clamp_cp(cp: i32) -> i32 {
    if cp >= Value::INFINITE.raw() {
        30000
    } else if cp <= -Value::INFINITE.raw() {
        -30000
    } else {
        cp.clamp(-30000, 30000)
    }
}

/// MCTS TopK 候補を αβ プローブ + 補正 + 最終手選択
///
/// `pos`: MCTS 探索を行った局面
/// `phase`: 既存 phase 判定の結果
/// `endgame_score`: 既存 compute_endgame_score の結果 [0, 1]
/// `mcts_topk`: 既に visits/mcts_value/policy 順にソート済みの TopK 候補
///
/// 戻り値: middle late で発動した場合のみ Some((bestmove, candidates))。
/// 発動しなかった (= 通常 PUCT で best 選択) 場合は None。
pub fn run_hybrid_selection(
    config: &HybridConfig,
    pos: &Position,
    phase: GamePhase,
    endgame_score: f32,
    mcts_topk: &[McTsCandidateInput],
) -> Option<(Move, Vec<HybridCandidate>)> {
    if !config.enabled {
        return None;
    }
    if !is_middle_late(phase, endgame_score) {
        return None;
    }
    if mcts_topk.is_empty() {
        return None;
    }

    let n = mcts_topk.len().min(config.topk);
    let mut cands: Vec<HybridCandidate> = Vec::with_capacity(n);

    if config.log_enabled {
        eprintln!("hybrid phase=middle_late score={:.3} topk={}", endgame_score, n);
    }

    // 各候補に αβ probe + 補正
    for input in mcts_topk.iter().take(n) {
        let mcts_norm = normalize_mcts_value(input.mcts_value_raw);
        let ab_cp = ab_probe_after_move(pos, input.mv, config.probe_ms);
        let ab_norm = normalize_ab_score_cp(ab_cp);
        let (corrected, flag) = correct_mcts_value(mcts_norm, ab_norm, config);

        let policy_bonus = 0.0_f32; // TODO: 拡張時に input.policy などから計算
        let visit_bonus = 0.0_f32; // TODO: 拡張時に visits 比率などから計算

        let final_score = config.mcts_weight * corrected
            + config.ab_weight * ab_norm
            + config.policy_weight * policy_bonus
            + config.visit_weight * visit_bonus;

        if config.log_enabled {
            eprintln!(
                "hybrid candidate move={:?} visits={} policy={:.3} mcts_raw={:.3} mcts_norm={:.3} ab_cp={} ab_norm={:.3} delta={:.3} corrected={:.3} final={:.3} flags={:?}",
                input.mv,
                input.visits,
                input.policy,
                input.mcts_value_raw,
                mcts_norm,
                ab_cp,
                ab_norm,
                ab_norm - mcts_norm,
                corrected,
                final_score,
                flag
            );
        }

        cands.push(HybridCandidate {
            mv: input.mv,
            mcts_value_raw: input.mcts_value_raw,
            mcts_value_norm: mcts_norm,
            policy: input.policy,
            visits: input.visits,
            ab_score_cp: Some(ab_cp),
            ab_norm: Some(ab_norm),
            corrected_mcts_value: corrected,
            final_score,
            flag,
        });
    }

    // final_score 最大を選ぶ
    let best = cands
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.final_score
                .partial_cmp(&b.1.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?
        .0;

    let bestmove = cands[best].mv;
    if config.log_enabled {
        eprintln!(
            "hybrid selected move={:?} final={:.3} reason={:?}",
            bestmove, cands[best].final_score, cands[best].flag
        );
    }

    Some((bestmove, cands))
}

// ────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_ab_score_cp_clamps() {
        assert!((normalize_ab_score_cp(0) - 0.0).abs() < 1e-6);
        // 600 cp で tanh(1) ≈ 0.762
        assert!((normalize_ab_score_cp(600) - 0.7616).abs() < 1e-3);
        // mate-level cp は 2000 cp で clamp
        let v_mate = normalize_ab_score_cp(30000);
        let v_2000 = normalize_ab_score_cp(2000);
        assert!((v_mate - v_2000).abs() < 1e-6);
        assert!(v_mate > 0.99);
    }

    #[test]
    fn normalize_mcts_value_clamps() {
        assert_eq!(normalize_mcts_value(0.5), 0.5);
        assert_eq!(normalize_mcts_value(-0.7), -0.7);
        assert_eq!(normalize_mcts_value(2.0), 1.0);
        assert_eq!(normalize_mcts_value(-3.0), -1.0);
    }

    #[test]
    fn correct_blunder_guard_severe() {
        let cfg = HybridConfig {
            correction_enabled: true,
            correction_rate: 0.35,
            blunder_guard: true,
            good_move_rescue: true,
            ..HybridConfig::default()
        };
        // MCTS 0.7 (高評価) + αβ -0.7 (悪手) → severe
        let (c, flag) = correct_mcts_value(0.7, -0.7, &cfg);
        assert_eq!(flag, CorrectionFlag::BlunderGuardSevere);
        assert!(c <= -0.30, "corrected should be ≤ -0.30, got {}", c);
    }

    #[test]
    fn correct_blunder_guard_mild() {
        let cfg = HybridConfig::default();
        // MCTS 0.4 + αβ -0.4 → mild blunder
        let (c, flag) = correct_mcts_value(0.4, -0.4, &cfg);
        assert_eq!(flag, CorrectionFlag::BlunderGuard);
        assert!(c <= -0.10, "corrected should be ≤ -0.10, got {}", c);
    }

    #[test]
    fn correct_good_move_rescue() {
        let cfg = HybridConfig::default();
        // MCTS -0.2 + αβ 0.5 → rescue
        let (c, flag) = correct_mcts_value(-0.2, 0.5, &cfg);
        assert_eq!(flag, CorrectionFlag::GoodMoveRescue);
        assert!(c >= 0.15, "corrected should be ≥ 0.15, got {}", c);
    }

    #[test]
    fn correct_normal_no_special() {
        let cfg = HybridConfig::default();
        // 同じ評価域 → normal
        let (c, flag) = correct_mcts_value(0.4, 0.5, &cfg);
        assert_eq!(flag, CorrectionFlag::Normal);
        // 0.4 + 0.35 * 0.1 = 0.435
        assert!((c - 0.435).abs() < 1e-5, "got {}", c);
    }

    #[test]
    fn correction_disabled_passthrough() {
        let cfg = HybridConfig {
            correction_enabled: false,
            ..HybridConfig::default()
        };
        let (c, flag) = correct_mcts_value(0.7, -0.7, &cfg);
        assert_eq!(flag, CorrectionFlag::Normal);
        assert_eq!(c, 0.7);
    }

    #[test]
    fn middle_late_phase_detection() {
        // Middlegame + score in [0.45, 0.65) → true
        assert!(is_middle_late(GamePhase::Middlegame, 0.50));
        assert!(is_middle_late(GamePhase::Middlegame, 0.45));
        // 境界外
        assert!(!is_middle_late(GamePhase::Middlegame, 0.65));
        assert!(!is_middle_late(GamePhase::Middlegame, 0.30));
        // Endgame は除外
        assert!(!is_middle_late(GamePhase::Endgame, 0.50));
        // Opening は除外
        assert!(!is_middle_late(GamePhase::Opening, 0.50));
    }

    #[test]
    fn run_hybrid_disabled_returns_none() {
        let cfg = HybridConfig {
            enabled: false,
            ..HybridConfig::default()
        };
        let pos = {
            let mut p = Position::new();
            p.set_hirate();
            p
        };
        let cands = vec![];
        let r = run_hybrid_selection(&cfg, &pos, GamePhase::Middlegame, 0.5, &cands);
        assert!(r.is_none());
    }

    #[test]
    fn run_hybrid_outside_phase_returns_none() {
        let cfg = HybridConfig::default();
        let pos = {
            let mut p = Position::new();
            p.set_hirate();
            p
        };
        let cands = vec![McTsCandidateInput {
            mv: Move::NONE,
            mcts_value_raw: 0.0,
            policy: 1.0,
            visits: 1,
        }];
        // Endgame (= middle_late ではない) は None
        let r = run_hybrid_selection(&cfg, &pos, GamePhase::Endgame, 0.8, &cands);
        assert!(r.is_none());
        // Opening も None
        let r = run_hybrid_selection(&cfg, &pos, GamePhase::Opening, 0.2, &cands);
        assert!(r.is_none());
    }
}
