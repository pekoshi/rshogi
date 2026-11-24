//! 既存 `PhaseTracker` / `EndgameDetector` への並立 hook。
//!
//! 既存 API を **壊さず** に ML 進行度を併用するための trait と enum。
//!
//! ## 使い方 (skeleton)
//!
//! ```ignore
//! use p2_progress_predictor::{MlProgressPredictor, ScoreSource, ProgressionExt};
//! use p2_search::phase::{PhaseTracker, compute_features, compute_endgame_score};
//!
//! let predictor = MlProgressPredictor::from_default();
//! let mut tracker = PhaseTracker::new();
//!
//! // 既存 rule-based を維持しつつ ML を併用
//! let f = compute_features(&pos, 0);
//! let rule = compute_endgame_score(&f);
//! let ml   = predictor.predict(&pos);
//! let blended = tracker.blended_score(rule, ml, 0.5);  // 0.5 で線形 blend
//! ```
//!
//! ## 設計上の注意
//!
//! - `PhaseTracker` 自体に depend するため、 既存 crate を壊さない範囲で
//!   extension trait のみを足す。 trait の default impl は計算なしで stub。
//! - `score_source` は呼び出し側で選択 (Rule / Ml / Blended)。 USI option での
//!   切替は後段別タスク。

use crate::ensemble::{EnsembleConfig, EnsemblePredictor, EnsembleResult};
use crate::hysteresis::{BackendChoice, HysteresisConfig, SwitchingStateMachine};
use crate::model::MlProgressPredictor;
use p2_search::phase::PhaseTracker;
use rshogi_core::position::Position;

/// 進行度の source (どこから score を採用するか)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreSource {
    /// 既存 rule-based (compute_endgame_score)
    Rule,
    /// ML model (MlProgressPredictor::predict)
    Ml,
    /// `(1-w) * rule + w * ml` (w は呼び出し側で指定)
    Blended,
}

/// PhaseTracker への blend / source 選択 hook。
///
/// 既存 PhaseTracker API は変更しないため、 trait の default 実装で完結する。
/// score source の切替自体は呼び出し側の責任。
pub trait ProgressionExt {
    /// 重み付き線形 blend を計算する。 trackerは値を保持しない (pure fn)。
    ///
    /// - `rule`: rule-based score (0..=1)
    /// - `ml`: ML score (0..=1)
    /// - `w_ml`: ML の混合比率 (0..=1)、 0.0 で rule のみ、 1.0 で ml のみ
    fn blended_score(&self, rule: f32, ml: f32, w_ml: f32) -> f32 {
        let w = w_ml.clamp(0.0, 1.0);
        (1.0 - w) * rule + w * ml
    }

    /// source 列挙から score を選択する helper。 Blended 時は等比 (0.5) で混ぜる。
    fn pick_score(&self, rule: f32, ml: f32, source: ScoreSource) -> f32 {
        match source {
            ScoreSource::Rule => rule,
            ScoreSource::Ml => ml,
            ScoreSource::Blended => self.blended_score(rule, ml, 0.5),
        }
    }
}

impl ProgressionExt for PhaseTracker {}

/// 3 source ensemble + hysteresis 適用後の最終 progress 推定結果
#[derive(Clone, Copy, Debug)]
pub struct CombinedProgress {
    /// ensemble 後 progress (0..=1)
    pub progress: f32,
    /// 現在の backend (hysteresis 経由)
    pub backend: BackendChoice,
    /// confidence (|progress - 0.5| * 2、 0..=1)
    /// 0.5 (中間) で 0、 0 or 1 (確信) で 1.0
    pub confidence: f32,
    /// ensemble 内訳 (debug / logging 用)
    pub raw: EnsembleResult,
}

/// 3 source ensemble + ヒステリシス backend 選択を統合した predictor。
///
/// allocation 0。 state machine だけが mutable、 他は read-only 引数。
pub fn predict_combined(
    ml: Option<&MlProgressPredictor>,
    progress_weights: Option<&[f32]>,
    pos: &Position,
    eval_cp: i32,
    state_machine: &mut SwitchingStateMachine,
    ensemble_config: &EnsembleConfig,
    hysteresis_config: &HysteresisConfig,
) -> CombinedProgress {
    let predictor = EnsemblePredictor::new(ml, progress_weights, *ensemble_config);
    let raw = predictor.predict(pos, eval_cp);

    // game_ply は i32 だが負にはならない (>=1 想定)。 念のため saturating cast。
    let current_ply = pos.game_ply().max(0) as u32;
    let backend = state_machine.update(raw.progress, current_ply, hysteresis_config);

    let confidence = ((raw.progress - 0.5).abs() * 2.0).clamp(0.0, 1.0);

    CombinedProgress {
        progress: raw.progress,
        backend,
        confidence,
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2_search::phase::PhaseTracker;

    #[test]
    fn blended_extremes() {
        let t = PhaseTracker::new();
        assert!((t.blended_score(0.2, 0.8, 0.0) - 0.2).abs() < 1e-6);
        assert!((t.blended_score(0.2, 0.8, 1.0) - 0.8).abs() < 1e-6);
        assert!((t.blended_score(0.2, 0.8, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pick_score_selects_source() {
        let t = PhaseTracker::new();
        assert_eq!(t.pick_score(0.3, 0.7, ScoreSource::Rule), 0.3);
        assert_eq!(t.pick_score(0.3, 0.7, ScoreSource::Ml), 0.7);
        let b = t.pick_score(0.3, 0.7, ScoreSource::Blended);
        assert!((b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn blend_weight_is_clamped() {
        let t = PhaseTracker::new();
        // w_ml > 1 でも全 ml と等価になる
        let y = t.blended_score(0.0, 1.0, 2.0);
        assert!((y - 1.0).abs() < 1e-6);
        // w_ml < 0 でも全 rule と等価
        let y = t.blended_score(0.0, 1.0, -1.0);
        assert!(y.abs() < 1e-6);
    }

    #[test]
    fn predict_combined_smoke_at_startpos() {
        use rshogi_core::position::Position;
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let mut sm = SwitchingStateMachine::new(BackendChoice::Puct);
        let ec = EnsembleConfig::default();
        let hc = HysteresisConfig::default();

        let r = predict_combined(None, None, &pos, 0, &mut sm, &ec, &hc);
        assert!((0.0..=1.0).contains(&r.progress), "progress out of range: {}", r.progress);
        assert!((0.0..=1.0).contains(&r.confidence), "confidence out of range: {}", r.confidence);
        // startpos なので backend は Puct のまま (低 progress)
        assert_eq!(r.backend, BackendChoice::Puct);
    }
}
