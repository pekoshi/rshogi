//! 3 種 progress signal の重み付け ensemble
//!
//! - **ML**: 訓練済み MLP (`MlProgressPredictor`、 V1、 32 dim 入力)
//! - **Rule**: ルールベース `compute_endgame_score` (`EndgameFeatures` → [0,1])
//! - **Nodchip**: NNUE progress8kpabs bucket (bucket index / 7.0)
//!
//! ## ホットパス制約
//!
//! - allocation 0、 スタックのみ
//! - V2 features を経由するため compute_features_v2 を 1 度呼ぶだけ
//!   - ただし V1 model 用に head32 を切り出す (zero-copy参照)

use crate::features::{compute_features, FEATURE_DIM};
use crate::features_v2::{compute_features_v2, NUM_PROGRESS_BUCKETS};
use crate::model::MlProgressPredictor;
use crate::model_v2::MlProgressPredictorV2;
use p2_search::phase::{compute_endgame_score, compute_features as compute_endgame_features};
use rshogi_core::nnue::{compute_progress8kpabs_sum, progress_sum_to_bucket};
use rshogi_core::position::Position;

/// 3 source の重み付け設定
#[derive(Clone, Copy, Debug)]
pub struct EnsembleConfig {
    /// ML model の重み
    pub w_ml: f32,
    /// rule-based PhaseTracker score の重み
    pub w_rule: f32,
    /// progress8kpabs bucket index の重み
    pub w_nodchip: f32,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            w_ml: 0.5,
            w_rule: 0.3,
            w_nodchip: 0.2,
        }
    }
}

impl EnsembleConfig {
    /// 正規化 (重み 0 のソースを実質無効化) して内部使用。
    /// 合計 0 の場合 (= 全ソース無効) は (0,0,0) のまま返す。
    #[inline]
    fn normalized(&self) -> (f32, f32, f32) {
        let total = self.w_ml + self.w_rule + self.w_nodchip;
        if total <= 0.0 {
            (0.0, 0.0, 0.0)
        } else {
            (self.w_ml / total, self.w_rule / total, self.w_nodchip / total)
        }
    }
}

/// 各 source signal を含む推論結果
#[derive(Clone, Copy, Debug)]
pub struct EnsembleResult {
    /// 最終 blend (重み正規化後)
    pub progress: f32,
    /// ML 単独 signal (model 無い場合は 0.5)
    pub ml_signal: f32,
    /// rule-based signal (compute_endgame_score の値)
    pub rule_signal: f32,
    /// nodchip bucket signal (bucket index / 7.0、 重みなし時は 0.5)
    pub nodchip_signal: f32,
}

/// progress signal を 3 source から推定する predictor。
///
/// `ml` / `progress_weights` は両方 None でも動く (rule のみで予測)。
pub struct EnsemblePredictor<'a> {
    pub ml: Option<&'a MlProgressPredictor>,
    pub progress_weights: Option<&'a [f32]>,
    pub config: EnsembleConfig,
}

impl<'a> EnsemblePredictor<'a> {
    /// 新規インスタンス。
    pub fn new(
        ml: Option<&'a MlProgressPredictor>,
        progress_weights: Option<&'a [f32]>,
        config: EnsembleConfig,
    ) -> Self {
        Self {
            ml,
            progress_weights,
            config,
        }
    }

    /// progress を推定。 allocation 0。
    pub fn predict(&self, pos: &Position, eval_cp: i32) -> EnsembleResult {
        // V1 features を直接計算 (V2 を呼ぶと nodchip も計算するが ml 経路では未使用)
        // ml.predict_features には ProgressFeatures (32 dim) が必要
        let v1 = compute_features(pos, eval_cp);

        // ML signal (model 無ければ 0.5 fallback)
        let ml_signal = match self.ml {
            Some(m) => m.predict_features(&v1),
            None => 0.5,
        };

        // Rule signal: compute_endgame_features 経由
        let ef = compute_endgame_features(pos, eval_cp);
        let rule_signal = compute_endgame_score(&ef).clamp(0.0, 1.0);

        // Nodchip signal: bucket index / 7.0
        let nodchip_signal = match self.progress_weights {
            Some(w) => {
                let sum = compute_progress8kpabs_sum(pos, w);
                let bucket = progress_sum_to_bucket(sum);
                bucket as f32 / (NUM_PROGRESS_BUCKETS - 1) as f32
            }
            None => 0.5,
        };

        let (wm, wr, wn) = self.config.normalized();
        let progress = (wm * ml_signal + wr * rule_signal + wn * nodchip_signal).clamp(0.0, 1.0);

        EnsembleResult {
            progress,
            ml_signal,
            rule_signal,
            nodchip_signal,
        }
    }
}

/// V2 ensemble predictor。
///
/// V1 `EnsemblePredictor` の V2 features + V2 model 並立版。
/// V1 (`EnsemblePredictor`) と異なり、 `compute_features_v2` を 1 回だけ呼び、
/// 48 dim 全てを V2 model に流す。 nodchip signal は V2 features の
/// 計算 path と共有 (zero-cost、 progress_weights は features_v2 内で消費)。
pub struct EnsemblePredictorV2<'a> {
    /// V2 model (48 dim 入力、 48→64→32→16→1)
    pub ml_v2: Option<&'a MlProgressPredictorV2>,
    /// nodchip progress8kpabs 重み (None なら nodchip signal 無効)
    pub progress_weights: Option<&'a [f32]>,
    /// 3 source の混合重み
    pub config: EnsembleConfig,
}

impl<'a> EnsemblePredictorV2<'a> {
    /// 新規インスタンス。
    pub fn new(
        ml_v2: Option<&'a MlProgressPredictorV2>,
        progress_weights: Option<&'a [f32]>,
        config: EnsembleConfig,
    ) -> Self {
        Self {
            ml_v2,
            progress_weights,
            config,
        }
    }

    /// V2 progress signal を 3 source から推定。 allocation 0。
    pub fn predict(&self, pos: &Position, eval_cp: i32) -> EnsembleResult {
        // V2 features を 1 度だけ計算 (nodchip soft 8 ch を内部で埋める)
        let v2 = compute_features_v2(pos, eval_cp, self.progress_weights);

        // ML signal (V2 model 無ければ 0.5 fallback)
        let ml_signal = match self.ml_v2 {
            Some(m) => m.predict_features(&v2),
            None => 0.5,
        };

        // Rule signal: compute_endgame_features 経由 (V1 と同じ実装)
        let ef = compute_endgame_features(pos, eval_cp);
        let rule_signal = compute_endgame_score(&ef).clamp(0.0, 1.0);

        // Nodchip signal: bucket index / 7.0 (V1 と同じ実装、 features_v2 とは別途
        // bucket index を取り出す。 features_v2 は soft channel 用なので別 path)
        let nodchip_signal = match self.progress_weights {
            Some(w) => {
                let sum = compute_progress8kpabs_sum(pos, w);
                let bucket = progress_sum_to_bucket(sum);
                bucket as f32 / (NUM_PROGRESS_BUCKETS - 1) as f32
            }
            None => 0.5,
        };

        let (wm, wr, wn) = self.config.normalized();
        let progress = (wm * ml_signal + wr * rule_signal + wn * nodchip_signal).clamp(0.0, 1.0);

        EnsembleResult {
            progress,
            ml_signal,
            rule_signal,
            nodchip_signal,
        }
    }
}

/// V2 features を ensemble に流したい場合の helper。
///
/// 軸 B (V2 model) を実装する前の暫定: V2 の先頭 32 dim を切り出して V1 model に流す。
///
/// 戻り値は `ProgressFeatures` ではなく内部の `[f32; FEATURE_DIM]` slice 参照。
#[inline]
pub fn v2_head_as_v1_slice(v2: &crate::features_v2::ProgressFeaturesV2) -> &[f32; FEATURE_DIM] {
    v2.head32()
}

/// V2 features をワンショットで計算 + ensemble (上記 helper のショートカット)。
///
/// nodchip signal は V2 features の bucket 計算と同一 path を共有する。
pub fn predict_via_v2(
    ml: Option<&MlProgressPredictor>,
    progress_weights: Option<&[f32]>,
    pos: &Position,
    eval_cp: i32,
    config: &EnsembleConfig,
) -> EnsembleResult {
    let v2 = compute_features_v2(pos, eval_cp, progress_weights);
    let head = v2.head32();

    let ml_signal = match ml {
        Some(m) => {
            // ProgressFeatures は [f32; 32] を中に持つ Copy 型なので、 単純コピー。
            let mut feats = crate::features::ProgressFeatures::default();
            feats.data.copy_from_slice(head);
            m.predict_features(&feats)
        }
        None => 0.5,
    };

    let ef = compute_endgame_features(pos, eval_cp);
    let rule_signal = compute_endgame_score(&ef).clamp(0.0, 1.0);

    let nodchip_signal = match progress_weights {
        Some(w) => {
            let sum = compute_progress8kpabs_sum(pos, w);
            let bucket = progress_sum_to_bucket(sum);
            bucket as f32 / (NUM_PROGRESS_BUCKETS - 1) as f32
        }
        None => 0.5,
    };

    let (wm, wr, wn) = config.normalized();
    let progress = (wm * ml_signal + wr * rule_signal + wn * nodchip_signal).clamp(0.0, 1.0);

    EnsembleResult {
        progress,
        ml_signal,
        rule_signal,
        nodchip_signal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::position::Position;

    fn startpos() -> Position {
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        pos
    }

    #[test]
    fn default_config_weights_sum_to_one() {
        let c = EnsembleConfig::default();
        let (wm, wr, wn) = c.normalized();
        assert!(((wm + wr + wn) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ensemble_runs_without_ml_or_nodchip() {
        let pos = startpos();
        let p = EnsemblePredictor::new(None, None, EnsembleConfig::default());
        let r = p.predict(&pos, 0);
        assert!((0.0..=1.0).contains(&r.progress));
        assert!((r.ml_signal - 0.5).abs() < 1e-6);
        assert!((r.nodchip_signal - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ensemble_w_ml_only_yields_ml_signal() {
        let pos = startpos();
        let cfg = EnsembleConfig {
            w_ml: 1.0,
            w_rule: 0.0,
            w_nodchip: 0.0,
        };
        let model = MlProgressPredictor::from_default();
        let p = EnsemblePredictor::new(Some(&model), None, cfg);
        let r = p.predict(&pos, 0);
        // ml signal が支配的
        assert!((r.progress - r.ml_signal).abs() < 1e-5);
    }

    #[test]
    fn ensemble_w_rule_only_yields_rule_signal() {
        let pos = startpos();
        let cfg = EnsembleConfig {
            w_ml: 0.0,
            w_rule: 1.0,
            w_nodchip: 0.0,
        };
        let p = EnsemblePredictor::new(None, None, cfg);
        let r = p.predict(&pos, 0);
        assert!((r.progress - r.rule_signal).abs() < 1e-5);
    }

    #[test]
    fn ensemble_all_zero_weights_returns_zero() {
        let pos = startpos();
        let cfg = EnsembleConfig {
            w_ml: 0.0,
            w_rule: 0.0,
            w_nodchip: 0.0,
        };
        let p = EnsemblePredictor::new(None, None, cfg);
        let r = p.predict(&pos, 0);
        assert_eq!(r.progress, 0.0);
    }

    #[test]
    fn ensemble_v2_smoke() {
        // V2 ensemble は V1 と並立で動作。 ml_v2 = None / progress_weights = None で
        // も rule signal のみで動く。
        let pos = startpos();
        let p = EnsemblePredictorV2::new(None, None, EnsembleConfig::default());
        let r = p.predict(&pos, 0);
        assert!((0.0..=1.0).contains(&r.progress));
        // ml_v2=None なので ml_signal は 0.5
        assert!((r.ml_signal - 0.5).abs() < 1e-6);
        // nodchip も None で 0.5
        assert!((r.nodchip_signal - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ensemble_v2_with_random_model() {
        // random V2 model を ml=1.0 で流すと progress = ml_signal が支配
        let pos = startpos();
        let cfg = EnsembleConfig {
            w_ml: 1.0,
            w_rule: 0.0,
            w_nodchip: 0.0,
        };
        let model = crate::model_v2::MlProgressPredictorV2::from_random(31);
        let p = EnsemblePredictorV2::new(Some(&model), None, cfg);
        let r = p.predict(&pos, 0);
        assert!((0.0..=1.0).contains(&r.progress));
        assert!((r.progress - r.ml_signal).abs() < 1e-5);
    }

    #[test]
    fn ensemble_mid_signals_yield_mid_progress() {
        // 全 source が 0.5 を返す状況: ml model なし (0.5)、 nodchip None (0.5)
        // rule は startpos で 0 近辺なので、 rule 重み下げて確認
        let pos = startpos();
        let cfg = EnsembleConfig {
            w_ml: 1.0,
            w_rule: 0.0,
            w_nodchip: 1.0,
        };
        let p = EnsemblePredictor::new(None, None, cfg);
        let r = p.predict(&pos, 0);
        // ml=0.5、 nodchip=0.5 平均 = 0.5
        assert!((r.progress - 0.5).abs() < 1e-5, "got {}", r.progress);
    }
}
