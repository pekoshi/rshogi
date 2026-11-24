use rshogi_core::position::Position;
use rshogi_core::search::{LimitsType, SearchInfo};

use crate::adapter::AlphaBetaAdapter;
use crate::detector::EndgameDetector;
use crate::phase::{self, EndgameFeatures, PhaseTracker, compute_features};
use crate::puct::{PuctBackend, PuctLimits};
use crate::types::UnifiedSearchResult;

/// CPU/GPU 動的切替探索コントローラー
///
/// root レベルで終盤判定を行い、αβ または PUCT にルーティングする。
/// PUCT 未設定時は常に αβ にフォールバックする。
pub struct SearchController {
    detector: EndgameDetector,
    alpha_beta: AlphaBetaAdapter,
    puct: Option<Box<dyn PuctBackend>>,
    phase_tracker: PhaseTracker,
}

impl SearchController {
    pub fn new(tt_size_mb: usize) -> Self {
        Self {
            detector: EndgameDetector::new(),
            alpha_beta: AlphaBetaAdapter::new(tt_size_mb),
            puct: None,
            phase_tracker: PhaseTracker::new(),
        }
    }

    /// PUCT バックエンドを設定
    pub fn set_puct(&mut self, puct: Box<dyn PuctBackend>) {
        self.puct = Some(puct);
    }

    /// 定跡ライン継続判定 (VPT/PUCT の自己判断)
    ///
    /// 条件:
    /// - VPT policy entropy < 0.30 (1 手に集中)
    /// - top1 visit / total visit > 0.70 (PUCT も同手に集中)
    /// 両方 met なら true、 αβ 切替を抑止し PUCT 維持。
    ///
    /// VPT 未統合時は false (現 placeholder)。 統合後 VPT trait に
    /// `policy_entropy()` と `top1_visit_ratio()` API を追加して実装。
    pub fn detect_joseki_line(&self) -> bool {
        // PUCT backend が VPT trait 拡張を持つかチェック (将来実装)
        // 今は false 固定
        const ENTROPY_THRESHOLD: f32 = 0.30;
        const VISIT_RATIO_THRESHOLD: f32 = 0.70;
        let _ = (ENTROPY_THRESHOLD, VISIT_RATIO_THRESHOLD);
        false
    }

    /// root レベル探索
    pub fn search_root<F>(
        &mut self,
        pos: &mut Position,
        limits: LimitsType,
        on_info: Option<F>,
    ) -> UnifiedSearchResult
    where
        F: FnMut(&SearchInfo),
    {
        let signal = self.detector.evaluate(pos);

        // signal.score は detector が計算済み (絶対終盤の即決時は 1.0)。
        // PhaseTracker / SearchMode 判定にそのまま再利用する。
        let _phase = phase::determine_phase(signal.score, &mut self.phase_tracker);
        let search_mode =
            phase::determine_search_mode(signal.score, pos.in_check(), 0, 0.0, 0, 0.0);

        // ★ 定跡ライン継続検出 (VPT 完成後に有効化)
        // 角換わり 100+ 手等の定跡を VPT が学習している場合、
        // policy entropy + visit ratio で「定跡レベル自信」 を判定し PUCT 維持。
        // book file 不参照、 全て VPT 学習結果の self-judgment。
        let is_in_joseki_line = self.detect_joseki_line();

        // αβ パス: 終盤 OR PUCT 未設定 (但し定跡ライン中は除外)
        if (signal.use_alpha_beta && !is_in_joseki_line) || self.puct.is_none() {
            log::info!(
                "[endgame-switch] backend=AlphaBeta reason={:?} search_mode={:?}",
                signal.reason,
                search_mode
            );
            let result = self.alpha_beta.search(pos, limits, on_info);
            return UnifiedSearchResult::from_ab(result, signal.reason);
        }

        // PUCT パス
        log::info!("[endgame-switch] backend=Puct search_mode={:?}", search_mode);
        let puct = self.puct.as_mut().unwrap();
        let puct_limits = PuctLimits {
            max_nodes: limits.nodes,
            max_time_ms: limits.movetime as u64,
            search_mode,
            game_phase: None,
            endgame_score: 0.0,
        };
        let result = puct.search(pos, &puct_limits);
        UnifiedSearchResult::from_puct(result)
    }

    /// 探索中断
    pub fn stop(&self) {
        self.alpha_beta.stop();
        if let Some(puct) = &self.puct {
            puct.stop();
        }
    }

    /// 内部 αβ アダプターへのアクセス
    pub fn alpha_beta_mut(&mut self) -> &mut AlphaBetaAdapter {
        &mut self.alpha_beta
    }

    /// EndgameDetector へのアクセス
    pub fn detector_mut(&mut self) -> &mut EndgameDetector {
        &mut self.detector
    }

    /// PhaseTracker へのアクセス
    pub fn phase_tracker_mut(&mut self) -> &mut PhaseTracker {
        &mut self.phase_tracker
    }

    /// 現在の EndgameFeatures を計算（デバッグ・レポート用）
    pub fn compute_current_features(&self, pos: &Position) -> EndgameFeatures {
        compute_features(pos, 0)
    }
}
