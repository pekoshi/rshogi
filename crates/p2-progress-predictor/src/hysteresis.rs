//! Backend choice (PUCT / αβ) のヒステリシス制御
//!
//! progress 値が閾値を越えた瞬間に切り替えるとフリッカ (連続切替) が起きるため、
//! 以下 2 段で安定化する:
//!
//! 1. **2 段閾値**: PUCT→αβ は 0.70 以上、 αβ→PUCT は 0.50 以下 (ヒステリシス幅 0.20)
//! 2. **最小滞在 ply** (`min_dwell_ply`): 切替後 N ply は再切替を抑止
//! 3. **連続同一 signal カウント** (`min_consecutive`): N 回連続で「切替したい」 signal が
//!    出てから実際に切替確定
//!
//! ## ホットパス制約
//!
//! - allocation 0、 struct field のみ
//! - update() は 1 回の compare で済むのが理想 (分岐 < 10)

/// 探索 backend の選択肢
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendChoice {
    /// 序中盤向け Monte-Carlo (PUCT)
    Puct,
    /// 終盤向け αβ
    AlphaBeta,
}

/// ヒステリシス設定。 ホットパス外で構築されることを想定 (`Copy`)。
#[derive(Clone, Copy, Debug)]
pub struct HysteresisConfig {
    /// PUCT → αβ への切替閾値 (progress がこれ以上で αβ 検討開始)
    pub puct_to_ab_threshold: f32,
    /// αβ → PUCT への切替閾値 (progress がこれより下で PUCT 検討開始)
    pub ab_to_puct_threshold: f32,
    /// 切替直後の固定 ply 数 (この間は signal を無視)
    pub min_dwell_ply: u32,
    /// 切替確定までに必要な同一 signal 連続回数
    pub min_consecutive: u32,
}

impl Default for HysteresisConfig {
    fn default() -> Self {
        Self {
            puct_to_ab_threshold: 0.70,
            ab_to_puct_threshold: 0.50,
            min_dwell_ply: 3,
            min_consecutive: 2,
        }
    }
}

/// backend 選択の状態機械。
///
/// `update(progress, current_ply, &config)` を毎手呼び、 確定した backend を取得する。
#[derive(Clone, Copy, Debug)]
pub struct SwitchingStateMachine {
    state: BackendChoice,
    last_switch_ply: u32,
    consecutive_same_signal: u32,
}

impl SwitchingStateMachine {
    /// 初期状態 (典型的には Puct から開始)。
    #[inline]
    pub fn new(initial: BackendChoice) -> Self {
        Self {
            state: initial,
            last_switch_ply: 0,
            consecutive_same_signal: 0,
        }
    }

    /// 現在確定している backend。
    #[inline]
    pub fn current(&self) -> BackendChoice {
        self.state
    }

    /// progress と現在 ply、 設定から backend を更新し、 最新値を返す。
    ///
    /// ## 動作
    ///
    /// - `min_dwell_ply` 経過前は signal 無視 (連続カウントもリセットしない)
    /// - 現在の state と逆方向への signal が `min_consecutive` 回連続したら切替
    /// - 順方向 (現在 state と同じ) signal は連続カウントを 0 にする
    pub fn update(
        &mut self,
        progress: f32,
        current_ply: u32,
        config: &HysteresisConfig,
    ) -> BackendChoice {
        // 切替してほしい signal を判定
        let want_switch = match self.state {
            BackendChoice::Puct => progress >= config.puct_to_ab_threshold,
            BackendChoice::AlphaBeta => progress < config.ab_to_puct_threshold,
        };

        // dwell 期間中は signal 評価をスキップ (連続カウントも維持)
        let dwell_active = current_ply.saturating_sub(self.last_switch_ply) < config.min_dwell_ply;
        if dwell_active {
            return self.state;
        }

        if want_switch {
            self.consecutive_same_signal = self.consecutive_same_signal.saturating_add(1);
            if self.consecutive_same_signal >= config.min_consecutive {
                // 切替確定
                self.state = match self.state {
                    BackendChoice::Puct => BackendChoice::AlphaBeta,
                    BackendChoice::AlphaBeta => BackendChoice::Puct,
                };
                self.last_switch_ply = current_ply;
                self.consecutive_same_signal = 0;
            }
        } else {
            // 切替不要なら累積をリセット
            self.consecutive_same_signal = 0;
        }
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> HysteresisConfig {
        HysteresisConfig::default()
    }

    #[test]
    fn puct_to_ab_after_consecutive_high_progress() {
        let mut sm = SwitchingStateMachine::new(BackendChoice::Puct);
        let c = cfg();
        // 連続 2 回 progress >= 0.70 で切替
        // dwell は last_switch_ply=0 開始なので current_ply >= 3 から signal 評価
        // ※ 最初の current_ply=0 から: 0-0=0 < 3 で dwell_active=true → skip
        //    current_ply=3: dwell 解除、 want_switch=true → consecutive=1
        //    current_ply=4: consecutive=2 == min_consecutive → 切替
        let r0 = sm.update(0.8, 0, &c);
        assert_eq!(r0, BackendChoice::Puct, "dwell suppresses ply 0");
        let r1 = sm.update(0.8, 3, &c);
        assert_eq!(r1, BackendChoice::Puct, "consecutive=1, not enough");
        let r2 = sm.update(0.8, 4, &c);
        assert_eq!(r2, BackendChoice::AlphaBeta, "consecutive=2 -> switch");
    }

    #[test]
    fn ab_to_puct_after_consecutive_low_progress() {
        let mut sm = SwitchingStateMachine::new(BackendChoice::AlphaBeta);
        // 初回切替を成立させたら last_switch_ply は 0 のまま
        let c = cfg();
        let r0 = sm.update(0.2, 3, &c);
        assert_eq!(r0, BackendChoice::AlphaBeta);
        let r1 = sm.update(0.2, 4, &c);
        assert_eq!(r1, BackendChoice::Puct);
    }

    #[test]
    fn dwell_suppresses_immediate_reswitch() {
        let mut sm = SwitchingStateMachine::new(BackendChoice::Puct);
        let c = cfg();
        // ply 3, 4 で Puct->AB 切替
        sm.update(0.8, 3, &c);
        let after_switch = sm.update(0.8, 4, &c);
        assert_eq!(after_switch, BackendChoice::AlphaBeta);
        // すぐに低 progress に振れても dwell 期間 (3 ply) は AB を維持
        let r5 = sm.update(0.2, 5, &c);
        assert_eq!(r5, BackendChoice::AlphaBeta, "dwell suppresses ply 5");
        let r6 = sm.update(0.2, 6, &c);
        assert_eq!(r6, BackendChoice::AlphaBeta, "dwell suppresses ply 6");
        // ply 7 = 4+3 で dwell 解除、 連続カウント開始
        let r7 = sm.update(0.2, 7, &c);
        assert_eq!(r7, BackendChoice::AlphaBeta, "consec=1");
        let r8 = sm.update(0.2, 8, &c);
        assert_eq!(r8, BackendChoice::Puct, "consec=2 -> switch back");
    }

    #[test]
    fn consecutive_resets_when_signal_flips() {
        let mut sm = SwitchingStateMachine::new(BackendChoice::Puct);
        let c = cfg();
        // ply 3: want_switch=true, consec=1
        sm.update(0.8, 3, &c);
        // ply 4: progress 落ちる -> reset (consec=0)
        let r4 = sm.update(0.4, 4, &c);
        assert_eq!(r4, BackendChoice::Puct);
        // ply 5: want_switch=true, consec=1 (まだ切替しない)
        let r5 = sm.update(0.8, 5, &c);
        assert_eq!(r5, BackendChoice::Puct, "consec restarted from 1");
        // ply 6: consec=2 -> switch
        let r6 = sm.update(0.8, 6, &c);
        assert_eq!(r6, BackendChoice::AlphaBeta);
    }

    #[test]
    fn initial_state_preserved() {
        let sm_p = SwitchingStateMachine::new(BackendChoice::Puct);
        assert_eq!(sm_p.current(), BackendChoice::Puct);
        let sm_a = SwitchingStateMachine::new(BackendChoice::AlphaBeta);
        assert_eq!(sm_a.current(), BackendChoice::AlphaBeta);
    }
}
