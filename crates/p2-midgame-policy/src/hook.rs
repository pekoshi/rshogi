//! αβ root に挿入する高位 API: [`MidgamePolicyHook`]。
//!
//! USI 層から呼び出して、 ID (Iterative Deepening) 1 回ごとに 1 度 root
//! 局面で Transformer を走らせ、 root move list を policy score 順に並び替える。
//!
//! ## なぜ root だけにしたか (第 1 段階)
//!
//! - 中盤の internal node に hook を入れると、 1 推論 100μs-1ms × ノード数で
//!   NPS が壊滅する。 cache hit が無いと致命的。
//! - 一方 root 1 回なら推論時間は ID 全体の中で <1% の overhead に収まる
//!   (depth 20 で数秒探索する文脈で 1ms は誤差)。
//! - root prior による戦術的見落とし削減は YOZ 系研究で実証済み (top-1
//!   accuracy が良ければ first move cut-off 率が顕著に上昇)。
//! - 第 2 段階 (depth >= MidgamePolicyMinDepth の internal node hook) は
//!   selfplay で root prior の効果を実証してから追加する。
//!
//! ## 呼び出し方
//!
//! ```ignore
//! // 1) Hook を作る (起動時、 USI option で controlled)
//! let mut hook = MidgamePolicyHook::new(
//!     MidgamePolicyConfig::default(),
//!     Box::new(my_evaluator),
//! );
//!
//! // 2) go の冒頭で root_moves を並び替える
//! let mut scored = PolicyScoreList::new();
//! hook.score_root_moves(&pos, &root_moves, &mut scored)?;
//! scored.sort_by_logit_desc();
//! // → scored.as_slice() の Move を新しい root order として上書きする
//! ```

use rshogi_core::position::Position;
use rshogi_core::types::Move;

use crate::cache::PolicyCache;
use crate::move_lookup::POLICY_SIZE;
use crate::ordering::{PolicyScoreList, score_moves};
use crate::runtime_iface::{InferenceError, PolicyEvaluator};

/// 中盤 policy hook の設定。
#[derive(Clone, Debug)]
pub struct MidgamePolicyConfig {
    /// 有効化フラグ。 false なら推論をスキップして即 fallback。
    pub enabled: bool,
    /// 中盤 internal node で hook を発火する最小 depth。
    /// **本クレートの第 1 段階実装では root 専用なので未参照。 第 2 段階の
    ///  internal node hook で参照する想定で予約する。**
    pub min_depth: i32,
    /// policy top-K を最優先に挿入する K。 0 なら全 root move を policy 順に
    /// 並べる (純粋な reorder)。 1-N なら top-K 以外は元順序を維持する hybrid。
    pub top_k: usize,
    /// `final_score = α * policy_logit + history` の α。 root 並び替えでは
    /// policy_logit 単独で sort するため未使用 (内部 node hook で参照予定)。
    pub policy_alpha: f32,
    /// Policy cache のエントリ数 (power-of-2 に切り上げ)。
    pub cache_entries: usize,
}

impl Default for MidgamePolicyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_depth: 8,
            top_k: 4,
            policy_alpha: 32.0,
            cache_entries: 1024,
        }
    }
}

/// Hook 本体。 evaluator (Box<dyn>) + cache + 推論用 scratch buffer を保有。
pub struct MidgamePolicyHook {
    config: MidgamePolicyConfig,
    evaluator: Option<Box<dyn PolicyEvaluator>>,
    cache: PolicyCache,
    /// 推論 scratch (heap 1 度確保、 以降 reuse)。
    scratch_logits: Box<[f32; POLICY_SIZE]>,
}

impl MidgamePolicyHook {
    /// Hook を生成。 `evaluator` が `None` なら hook は無効状態。
    pub fn new(config: MidgamePolicyConfig, evaluator: Option<Box<dyn PolicyEvaluator>>) -> Self {
        let cache = PolicyCache::new(config.cache_entries);
        let scratch_logits = Box::new([0f32; POLICY_SIZE]);
        Self {
            config,
            evaluator,
            cache,
            scratch_logits,
        }
    }

    /// 設定を取り出す (USI option 経由で再設定するため mut)。
    pub fn config_mut(&mut self) -> &mut MidgamePolicyConfig {
        &mut self.config
    }

    /// 設定参照。
    pub fn config(&self) -> &MidgamePolicyConfig {
        &self.config
    }

    /// evaluator が設定されていて、 hook が有効か。
    pub fn is_active(&self) -> bool {
        self.config.enabled && self.evaluator.is_some()
    }

    /// evaluator を差し替える (model 切替用)。
    pub fn set_evaluator(&mut self, evaluator: Option<Box<dyn PolicyEvaluator>>) {
        self.evaluator = evaluator;
        // 別 model に切り替わると logits が無効になるため cache をクリア。
        self.cache.clear();
    }

    /// Cache 統計 (info 文字列で USI に流す用途)。
    pub fn cache_stats(&self) -> crate::cache::PolicyCacheStats {
        self.cache.stats()
    }

    /// Root move 列を policy score でソートして `out` に書き込む。
    ///
    /// - hit (cache 経由) なら推論をスキップ。
    /// - hook 無効時 (enabled=false / evaluator なし) は `Err(NotConfigured)` を返す。
    ///   呼び出し側は元の root order を維持する fallback を取る。
    ///
    /// `out` は呼び出し前に `clear()` される。
    pub fn score_root_moves(
        &mut self,
        pos: &Position,
        moves: &[Move],
        out: &mut PolicyScoreList,
    ) -> Result<(), InferenceError> {
        if !self.is_active() {
            out.clear();
            return Err(InferenceError::NotConfigured);
        }

        let key = pos.key();
        let logits_slice: &[f32] = if let Some(cached) = self.cache.probe(key) {
            // hit
            &cached[..]
        } else {
            // miss: 推論
            let ev = self.evaluator.as_mut().ok_or(InferenceError::NotConfigured)?;
            ev.evaluate(pos, &mut self.scratch_logits[..])?;
            self.cache.store(key, &self.scratch_logits[..]);
            &self.scratch_logits[..]
        };
        score_moves(moves, pos.side_to_move(), logits_slice, out);
        Ok(())
    }

    /// `score_root_moves` + sort をまとめて行う。
    pub fn score_and_sort(
        &mut self,
        pos: &Position,
        moves: &[Move],
        out: &mut PolicyScoreList,
    ) -> Result<(), InferenceError> {
        self.score_root_moves(pos, moves, out)?;
        out.sort_by_logit_desc();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_iface::ConstantPolicyEvaluator;
    use rshogi_core::movegen::{MoveList, generate_legal};
    use rshogi_core::position::{Position, SFEN_HIRATE};
    use rshogi_core::types::{File, Rank, Square};

    fn hirate() -> Position {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        pos
    }

    fn legal_moves(pos: &Position) -> Vec<Move> {
        let mut list = MoveList::new();
        generate_legal(pos, &mut list);
        list.iter().copied().collect()
    }

    #[test]
    fn hook_disabled_returns_not_configured() {
        let mut hook = MidgamePolicyHook::new(MidgamePolicyConfig::default(), None);
        let pos = hirate();
        let moves = legal_moves(&pos);
        let mut out = PolicyScoreList::new();
        let err = hook.score_root_moves(&pos, &moves, &mut out).unwrap_err();
        assert!(matches!(err, InferenceError::NotConfigured));
    }

    #[test]
    fn hook_with_constant_evaluator_returns_top_k() {
        // 全 logits=0 で constant evaluator → score_moves は全手 logit=0 を返す。
        let config = MidgamePolicyConfig {
            enabled: true,
            top_k: 4,
            ..Default::default()
        };
        let evaluator: Box<dyn PolicyEvaluator> = Box::new(ConstantPolicyEvaluator::zeros());
        let mut hook = MidgamePolicyHook::new(config, Some(evaluator));
        assert!(hook.is_active());

        let pos = hirate();
        let moves = legal_moves(&pos);
        let mut out = PolicyScoreList::new();
        hook.score_root_moves(&pos, &moves, &mut out).unwrap();
        assert_eq!(out.len(), moves.len());
        // すべて logit=0
        for sm in out.as_slice() {
            assert_eq!(sm.logit, 0.0);
        }
    }

    /// 偏った logits を返す test evaluator。 7g7f だけ高 logit。
    struct BiasedEvaluator {
        target_idx: usize,
    }
    impl PolicyEvaluator for BiasedEvaluator {
        fn evaluate(
            &mut self,
            _pos: &Position,
            out_logits: &mut [f32],
        ) -> Result<(), InferenceError> {
            crate::runtime_iface::check_logits_buffer(out_logits)?;
            out_logits.fill(0.0);
            out_logits[self.target_idx] = 50.0;
            Ok(())
        }
        fn name(&self) -> &'static str {
            "biased"
        }
    }

    #[test]
    fn policy_prior_returns_top_k_test() {
        // 7g7f を最高 logit にする → sort 後の先頭が 7g7f
        let pos = hirate();
        let color = pos.side_to_move();
        let target = Move::new_move(
            Square::new(File::File7, Rank::Rank7),
            Square::new(File::File7, Rank::Rank6),
            false,
        );
        let target_idx = crate::move_lookup::make_move_label(target, color);

        let config = MidgamePolicyConfig {
            enabled: true,
            ..Default::default()
        };
        let evaluator: Box<dyn PolicyEvaluator> = Box::new(BiasedEvaluator { target_idx });
        let mut hook = MidgamePolicyHook::new(config, Some(evaluator));

        let moves = legal_moves(&pos);
        let mut out = PolicyScoreList::new();
        hook.score_and_sort(&pos, &moves, &mut out).unwrap();
        // generate_legal は piece info 付きの Move を返すため raw(16bit) で比較。
        assert_eq!(out.as_slice()[0].mv.raw(), target.raw());
        assert!((out.as_slice()[0].logit - 50.0).abs() < 1e-5);
    }

    #[test]
    fn policy_cache_hit_test() {
        // 同一局面で 2 度呼んで 2 回目が cache hit になることを確認
        let config = MidgamePolicyConfig {
            enabled: true,
            ..Default::default()
        };
        let evaluator: Box<dyn PolicyEvaluator> = Box::new(ConstantPolicyEvaluator { fill: 0.0 });
        let mut hook = MidgamePolicyHook::new(config, Some(evaluator));

        let pos = hirate();
        let moves = legal_moves(&pos);
        let mut out = PolicyScoreList::new();
        hook.score_root_moves(&pos, &moves, &mut out).unwrap();
        let stats_after_first = hook.cache_stats();
        assert_eq!(stats_after_first.misses, 1);
        assert_eq!(stats_after_first.hits, 0);
        assert_eq!(stats_after_first.inserts, 1);

        hook.score_root_moves(&pos, &moves, &mut out).unwrap();
        let stats_after_second = hook.cache_stats();
        assert_eq!(stats_after_second.hits, 1);
        assert_eq!(stats_after_second.misses, 1);
        // inserts は増えない (cache hit のため)
        assert_eq!(stats_after_second.inserts, 1);
    }

    #[test]
    fn move_lookup_roundtrip_test() {
        // legal moves に make_move_label を回し、 全 idx が 0..POLICY_SIZE 内であること
        let pos = hirate();
        let color = pos.side_to_move();
        let moves = legal_moves(&pos);
        for mv in moves {
            let idx = crate::move_lookup::make_move_label(mv, color);
            assert!(idx < POLICY_SIZE);
        }
    }

    #[test]
    fn set_evaluator_clears_cache() {
        let config = MidgamePolicyConfig {
            enabled: true,
            ..Default::default()
        };
        let evaluator: Box<dyn PolicyEvaluator> = Box::new(ConstantPolicyEvaluator { fill: 0.0 });
        let mut hook = MidgamePolicyHook::new(config, Some(evaluator));

        let pos = hirate();
        let moves = legal_moves(&pos);
        let mut out = PolicyScoreList::new();
        hook.score_root_moves(&pos, &moves, &mut out).unwrap();
        assert_eq!(hook.cache_stats().inserts, 1);

        // 別 model に切り替え → cache クリア → probe は miss
        let new_ev: Box<dyn PolicyEvaluator> = Box::new(ConstantPolicyEvaluator { fill: 1.0 });
        hook.set_evaluator(Some(new_ev));

        hook.score_root_moves(&pos, &moves, &mut out).unwrap();
        // 1 度目の insert + 切替後 1 度目の insert = 2 inserts
        // 切替後の probe は miss (cache クリアされたので)
        let stats = hook.cache_stats();
        assert_eq!(stats.inserts, 2);
    }
}
