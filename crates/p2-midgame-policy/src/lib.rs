//! # p2-midgame-policy
//!
//! αβ 中盤探索の move ordering を Transformer policy で補完するモジュール。
//!
//! ## 設計方針 (default OFF / opt-in)
//!
//! - αβ NPS への影響を最小化するため、 default は完全 OFF。
//! - USI option `UseMidgamePolicy` で opt-in。
//! - 推論コストは「policy cache hit 時は 0」「miss 時は ~100μs-1ms」 と
//!   非対称なので、 cache hit 率を稼ぐために TT key (Zobrist) 直結 cache を採用。
//! - policy → Move 変換は dlshogi 互換 `make_move_label` の **forward 計算**
//!   で代替する (legal moves 30-100 個に対して O(N) 計算、 逆引きテーブル不要)。
//!
//! ## node hook 位置
//!
//! 現在の本クレートは **root レベル ordering** に絞った実装を提供する。
//! 中盤 αβ の各 internal node に hook を入れる方式は invasive かつ NPS への
//! 影響が大きいため、 selfplay で root prior の効果を実証してから拡張する。
//!
//! - 第 1 段階 (本クレート、 本タスク): root move ordering の policy 補正
//!   - 1 go コマンドにつき 1 回推論、 root の `RootMoves` 順序を policy で再ソート
//!   - α 比 = `MidgamePolicyAlpha` で history と混合
//! - 第 2 段階 (将来): depth >= `MidgamePolicyMinDepth` かつ PV node の場合のみ
//!   internal node の TT 手探索後に policy top-K を最優先に挿入
//!   - これは `rshogi-core::search::MovePicker` への `policy_hook: Option<&dyn PolicyHook>`
//!     注入が必要。 本タスクでは trait の skeleton のみ提供。
//!
//! ## crate module 構成
//!
//! - [`runtime_iface`] — Transformer 推論 backend の共通 trait
//!   ([`PolicyEvaluator`])。 並行 Agent B の `p2-transformer-runtime` に依存
//!   しないように、 内部 trait を切る。 `transformer-runtime` feature 有効時
//!   のみ adapter を提供。
//! - [`move_lookup`] — `Move` ↔ dlshogi 2187 idx の双方向 mapping。
//! - [`cache`] — policy cache (Zobrist key 直結、 直接マップ式)。
//! - [`ordering`] — legal move list を policy score で並び替える core。
//! - [`hook`] — αβ root entry に注入する高位 API ([`MidgamePolicyHook`])。

pub mod cache;
pub mod hook;
pub mod move_lookup;
pub mod ordering;
pub mod runtime_iface;

pub use cache::{PolicyCache, PolicyCacheEntry, PolicyCacheStats};
pub use hook::{MidgamePolicyConfig, MidgamePolicyHook};
pub use move_lookup::{POLICY_SIZE, make_move_label};
pub use ordering::{PolicyScoreList, ScoredMove, score_moves};
pub use runtime_iface::{InferenceError, PolicyEvaluator};

#[cfg(feature = "transformer-runtime")]
pub mod runtime_adapter;
