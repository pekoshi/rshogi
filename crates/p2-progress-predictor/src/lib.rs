//! # p2-progress-predictor
//!
//! 将棋局面の **進行度 (game progression)** を機械学習で予測する小型 MLP。
//!
//! ## ゴール
//!
//! Hybrid 探索 (序中盤 PUCT / 終盤 αβ) における routing 判定を、
//! ルールベース [`p2_search::phase::compute_endgame_score`] よりも
//! 正確に行うための ML 補助。
//!
//! ## 設計判断 (採用案)
//!
//! - **モデル**: 小型 MLP (input=32, hidden=32 → 16, output=1)
//!   - 約 1.6K params、 推論 < 1 μs (pure Rust + 静的 array)
//!   - candle はあくまで訓練側のみ依存。 推論は heap allocation 0
//! - **target label**: `progress(t) = game_ply(t) / total_plies(game)`
//!   - 1 局単位で正規化、 [0, 1] の連続値
//!   - 棋譜終局時点の総手数を分母とする
//!   - 局単独で求められない PSV の場合は heuristic 疑似 label にフォールバック
//! - **入力**: hand-crafted feature 32 次元 (allocation 0)
//!   - 既存 [`p2_search::phase::EndgameFeatures`] の 10 次元 +
//!     駒種別 board count 13 次元 + 王安全 / 入玉特徴 9 次元
//!   - Transformer tokens は使わない (1 μs 目標を超える)
//! - **既存系との関係**: 並立で hook (`MlProgressPredictor::predict`) を提供。
//!   `PhaseTracker` / `EndgameDetector` の API は変更しない。
//!   呼び出し側で `compute_endgame_score` か `predict` か選択する。
//!
//! ## 不採用案
//!
//! - **既存 progress8kpabs 拡張**: NNUE 内部の 8-bucket selector に特化されており
//!   汎用化が難しい。 別 trainer line になるため新規 MLP の方が clean。
//! - **小型 Transformer**: 推論 1 μs 達成困難 (token 構築 1 μs + attention)。
//! - **PhaseTracker 全置換**: 訓練データなしで動かない、 fallback 必須 →
//!   並立 hook の方が安全。
//!
//! ## モジュール構成
//!
//! - [`features`]: `Position` → `[f32; FEATURE_DIM]` の抽出 (allocation 0)
//! - [`model`]: 推論専用 MLP (重み array + forward)
//! - [`integration`]: `PhaseTracker` 互換 hook trait
//! - [`label`]: target label の生成ロジック (heuristic fallback)
//! - [`weights`]: safetensors weight serialization (trainer feature 限定)
//!
//! ## API 例
//!
//! ```ignore
//! use p2_progress_predictor::{MlProgressPredictor, features::compute_features};
//! let predictor = MlProgressPredictor::from_default(); // identity init
//! let score = predictor.predict(&position);            // f32 in [0,1]
//! ```

pub mod ensemble;
pub mod features;
pub mod features_v2;
pub mod hysteresis;
pub mod integration;
pub mod label;
pub mod model;
pub mod model_v2;

/// AVX2 SIMD microkernel (x86 / x86_64 のみ)。
///
/// runtime detection は `MlProgressPredictor::predict_features` 内で行う。
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod avx2;

/// V2 model 用 AVX2 SIMD microkernel (48→64→32→16→1)。
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod avx2_v2;

#[cfg(feature = "trainer")]
pub mod weights;

/// V2 weights serialization (trainer feature 専用)
#[cfg(feature = "trainer")]
pub mod weights_v2;

pub use ensemble::{
    predict_via_v2, EnsembleConfig, EnsemblePredictor, EnsemblePredictorV2, EnsembleResult,
};
pub use features::{compute_features, ProgressFeatures, FEATURE_DIM};
pub use features_v2::{compute_features_v2, ProgressFeaturesV2, FEATURE_DIM_V2};
pub use hysteresis::{BackendChoice, HysteresisConfig, SwitchingStateMachine};
pub use integration::{ProgressionExt, ScoreSource};
pub use label::{compute_label, LabelMode};
pub use model::{MlProgressPredictor, ProgressWeights, HIDDEN1, HIDDEN2};
pub use model_v2::{
    MlProgressPredictorV2, WeightsV2, HIDDEN1_V2, HIDDEN2_V2, HIDDEN3_V2, OUTPUT_V2,
};
