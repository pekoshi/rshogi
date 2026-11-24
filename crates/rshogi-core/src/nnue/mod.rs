//! NNUE評価関数モジュール
//!
//! Efficiently Updatable Neural Network による局面評価。
//! YaneuraOu の HalfKP 256x2-32-32 アーキテクチャを Rust で実装する。
//!
//! サポートするアーキテクチャ:
//! - **HalfKP**: 従来のclassic NNUE（水匠/tanuki互換）
//! - **HalfKA**: nnue-pytorch互換（Non-mirror）
//! - **HalfKA_hm^**: nnue-pytorch互換（Half-Mirror + Factorization）
//!
//! # const generics 版統一実装
//!
//! `NetworkHalfKA_hm<L1, L2, L3, A>` で複数のアーキテクチャに対応:
//! - L1: FT出力次元（256, 512, 1024）
//! - L2: 隠れ層1出力次元（8, 32）
//! - L3: 隠れ層2出力次元（32, 96）
//! - A: 活性化関数（CReLU, SCReLU, PairwiseCReLU）
//!
//! - ネットワーク構造の読み込み（`Network::load` / `init_nnue`）
//! - 入力特徴量（HalfKP: 自玉×駒配置）の計算と変換（`BonaPiece` / `FeatureTransformer`）
//! - Accumulator による差分更新可能な中間表現の保持（`diff::get_changed_features` を用いた増分更新 + フォールバック全計算）
//! - AffineTransform + ClippedReLU による 512→32→32→1 の多層パーセプトロン
//! - NNUE 未初期化時のフォールバック駒得評価

// LMAS (Lightweight Multi-Aggregate Shortcut) は 2026-05-25 廃止 (= 物理削除)。
// 代替の革新 bit は `nnue-pin-aware` (pin_features) と `nnue-dual-psqt` (dual_psqt) で実装。

pub mod ablation;
mod accumulator;
pub mod accumulator_layer_stacks;
mod accumulator_stack_variant;
pub mod activation;
pub mod aliases;
mod bona_piece;
mod bona_piece_halfka;
mod bona_piece_halfka_hm;
mod constants;
pub mod dense_features;
mod diff;
mod evaluator;
mod feature_transformer;
mod feature_transformer_layer_stacks;
pub mod features;
pub(crate) mod halfka;
pub(crate) mod halfka_hm;
pub(crate) mod halfkp;
mod layer_stacks;
mod layers;
mod leb128;
#[macro_use]
pub mod macros;
mod network;
pub(crate) mod network_halfka;
pub(crate) mod network_halfka_hm;
pub(crate) mod network_halfkp;
mod network_layer_stacks;
// shogi2344_v16 系は 2026-05-12 削除 (HalfKAv2_hm 1536-8-96 に完全切替)
#[cfg(feature = "nnue-activity-risk-psqt")]
pub mod activity_risk;
#[cfg(feature = "nnue-attack-dev-psqt")]
pub mod attack_dev;
#[cfg(feature = "nnue-attack-relay")]
pub mod attack_relay;
#[cfg(feature = "nnue-attack-relay-v2")]
pub mod attack_relay_v2;
#[cfg(feature = "nnue-drop-pressure-psqt")]
pub mod drop_pressure;
#[cfg(feature = "nnue-dual-psqt")]
pub mod dual_cross;
#[cfg(feature = "nnue-dual-cross-geometry")]
pub mod dual_cross_geometry;
#[cfg(any(
    feature = "nnue-dual-psqt",
    feature = "nnue-kingring-psqt",
    feature = "nnue-pairshape-psqt"
))]
pub mod dual_delta;
#[cfg(feature = "nnue-dual-psqt")]
pub mod dual_psqt;
#[cfg(feature = "nnue-eval-bias-corr")]
pub mod eval_bias_corr;
#[cfg(feature = "nnue-king-danger-psqt")]
pub mod king_danger;
#[cfg(feature = "nnue-king-escape-psqt")]
pub mod king_escape;
#[cfg(feature = "nnue-king-mobility-psqt")]
pub mod king_mobility;
#[cfg(feature = "nnue-kingring-psqt")]
pub mod kingring;
#[cfg(any(feature = "nnue-kzt-psqt", feature = "nnue-kzt-ft"))]
pub mod kzt;
#[cfg(feature = "nnue-odh")]
pub mod odh;
#[cfg(feature = "nnue-pairshape-psqt")]
pub mod pairshape;
pub mod piece_list;
#[cfg(feature = "nnue-pin-aware")]
pub mod pin_features;
pub mod prelude;
#[cfg(feature = "nnue-scalar-threat")]
pub mod scalar_threat;
#[cfg(feature = "nnue-slider-xray-psqt")]
pub mod slider_xray;
pub mod spec;
pub mod stats;
#[cfg(any(feature = "nnue-threat", feature = "nnue-scalar-threat"))]
pub(crate) mod threat_exclusion;
#[cfg(any(feature = "nnue-threat", feature = "nnue-scalar-threat"))]
pub(crate) mod threat_features;

pub use accumulator::{
    Accumulator, AccumulatorStack, ChangedBonaPiece, DirtyPiece, IndexList, MAX_ACTIVE_FEATURES,
    MAX_CHANGED_FEATURES, StackEntry,
};
pub use accumulator_layer_stacks::{
    AccumulatorCacheLayerStacks, AccumulatorLayerStacks, AccumulatorStackLayerStacks,
    LayerStacksAccCache, LayerStacksAccStack, StackEntryLayerStacks,
};
pub use accumulator_stack_variant::AccumulatorStackVariant;
pub use bona_piece::{BonaPiece, ExtBonaPiece, FE_END, halfkp_index};
pub use bona_piece_halfka_hm::{
    BonaPieceHalfKA_hm, E_KING, F_KING, FE_HAND_END, FE_OLD_END, PIECE_INPUTS, halfka_index,
    is_hm_mirror, king_bucket, pack_bonapiece,
};
pub use constants::*;
pub use diff::get_changed_features;
pub use feature_transformer::FeatureTransformer;
pub use feature_transformer_layer_stacks::FeatureTransformerLayerStacks;
pub use features::{
    Feature, FeatureSet, HalfKA, HalfKA_hm, HalfKA_hm_FeatureSet, HalfKAFeatureSet, HalfKP,
    HalfKPFeatureSet, TriggerEvent,
};
pub use layer_stacks::{
    LayerStackBucket, LayerStacks, compute_bucket_index, compute_king_ranks,
    sqr_clipped_relu_transform,
};
pub use layers::{AffineTransform, ClippedReLU};
#[cfg(feature = "layerstack-only")]
pub(crate) use network::update_and_evaluate_layer_stacks_cached;
pub use network::{
    LayerStackBucketMode, NNUENetwork, NnueFormatInfo, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
    compute_layer_stack_progress8kpabs_bucket_index, compute_progress8kpabs_sum, detect_format,
    ensure_accumulator_computed, evaluate_dispatch, evaluate_layer_stacks,
    get_bucket_index_and_frac_for_pos, get_bucket_index_for_pos, get_bucket_interp,
    get_fv_scale_override, get_layer_stack_bucket_mode, get_layer_stack_progress_kpabs_weights,
    get_network, init_nnue, init_nnue_from_bytes, is_halfka_256_loaded, is_halfka_512_loaded,
    is_halfka_1024_loaded, is_halfka_hm_256_loaded, is_halfka_hm_512_loaded,
    is_halfka_hm_1024_loaded, is_layer_stacks_loaded, is_nnue_initialized,
    parse_layer_stack_bucket_mode, parse_nnue_architecture, progress_sum_to_bucket,
    progress_sum_to_bucket_and_frac, reset_layer_stack_progress_kpabs_weights, set_bucket_interp,
    set_fv_scale_override, set_layer_stack_bucket_mode, set_layer_stack_progress_kpabs_weights,
    set_nnue_architecture_override,
};
pub use network_layer_stacks::{LayerStacksNetwork, NetworkLayerStacks};
pub use piece_list::{PieceList, PieceNumber};

// const generics 版統一実装（内部型は pub(crate) に隠蔽）
pub use activation::{CReLU, FtActivation, PairwiseCReLU, SCReLU, detect_activation_from_arch};
pub use spec::{Activation, ArchitectureSpec, FeatureSet as SpecFeatureSet};

// 型エイリアス（HalfKA*/HalfKP* の全バリアント）は pub(crate) に隠蔽
// 外部からは NNUEEvaluator を通じてのみ NNUE 評価を行う
// 内部モジュールは crate::nnue::aliases 経由で直接インポート

// Phase 2: 外部 API 統一
pub use evaluator::NNUEEvaluator;
pub use network::clear_nnue;

// 統計カウンタ（デバッグ・チューニング用）
pub use stats::{NnueStatsSnapshot, get_nnue_stats, print_nnue_stats, reset_nnue_stats};
