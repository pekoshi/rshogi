//! # nnue-b
//!
//! PUCT hybrid 用の NNUE-B (Candle) Transformer モデル定義 + 訓練 + ONNX export。
//!
//! ## 位置づけ
//!
//! - `p2-search` の `PuctEngine` から呼ばれる leaf 評価器。
//! - 既存 `rshogi-core` の αβ NNUE (HalfKAv2_hm LayerStack) とは別系統。
//! - 訓練は本 crate 内 `train_nnue_b` binary、 推論は ONNX export → ORT (別 crate) で行う。
//!
//! ## 設計指針 (memory `nnue_b_transformer_plan_20260512` 準拠)
//!
//! - arch: Candle Transformer encoder
//! - d_model=192, n_layer=6, n_heads=6, d_ff=768
//! - 推定 params 5-8M (GTX 1050 Ti 4 GB 訓練可能)
//! - output:
//!   - value head: `[B, 1]` (raw、 loss 側で tanh / sigmoid を適用)
//!   - policy head: `[B, 2187]` (dlshogi `make_move_label` 互換 logits)
//!
//! ## 入力 feature 設計
//!
//! Position 入力を **dense board representation** で受ける。
//!
//! - 81 squares × 28 channel = 2,268 dim (盤面 14 ch × 2 perspective)
//! - 14 channel: empty + (Pawn..King + Pawn-promoted..Rook-promoted) = 1+13 = 14
//! - perspective 2: own (手番側視点) / enemy
//! - 持ち駒: 7 種 × 2 side = 14 値 (count を float に正規化、 別 token として attach)
//! - meta: side_to_move 1 + ply bucket 1 + king 自陣 1 + king 敵陣 1 = 4 値
//!
//! ただし 1 token = 1 square の attention 入力に揃えるため、 dense feature は
//! 「per-square embedding」として Transformer に渡す:
//!
//! ```text
//! tokens layout (T = 1 + 81 + 14 + 4 = 100):
//!   [0]       CLS
//!   [1..82)   board square (81 tokens、 piece_type id をそのまま埋め込む)
//!   [82..96)  hand bucket (Black 7 + White 7 = 14 tokens)
//!   [96..100) meta (side / ply_bucket / king_b_sq / king_w_sq)
//! ```
//!
//! piece embedding vocab は以下:
//!
//! - 0: PAD
//! - 1: CLS
//! - 2: EMPTY
//! - 3..17: Black piece (Pawn..Dragon、 14 種)
//! - 17..31: White piece
//! - 31..43: side / ply / king special tokens (今は簡易、 拡張余地あり)
//!
//! vocab_size = 64 (余裕含む)。
//!
//! ## 各モジュール
//!
//! - [`input`]: Position → tokens (i64 tensor) 変換
//! - [`model`]: Candle Transformer 定義 (encoder 6 layer + 2 head)
//! - [`train`]: 訓練 loop + loss + ckpt 保存
//! - [`onnx`]: ONNX export hook (feature = "onnx" でのみ pull)
//!
//! ## 注意
//!
//! - 推論 hot path での `Vec` heap alloc は **禁止** (CLAUDE.md ルール)。
//! - `unsafe` 追加禁止。
//! - GPU は optional (`--features cuda`)、 default は CPU で build 可能。

pub mod input;
pub mod model;
pub mod psv;

#[cfg(feature = "train")]
pub mod dataset;

#[cfg(feature = "train")]
pub mod train;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use input::{
    encode_position_to_tokens, NUM_BOARD_TOKENS, NUM_HAND_TOKENS, NUM_META_TOKENS, NUM_TOKENS,
    VOCAB_SIZE,
};
pub use model::{NnueBConfig, NnueBModel, POLICY_SIZE};
pub use psv::{PsvRecord, PSV_RECORD_SIZE};

#[cfg(feature = "train")]
pub use dataset::{decode_psv_record, PsvSample};
