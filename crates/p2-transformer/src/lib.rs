//! # p2-transformer
//!
//! 序中盤専用 Transformer + PUCT 用の token system。
//!
//! ## 目的
//!
//! 将棋局面 (`rshogi_core::position::Position`) を、 Transformer の入力に直接
//! 投げ込める固定長 token 列 ([`TransformerTokens`]) へ branch-less / allocation
//! 0 で変換する。 NLP 風の string tokenization ではなく、 「局面構造の
//! tokenization」として設計する。
//!
//! ## token 配置 (max_tokens = 128)
//!
//! ```text
//! [0]        CLS
//! [1..82)    board tokens     (81 squares、 raster: file_major / rank_minor)
//! [82..96)   hand count tokens (Black 7 + White 7 = 14 entries)
//! [96..98)   king tokens      (token[96] = Black king、 token[97] = White king)
//! [98..116)  summary tokens   (18 entries、 side / phase / material / ply / check / 予約)
//! [116..128) reserved/padding (12 entries、 PAD 埋め)
//! ```
//!
//! ## SoA (Struct of Arrays) 設計
//!
//! [`TransformerTokens`] は piece embedding 用 vocab id だけでなく、 square
//! embedding / own-king-relative / enemy-king-relative / promotion zone の lookup
//! id を SoA 形式で連続メモリ上に保持する。 これにより:
//!
//! - Transformer 側で `x = piece_emb + square_emb + own_king_rel_emb + enemy_king_rel_emb + zone_emb`
//!   を 5 つの connected memory chunk から並列 gather できる。
//! - 将来的な CUDA batch inference 時、 device 転送は 5 つの contiguous slice
//!   を `cudaMemcpyAsync` するだけで済む。
//! - SIMD broadcast 加算が gather 不要で実行可能 (各配列は線形 layout)。
//!
//! ## 設計理念
//!
//! - **allocation 0**: `position_to_tokens` は stack 上で `TransformerTokens`
//!   を構築して return する。 内部に Vec / HashMap / Box は使わない。
//! - **branch minimised**: piece の vocab 化は `Piece::raw()` を index にした
//!   lookup table を使う (`board::piece_vocab_id`)。 巨大 match nest は禁じる。
//! - **vocab 拡張余地**: 現在 328/512 を使用、 残り 184 slots は将来の summary
//!   feature や新 token kind 用に予約。 後方互換のため既存 id は変更しない。
//! - **Matryoshka 互換**: token vocab 自体が n_layer 数に依存しないため、 同一
//!   token を異なる depth の Transformer (e.g. 4-layer / 6-layer / 8-layer) で
//!   nested に使い回せる。 PUCT 側で時間予算に応じて depth を選択する。
//!
//! ## 制約
//!
//! - ホットパス (`position_to_tokens`) では heap allocation / dynamic dispatch
//!   を **禁止**。
//! - vocab id は `u16` で十分 (現在 max 328 < 512 < 65535)。
//! - sentinel: `squares` / `zones` で 255、 king-relative で `u16::MAX` を「emb
//!   なし」として表現する。 Transformer 側で対応する emb table の最終 row を
//!   zero vector にしておけば mask 不要。
//!
//! ## 将来拡張
//!
//! - [`token::summary::SummarySlot`] に `KingSafety` / `AttackPressure` などを
//!   追加する場合、 vocab id 範囲を [`token::vocab::SUMMARY_RESERVED_BASE`]
//!   以降から取る。
//! - policy head (dlshogi 2187 互換) は本 crate の責務外。 Transformer 出力を
//!   そのまま softmax + CE で学習する。 legal-move masking は PUCT engine 側。

pub mod token;

pub use token::convert::{TransformerTokens, position_to_tokens};
pub use token::vocab::{MAX_TOKENS, POLICY_SIZE, VOCAB_SIZE};
