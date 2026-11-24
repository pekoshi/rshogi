//! shogi2344_v8 NNUE 訓練 (candle + custom CUDA kernels)
//!
//! 設計目標:
//! - Bullet 並み速度 (250K pos/sec) を candle 独自 CUDA で達成
//! - PyTorch 並み flexibility (raw MSE / 任意 loss / grad clip)
//! - rshogi shogi2344_v8 互換 quantised.bin export
//!
//! 構成:
//! - model: shogi2344_v8 (sparse FT 512 + L1=64 + L2=768 + L3=8、 8 buckets)
//! - data: features.bin (164B/record) memmap reader
//! - train: AdamW + cosine LR + grad clip + raw MSE
//! - export: i16 quantised.bin (FT_SCALE=255、 layer scale=64)

pub mod dataset;
pub mod model;
pub mod transformer;
pub mod transformer_dataset;
