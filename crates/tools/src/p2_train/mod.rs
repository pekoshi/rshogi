//! P2-NNUE 学習パイプライン
//!
//! f32 順伝播・逆伝播・AdamW オプティマイザによる NNUE 学習。

pub mod backward;
pub mod checkpoint;
pub mod data_loader;
pub mod forward_f32;
pub mod optimizer;
pub mod train_report;
pub mod value_safe;
