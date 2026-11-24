//! P2-NNUE 学習データセット構築パイプライン
//!
//! 複数入力形式・dedup・正規化・policy 生成・層別サンプリングを
//! 一貫して行い、train/valid/test JSONL を生成する。

pub mod config;
pub mod conflict;
pub mod dedup;
pub mod detect;
pub mod filter;
pub mod ingest;
pub mod normalize;
pub mod phase;
pub mod pipeline;
pub mod policy;
pub mod report;
pub mod sample;
pub mod shuffle;
pub mod types;
