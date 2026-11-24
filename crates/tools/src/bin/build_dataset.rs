//! P2-NNUE 学習データセット構築 CLI
//!
//! Usage:
//!   cargo run --release --bin build_dataset -- --config dataset_config.toml
//!   cargo run --release --bin build_dataset -- --config dataset_config.toml --limit 10000

use clap::Parser;
use std::path::PathBuf;
use tools::dataset::config::DatasetConfig;
use tools::dataset::pipeline::{resume_pipeline, run_pipeline};

#[derive(Parser, Debug)]
#[command(name = "build_dataset", about = "P2-NNUE 学習データセット構築")]
struct Args {
    /// TOML 設定ファイルのパス
    #[arg(long)]
    config: PathBuf,

    /// 処理するレコード数の上限（テスト用）
    #[arg(long)]
    limit: Option<u64>,

    /// 既存の中間ファイルから Pass 2 以降を再開する
    #[arg(long)]
    resume: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let config = DatasetConfig::load(&args.config)?;
    eprintln!("Config loaded: {} sources", config.sources.len());

    let output_dir = if let Some(resume_dir) = &args.resume {
        resume_pipeline(&config, resume_dir)?
    } else {
        run_pipeline(&config, args.limit)?
    };
    println!("Dataset written to: {}", output_dir.display());

    Ok(())
}
