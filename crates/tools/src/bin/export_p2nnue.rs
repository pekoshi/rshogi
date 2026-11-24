//! チェックポイント → .p2nn エクスポート CLI

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use p2_nnue::network::P2Network;
use p2_nnue::types::P2_INPUT_DIM;
use tools::p2_train::checkpoint::load_checkpoint;
use tools::p2_train::forward_f32::{F32Weights, forward};
use tools::p2_train::optimizer::AdamWConfig;

#[derive(Parser)]
#[command(
    name = "export_p2nnue",
    about = "チェックポイントから .p2nn をエクスポート"
)]
struct Args {
    /// チェックポイントファイル
    #[arg(long)]
    checkpoint: PathBuf,

    /// 出力ファイル
    #[arg(long)]
    output: PathBuf,

    /// パリティチェック（エクスポート後に再ロードして比較）
    #[arg(long, default_value = "false")]
    parity_check: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("チェックポイント読み込み: {:?}", args.checkpoint);
    let (meta, weights_flat, _optimizer) =
        load_checkpoint(&args.checkpoint, AdamWConfig::default())
            .context("チェックポイント読み込み失敗")?;

    println!(
        "  epoch={}, step={}, train_loss={:.6}, val_loss={:.6}",
        meta.epoch, meta.step, meta.train_loss, meta.val_loss
    );

    // 量子化して .p2nn に書き出す
    let mut net = P2Network::new_zeroed();
    net.set_weights_from_f32(&weights_flat);

    let file = std::fs::File::create(&args.output)?;
    let mut writer = std::io::BufWriter::new(file);
    net.write(&mut writer)?;
    drop(writer);

    println!("エクスポート完了: {:?}", args.output);

    // パリティチェック
    if args.parity_check {
        println!("パリティチェック...");
        let file = std::fs::File::open(&args.output)?;
        let mut reader = std::io::BufReader::new(file);
        let loaded = P2Network::load(&mut reader)?;

        let original_weights = F32Weights::from_flat(&weights_flat);
        let loaded_weights = F32Weights::from_network(&loaded);

        // テスト入力で比較
        let test_input = [1.0f32; P2_INPUT_DIM];
        let s1 = forward(&original_weights, &test_input);
        let s2 = forward(&loaded_weights, &test_input);

        let diff = (s1.logit - s2.logit).abs();
        println!(
            "  f32 logit: {:.6}, quantized logit: {:.6}, diff: {diff:.6}",
            s1.logit, s2.logit
        );

        if diff > 100.0 {
            eprintln!("警告: 量子化誤差が大きい");
        } else {
            println!("  パリティチェック OK");
        }
    }

    Ok(())
}
