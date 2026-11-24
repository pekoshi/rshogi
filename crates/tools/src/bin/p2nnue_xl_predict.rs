//! Stage 4 学習済み ONNX モデルで局面評価
//!
//! ビルド:
//!   cargo build --release --bin p2nnue_xl_predict --features aobazero-onnx
//!
//! 使用例:
//!   p2nnue_xl_predict --model model.onnx --sfen "lnsgk..."

#[cfg(not(feature = "aobazero-onnx"))]
fn main() {
    eprintln!("p2nnue_xl_predict は aobazero-onnx feature が必要です");
    eprintln!("  cargo build --release --bin p2nnue_xl_predict --features aobazero-onnx");
    std::process::exit(1);
}

#[cfg(feature = "aobazero-onnx")]
fn main() -> anyhow::Result<()> {
    use std::path::PathBuf;

    use anyhow::Context;
    use clap::Parser;
    use ort::session::{Session, builder::GraphOptimizationLevel};
    use ort::value::Tensor;
    use p2_nnue::features::{P2_FEATURE_DIM, extract_features};
    use rshogi_core::movegen::{MoveList, generate_legal_all};
    use rshogi_core::position::Position;
    use tools::dlshogi_features::make_move_label;

    #[derive(Parser, Debug)]
    #[command(name = "p2nnue_xl_predict")]
    struct Args {
        #[arg(long)]
        model: PathBuf,

        /// SFEN 局面 (省略時は startpos)
        #[arg(
            long,
            default_value = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )]
        sfen: String,

        /// game phase index (0..3、 0=opening)
        #[arg(long, default_value = "0")]
        phase: i64,
    }

    let args = Args::parse();

    // ONNX session 構築
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(&args.model)
        .with_context(|| format!("ONNX 読み込み失敗: {:?}", args.model))?;
    println!("Loaded model: {}", args.model.display());

    // Position 解析
    let mut pos = Position::new();
    pos.set_sfen(&args.sfen).context("SFEN parse")?;
    let color = pos.side_to_move();

    // features 抽出
    let input = extract_features(&pos);
    let mut features = vec![0.0f32; P2_FEATURE_DIM];
    for (i, &v) in input.piece_features.iter().enumerate() {
        features[i] = v as f32;
    }
    // shape + flat slice 形式で Tensor を作る (ort 2.0 系 API)
    let features_tensor =
        Tensor::from_array(([1usize, P2_FEATURE_DIM], features.into_boxed_slice()))?;
    let phase_tensor = Tensor::from_array(([1usize], vec![args.phase].into_boxed_slice()))?;

    let outputs = session.run(ort::inputs![
      "features" => features_tensor,
      "phase_idx" => phase_tensor,
    ])?;

    let (_, value_data) = outputs["value"].try_extract_tensor::<f32>()?;
    let (_, outlook_data) = outputs["outlook"].try_extract_tensor::<f32>()?;
    let (_, policy_data) = outputs["policy"].try_extract_tensor::<f32>()?;

    let value_v = value_data.first().copied().unwrap_or(0.0);
    let outlook_v = outlook_data.first().copied().unwrap_or(0.0);
    let value_winrate = 1.0 / (1.0 + (-value_v).exp());
    let outlook_winrate = 1.0 / (1.0 + (-outlook_v).exp());

    println!("\n=== 評価結果 ===");
    println!("SFEN:    {}", args.sfen);
    println!("phase:   {}", args.phase);
    println!(
        "value:   logit={:.4} winrate={:.4} cp_est={:+.0}",
        value_v,
        value_winrate,
        value_v * 800.0
    );
    println!("outlook: logit={:.4} winrate={:.4}", outlook_v, outlook_winrate);

    // policy top-5 (legal mask 適用)
    let mut legal = MoveList::new();
    generate_legal_all(&pos, &mut legal);

    let mut entries: Vec<(String, f32)> = legal
        .iter()
        .map(|&mv| {
            let label = make_move_label(mv, color);
            (mv.to_usi(), policy_data[label])
        })
        .collect();

    // softmax
    let max_l = entries.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, l) in &mut entries {
        *l = (*l - max_l).exp();
        sum += *l;
    }
    if sum > 0.0 {
        for (_, p) in &mut entries {
            *p /= sum;
        }
    }
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("policy top-5 (legal masked):");
    for (i, (usi, prob)) in entries.iter().take(5).enumerate() {
        println!("  {}. {} (prob={:.4})", i + 1, usi, prob);
    }
    println!("legal_count: {}", legal.len());

    Ok(())
}
