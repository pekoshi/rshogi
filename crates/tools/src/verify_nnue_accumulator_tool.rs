//! NNUE accumulator 讀懆ｨｼ繝・・繝ｫ (refresh vs differential update 荳閾ｴ繝・せ繝・
//!
//! quantised.bin 繧定ｪｭ縺ｿ霎ｼ縺ｿ縲《tartpos 縺九ｉ謇九ｒ騾ｲ繧√↑縺後ｉ:
//! 1. move 蜑阪・螻髱｢縺ｧ refresh 竊・accumulator 繧剃ｿ晏ｭ・//! 2. do_move + update (differential) 竊・evaluate
//! 3. do_move 蠕後↓ refresh 竊・evaluate
//! 4. (2) 縺ｨ (3) 縺ｮ隧穂ｾ｡蛟､縺悟ｮ悟・荳閾ｴ縺吶ｋ縺薙→繧呈､懆ｨｼ
//!
//! PSQT / Threat / PSQT+Threat / 邏縺ｮ LayerStacks 蜈ｨ縺ｦ縺ｫ蟇ｾ蠢懊・//!
//! ```bash
//! cargo run --release --bin verify_nnue_accumulator -- \
//!   --nnue-file path/to/quantised.bin \
//!   --ls-progress-coeff path/to/nodchip_progress_e1_f1_cuda.bin
//! ```

use anyhow::{Context, Result, bail};
use clap::Parser;
use std::mem::size_of;
use std::path::PathBuf;

use rshogi_core::movegen::{MoveList, generate_legal_all};
use rshogi_core::nnue::{
    AccumulatorLayerStacks, LayerStackBucketMode, LayerStacksNetwork, NNUENetwork,
    NetworkLayerStacks, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, set_layer_stack_bucket_mode,
    set_layer_stack_progress_kpabs_weights,
};
use rshogi_core::position::Position;

#[derive(Parser, Debug)]
#[command(
    name = "verify_nnue_accumulator",
    about = "NNUE accumulator 讀懆ｨｼ (refresh vs differential update 荳閾ｴ繝・せ繝・"
)]
struct Cli {
    #[arg(long)]
    nnue_file: PathBuf,

    #[arg(long)]
    ls_progress_coeff: Option<PathBuf>,

    /// 繝・せ繝域焔謨ｰ (default: 50)
    #[arg(long, default_value = "50")]
    moves: usize,
}

fn verify_with_network<
    const L1: usize,
    const LS_L1_OUT: usize,
    const LS_L2_IN: usize,
    const LS_L2_PADDED_INPUT: usize,
    const LS_L3: usize,
>(
    cli: &Cli,
    net: &NetworkLayerStacks<L1, LS_L1_OUT, LS_L2_IN, LS_L2_PADDED_INPUT, LS_L3>,
) -> Result<(usize, usize)> {
    let sfens = ["lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"];
    let mut total_tests = 0usize;
    let mut fail = 0usize;

    for sfen in &sfens {
        let mut pos = Position::new();
        pos.set_sfen(sfen).with_context(|| format!("Bad SFEN: {sfen}"))?;

        for step in 0..cli.moves {
            let mut moves = MoveList::new();
            generate_legal_all(&pos, &mut moves);
            if moves.is_empty() {
                println!("  No legal moves at step {step}, restarting.");
                pos.set_sfen(sfen)?;
                continue;
            }

            let m = moves[0];
            let gc = pos.gives_check(m);

            let mut acc_before = AccumulatorLayerStacks::new();
            net.refresh_accumulator(&pos, &mut acc_before);

            let dirty = pos.do_move(m, gc);

            let mut acc_refresh = AccumulatorLayerStacks::new();
            net.refresh_accumulator(&pos, &mut acc_refresh);
            let eval_refresh = net.evaluate(&mut pos, &acc_refresh);

            let mut acc_update = AccumulatorLayerStacks::new();
            net.update_accumulator(&pos, &dirty, &mut acc_update, &acc_before);
            let eval_update = net.evaluate(&mut pos, &acc_update);

            total_tests += 1;

            if eval_refresh != eval_update {
                fail += 1;
                eprintln!(
                    "MISMATCH step={step} move={m:?}: refresh={} update={}",
                    eval_refresh.raw(),
                    eval_update.raw()
                );
                for p in 0..2 {
                    let r = acc_refresh.get(p);
                    let u = acc_update.get(p);
                    let diffs: usize = r.iter().zip(u.iter()).filter(|(a, b)| a != b).count();
                    if diffs > 0 {
                        eprintln!("  piece_acc[{p}]: {diffs}/{} differ", r.len());
                    }
                    #[cfg(feature = "nnue-threat")]
                    {
                        let rt = acc_refresh.get_threat(p);
                        let ut = acc_update.get_threat(p);
                        let tdiffs: usize =
                            rt.iter().zip(ut.iter()).filter(|(a, b)| a != b).count();
                        if tdiffs > 0 {
                            eprintln!("  threat_acc[{p}]: {tdiffs}/{} differ", rt.len());
                        }
                    }
                }
                if fail >= 10 {
                    eprintln!("Too many failures, stopping.");
                    break;
                }
            }
        }
    }

    Ok((total_tests, fail))
}

fn load_progress_coeff_weights(coeff_path: &PathBuf) -> Result<Box<[f32]>> {
    let data = std::fs::read(coeff_path)
        .with_context(|| format!("Failed to read progress coeff: {}", coeff_path.display()))?;

    let expected_f32_bytes = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f32>();
    let expected_f64_bytes = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f64>();

    if data.len() == expected_f32_bytes {
        let weights: Vec<f32> = data
            .chunks_exact(size_of::<f32>())
            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk size is checked")))
            .collect();
        return Ok(weights.into_boxed_slice());
    }

    if data.len() == expected_f64_bytes {
        let weights: Vec<f32> = data
            .chunks_exact(size_of::<f64>())
            .map(|c| f64::from_le_bytes(c.try_into().expect("chunk size is checked")) as f32)
            .collect();
        return Ok(weights.into_boxed_slice());
    }

    bail!(
        "Invalid progress coeff size: {} bytes (expected {} bytes for {} f32 weights or {} bytes for {} f64 weights): {}",
        data.len(),
        expected_f32_bytes,
        SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
        expected_f64_bytes,
        SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
        coeff_path.display()
    );
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();

    // Bucket mode 設定
    if let Some(ref coeff_path) = cli.ls_progress_coeff {
        let weights = load_progress_coeff_weights(coeff_path)?;
        set_layer_stack_progress_kpabs_weights(weights).map_err(anyhow::Error::msg)?;
        set_layer_stack_bucket_mode(LayerStackBucketMode::Progress8KPAbs);
        println!("Bucket mode: progress8kpabs");
    }

    // NNUE 繝｢繝・Ν隱ｭ縺ｿ霎ｼ縺ｿ
    let network = NNUENetwork::load(&cli.nnue_file)
        .with_context(|| format!("Failed to load NNUE: {}", cli.nnue_file.display()))?;

    let net = match &network {
        NNUENetwork::LayerStacks(n) => n,
        _ => anyhow::bail!("Expected LayerStacks network"),
    };

    println!("Model loaded successfully (L1={}).", net.l1_size());

    let (total_tests, fail): (usize, usize) = match net {
        #[cfg(feature = "layerstacks-1536x16x32")]
        LayerStacksNetwork::L1536x16x32(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-1536x32x32")]
        LayerStacksNetwork::L1536x32x32(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-1536x8x96")]
        LayerStacksNetwork::L1536x8x96(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-1536x16x96")]
        LayerStacksNetwork::L1536x16x96(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-1536x32x96")]
        LayerStacksNetwork::L1536x32x96(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-1536x48x96")]
        LayerStacksNetwork::L1536x48x96(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-1536x64x96")]
        LayerStacksNetwork::L1536x64x96(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-1536x16x48")]
        LayerStacksNetwork::L1536x16x48(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-1024x16x48")]
        LayerStacksNetwork::L1024x16x48(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-768x32x96")]
        LayerStacksNetwork::L768x32x96(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-768x16x32")]
        LayerStacksNetwork::L768x16x32(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-768x8x32")]
        LayerStacksNetwork::L768x8x32(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[cfg(feature = "layerstacks-512x16x32")]
        LayerStacksNetwork::L512x16x32(concrete_net) => verify_with_network(&cli, concrete_net)?,
        #[allow(unreachable_patterns)]
        _ => anyhow::bail!("no enabled LayerStacks variant"),
    };

    println!("\n=== Golden Forward Test Results ===");
    println!("Total: {total_tests}, Pass: {}, Fail: {fail}", total_tests - fail);

    if fail > 0 {
        anyhow::bail!("{fail}/{total_tests} tests FAILED");
    }
    println!("ALL PASSED");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_entry_is_reachable_in_tests() {
        let _ = run as fn() -> Result<()>;
    }
}
