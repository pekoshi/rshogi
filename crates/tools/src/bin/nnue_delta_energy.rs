use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use rshogi_core::movegen::is_legal_with_pass;
use rshogi_core::nnue::{
    AccumulatorLayerStacks, LayerStacksNetwork, NNUENetwork, NUM_LAYER_STACK_BUCKETS,
    NetworkLayerStacks,
};
use rshogi_core::position::Position;
use rshogi_core::types::Move;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(about = "LayerStack内部表現の親子局面間変化量をmT/mSで比較する")]
struct Cli {
    #[arg(long)]
    nnue_file: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct FocusCase {
    #[serde(rename = "mT")]
    teacher_move: String,
    #[serde(rename = "mS")]
    student_move: String,
    #[serde(default)]
    net: String,
    #[serde(default)]
    ply: u32,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct Energy {
    raw_l1: u64,
    raw_l2: u64,
    activation_l1: u64,
    activation_l2: u64,
    activation_changed: u32,
    activation_max_delta: u16,
}

#[derive(Debug, Serialize)]
struct MoveMeasurement {
    move_usi: String,
    eval_after: i32,
    energy: Energy,
    bucket_scores: Vec<i32>,
    bucket_span: i32,
    head_abs_max: u64,
    head_cancellation_max: u64,
}

#[derive(Debug, Serialize)]
struct CaseMeasurement {
    sfen: String,
    net: String,
    ply: u32,
    eval_before: i32,
    teacher: MoveMeasurement,
    student: MoveMeasurement,
}

fn activation_value(a: i16, b: i16) -> i32 {
    let a = i32::from(a).clamp(0, 127);
    let b = i32::from(b).clamp(0, 127);
    (a * b) >> 7
}

fn energy<const L1: usize>(
    parent: &AccumulatorLayerStacks<L1>,
    child: &AccumulatorLayerStacks<L1>,
) -> Energy {
    let mut raw_l1 = 0u64;
    let mut raw_l2 = 0u64;
    let mut activation_l1 = 0u64;
    let mut activation_l2 = 0u64;
    let mut activation_changed = 0u32;
    let mut activation_max_delta = 0u16;
    let half = L1 / 2;

    for perspective in 0..2 {
        let before = parent.get(perspective);
        let after = child.get(perspective);

        for (&a, &b) in before.iter().zip(after.iter()) {
            let delta = i64::from(b) - i64::from(a);
            raw_l1 += delta.unsigned_abs();
            raw_l2 += (delta * delta) as u64;
        }

        for i in 0..half {
            let before_value = activation_value(before[i], before[i + half]);
            let after_value = activation_value(after[i], after[i + half]);
            let delta = after_value - before_value;
            let abs_delta = delta.unsigned_abs();
            activation_l1 += u64::from(abs_delta);
            activation_l2 += i64::from(delta * delta) as u64;
            activation_changed += u32::from(delta != 0);
            activation_max_delta = activation_max_delta.max(abs_delta as u16);
        }
    }

    Energy {
        raw_l1,
        raw_l2,
        activation_l1,
        activation_l2,
        activation_changed,
        activation_max_delta,
    }
}

fn measure_move<
    const L1: usize,
    const LS_L1_OUT: usize,
    const LS_L2_IN: usize,
    const LS_L2_PADDED_INPUT: usize,
    const LS_L3: usize,
>(
    net: &NetworkLayerStacks<L1, LS_L1_OUT, LS_L2_IN, LS_L2_PADDED_INPUT, LS_L3>,
    pos: &Position,
    parent_acc: &AccumulatorLayerStacks<L1>,
    move_usi: &str,
) -> Result<MoveMeasurement> {
    let mv = Move::from_usi(move_usi).with_context(|| format!("不正なUSI手: {move_usi}"))?;
    if !is_legal_with_pass(pos, mv) {
        bail!("非合法手: {move_usi}");
    }

    let mut child_pos = pos.clone();
    let gives_check = child_pos.gives_check(mv);
    child_pos.do_move(mv, gives_check);
    let mut child_acc = AccumulatorLayerStacks::<L1>::new();
    net.refresh_accumulator(&child_pos, &mut child_acc);
    let eval_after = net.evaluate(&mut child_pos, &child_acc).raw();
    let mut bucket_scores = Vec::with_capacity(NUM_LAYER_STACK_BUCKETS);
    let mut head_abs_max = 0u64;
    let mut head_cancellation_max = 0u64;
    for bucket in 0..NUM_LAYER_STACK_BUCKETS {
        bucket_scores.push(net.evaluate_with_bucket(&mut child_pos, &child_acc, bucket).raw());
        let breakdown = net.evaluate_with_breakdown(&mut child_pos, &child_acc, bucket);
        let values = [
            breakdown.psqt_value,
            breakdown.pin_value,
            breakdown.dual_psqt_value,
            breakdown.dual_cross_value,
            breakdown.kingring_value,
            breakdown.pairshape_value,
            breakdown.activity_risk_value,
            breakdown.king_escape_value,
            breakdown.attack_relay_value,
            breakdown.attack_relay_v2_value,
            breakdown.king_danger_value,
            breakdown.king_danger_analytic_value,
            breakdown.attack_dev_value,
            breakdown.king_mobility_value,
            breakdown.drop_pressure_value,
        ];
        let head_abs: u64 = values.iter().map(|v| i64::from(*v).unsigned_abs()).sum();
        let all_abs = head_abs + i64::from(breakdown.raw_score).unsigned_abs();
        let cancellation =
            all_abs.saturating_sub(i64::from(breakdown.raw_total_pre_scale()).unsigned_abs());
        head_abs_max = head_abs_max.max(head_abs);
        head_cancellation_max = head_cancellation_max.max(cancellation);
    }
    let bucket_span = bucket_scores.iter().max().unwrap() - bucket_scores.iter().min().unwrap();

    Ok(MoveMeasurement {
        move_usi: move_usi.to_owned(),
        eval_after,
        energy: energy(parent_acc, &child_acc),
        bucket_scores,
        bucket_span,
        head_abs_max,
        head_cancellation_max,
    })
}

fn run_for_network<
    const L1: usize,
    const LS_L1_OUT: usize,
    const LS_L2_IN: usize,
    const LS_L2_PADDED_INPUT: usize,
    const LS_L3: usize,
>(
    net: &NetworkLayerStacks<L1, LS_L1_OUT, LS_L2_IN, LS_L2_PADDED_INPUT, LS_L3>,
    cases: BTreeMap<String, FocusCase>,
) -> Result<Vec<CaseMeasurement>> {
    let mut output = Vec::with_capacity(cases.len());

    for (sfen, case) in cases {
        let mut pos = Position::new();
        pos.set_sfen(&sfen).with_context(|| format!("不正なSFEN: {sfen}"))?;
        let mut parent_acc = AccumulatorLayerStacks::<L1>::new();
        net.refresh_accumulator(&pos, &mut parent_acc);
        let eval_before = net.evaluate(&mut pos, &parent_acc).raw();
        let teacher = measure_move(net, &pos, &parent_acc, &case.teacher_move)
            .with_context(|| format!("mT測定失敗: {sfen}"))?;
        let student = measure_move(net, &pos, &parent_acc, &case.student_move)
            .with_context(|| format!("mS測定失敗: {sfen}"))?;

        output.push(CaseMeasurement {
            sfen,
            net: case.net,
            ply: case.ply,
            eval_before,
            teacher,
            student,
        });
    }

    Ok(output)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cases: BTreeMap<String, FocusCase> = serde_json::from_reader(
        File::open(&cli.input)
            .with_context(|| format!("入力を開けません: {}", cli.input.display()))?,
    )
    .with_context(|| format!("JSONを読めません: {}", cli.input.display()))?;
    let network = NNUENetwork::load(&cli.nnue_file)
        .with_context(|| format!("NNUEを読めません: {}", cli.nnue_file.display()))?;
    let NNUENetwork::LayerStacks(layer_stacks) = network else {
        bail!("LayerStacks NNUEではありません");
    };

    let measurements = match &layer_stacks {
        #[cfg(feature = "layerstacks-1536x16x32")]
        LayerStacksNetwork::L1536x16x32(net) => run_for_network(net, cases)?,
        #[allow(unreachable_patterns)]
        _ => bail!("このbuildで未対応のLayerStacks variantです"),
    };

    if let Some(path) = cli.output {
        serde_json::to_writer_pretty(
            BufWriter::new(
                File::create(&path)
                    .with_context(|| format!("出力を作れません: {}", path.display()))?,
            ),
            &measurements,
        )?;
    } else {
        serde_json::to_writer_pretty(std::io::stdout().lock(), &measurements)?;
    }

    Ok(())
}
