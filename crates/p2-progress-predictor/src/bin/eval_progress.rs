//! Progress predictor 評価 binary
//!
//! ground-truth label (棋譜由来 ply_ratio) vs rule-based PhaseTracker vs ML model
//! を比較して MAE / MSE / Spearman 風 monotonicity を出力する。
//!
//! 使用例 (smoke synthetic):
//!   eval_progress --synthetic 1000 --weights progress.bin
//!
//! 実 csa/hcpe pipeline は TODO。

use anyhow::{bail, Result};
use clap::Parser;
use p2_progress_predictor::features::FEATURE_DIM;
use p2_progress_predictor::model::MlProgressPredictor;
use p2_progress_predictor::weights;

#[derive(Parser, Debug)]
#[command(name = "eval_progress", about = "compare rule vs ML progress")]
struct Args {
    /// 訓練済み weight path (.bin)、 空なら default_baseline
    #[arg(long, default_value = "")]
    weights: String,

    /// 実データ source (TODO)
    #[arg(long, default_value = "")]
    data: String,

    /// 合成データ件数 (smoke)
    #[arg(long, default_value_t = 0)]
    synthetic: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn make_synthetic(n: usize, seed: u64) -> (Vec<[f32; FEATURE_DIM]>, Vec<f32>) {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut rng = seed;
    for _ in 0..n {
        let mut row = [0.0f32; FEATURE_DIM];
        for v in row.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng & 0xFFFF) as f32 / 65536.0;
            *v = u;
        }
        let ply = row[31];
        let mat = row[0];
        let logit = 2.0 * (ply - 0.5) + 1.5 * (mat - 0.5);
        let y = 1.0 / (1.0 + (-logit).exp());
        xs.push(row);
        ys.push(y);
    }
    (xs, ys)
}

fn mae(pred: &[f32], target: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (p, t) in pred.iter().zip(target.iter()) {
        s += (p - t).abs();
    }
    s / pred.len() as f32
}

fn mse(pred: &[f32], target: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (p, t) in pred.iter().zip(target.iter()) {
        let d = p - t;
        s += d * d;
    }
    s / pred.len() as f32
}

/// 簡易 Spearman 風: pred と target の rank correlation
fn rank_correlation(pred: &[f32], target: &[f32]) -> f32 {
    let n = pred.len();
    if n < 2 {
        return 0.0;
    }
    let mut p_rank: Vec<(f32, usize)> = pred.iter().copied().zip(0..n).collect();
    let mut t_rank: Vec<(f32, usize)> = target.iter().copied().zip(0..n).collect();
    p_rank.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    t_rank.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut pr = vec![0.0f32; n];
    let mut tr = vec![0.0f32; n];
    for i in 0..n {
        pr[p_rank[i].1] = i as f32;
        tr[t_rank[i].1] = i as f32;
    }
    let mean = (n - 1) as f32 / 2.0;
    let mut num = 0.0f32;
    let mut den_p = 0.0f32;
    let mut den_t = 0.0f32;
    for i in 0..n {
        let dp = pr[i] - mean;
        let dt = tr[i] - mean;
        num += dp * dt;
        den_p += dp * dp;
        den_t += dt * dt;
    }
    if den_p > 0.0 && den_t > 0.0 {
        num / (den_p.sqrt() * den_t.sqrt())
    } else {
        0.0
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let predictor = if args.weights.is_empty() {
        eprintln!("[eval_progress] no --weights, using default_baseline");
        MlProgressPredictor::from_default()
    } else {
        let w = weights::load(std::path::Path::new(&args.weights))?;
        MlProgressPredictor::from_weights(w)
    };

    if args.synthetic == 0 && args.data.is_empty() {
        bail!("--synthetic か --data のいずれかを指定してください。");
    }

    if !args.data.is_empty() {
        bail!("実データ loader は未実装です。 --synthetic で smoke test してください。");
    }

    let (xs, ys) = make_synthetic(args.synthetic, args.seed);

    let mut ml_preds = Vec::with_capacity(xs.len());
    // synthetic 経路では Position が無いため、 features を直接埋めて predict_features を呼ぶ
    for row in &xs {
        let feats = p2_progress_predictor::ProgressFeatures { data: *row };
        ml_preds.push(predictor.predict_features(&feats));
    }

    // rule-based 比較: synthetic では row[0]=material_phase をそのまま score の
    // crude proxy として使う (実 dataset では compute_endgame_score を呼ぶ)
    let rule_preds: Vec<f32> = xs.iter().map(|r| r[0]).collect();

    let m_mae = mae(&ml_preds, &ys);
    let m_mse = mse(&ml_preds, &ys);
    let m_spr = rank_correlation(&ml_preds, &ys);
    let r_mae = mae(&rule_preds, &ys);
    let r_mse = mse(&rule_preds, &ys);
    let r_spr = rank_correlation(&rule_preds, &ys);

    println!("[eval] ML   MAE={:.4} MSE={:.4} rank_corr={:.4}", m_mae, m_mse, m_spr);
    println!("[eval] Rule MAE={:.4} MSE={:.4} rank_corr={:.4}", r_mae, r_mse, r_spr);
    Ok(())
}
