//! Phase B/C: value-safe dual head trainer
//!
//! train_p2nnue を base に、以下の value 保護機能を追加:
//!   - .p2nn からの resume (既存モデル継続学習)
//!   - per-head LR multiplier (shared / value head / policy head)
//!   - Huber loss (sigmoid 出力空間)
//!   - global gradient clipping (L2 norm)
//!   - early stop (val_value_loss 監視)
//!   - cp_scale 設定可能 (target = sigmoid(clamp(cp,±cp_clamp)/cp_scale))
//!   - value_weight / policy_weight (損失重み)
//!   - best_value.p2nn の自動保存

use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;

use p2_nnue::network::P2Network;
use p2_nnue::types::P2_INPUT_DIM;
use tools::p2_train::backward::{
    Gradients, backward, bce_with_logits_loss, masked_softmax_ce_loss,
};
use tools::p2_train::checkpoint::{CheckpointMeta, save_checkpoint};
use tools::p2_train::data_loader::{
    JsonlRecord, cp_to_winrate_scaled, decode_batch_dual, load_jsonl_records,
};
use tools::p2_train::forward_f32::{F32Weights, forward, sigmoid};
use tools::p2_train::optimizer::{AdamWConfig, AdamWState};
use tools::p2_train::train_report::{EpochRecord, TrainReport, write_report_json, write_report_md};
use tools::p2_train::value_safe::{EarlyStop, LrMultipliers, ValueLoss, clip_grad_norm};

#[derive(Parser, Debug)]
#[command(
    name = "train_dual_safe",
    about = "Phase B/C: value-safe dual head trainer"
)]
struct Args {
    /// 設定 TOML
    #[arg(long)]
    config: PathBuf,

    /// 出力ディレクトリ override
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// epochs override
    #[arg(long)]
    epochs: Option<u32>,

    /// seed override
    #[arg(long)]
    seed: Option<u64>,

    /// resume_p2nn override (空文字で無効化、 'none' でも可)
    #[arg(long)]
    resume_p2nn: Option<String>,

    /// max_train_records override (試走用)
    #[arg(long)]
    max_train_records: Option<usize>,
}

#[derive(Deserialize)]
struct Config {
    data: DataCfg,
    value_target: ValueTargetCfg,
    training: TrainingCfg,
}

#[derive(Deserialize)]
struct DataCfg {
    train_jsonl: PathBuf,
    max_train_records: usize,
}

#[derive(Deserialize)]
struct ValueTargetCfg {
    cp_scale: f32,
    cp_clamp: i32,
}

#[derive(Deserialize)]
struct TrainingCfg {
    resume_p2nn: Option<PathBuf>,
    output_dir: PathBuf,
    epochs: u32,
    batch_size: usize,
    val_ratio: f32,
    base_lr: f32,
    shared_lr_mult: f32,
    value_head_lr_mult: f32,
    policy_head_lr_mult: f32,
    value_weight: f32,
    policy_weight: f32,
    #[serde(default)]
    outlook_weight: f32,
    huber_delta: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    grad_clip_norm: f32,
    early_stop_patience: u32,
    seed: u64,
    /// 序中盤特化フィルタ (空 = フィルタなし)
    #[serde(default)]
    phase_filter: Vec<String>,
    /// game_result blend 比率 (0.0..1.0)
    #[serde(default)]
    result_blend: f32,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let cfg_text = fs::read_to_string(&args.config)
        .with_context(|| format!("config 読み込み失敗: {:?}", args.config))?;
    let cfg: Config = toml::from_str(&cfg_text).context("TOML パース失敗")?;

    let output_dir = args.output_dir.clone().unwrap_or_else(|| cfg.training.output_dir.clone());
    let epochs = args.epochs.unwrap_or(cfg.training.epochs);
    let seed = args.seed.unwrap_or(cfg.training.seed);
    let resume_p2nn = match args.resume_p2nn.as_deref() {
        Some("none") | Some("") => None,
        Some(p) => Some(PathBuf::from(p)),
        None => cfg.training.resume_p2nn.clone(),
    };

    fs::create_dir_all(&output_dir)
        .with_context(|| format!("出力ディレクトリ作成失敗: {:?}", output_dir))?;

    let start_time = Instant::now();

    // データロード
    println!("学習データロード: {:?}", cfg.data.train_jsonl);
    let max_train = args.max_train_records.unwrap_or(cfg.data.max_train_records);
    let limit = if max_train == 0 {
        None
    } else {
        Some(max_train)
    };
    let load_start = Instant::now();
    let mut records =
        load_jsonl_records(&cfg.data.train_jsonl, limit).context("JSONL 読み込み失敗")?;
    println!("レコード数: {} ({:.1}s)", records.len(), load_start.elapsed().as_secs_f64());
    if records.is_empty() {
        bail!("学習データが空です");
    }

    // Stage 1: phase filter (序中盤特化)
    if !cfg.training.phase_filter.is_empty() {
        let before = records.len();
        let allowed: Vec<&str> = cfg.training.phase_filter.iter().map(|s| s.as_str()).collect();
        records.retain(|r| allowed.iter().any(|p| *p == r.phase.as_str()));
        println!("Phase filter {:?}: {} → {} records", allowed, before, records.len());
        if records.is_empty() {
            bail!("phase_filter で全件除外されました");
        }
    }

    // Train/Val 分割
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..records.len()).collect();
    indices.shuffle(&mut rng);

    let val_count = (records.len() as f32 * cfg.training.val_ratio) as usize;
    let val_count = val_count.max(1).min(records.len() - 1);
    let val_indices: Vec<usize> = indices[..val_count].to_vec();
    let train_indices: Vec<usize> = indices[val_count..].to_vec();
    drop(indices);

    println!("Train: {} samples, Val: {} samples", train_indices.len(), val_indices.len());
    let val_records: Vec<JsonlRecord> = val_indices.iter().map(|&i| records[i].clone()).collect();
    drop(val_indices);

    // 重み初期化 / resume
    let adam_config = AdamWConfig {
        lr: cfg.training.base_lr,
        beta1: cfg.training.beta1,
        beta2: cfg.training.beta2,
        epsilon: 1e-8,
        weight_decay: cfg.training.weight_decay,
    };

    let (mut weights_flat, mut optimizer) = if let Some(p2nn_path) = &resume_p2nn {
        println!("既存 .p2nn から resume: {:?}", p2nn_path);
        let file = fs::File::open(p2nn_path)
            .with_context(|| format!(".p2nn オープン失敗: {p2nn_path:?}"))?;
        let mut reader = BufReader::new(file);
        let net = P2Network::load(&mut reader).context(".p2nn ロード失敗")?;
        let weights = net.weights_to_f32();
        println!("  weights len = {}", weights.len());
        (weights, AdamWState::new(adam_config.clone()))
    } else {
        println!("He 初期化");
        let w = F32Weights::he_init(&mut rng);
        (w.to_flat(), AdamWState::new(adam_config.clone()))
    };

    let lr_mult = LrMultipliers {
        shared: cfg.training.shared_lr_mult,
        value_head: cfg.training.value_head_lr_mult,
        policy_head: cfg.training.policy_head_lr_mult,
        // outlook head は新規 head なので value head と同じ multiplier を使う
        outlook_head: cfg.training.value_head_lr_mult,
    };
    let value_loss_kind = ValueLoss::Huber {
        delta: cfg.training.huber_delta,
    };
    let cp_clamp = cfg.value_target.cp_clamp;
    let cp_scale = cfg.value_target.cp_scale;
    let value_weight = cfg.training.value_weight;
    let policy_weight = cfg.training.policy_weight;
    let outlook_weight = cfg.training.outlook_weight;
    let grad_clip = cfg.training.grad_clip_norm;
    let result_blend = cfg.training.result_blend.clamp(0.0, 1.0);

    println!("=== Training config ===");
    println!("  base_lr: {}", cfg.training.base_lr);
    println!(
        "  lr_mult: shared={} value={} policy={}",
        lr_mult.shared, lr_mult.value_head, lr_mult.policy_head
    );
    println!(
        "  value_weight={} policy_weight={} outlook_weight={} huber_delta={} grad_clip={}",
        value_weight, policy_weight, outlook_weight, cfg.training.huber_delta, grad_clip
    );
    println!("  cp_scale={} cp_clamp={}", cp_scale, cp_clamp);
    println!(
        "  early_stop_patience={} weight_decay={}",
        cfg.training.early_stop_patience, cfg.training.weight_decay
    );
    println!("  result_blend={result_blend} phase_filter={:?}", cfg.training.phase_filter);

    let mut early_stop = EarlyStop::new(cfg.training.early_stop_patience, 0.001);
    let mut epoch_history: Vec<EpochRecord> = Vec::new();
    let mut best_val_v_loss = f32::INFINITY;

    // 学習ループ
    for epoch in 1..=epochs {
        let epoch_start = Instant::now();

        let mut train_order: Vec<usize> = (0..train_indices.len()).collect();
        train_order.shuffle(&mut rng);

        let mut value_loss_sum = 0.0f64;
        let mut policy_loss_sum = 0.0f64;
        let mut outlook_loss_sum = 0.0f64;
        let mut policy_correct = 0u64;
        let mut samples_total = 0u64;
        let num_batches = train_indices.len().div_ceil(cfg.training.batch_size);
        let mut last_log = Instant::now();
        let mut grad_norm_sum = 0.0f64;
        let mut grad_norm_clipped_count = 0u64;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * cfg.training.batch_size;
            let batch_end = (batch_start + cfg.training.batch_size).min(train_indices.len());
            let batch_local = &train_order[batch_start..batch_end];

            let batch_records: Vec<JsonlRecord> =
                batch_local.iter().map(|&i| records[train_indices[i]].clone()).collect();

            let samples = decode_batch_dual(&batch_records);
            if samples.is_empty() {
                continue;
            }

            let weights = F32Weights::from_flat(&weights_flat);
            let mut batch_grads = Gradients::zeros();
            let mut batch_v_loss = 0.0f64;
            let mut batch_p_loss = 0.0f64;
            let mut batch_o_loss = 0.0f64;
            let mut batch_correct = 0u64;

            for sample in &samples {
                let input: &[f32; P2_INPUT_DIM] = &sample.features;
                let state = forward(&weights, input);

                // configured cp_scale で target を再計算 + game_result blend (大局観)
                let cp_target = cp_to_winrate_scaled(sample.raw_value_cp, cp_clamp, cp_scale);
                let target_winrate = blend_target(cp_target, sample.game_result, result_blend);

                let v_loss = value_loss_kind.forward(state.logit, target_winrate);
                let d_v_logit = value_loss_kind.d_logit(state.logit, target_winrate);

                // outlook target = (game_result + 1) / 2 (BCE on sigmoid)
                let outlook_target = match sample.game_result {
                    1 => 1.0f32,
                    0 => 0.5,
                    -1 => 0.0,
                    _ => 0.5,
                };
                let o_loss = bce_with_logits_loss(state.outlook_logit, outlook_target);
                let d_o_logit = sigmoid(state.outlook_logit) - outlook_target;

                let p_loss = masked_softmax_ce_loss(
                    &state.policy_logits,
                    sample.target_label,
                    &sample.legal_labels,
                );

                batch_v_loss += v_loss as f64;
                batch_p_loss += p_loss as f64;
                batch_o_loss += o_loss as f64;

                // policy top-1
                if !sample.legal_labels.is_empty() {
                    let mut best = sample.legal_labels[0] as usize;
                    let mut best_v = state.policy_logits[best];
                    for &lbl in &sample.legal_labels[1..] {
                        let v = state.policy_logits[lbl as usize];
                        if v > best_v {
                            best_v = v;
                            best = lbl as usize;
                        }
                    }
                    if best == sample.target_label {
                        batch_correct += 1;
                    }
                }

                // 各 head 損失重みを d_logit に乗じる
                let weighted_d_v = d_v_logit * value_weight;
                let weighted_d_o = d_o_logit * outlook_weight;
                let grad = backward(
                    &weights,
                    &state,
                    input,
                    weighted_d_v,
                    weighted_d_o,
                    sample.target_label,
                    &sample.legal_labels,
                    policy_weight,
                );
                batch_grads.accumulate(&grad);
            }

            // バッチ平均
            let n = samples.len() as f32;
            let mut grad_flat = batch_grads.to_flat();
            for g in &mut grad_flat {
                *g /= n;
            }

            // per-head LR
            lr_mult.apply(&mut grad_flat);

            // global grad clip
            let (norm_before, norm_after) = clip_grad_norm(&mut grad_flat, grad_clip);
            grad_norm_sum += norm_before as f64;
            if norm_after < norm_before {
                grad_norm_clipped_count += 1;
            }

            optimizer.step(&mut weights_flat, &grad_flat);

            value_loss_sum += batch_v_loss;
            policy_loss_sum += batch_p_loss;
            outlook_loss_sum += batch_o_loss;
            policy_correct += batch_correct;
            samples_total += samples.len() as u64;

            if last_log.elapsed().as_secs_f64() > 60.0 {
                let avg_v = value_loss_sum / samples_total as f64;
                let avg_p = policy_loss_sum / samples_total as f64;
                let avg_o = outlook_loss_sum / samples_total as f64;
                let acc = policy_correct as f64 / samples_total.max(1) as f64;
                let avg_grad = grad_norm_sum / (batch_idx + 1) as f64;
                println!(
                    "  [batch {}/{}] samples={} v={:.4} p={:.4} o={:.4} acc={:.3} grad_norm_avg={:.3} clipped={}/{} elapsed={:.0}s",
                    batch_idx + 1,
                    num_batches,
                    samples_total,
                    avg_v,
                    avg_p,
                    avg_o,
                    acc,
                    avg_grad,
                    grad_norm_clipped_count,
                    batch_idx + 1,
                    epoch_start.elapsed().as_secs_f64()
                );
                last_log = Instant::now();
            }
        }

        let train_v_loss = (value_loss_sum / samples_total.max(1) as f64) as f32;
        let train_p_loss = (policy_loss_sum / samples_total.max(1) as f64) as f32;
        let train_o_loss = (outlook_loss_sum / samples_total.max(1) as f64) as f32;
        let train_p_acc = policy_correct as f32 / samples_total.max(1) as f32;

        let (val_v_loss, val_p_loss, val_o_loss, val_p_acc) = compute_val_metrics(
            &weights_flat,
            &val_records,
            cp_clamp,
            cp_scale,
            value_loss_kind,
            result_blend,
        );

        let elapsed = epoch_start.elapsed();
        println!(
            "Epoch {epoch}/{epochs}: train v={train_v_loss:.4} p={train_p_loss:.4} o={train_o_loss:.4} acc={train_p_acc:.3} | val v={val_v_loss:.4} p={val_p_loss:.4} o={val_o_loss:.4} acc={val_p_acc:.3} | time={:.1}s",
            elapsed.as_secs_f64()
        );

        epoch_history.push(EpochRecord {
            epoch,
            train_loss: train_v_loss * value_weight
                + train_p_loss * policy_weight
                + train_o_loss * outlook_weight,
            val_loss: val_v_loss * value_weight
                + val_p_loss * policy_weight
                + val_o_loss * outlook_weight,
        });

        // Checkpoint 保存
        let ckpt_path = output_dir.join(format!("checkpoint_epoch{epoch}.bin"));
        let meta = CheckpointMeta {
            epoch,
            step: optimizer.t,
            train_loss: train_v_loss,
            val_loss: val_v_loss,
            lr: cfg.training.base_lr,
        };
        save_checkpoint(&ckpt_path, &meta, &weights_flat, &optimizer)
            .with_context(|| format!("checkpoint 保存失敗: {ckpt_path:?}"))?;

        // .p2nn 出力
        let model_path = output_dir.join(format!("model_epoch{epoch}.p2nn"));
        export_p2nn(&weights_flat, &model_path)?;
        println!("  → {} / {}", ckpt_path.display(), model_path.display());

        // best_value.p2nn 更新 (val_v_loss 基準)
        if val_v_loss < best_val_v_loss {
            best_val_v_loss = val_v_loss;
            let best_path = output_dir.join("best_value.p2nn");
            export_p2nn(&weights_flat, &best_path)?;
            println!("  → best_value.p2nn (val_v_loss={val_v_loss:.4})");
        }

        let (improved, should_stop) = early_stop.step(val_v_loss);
        if !improved {
            println!(
                "  early_stop: bad_count={}/{}",
                early_stop.bad_count, cfg.training.early_stop_patience
            );
        }
        if should_stop {
            println!("Early stop triggered at epoch {epoch}");
            break;
        }
    }

    // 最終レポート
    let report = TrainReport {
        data_file: cfg.data.train_jsonl.display().to_string(),
        train_samples: train_indices.len(),
        val_samples: val_records.len(),
        epochs,
        batch_size: cfg.training.batch_size,
        lr: cfg.training.base_lr,
        weight_decay: cfg.training.weight_decay,
        seed: Some(seed),
        final_train_loss: epoch_history.last().map(|r| r.train_loss).unwrap_or(f32::NAN),
        final_val_loss: epoch_history.last().map(|r| r.val_loss).unwrap_or(f32::NAN),
        epoch_history,
        output_dir: output_dir.display().to_string(),
        elapsed_secs: start_time.elapsed().as_secs_f64(),
    };
    write_report_md(&output_dir.join("RUN_REPORT.md"), &report)?;
    write_report_json(&output_dir.join("report.json"), &report)?;
    println!("完了: {:.1}s", start_time.elapsed().as_secs_f64());
    println!("最良 val_value_loss: {:.6} → best_value.p2nn", best_val_v_loss);

    Ok(())
}

fn compute_val_metrics(
    weights_flat: &[f32],
    val_records: &[JsonlRecord],
    cp_clamp: i32,
    cp_scale: f32,
    value_loss_kind: ValueLoss,
    result_blend: f32,
) -> (f32, f32, f32, f32) {
    let samples = decode_batch_dual(val_records);
    if samples.is_empty() {
        return (f32::NAN, f32::NAN, f32::NAN, 0.0);
    }
    let weights = F32Weights::from_flat(weights_flat);
    let mut v_loss = 0.0f64;
    let mut p_loss = 0.0f64;
    let mut o_loss = 0.0f64;
    let mut correct = 0u64;
    for sample in &samples {
        let state = forward(&weights, &sample.features);
        let cp_target = cp_to_winrate_scaled(sample.raw_value_cp, cp_clamp, cp_scale);
        let target = blend_target(cp_target, sample.game_result, result_blend);
        v_loss += value_loss_kind.forward(state.logit, target) as f64;
        p_loss +=
            masked_softmax_ce_loss(&state.policy_logits, sample.target_label, &sample.legal_labels)
                as f64;

        let outlook_target = match sample.game_result {
            1 => 1.0f32,
            0 => 0.5,
            -1 => 0.0,
            _ => 0.5,
        };
        o_loss += bce_with_logits_loss(state.outlook_logit, outlook_target) as f64;

        if !sample.legal_labels.is_empty() {
            let mut best = sample.legal_labels[0] as usize;
            let mut best_v = state.policy_logits[best];
            for &lbl in &sample.legal_labels[1..] {
                let v = state.policy_logits[lbl as usize];
                if v > best_v {
                    best_v = v;
                    best = lbl as usize;
                }
            }
            if best == sample.target_label {
                correct += 1;
            }
        }
    }
    let n = samples.len() as f64;
    let acc = correct as f32 / samples.len() as f32;
    ((v_loss / n) as f32, (p_loss / n) as f32, (o_loss / n) as f32, acc)
}

/// cp 由来 target と対局結果を線形ブレンドする (大局観の方向性を value target に染み込ませる)
///
/// `target = (1 - blend) * cp_target + blend * result_target`
/// ただし result_target は (game_result + 1) / 2 = win:1.0 / draw:0.5 / loss:0.0
#[inline]
fn blend_target(cp_target: f32, game_result: i8, blend: f32) -> f32 {
    if blend <= 0.0 {
        return cp_target;
    }
    let result_target = match game_result {
        1 => 1.0,
        0 => 0.5,
        -1 => 0.0,
        _ => 0.5,
    };
    let b = blend.clamp(0.0, 1.0);
    (1.0 - b) * cp_target + b * result_target
}

fn export_p2nn(weights_flat: &[f32], path: &PathBuf) -> Result<()> {
    let mut net = P2Network::new_zeroed();
    net.set_weights_from_f32(weights_flat);
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    net.write(&mut writer)?;
    Ok(())
}
