//! P2-NNUE 学習 CLI
//!
//! JSONL データから P2-NNUE (2048→512→256→1, value) と policy head (256→2187) を
//! 同時に学習する dual head トレーナー。

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use p2_nnue::network::P2Network;
use p2_nnue::types::P2_INPUT_DIM;
use tools::p2_train::backward::{
    Gradients, backward, bce_with_logits_loss, masked_softmax_ce_loss,
};
use tools::p2_train::checkpoint::{CheckpointMeta, load_checkpoint, save_checkpoint};
use tools::p2_train::data_loader::{JsonlRecord, decode_batch_dual, load_jsonl_records};
use tools::p2_train::forward_f32::{F32Weights, TOTAL_PARAMS, forward, sigmoid};
use tools::p2_train::optimizer::{AdamWConfig, AdamWState};
use tools::p2_train::train_report::{EpochRecord, TrainReport, write_report_json, write_report_md};

#[derive(Parser)]
#[command(
    name = "train_p2nnue",
    about = "P2-NNUE dual head 学習 (value + policy)"
)]
struct Args {
    /// 学習データ（JSONL ファイル、build_dataset の出力）
    #[arg(long)]
    data: PathBuf,

    /// エポック数
    #[arg(long, default_value = "1")]
    epochs: u32,

    /// バッチサイズ
    #[arg(long, default_value = "4096")]
    batch_size: usize,

    /// 学習率
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// Weight decay
    #[arg(long, default_value = "0.01")]
    weight_decay: f32,

    /// Policy 損失係数（0.0 で value のみ、1.0 で同等）
    #[arg(long, default_value = "1.0")]
    lambda_policy: f32,

    /// 検証用データの割合
    #[arg(long, default_value = "0.01")]
    val_ratio: f32,

    /// 読み込む最大レコード数（メモリ節約。0 = 制限なし）
    #[arg(long, default_value = "0")]
    max_records: usize,

    /// 出力ディレクトリ
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,

    /// 初期重み（.p2nn ファイル）
    #[arg(long)]
    initial_weights: Option<PathBuf>,

    /// チェックポイントから再開
    #[arg(long)]
    resume: Option<PathBuf>,

    /// ランダムシード
    #[arg(long)]
    seed: Option<u64>,

    /// スレッド数（0 = 自動）
    #[arg(long, default_value = "0")]
    threads: usize,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let start_time = Instant::now();

    // スレッドプール設定
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().ok();
    }

    // 出力ディレクトリ作成
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("出力ディレクトリ作成失敗: {:?}", args.output_dir))?;

    // データロード
    println!("JSONL データ読み込み: {:?}", args.data);
    let limit = if args.max_records == 0 {
        None
    } else {
        Some(args.max_records)
    };
    let load_start = Instant::now();
    let records = load_jsonl_records(&args.data, limit).context("JSONL ファイル読み込み失敗")?;
    println!(
        "レコード数: {} (読み込み {:.1}s)",
        records.len(),
        load_start.elapsed().as_secs_f64()
    );

    if records.is_empty() {
        bail!("学習データが空です");
    }

    // Train/Val 分割
    let mut rng = if let Some(seed) = args.seed {
        ChaCha8Rng::seed_from_u64(seed)
    } else {
        ChaCha8Rng::from_os_rng()
    };

    let mut indices: Vec<usize> = (0..records.len()).collect();
    indices.shuffle(&mut rng);

    let val_count = (records.len() as f32 * args.val_ratio) as usize;
    let val_count = val_count.max(1).min(records.len() - 1);
    let val_indices: Vec<usize> = indices[..val_count].to_vec();
    let train_indices: Vec<usize> = indices[val_count..].to_vec();
    drop(indices);

    println!("Train: {} samples, Val: {} samples", train_indices.len(), val_indices.len());

    // val 用にだけは小さいサンプルなので clone で持つ (検証ループの効率のため)
    let val_records: Vec<JsonlRecord> = val_indices.iter().map(|&i| records[i].clone()).collect();
    drop(val_indices);

    // 重み初期化
    let adam_config = AdamWConfig {
        lr: args.lr,
        weight_decay: args.weight_decay,
        ..Default::default()
    };

    let (mut weights_flat, mut optimizer, start_epoch) = if let Some(ref ckpt_path) = args.resume {
        println!("チェックポイントから再開: {:?}", ckpt_path);
        let (meta, weights, state) = load_checkpoint(ckpt_path, adam_config.clone())
            .context("チェックポイント読み込み失敗")?;
        println!("  epoch={}, step={}, train_loss={:.6}", meta.epoch, meta.step, meta.train_loss);
        (weights, state, meta.epoch + 1)
    } else if let Some(ref model_path) = args.initial_weights {
        println!("初期重みロード: {:?}", model_path);
        let file = std::fs::File::open(model_path).context("モデルファイルオープン失敗")?;
        let mut reader = std::io::BufReader::new(file);
        let net = P2Network::load(&mut reader).context("モデル読み込み失敗")?;
        let w = F32Weights::from_network(&net);
        (w.to_flat(), AdamWState::new(adam_config.clone()), 1)
    } else {
        println!("He 初期化");
        let w = F32Weights::he_init(&mut rng);
        (w.to_flat(), AdamWState::new(adam_config.clone()), 1)
    };

    println!(
        "パラメータ数: {} ({:.1} MB f32)",
        TOTAL_PARAMS,
        TOTAL_PARAMS as f64 * 4.0 / 1024.0 / 1024.0
    );

    // 学習ループ
    let mut epoch_history = Vec::new();
    let mut global_step = optimizer.t;
    let lambda_policy = args.lambda_policy;

    for epoch in start_epoch..=args.epochs {
        let epoch_start = Instant::now();

        // エポックごとにシャッフル (train_indices の中での順序)
        let mut train_order: Vec<usize> = (0..train_indices.len()).collect();
        train_order.shuffle(&mut rng);

        let mut value_loss_sum = 0.0f64;
        let mut policy_loss_sum = 0.0f64;
        let mut policy_correct = 0u64;
        let mut policy_total = 0u64;
        let mut epoch_samples = 0usize;
        let num_batches = train_indices.len().div_ceil(args.batch_size);
        let mut last_log = Instant::now();

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * args.batch_size;
            let batch_end = (batch_start + args.batch_size).min(train_indices.len());
            let batch_local = &train_order[batch_start..batch_end];

            // 一時的な batch (4096 records ≈ 12 MB)。元の records は共有。
            let batch_records: Vec<JsonlRecord> =
                batch_local.iter().map(|&i| records[train_indices[i]].clone()).collect();

            let samples = decode_batch_dual(&batch_records);
            if samples.is_empty() {
                continue;
            }

            let weights = F32Weights::from_flat(&weights_flat);
            let mut batch_grads = Gradients::zeros();
            let mut batch_value_loss = 0.0f64;
            let mut batch_policy_loss = 0.0f64;
            let mut batch_policy_correct = 0u64;

            for sample in &samples {
                let input: &[f32; P2_INPUT_DIM] = &sample.features;
                let state = forward(&weights, input);

                let v_loss = bce_with_logits_loss(state.logit, sample.target_winrate);
                batch_value_loss += v_loss as f64;

                let p_loss = masked_softmax_ce_loss(
                    &state.policy_logits,
                    sample.target_label,
                    &sample.legal_labels,
                );
                batch_policy_loss += p_loss as f64;

                // policy top-1 accuracy (legal 内 argmax = target_label?)
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
                        batch_policy_correct += 1;
                    }
                }

                let d_value_logit = sigmoid(state.logit) - sample.target_winrate;
                let grad = backward(
                    &weights,
                    &state,
                    input,
                    d_value_logit,
                    0.0, // train_p2nnue では outlook 学習無効
                    sample.target_label,
                    &sample.legal_labels,
                    lambda_policy,
                );
                batch_grads.accumulate(&grad);
            }

            // 勾配の平均
            let n = samples.len() as f32;
            let mut grad_flat = batch_grads.to_flat();
            for g in &mut grad_flat {
                *g /= n;
            }

            optimizer.step(&mut weights_flat, &grad_flat);
            global_step = optimizer.t;

            value_loss_sum += batch_value_loss;
            policy_loss_sum += batch_policy_loss;
            policy_correct += batch_policy_correct;
            policy_total += samples.len() as u64;
            epoch_samples += samples.len();

            // 進捗ログ (60秒ごと)
            if last_log.elapsed().as_secs_f64() > 60.0 {
                let avg_v = value_loss_sum / epoch_samples as f64;
                let avg_p = policy_loss_sum / epoch_samples as f64;
                let acc = policy_correct as f64 / policy_total.max(1) as f64;
                println!(
                    "  [batch {}/{}] samples={} v_loss={:.4} p_loss={:.4} p_acc={:.3} elapsed={:.0}s",
                    batch_idx + 1,
                    num_batches,
                    epoch_samples,
                    avg_v,
                    avg_p,
                    acc,
                    epoch_start.elapsed().as_secs_f64()
                );
                last_log = Instant::now();
            }
        }

        let value_loss = if epoch_samples > 0 {
            (value_loss_sum / epoch_samples as f64) as f32
        } else {
            f32::NAN
        };
        let policy_loss = if epoch_samples > 0 {
            (policy_loss_sum / epoch_samples as f64) as f32
        } else {
            f32::NAN
        };
        let policy_acc = if policy_total > 0 {
            policy_correct as f32 / policy_total as f32
        } else {
            0.0
        };
        let train_loss = value_loss + lambda_policy * policy_loss;

        // Validation loss
        let (val_value, val_policy, val_acc) = compute_val_loss(&weights_flat, &val_records);
        let val_loss = val_value + lambda_policy * val_policy;

        let elapsed = epoch_start.elapsed();
        println!(
            "Epoch {epoch}/{}: train v={value_loss:.4} p={policy_loss:.4} acc={policy_acc:.3} | \
             val v={val_value:.4} p={val_policy:.4} acc={val_acc:.3} | total_train={train_loss:.4} \
             samples={epoch_samples} time={:.1}s",
            args.epochs,
            elapsed.as_secs_f64()
        );

        epoch_history.push(EpochRecord {
            epoch,
            train_loss,
            val_loss,
        });

        // チェックポイント保存
        let ckpt_path = args.output_dir.join(format!("checkpoint_epoch{epoch}.bin"));
        let meta = CheckpointMeta {
            epoch,
            step: global_step,
            train_loss,
            val_loss,
            lr: args.lr,
        };
        save_checkpoint(&ckpt_path, &meta, &weights_flat, &optimizer)
            .with_context(|| format!("チェックポイント保存失敗: {ckpt_path:?}"))?;

        // .p2nn エクスポート
        let model_path = args.output_dir.join(format!("model_epoch{epoch}.p2nn"));
        export_p2nn(&weights_flat, &model_path)?;
        println!("  → {:?}, {:?}", ckpt_path, model_path);
    }

    // パリティチェック: エクスポートした .p2nn を再ロードして forward 確認
    let final_model_path = args.output_dir.join(format!("model_epoch{}.p2nn", args.epochs));
    if final_model_path.exists() {
        println!("パリティチェック...");
        let file = std::fs::File::open(&final_model_path)?;
        let mut reader = std::io::BufReader::new(file);
        let net = P2Network::load(&mut reader)?;

        // 簡易チェック: 量子化後の重みを f32 に戻して logit を比較
        let net_weights = F32Weights::from_network(&net);
        let test_input = [1.0f32; P2_INPUT_DIM];
        let s1 = forward(&F32Weights::from_flat(&weights_flat), &test_input);
        let s2 = forward(&net_weights, &test_input);

        let logit_diff = (s1.logit - s2.logit).abs();
        println!(
            "パリティ: f32 logit={:.4}, quantized logit={:.4}, diff={logit_diff:.4}",
            s1.logit, s2.logit
        );
        if logit_diff > 100.0 {
            eprintln!("警告: 量子化誤差が大きい（diff={logit_diff:.4}）");
        }
    }

    // レポート出力
    let final_train_loss = epoch_history.last().map(|r| r.train_loss).unwrap_or(f32::NAN);
    let final_val_loss = epoch_history.last().map(|r| r.val_loss).unwrap_or(f32::NAN);
    let report = TrainReport {
        data_file: args.data.display().to_string(),
        train_samples: train_indices.len(),
        val_samples: val_records.len(),
        epochs: args.epochs,
        batch_size: args.batch_size,
        lr: args.lr,
        weight_decay: args.weight_decay,
        seed: args.seed,
        final_train_loss,
        final_val_loss,
        epoch_history,
        output_dir: args.output_dir.display().to_string(),
        elapsed_secs: start_time.elapsed().as_secs_f64(),
    };

    write_report_md(&args.output_dir.join("RUN_REPORT.md"), &report)?;
    write_report_json(&args.output_dir.join("report.json"), &report)?;
    println!("完了: {:.1}s", start_time.elapsed().as_secs_f64());

    Ok(())
}

/// Validation loss を計算 (value, policy, policy_acc)
fn compute_val_loss(weights_flat: &[f32], val_records: &[JsonlRecord]) -> (f32, f32, f32) {
    let samples = decode_batch_dual(val_records);
    if samples.is_empty() {
        return (f32::NAN, f32::NAN, 0.0);
    }

    let weights = F32Weights::from_flat(weights_flat);
    let mut value_loss = 0.0f64;
    let mut policy_loss = 0.0f64;
    let mut correct = 0u64;
    for sample in &samples {
        let state = forward(&weights, &sample.features);
        value_loss += bce_with_logits_loss(state.logit, sample.target_winrate) as f64;
        policy_loss +=
            masked_softmax_ce_loss(&state.policy_logits, sample.target_label, &sample.legal_labels)
                as f64;

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
    ((value_loss / n) as f32, (policy_loss / n) as f32, acc)
}

/// f32 重みを量子化して .p2nn に書き出す
fn export_p2nn(weights_flat: &[f32], path: &PathBuf) -> Result<()> {
    let mut net = P2Network::new_zeroed();
    net.set_weights_from_f32(weights_flat);

    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    net.write(&mut writer)?;
    Ok(())
}
