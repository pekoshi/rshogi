//! NNUE-B Transformer 訓練 binary。
//!
//! 使い方:
//!
//! ```bash
//! # PSV file から訓練 (default 動作)
//! train_nnue_b \
//!   --data E:\shogi\shogi_hao_depth9\suisho.psv \
//!   --epochs 1 --steps-per-epoch 5000 --batch 128 \
//!   --lr 3e-4 --policy-weight 1.0 \
//!   --output E:\shogi\p2shogi\nnue-b\ckpts \
//!   --device cuda
//!
//! # dry-run (random tensor、 動作確認用)
//! train_nnue_b --dry-run --steps-per-epoch 50 --batch 64
//! ```
//!
//! Loss / optimizer:
//! - AdamW (lr / weight_decay は CLI 指定)
//! - value loss: MSE(sigmoid(value_raw), wdl * result + (1-wdl) * sigmoid(score/scale))
//! - policy loss: CE(log_softmax(policy_logits), policy_idx)、 policy_idx=-1 mask
//!
//! ## PSV ストリーミング
//!
//! - `--data` の PSV file を BufReader で先頭から読み続け、 batch 単位で
//!   `decode_psv_record` → tokens / score / result / policy_idx に変換する。
//! - 不正 record (decode 失敗) は skip し、 1 batch あたり最大
//!   [`MAX_DECODE_RETRIES`] 回 retry する。
//! - EOF に達したら file を rewind して継続 (`--loop` でなくても自動)。 1 epoch
//!   は `--steps-per-epoch * --batch` records 相当。

use anyhow::{anyhow, bail, Result};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use clap::Parser;
use nnue_b::dataset::{decode_psv_record, PsvSample};
use nnue_b::input::{NUM_TOKENS, VOCAB_SIZE};
use nnue_b::model::{NnueBConfig, NnueBModel, POLICY_SIZE};
use nnue_b::psv::{PsvReader, PsvRecord};
use nnue_b::train::{compute_loss, TrainBatch};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// 1 batch 構築時に許容する最大 decode 失敗回数。 これを超えたら abort。
const MAX_DECODE_RETRIES_PER_BATCH: usize = 1024;

#[derive(Parser, Debug)]
#[command(
    name = "train_nnue_b",
    about = "NNUE-B Candle Transformer 訓練 (PUCT hybrid 用)"
)]
struct Args {
    /// PSV ファイル (Bullet 互換、 40 byte/record)。 複数指定で round-robin 読み。
    /// dry-run 時は無視される。
    #[arg(long, num_args = 0..)]
    data: Vec<PathBuf>,

    /// 複数 PSV file を round-robin する際、 1 file から連続で読む record 数。
    /// 1 = 完全 interleave (slow, 多 disk seek)、 大きいほど batch 単位 mix。
    #[arg(long, default_value_t = 1)]
    interleave_per_file: usize,

    /// ckpt 出力ディレクトリ
    #[arg(long, default_value = "checkpoints/nnue_b")]
    output: PathBuf,

    /// 訓練 epoch 数
    #[arg(long, default_value_t = 1)]
    epochs: usize,

    /// 1 epoch あたりのステップ数
    #[arg(long, default_value_t = 1_000)]
    steps_per_epoch: usize,

    /// バッチサイズ
    #[arg(long, default_value_t = 64)]
    batch: usize,

    /// 学習率
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,

    /// weight decay
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,

    /// cp → sigmoid scale
    #[arg(long, default_value_t = 600.0)]
    scale: f32,

    /// WDL lambda: target = wdl * result + (1-wdl) * sigmoid(score/scale)
    #[arg(long, default_value_t = 0.5)]
    wdl: f32,

    /// policy CE 重み (0 で policy 学習無効)
    #[arg(long, default_value_t = 1.0)]
    policy_weight: f32,

    /// device (`cpu` または `cuda`)
    #[arg(long, default_value = "cpu")]
    device: String,

    /// 1 epoch ごとに ckpt 保存
    #[arg(long, default_value_t = true)]
    save_per_epoch: bool,

    /// log step 間隔
    #[arg(long, default_value_t = 50)]
    log_every: usize,

    /// dry-run (PSV を読まず random tensor で loop、 動作確認用)
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// CUDA を有効化するためのフラグ (上書き、 `--device cuda` と同義)
    #[arg(long, default_value_t = false)]
    cuda: bool,
}

fn pick_device(args: &Args) -> Result<Device> {
    let want_cuda = args.cuda || args.device == "cuda";
    if want_cuda {
        match Device::new_cuda(0) {
            Ok(d) => Ok(d),
            Err(e) => {
                eprintln!("[train_nnue_b] CUDA init failed ({e}), falling back to CPU");
                Ok(Device::Cpu)
            }
        }
    } else {
        Ok(Device::Cpu)
    }
}

fn next_xs(state: &mut u64, n: usize) -> usize {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as usize) % n
}

/// 1 file 分の rewind 付き PSV reader。
struct LoopingPsvReader {
    path: PathBuf,
    reader: PsvReader<File>,
    total_read: u64,
    wrap_count: u64,
}

impl LoopingPsvReader {
    fn open(path: &Path) -> Result<Self> {
        let reader = PsvReader::open(path)
            .map_err(|e| anyhow!("open PSV {} failed: {e}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            reader,
            total_read: 0,
            wrap_count: 0,
        })
    }

    fn next_record(&mut self) -> Result<PsvRecord> {
        loop {
            match self.reader.next_record()? {
                Some(r) => {
                    self.total_read += 1;
                    return Ok(r);
                }
                None => {
                    self.wrap_count += 1;
                    self.reader = PsvReader::open(&self.path)?;
                }
            }
        }
    }
}

/// 複数 PSV file を round-robin (per record) で読む reader。
///
/// `interleave_per_file` records を 1 file から連続で読み、 次の file に切替。
/// `interleave_per_file=1` で完全 record 単位 interleave。 大きいほど disk seek が
/// 減るが local 順序依存が残る。 default 1 (完全 interleave)。
struct MultiPsvReader {
    files: Vec<LoopingPsvReader>,
    /// 現在 active な file index
    cursor: usize,
    /// 現在 active file から連続して読んだ record 数
    consumed_in_current: usize,
    interleave_per_file: usize,
}

impl MultiPsvReader {
    fn open(paths: &[PathBuf], interleave_per_file: usize) -> Result<Self> {
        if paths.is_empty() {
            bail!("--data に PSV file を 1 つ以上指定してください");
        }
        let mut files = Vec::with_capacity(paths.len());
        for p in paths {
            files.push(LoopingPsvReader::open(p)?);
        }
        Ok(Self {
            files,
            cursor: 0,
            consumed_in_current: 0,
            interleave_per_file: interleave_per_file.max(1),
        })
    }

    fn next_record(&mut self) -> Result<PsvRecord> {
        let n = self.files.len();
        if self.consumed_in_current >= self.interleave_per_file {
            self.cursor = (self.cursor + 1) % n;
            self.consumed_in_current = 0;
        }
        let r = self.files[self.cursor].next_record()?;
        self.consumed_in_current += 1;
        Ok(r)
    }

    fn total_read(&self) -> u64 {
        self.files.iter().map(|f| f.total_read).sum()
    }

    fn wrap_count(&self) -> u64 {
        self.files.iter().map(|f| f.wrap_count).sum()
    }
}

/// PSV から `batch` 件の `PsvSample` を取り出す。 decode 失敗は skip。
fn collect_psv_batch(reader: &mut MultiPsvReader, batch: usize) -> Result<Vec<PsvSample>> {
    let mut out = Vec::with_capacity(batch);
    let mut retries = 0usize;
    while out.len() < batch {
        if retries > MAX_DECODE_RETRIES_PER_BATCH {
            bail!(
                "PSV decode 失敗が多すぎます ({}/{}件、 retries={})。 PSV format を確認してください。",
                out.len(),
                batch,
                retries
            );
        }
        let rec = reader.next_record()?;
        match decode_psv_record(&rec) {
            Ok(s) => out.push(s),
            Err(_e) => retries += 1,
        }
    }
    Ok(out)
}

fn samples_to_batch(samples: &[PsvSample], device: &Device) -> Result<TrainBatch> {
    let b = samples.len();
    let mut tokens_vec = Vec::with_capacity(b * NUM_TOKENS);
    let mut score_vec = Vec::with_capacity(b);
    let mut result_vec = Vec::with_capacity(b);
    let mut policy_vec = Vec::with_capacity(b);
    for s in samples {
        tokens_vec.extend_from_slice(&s.tokens);
        score_vec.push(s.score_cp);
        result_vec.push(s.result_stm);
        policy_vec.push(s.policy_idx);
    }
    Ok(TrainBatch {
        tokens: Tensor::from_vec(tokens_vec, (b, NUM_TOKENS), device)?,
        score: Tensor::from_vec(score_vec, b, device)?,
        result: Tensor::from_vec(result_vec, b, device)?,
        policy_idx: Tensor::from_vec(policy_vec, b, device)?,
    })
}

fn make_random_batch(batch: usize, seed: &mut u64, device: &Device) -> Result<TrainBatch> {
    let mut tokens_vec = vec![0i64; batch * NUM_TOKENS];
    for v in tokens_vec.iter_mut() {
        *v = (next_xs(seed, VOCAB_SIZE)) as i64;
    }
    let mut score_vec = vec![0f32; batch];
    let mut result_vec = vec![0f32; batch];
    let mut policy_vec = vec![0i64; batch];
    for i in 0..batch {
        let s = next_xs(seed, 2000) as i32 - 1000;
        score_vec[i] = s as f32;
        result_vec[i] = (next_xs(seed, 3) as f32) * 0.5;
        let p = next_xs(seed, POLICY_SIZE + 1);
        policy_vec[i] = if p == POLICY_SIZE { -1 } else { p as i64 };
    }
    Ok(TrainBatch {
        tokens: Tensor::from_vec(tokens_vec, (batch, NUM_TOKENS), device)?,
        score: Tensor::from_vec(score_vec, batch, device)?,
        result: Tensor::from_vec(result_vec, batch, device)?,
        policy_idx: Tensor::from_vec(policy_vec, batch, device)?,
    })
}

/// PSV file 全体の record 数を推定 (size / 40)。 不正サイズなら 0。
fn estimate_psv_records(path: &Path) -> Result<u64> {
    let meta = std::fs::metadata(path)?;
    let bytes = meta.len();
    Ok(bytes / 40)
}

/// PSV 先頭 1 件だけ peek (decode せず raw 表示)。
fn peek_psv_header(path: &Path) -> Result<()> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut buf = [0u8; 40];
    r.read_exact(&mut buf)?;
    // 一旦 seek back を試す (debug 目的)
    let _ = r.seek(SeekFrom::Start(0));
    if let Some(rec) = PsvRecord::parse(&buf) {
        eprintln!(
            "[train_nnue_b] PSV peek score={} move_u16=0x{:04x} ply={} result={}",
            rec.score, rec.move_u16, rec.game_ply, rec.game_result
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = pick_device(&args)?;

    eprintln!("[train_nnue_b] device={:?}", device);
    eprintln!(
        "[train_nnue_b] batch={} lr={} wdl={} policy_w={} scale={} dry_run={}",
        args.batch, args.lr, args.wdl, args.policy_weight, args.scale, args.dry_run
    );

    std::fs::create_dir_all(&args.output)?;

    let cfg = NnueBConfig::nnue_b_default();
    eprintln!(
        "[train_nnue_b] arch d_model={} n_layers={} n_heads={} d_ff={}",
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff
    );
    let model = NnueBModel::new(cfg, device.clone())?;
    let mut opt = AdamW::new(
        model.varmap().all_vars(),
        ParamsAdamW {
            lr: args.lr,
            weight_decay: args.weight_decay,
            ..Default::default()
        },
    )?;

    let mut rng = args.seed.max(1);
    let mut psv_reader = if !args.dry_run {
        if args.data.is_empty() {
            bail!("--data <PSV> を 1 つ以上指定してください (または --dry-run)");
        }
        // 先頭 file の header peek + 全 file の record 数推定
        peek_psv_header(&args.data[0])?;
        let mut total_estimated: u64 = 0;
        for p in &args.data {
            let n = estimate_psv_records(p)?;
            total_estimated += n;
            eprintln!("[train_nnue_b] PSV {} 推定 {} records", p.display(), n);
        }
        eprintln!(
            "[train_nnue_b] 全 {} files 合計推定 {} records (interleave_per_file={})",
            args.data.len(),
            total_estimated,
            args.interleave_per_file
        );
        Some(MultiPsvReader::open(&args.data, args.interleave_per_file)?)
    } else {
        None
    };

    let mut global_step: u64 = 0;
    for epoch in 1..=args.epochs {
        let mut avg_v = 0.0f64;
        let mut avg_p = 0.0f64;
        let mut count = 0u64;
        let epoch_started = Instant::now();
        for step in 1..=args.steps_per_epoch {
            let batch = match psv_reader.as_mut() {
                Some(r) => {
                    let samples = collect_psv_batch(r, args.batch)?;
                    samples_to_batch(&samples, &device)?
                }
                None => make_random_batch(args.batch, &mut rng, &device)?,
            };
            let (v_pred, p_pred) = model.forward(&batch.tokens)?;
            let loss = compute_loss(
                &v_pred,
                &p_pred,
                &batch,
                args.scale,
                args.wdl,
                args.policy_weight,
                &device,
            )?;
            let v_val = loss.value_loss.to_scalar::<f32>()? as f64;
            let p_val = if args.policy_weight > 0.0 {
                loss.policy_loss.to_scalar::<f32>()? as f64
            } else {
                0.0
            };
            opt.backward_step(&loss.total)?;

            avg_v += v_val;
            avg_p += p_val;
            count += 1;
            global_step += 1;
            if step % args.log_every == 0 {
                let elapsed = epoch_started.elapsed().as_secs_f64();
                let pos_per_sec = (count as f64 * args.batch as f64) / elapsed.max(1e-6);
                eprintln!(
                    "[epoch {epoch} step {step}] loss_v={:.4} loss_p={:.4} pos/sec={:.0}",
                    avg_v / count as f64,
                    avg_p / count as f64,
                    pos_per_sec
                );
            }
        }
        eprintln!(
            "[epoch {epoch}/{}] avg loss_v={:.4} loss_p={:.4} (steps={})",
            args.epochs,
            avg_v / count as f64,
            avg_p / count as f64,
            count
        );
        if let Some(r) = psv_reader.as_ref() {
            eprintln!("[train_nnue_b] PSV read total={} wrap={}", r.total_read(), r.wrap_count());
        }
        if args.save_per_epoch {
            let ckpt = args.output.join(format!("nnue_b-ep{epoch}.safetensors"));
            model.save_safetensors(&ckpt)?;
            eprintln!("[train_nnue_b] saved {}", ckpt.display());
        }
    }

    eprintln!("[train_nnue_b] done, total steps={global_step}");
    Ok(())
}
