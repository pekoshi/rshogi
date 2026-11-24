//! progress8kpabs (進行度) bucket 重み `progress.bin` の自前訓練ツール
//!
//! NNUE の bucket 選択に使う「進行度」線形モデルを floodgate CSA 棋譜から訓練する。
//! 目的は license クリーンな provenance (現行 progress.bin は tanuki/nodchip 系統由来)。
//!
//! # モデル
//!
//! `progress_pred = sigmoid(w · x)`。`w` は長さ `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS`
//! (= 81 * FE_OLD_END = 125388) の重み配列。`x` は局面の疎特徴 (黒玉視点 + 白玉視点
//! 両方が同一 `w` を指す) で、ランタイムの `compute_progress8kpabs_sum` と完全一致する
//! index を emit する。
//!
//! # 教師信号 `t ∈ [0, 1]`
//!
//! - `p_ply = ply / total_ply`
//! - 引分局 (引分・千日手・持将棋など非決着): `p_res = p_ply`
//! - 決着局 (勝敗が付いた局): `p_res = p_ply.powf(gamma)` (gamma ∈ (0,1])
//!   凹 boost で終盤 (bucket 6,7) を持ち上げる。`gamma = 1` で `p_res = p_ply`。
//! - `t = alpha * p_ply + (1 - alpha) * p_res`
//!
//! 既定: `alpha = 0.5`, `decisive-gamma = 0.5`。
//!
//! # 損失
//!
//! soft binary cross-entropy `-(t*ln(ŷ) + (1-t)*ln(1-ŷ))` + L2 正則化 `lambda*||w||^2`。
//! 最適化は Adam。
//!
//! # 出力
//!
//! `progress.bin` = `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS` 個の f64 little-endian。
//! index = `king_sq * FE_OLD_END + bonapiece_value`。
//!
//! # 使用例
//!
//! ```bash
//! # 軽量 smoke
//! cargo run --release -p tools --bin train_progress_kpabs -- \
//!   --csa-dir data/training/floodgate/csa \
//!   --out target/progress_trained.bin \
//!   --max-games 50
//!
//! # 本訓練
//! cargo run --release -p tools --bin train_progress_kpabs -- \
//!   --csa-dir data/training/floodgate/csa \
//!   --out target/progress_trained.bin \
//!   --alpha 0.5 --decisive-gamma 0.5 --l2 1e-7 --lr 1e-2 --epochs 10
//! ```

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::PathBuf;

use clap::Parser;
use rshogi_core::movegen::{MoveList, generate_legal_all};
use rshogi_core::nnue::{
    BonaPiece, FE_OLD_END, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, progress_sum_to_bucket,
};
use rshogi_core::position::Position;
use rshogi_core::types::{Color, Move, PieceType};
use tools::common::dedup::collect_input_paths;

#[derive(Parser, Debug)]
#[command(
    name = "train_progress_kpabs",
    about = "progress8kpabs (進行度) bucket 重み progress.bin の自前訓練"
)]
struct Args {
    /// 入力 CSA ディレクトリ (再帰探索)。完全な棋譜 (ply / total) が必要なので
    /// shuffle 済 PSV ではなく CSA を使うこと。
    #[arg(long)]
    csa_dir: PathBuf,

    /// --csa-dir 使用時の glob パターン
    #[arg(long, default_value = "*.csa")]
    pattern: String,

    /// 出力ファイルパス (f64 LE)。既存 eval/progress.bin を上書きしないよう
    /// 新規ファイル名を既定にする。
    #[arg(long, default_value = "progress_trained.bin")]
    out: PathBuf,

    /// 教師信号の形状指数 beta (t = (ply/total_ply)^beta)。
    /// 既定 1.0 = 純手数比 (tanuki/技巧 と同一)。実験用ノブとして残す。
    #[arg(long, default_value_t = 1.0)]
    beta: f64,

    /// L2 正則化係数 lambda
    #[arg(long, default_value_t = 1e-7)]
    l2: f64,

    /// Adam 学習率
    #[arg(long, default_value_t = 1e-2)]
    lr: f64,

    /// エポック数
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// 処理する最大棋譜数 (0 = 無制限)
    #[arg(long, default_value_t = 0)]
    max_games: usize,

    /// この ply 未満の局面はスキップ (序盤定跡カット用)
    #[arg(long, default_value_t = 0)]
    min_ply: u16,

    /// この ply 超過の局面はスキップ
    #[arg(long, default_value_t = 512)]
    max_ply: u16,

    /// 指定した既存 progress.bin の bucket 分布を、同一 CSA サンプル上で集計して
    /// 表示するだけのモード (訓練しない)。tanuki 系 eval/progress.bin の基準分布取得用。
    #[arg(long)]
    hist_weights: Option<PathBuf>,

    /// tanuki/技巧 流儀のデータフィルタ: 引分局を除外し、256 手以上の対局も除外する。
    /// (投了短手数局の終盤質量過多を抑え、tanuki の単調減少分布に近づける。)
    #[arg(long, default_value_t = false)]
    decisive_only: bool,
}

/// 1 局面の訓練サンプル: 疎特徴 index 群と教師 target。
struct Sample {
    /// progress8kpabs の active index 群 (重複あり = 同一 index が複数回 +1 されることもある)。
    indices: Vec<u32>,
    /// 教師信号 t ∈ [0, 1]
    target: f64,
}

/// `compute_progress8kpabs_sum` の loop 構造を複製して active index 群を emit する。
///
/// network.rs を refactor せず、rshogi-core の public `BonaPiece` API を呼んで
/// bit-exact に同じ index を生成する。返す index 群を w で引いて加算すれば
/// `compute_progress8kpabs_sum(pos, w)` と完全一致する。
fn emit_progress_indices(pos: &Position) -> Vec<u32> {
    let sq_bk = pos.king_square(Color::Black).index();
    let sq_wk = pos.king_square(Color::White).inverse().index();
    let base_b = (sq_bk * FE_OLD_END) as u32;
    let base_w = (sq_wk * FE_OLD_END) as u32;

    let mut indices = Vec::new();

    // 盤上の駒 (玉以外)。黒玉視点 (Black) と白玉視点 (White) の両方を同一 w に加算。
    for sq in pos.occupied().iter() {
        let pc = pos.piece_on(sq);
        if pc.is_none() || pc.piece_type() == PieceType::King {
            continue;
        }

        let bp_b = BonaPiece::from_piece_square(pc, sq, Color::Black);
        if bp_b != BonaPiece::ZERO {
            indices.push(base_b + bp_b.value() as u32);
        }

        let bp_w = BonaPiece::from_piece_square(pc, sq, Color::White);
        if bp_w != BonaPiece::ZERO {
            indices.push(base_w + bp_w.value() as u32);
        }
    }

    // 持駒。HAND_PIECES を 1..=count で展開。
    for owner in [Color::Black, Color::White] {
        let hand = pos.hand(owner);
        for &pt in &PieceType::HAND_PIECES {
            let count = hand.count(pt);
            for c in 1..=count {
                let c_u8 = u8::try_from(c).expect("hand count fits in u8");

                let bp_b = BonaPiece::from_hand_piece(Color::Black, owner, pt, c_u8);
                if bp_b != BonaPiece::ZERO {
                    indices.push(base_b + bp_b.value() as u32);
                }

                let bp_w = BonaPiece::from_hand_piece(Color::White, owner, pt, c_u8);
                if bp_w != BonaPiece::ZERO {
                    indices.push(base_w + bp_w.value() as u32);
                }
            }
        }
    }

    indices
}

/// 教師信号 t を計算する。
///
/// - `p_ply = ply / total_ply`
/// - `t = p_ply.powf(beta)`。`beta > 1` で低 progress 側 (bucket 0 寄り) に歪め、
///   tanuki の単調減少分布に一致させる。`beta = 1` で線形 (= p_ply)。
fn compute_target(ply: u16, total_ply: u16, beta: f64) -> f64 {
    let l = f64::from(total_ply);
    let ply_f = f64::from(ply);
    let p_ply = if l > 0.0 { ply_f / l } else { 0.0 };
    p_ply.powf(beta)
}

/// 引分 (千日手/持将棋/Draw) で終わった対局かを判定する。決着局なら false。
fn is_draw_game(moves: &[rshogi_csa::ParsedMove]) -> bool {
    use rshogi_csa::{ParsedMove, SpecialMove};
    for mv in moves.iter().rev() {
        if let ParsedMove::Special(s) = mv {
            return matches!(s, SpecialMove::Draw | SpecialMove::Sennichite | SpecialMove::Jishogi);
        }
    }
    false
}

/// CSA テキストを個別の棋譜に分割。
fn split_csa_games(content: &str) -> Vec<&str> {
    let mut games = Vec::new();
    let mut start = 0;

    for (i, _) in content.match_indices("\nV2") {
        if start < i {
            let segment = content[start..i].trim();
            if !segment.is_empty() {
                games.push(segment);
            }
        }
        start = i + 1;
    }

    let segment = content[start..].trim();
    if !segment.is_empty() {
        games.push(segment);
    }

    if games.is_empty() && !content.trim().is_empty() {
        games.push(content.trim());
    }

    games
}

/// 1 棋譜分を Sample に変換し samples に追記。エラー時は途中まで進めて break。
fn collect_game_samples(text: &str, args: &Args, samples: &mut Vec<Sample>) -> Result<(), String> {
    let (csa_init_pos, moves, _info) =
        rshogi_csa::parse_csa_full(text).map_err(|e| format!("parse: {e}"))?;

    let init_sfen = csa_init_pos.to_sfen();
    let mut pos = Position::new();
    pos.set_sfen(&init_sfen).map_err(|e| format!("set_sfen: {e:?}"))?;

    let mut csa_pos = csa_init_pos.clone();

    let normal_moves: Vec<String> = moves
        .iter()
        .filter_map(|m| match m {
            rshogi_csa::ParsedMove::Normal(cm) => Some(cm.mv.clone()),
            rshogi_csa::ParsedMove::Special(_) => None,
        })
        .collect();

    let total_ply = normal_moves.len() as u16;

    // tanuki/技巧 流儀のデータフィルタ: 引分除外・超長手数(>=256)除外。
    if args.decisive_only && (total_ply >= 256 || is_draw_game(&moves)) {
        return Ok(());
    }

    for csa_mv in normal_moves.iter() {
        let usi = match rshogi_csa::csa_move_to_usi(csa_mv, &csa_pos) {
            Ok(u) => u,
            Err(_) => break,
        };

        let parsed = match Move::from_usi(&usi) {
            Some(m) => m,
            None => break,
        };

        let mut legal_moves = MoveList::new();
        generate_legal_all(&pos, &mut legal_moves);

        let mv = legal_moves
            .iter()
            .find(|m| {
                m.to() == parsed.to()
                    && m.is_drop() == parsed.is_drop()
                    && m.is_promotion() == parsed.is_promotion()
                    && if parsed.is_drop() {
                        m.drop_piece_type() == parsed.drop_piece_type()
                    } else {
                        m.from() == parsed.from()
                    }
            })
            .copied();

        let mv = match mv {
            Some(m) => m,
            None => break,
        };

        let game_ply = pos.game_ply() as u16;

        if game_ply >= args.min_ply && game_ply <= args.max_ply {
            let target = compute_target(game_ply, total_ply, args.beta);
            samples.push(Sample {
                indices: emit_progress_indices(&pos),
                target,
            });
        }

        let gives_check = pos.gives_check(mv);
        pos.do_move(mv, gives_check);
        csa_pos.apply_csa_move(csa_mv).map_err(|e| format!("apply_csa: {e}"))?;
    }

    Ok(())
}

/// 数値安定な sigmoid。
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Adam で線形モデルを訓練し、f64 の重み配列を返す。
fn train(samples: &[Sample], args: &Args) -> Vec<f64> {
    let n = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;
    let mut w = vec![0.0f64; n];
    let mut m = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];

    let beta1 = 0.9f64;
    let beta2 = 0.999f64;
    let eps = 1e-8f64;
    let lr = args.lr;
    let lambda = args.l2;

    // 全サンプル一括の full-batch gradient + Adam。サンプル数は CSA 棋譜から
    // 数万〜数十万局面、重み 12.5 万なので軽い。
    let mut t_step = 0u64;

    for epoch in 0..args.epochs {
        // 疎勾配を蓄積 (active index 以外は L2 のみ)。
        let mut grad: BTreeMap<u32, f64> = BTreeMap::new();
        let mut total_loss = 0.0f64;

        let inv_n = if samples.is_empty() {
            0.0
        } else {
            1.0 / samples.len() as f64
        };

        for s in samples {
            // 前向き: z = Σ w[idx]
            let mut z = 0.0f64;
            for &idx in &s.indices {
                z += w[idx as usize];
            }
            let y = sigmoid(z);

            // MSE on sigmoid 出力 (tanuki/技巧 と同形)。BCE だと出力が 0/1 に
            // 押し出され分布が bimodal 化するため、tanuki 一致には MSE を使う。
            let t = s.target;
            let err = y - t;
            total_loss += err * err;

            // dL/dz = (y - t) * sigmoid'(z) = (y - t) * y * (1 - y)。
            // 各 active index に同じ勾配を加算 (重複 index は重複加算)。
            let dz = err * y * (1.0 - y) * inv_n;
            for &idx in &s.indices {
                *grad.entry(idx).or_insert(0.0) += dz;
            }
        }

        // Adam step。L2 正則化は全重みに掛かるが、勾配が 0 の重みは
        // L2 のみで Adam を進めると全重みを毎 step 更新する必要が生じ重い。
        // ここでは active 勾配が立った重みのみ更新し、L2 はその重みにのみ適用する
        // (疎モデルの慣例。lambda は小さく実害なし)。
        t_step += 1;
        let bc1 = 1.0 - beta1.powi(t_step as i32);
        let bc2 = 1.0 - beta2.powi(t_step as i32);

        for (&idx, &g_data) in &grad {
            let i = idx as usize;
            let g = g_data + 2.0 * lambda * w[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        let rmse = if samples.is_empty() {
            0.0
        } else {
            (total_loss / samples.len() as f64).sqrt()
        };
        eprintln!(
            "epoch {:>3}/{}: RMSE = {:.6}, active weights updated = {}",
            epoch + 1,
            args.epochs,
            rmse,
            grad.len()
        );
    }

    w
}

/// 全 samples を等間隔 stride で最大 5 万件サブサンプルし bucket 分布を集計する。
/// 決定論的 (rand 非依存)。先頭バイアスを避けるため stride サンプリング。
fn bucket_histogram(samples: &[Sample], w_f32: &[f32]) -> [u64; 8] {
    let target_count = 50_000usize;
    let total = samples.len();
    let stride = total.div_ceil(target_count).max(1);
    let mut hist = [0u64; 8];
    let mut i = 0usize;
    while i < total {
        let mut sum = 0.0f32;
        for &idx in &samples[i].indices {
            sum += w_f32[idx as usize];
        }
        let b = progress_sum_to_bucket(sum);
        if b < 8 {
            hist[b] += 1;
        }
        i += stride;
    }
    hist
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    if args.hist_weights.is_none() && args.out.exists() {
        eprintln!(
            "ERROR: 出力先 '{}' は既に存在します。既存ファイルの上書きを避けるため別名を指定してください。",
            args.out.display()
        );
        std::process::exit(1);
    }

    let paths = collect_input_paths(None, Some(&args.csa_dir), &args.pattern)?;
    if paths.is_empty() {
        eprintln!("入力 CSA ファイルが見つかりません: {}", args.csa_dir.display());
        std::process::exit(1);
    }

    eprintln!("CSA files found: {}", paths.len());
    eprintln!(
        "beta={}, l2={}, lr={}, epochs={}, max_games={}",
        args.beta, args.l2, args.lr, args.epochs, args.max_games
    );

    // 棋譜を読み込み Sample 化。
    let mut samples: Vec<Sample> = Vec::new();
    let mut games_done = 0usize;
    let mut parse_errors = 0usize;

    'outer: for path in &paths {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };
        let cleaned: String =
            bytes.into_iter().map(|b| if b < 0x80 { b as char } else { ' ' }).collect();

        for game_text in split_csa_games(&cleaned) {
            if args.max_games > 0 && games_done >= args.max_games {
                break 'outer;
            }
            match collect_game_samples(game_text, &args, &mut samples) {
                Ok(()) => games_done += 1,
                Err(_) => parse_errors += 1,
            }
        }
    }

    eprintln!(
        "games processed: {}, parse errors: {}, samples: {}",
        games_done,
        parse_errors,
        samples.len()
    );

    if samples.is_empty() {
        eprintln!("ERROR: 訓練サンプルが 0 件です。CSA ディレクトリを確認してください。");
        std::process::exit(1);
    }

    // --hist-weights: 既存 progress.bin を読み、同一サンプル上で bucket 分布のみ
    // 集計して終了 (訓練しない)。tanuki 系の基準分布を取得するため。
    if let Some(hw_path) = &args.hist_weights {
        let bytes = std::fs::read(hw_path)?;
        let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f64>();
        if bytes.len() != expected {
            eprintln!(
                "ERROR: {} のサイズが不正: {} bytes (期待 {})",
                hw_path.display(),
                bytes.len(),
                expected
            );
            std::process::exit(1);
        }
        let w_f32: Vec<f32> = bytes
            .chunks_exact(size_of::<f64>())
            .map(|c| f64::from_le_bytes(c.try_into().expect("chunk size is checked")) as f32)
            .collect();
        let hist = bucket_histogram(&samples, &w_f32);
        eprintln!("hist-weights {} bucket histogram: {hist:?}", hw_path.display());
        println!("OK: histogram only (no training)");
        return Ok(());
    }

    // 訓練。
    let w = train(&samples, &args);

    // NaN 検査。
    if w.iter().any(|x| x.is_nan() || x.is_infinite()) {
        eprintln!("ERROR: 学習後の重みに NaN/Inf が含まれます。lr/l2 を調整してください。");
        std::process::exit(1);
    }

    // f64 LE で書き出し。
    let mut out = io::BufWriter::new(std::fs::File::create(&args.out)?);
    for &x in &w {
        out.write_all(&x.to_le_bytes())?;
    }
    out.flush()?;
    drop(out);

    let written = std::fs::metadata(&args.out)?.len();
    let expected = (SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f64>()) as u64;
    eprintln!("wrote {} ({} bytes, expected {})", args.out.display(), written, expected);
    assert_eq!(written, expected, "progress.bin サイズが期待値と不一致");

    // bucket 分布の簡易レポート (f32 化してランタイムと同じ閾値で集計)。
    let w_f32: Vec<f32> = w.iter().map(|&x| x as f32).collect();
    let hist = bucket_histogram(&samples, &w_f32);
    eprintln!("bucket histogram: {hist:?}");

    println!("OK: progress.bin written to {}", args.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rshogi_core::nnue::compute_progress8kpabs_sum;

    /// 複数のランダム局面 + ランダム重み w について、
    /// 自前 emit した index 群で Σ w[idx] が rshogi-core の
    /// compute_progress8kpabs_sum(pos, w) と完全一致することを確認する。
    /// これが feature parity (bit-exact) の保証。
    #[test]
    fn test_emit_indices_bit_exact() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);

        // ランダム重み配列。
        let mut w = vec![0.0f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
        for x in w.iter_mut() {
            *x = rng.random_range(-1.0f32..1.0f32);
        }

        // 複数の局面: startpos からランダムに合法手を進める。
        let mut positions: Vec<Position> = Vec::new();
        for seed in 0..8u64 {
            let mut pos = Position::new();
            pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
                .expect("startpos sfen");

            let mut local = StdRng::seed_from_u64(seed.wrapping_mul(2654435761));
            let steps = 2 + (seed as usize % 30);
            for _ in 0..steps {
                let mut legal = MoveList::new();
                generate_legal_all(&pos, &mut legal);
                if legal.is_empty() {
                    break;
                }
                let pick = local.random_range(0..legal.len());
                let mv = legal.iter().nth(pick).copied().expect("picked move");
                let gives_check = pos.gives_check(mv);
                pos.do_move(mv, gives_check);
            }
            positions.push(pos);
        }

        for (i, pos) in positions.iter().enumerate() {
            // 自前 emit。
            let indices = emit_progress_indices(pos);
            let mut my_sum = 0.0f32;
            for &idx in &indices {
                my_sum += w[idx as usize];
            }

            // ランタイム参照実装。
            let ref_sum = compute_progress8kpabs_sum(pos, &w);

            assert_eq!(
                my_sum.to_bits(),
                ref_sum.to_bits(),
                "position #{i}: emit sum {my_sum} != reference {ref_sum} (bit mismatch)"
            );
        }
    }

    #[test]
    fn test_compute_target_beta_one_linear() {
        // beta=1: t = p_ply。
        let t = compute_target(30, 100, 1.0);
        assert!((t - 0.3).abs() < 1e-9, "beta=1 target = {t}");
    }

    #[test]
    fn test_compute_target_beta_skew_low() {
        // beta=2: p_ply=0.5 → t=0.25 (低 progress 側へ歪む)。
        let t = compute_target(50, 100, 2.0);
        assert!((t - 0.25).abs() < 1e-9, "beta=2 target = {t}");
        // ply=total → t=1 (beta 無関係)。
        let t2 = compute_target(100, 100, 2.0);
        assert!((t2 - 1.0).abs() < 1e-9, "last ply target = {t2}");
    }
}
