//! build_washiun_train - washiun 教師データから nodchip 学習用 psv を構築する
//!
//! washiun (`split_*.bin`, PackedSfenValue 40バイト/レコード) を複数ストリーム処理し、
//! 以下を単一パスで適用して均衡化した psv を出力する。
//!
//! - mate 除外: `|score| >= mate_threshold` を drop
//! - 居玉フィルタ: `ply >= igyo_min_ply` かつ「手番側の玉が自陣本来位置(先手5九/後手5一)」を drop
//! - 重複排除: PackedSfen(32バイト) の bloom filter dedup
//! - フェーズ均衡: 序盤/中盤/終盤を材料+手数で分類し、フェーズ別 keep 確率で downsample
//!
//! keep 判定は局面ハッシュ由来の決定的疑似乱数を使うため、スレッド数・実行順に依存しない。
//!
//! # 使用例
//!
//! ```bash
//! # 事前計測（フェーズ分布・フィルタ通過率を把握）
//! cargo run -p tools --release --bin build_washiun_train -- \
//!   --inputs E:/shogi/p2shogi/data/suisho_ft/washiun_500m.bin \
//!   --measure-only --sample 100000000
//!
//! # 本番構築
//! cargo run -p tools --release --bin build_washiun_train -- \
//!   --inputs a.bin b.bin c.bin --output out.psv \
//!   --keep-opening 1.0 --keep-middle 0.545 --keep-endgame 0.622 \
//!   --delete-after-read
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use tools::common::bloom::{BloomFilter, hash_pair_bytes};
use tools::dataset::phase::classify_phase;
use tools::dataset::types::Phase;
use tools::packed_sfen::unpack_sfen;

const REC: usize = 40;

/// 先手玉の本来位置 5九 = (筋5-1)*9 + (段9-1) = 44
const BLACK_KING_HOME: u8 = 44;
/// 後手玉の本来位置 5一 = (筋5-1)*9 + (段1-1) = 36
const WHITE_KING_HOME: u8 = 36;
/// 先手 中住まい 5八 = (筋5-1)*9 + (段8-1) = 43
const BLACK_KING_MIDDLE: u8 = 43;
/// 後手 中住まい 5二 = (筋5-1)*9 + (段2-1) = 37
const WHITE_KING_MIDDLE: u8 = 37;

#[derive(Parser)]
#[command(
    name = "build_washiun_train",
    version,
    about = "washiun 教師データから mate除外/居玉除外/dedup/フェーズ均衡を適用した学習 psv を構築"
)]
struct Cli {
    /// 入力 PackedSfenValue ファイル（複数指定可、指定順に処理）
    #[arg(long, num_args = 1.., required = true)]
    inputs: Vec<PathBuf>,

    /// 出力 psv（--measure-only 時は不要）
    #[arg(long)]
    output: Option<PathBuf>,

    /// 居玉除外の手数しきい値（この手数以上で手番玉が本来位置なら drop）
    #[arg(long, default_value_t = 24)]
    igyo_min_ply: u16,

    /// 居玉かつ敗北(result<0=手番側が負け)の局面は除外せず残す（居玉の危険性を学習）
    #[arg(long, default_value_t = false)]
    keep_igyo_loss: bool,

    /// 居玉敗北局面の複製回数（>1 で oversample、phase-keep をバイパスして必ず書く）
    #[arg(long, default_value_t = 1)]
    igyo_loss_oversample: u32,

    /// mate 除外しきい値（|score| がこれ以上を drop）
    #[arg(long, default_value_t = 29000)]
    mate_threshold: i32,

    /// cp帯 hard filter: 通常局面で |score| がこの範囲[cp_lo,cp_hi]外を drop する。cp_hi=0 で無効。
    /// 例: cp0-250帯なら --cp-lo 0 --cp-hi 250。
    #[arg(long, default_value_t = 0)]
    cp_lo: i32,
    #[arg(long, default_value_t = 0)]
    cp_hi: i32,

    /// 中央玉除去: 手番玉が居玉(5九/5一)または中住まい(5八/5二)の局面を、
    /// score >= central_king_keep_score のとき除外する(評価が良い/互角の中央玉を捨てる)。
    /// score < keep_score(評価が悪い)の中央玉は cp帯を免除して残す。ply >= igyo_min_ply で適用。
    #[arg(long, default_value_t = false)]
    central_king_drop: bool,
    /// 中央玉を残す評価しきい値。score がこれ未満なら残す(default -100)。
    #[arg(long, default_value_t = -100)]
    central_king_keep_score: i32,

    /// eval帯 重点: |score| がこの範囲[lo,hi]の中盤局面を oversample する。hi=0 で無効。
    #[arg(long, default_value_t = 0)]
    eval_band_lo: i32,
    #[arg(long, default_value_t = 0)]
    eval_band_hi: i32,
    /// eval帯内 中盤局面の複製回数（>1 で oversample、phase-keep をバイパスして必ず書く）。
    #[arg(long, default_value_t = 1)]
    eval_band_oversample: u32,

    /// フェーズ別 keep 確率（序盤 / 中盤 / 終盤）。dedup 通過後に適用。
    #[arg(long, default_value_t = 1.0)]
    keep_opening: f64,
    #[arg(long, default_value_t = 1.0)]
    keep_middle: f64,
    #[arg(long, default_value_t = 1.0)]
    keep_endgame: f64,

    /// 全体スケール（全フェーズ keep 確率に一律乗算。総量微調整用）
    #[arg(long, default_value_t = 1.0)]
    global_scale: f64,

    /// keep 判定の乱数シード
    #[arg(long, default_value_t = 20260708)]
    seed: u64,

    /// bloom filter の想定要素数（dedup サイズ設計）
    #[arg(long, default_value_t = 4_000_000_000)]
    bloom_elems: u64,

    /// bloom filter の偽陽性率
    #[arg(long, default_value_t = 0.01)]
    bloom_fpr: f64,

    /// 計測のみ（出力しない）。フェーズ分布とフィルタ通過率を報告。
    #[arg(long, default_value_t = false)]
    measure_only: bool,

    /// 各ファイル先頭からの計測サンプル数（0=全件）。measure_only 用。
    #[arg(long, default_value_t = 0)]
    sample: u64,

    /// 読み終えた入力ファイルを削除する（ディスク節約）
    #[arg(long, default_value_t = false)]
    delete_after_read: bool,

    /// rayon スレッド数（0=自動）
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// 1チャンクあたりのレコード数
    #[arg(long, default_value_t = 4_000_000)]
    chunk: usize,
}

/// レコード単位の判定結果（並列計算パートの出力）
#[derive(Clone, Copy)]
struct Decision {
    /// mate / 居玉 を通過したか
    survives: bool,
    /// 居玉かつ敗北で残した局面（phase-keep をバイパスして oversample する対象）
    igyo_loss: bool,
    /// eval帯内の中盤局面（重点 oversample 対象）
    in_band: bool,
    /// フェーズ（0=序盤,1=中盤,2=終盤）
    phase: u8,
    /// PackedSfen ハッシュ（dedup + keep 判定用）
    h1: u64,
    h2: u64,
}

/// 手番玉が本来位置にいる「居玉」か判定する。
///
/// PackedSfen の先頭ビット配置（LSB-first）:
/// bit0 = 手番(0=先手/1=後手), bit1..7 = 先手玉7bit, bit8..14 = 後手玉7bit。
#[inline]
fn is_igyo_side_to_move(sfen: &[u8]) -> bool {
    let turn_black = (sfen[0] & 1) == 0;
    let black_king = (sfen[0] >> 1) & 0x7F;
    let white_king = sfen[1] & 0x7F;
    if turn_black {
        black_king == BLACK_KING_HOME
    } else {
        white_king == WHITE_KING_HOME
    }
}

/// 手番玉が「居玉(5九/5一)または中住まい(5八/5二)」= 中央玉かを判定する。
#[inline]
fn is_central_king_side_to_move(sfen: &[u8]) -> bool {
    let turn_black = (sfen[0] & 1) == 0;
    let black_king = (sfen[0] >> 1) & 0x7F;
    let white_king = sfen[1] & 0x7F;
    if turn_black {
        black_king == BLACK_KING_HOME || black_king == BLACK_KING_MIDDLE
    } else {
        white_king == WHITE_KING_HOME || white_king == WHITE_KING_MIDDLE
    }
}

/// 局面ハッシュ由来の決定的 [0,1) 疑似乱数
#[inline]
fn keep_fraction(h1: u64, seed: u64) -> f64 {
    let mut x = h1 ^ seed.wrapping_mul(0x9E3779B97F4A7C15);
    // splitmix64 finalizer
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    ((x >> 11) as f64) / ((1u64 << 53) as f64)
}

const DROP: Decision = Decision {
    survives: false,
    igyo_loss: false,
    in_band: false,
    phase: 0,
    h1: 0,
    h2: 0,
};

/// 1レコードを並列に判定する（unpack は生存レコードのみ）
fn decide(
    rec: &[u8],
    mate_threshold: i32,
    igyo_min_ply: u16,
    keep_igyo_loss: bool,
    eval_band_lo: i32,
    eval_band_hi: i32,
    cp_lo: i32,
    cp_hi: i32,
    central_king_drop: bool,
    central_king_keep_score: i32,
) -> Decision {
    let sfen = &rec[0..32];
    let score = i16::from_le_bytes([rec[32], rec[33]]) as i32;
    let ply = u16::from_le_bytes([rec[36], rec[37]]);
    // result byte(offset38) = int8 stm視点 win=+1/draw=0/loss=-1
    let result = rec[38] as i8;

    // mate 除外
    if score.abs() >= mate_threshold {
        return DROP;
    }

    // 中央玉(居玉5九/5一 + 中住まい5八/5二)処理: --central-king-drop 有効時。
    // 評価が悪い(score < keep_score, 例 -100)局面のみ cp帯を免除して残し、
    // それ以外(良い/普通)の中央玉局面は除外する。
    // 有効時は下段の従来 igyo ロジックと cp帯フィルタをバイパスする。
    let mut igyo_loss = false;
    if central_king_drop {
        if ply >= igyo_min_ply && is_central_king_side_to_move(sfen) {
            if score >= central_king_keep_score {
                return DROP; // 得している/互角の中央玉 → 除外
            }
            // 評価の悪い中央玉 → cp帯を免除して残す
        } else {
            // 通常局面: cp帯 [cp_lo, cp_hi] 外を除外(cp_hi=0 で無効)
            if cp_hi > 0 && !(score.abs() >= cp_lo && score.abs() <= cp_hi) {
                return DROP;
            }
        }
    } else {
        // 従来ロジック: 居玉判定(通常除外、keep_igyo_loss かつ敗北なら残す)
        if ply >= igyo_min_ply && is_igyo_side_to_move(sfen) {
            if keep_igyo_loss && result < 0 {
                igyo_loss = true; // 居玉で負けた局面 → 学習対象として残す
            } else {
                return DROP;
            }
        }
        // cp帯 [cp_lo, cp_hi] 外を除外(cp_hi=0 で無効)
        if cp_hi > 0 && !(score.abs() >= cp_lo && score.abs() <= cp_hi) {
            return DROP;
        }
    }

    let phase = match unpack_sfen(sfen.try_into().unwrap()) {
        Ok(s) => match classify_phase(&s, ply as u32) {
            Phase::Opening => 0u8,
            Phase::Middle => 1u8,
            Phase::Endgame => 2u8,
        },
        // unpack 失敗は破損レコードとして drop
        Err(_) => return DROP,
    };

    // eval帯 重点: |score| ∈ [lo,hi] かつ中盤(phase==1) を oversample 対象にする。
    let in_band = eval_band_hi > 0
        && phase == 1
        && score.abs() >= eval_band_lo
        && score.abs() <= eval_band_hi;

    let (h1, h2) = hash_pair_bytes(sfen);
    Decision {
        survives: true,
        igyo_loss,
        in_band,
        phase,
        h1,
        h2,
    }
}

#[derive(Default, Clone, Copy)]
struct Stats {
    seen: u64,
    mate_or_igyo_dropped: u64,
    dedup_dropped: u64,
    phase_survivor: [u64; 3],
    phase_written: [u64; 3],
    /// 居玉敗北で残した survivor 数（複製前）と、書き込んだ延べ数（複製後）
    igyo_loss_survivor: u64,
    igyo_loss_written: u64,
    /// eval帯 重点で oversample した中盤局面（複製前 survivor / 複製後 written）
    band_survivor: u64,
    band_written: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(cli.threads).build_global().ok();
    }

    if !cli.measure_only && cli.output.is_none() {
        anyhow::bail!("--output が必要です（--measure-only でない場合）");
    }

    let params = BloomFilter::estimate(cli.bloom_elems, cli.bloom_fpr);
    eprintln!(
        "bloom: {:.2} GiB, k={}, elems={}, fpr={}",
        params.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        params.num_hashes,
        cli.bloom_elems,
        cli.bloom_fpr
    );
    let mut bloom = BloomFilter::allocate(&params);

    let keep_prob = [
        cli.keep_opening * cli.global_scale,
        cli.keep_middle * cli.global_scale,
        cli.keep_endgame * cli.global_scale,
    ];

    let mut writer = match &cli.output {
        Some(p) => {
            let f = File::create(p).with_context(|| format!("出力作成失敗: {}", p.display()))?;
            Some(BufWriter::with_capacity(1 << 24, f))
        }
        None => None,
    };

    let mut stats = Stats::default();
    let chunk_bytes = cli.chunk * REC;
    let mut buf = vec![0u8; chunk_bytes];

    for path in &cli.inputs {
        let mut f =
            File::open(path).with_context(|| format!("入力オープン失敗: {}", path.display()))?;
        eprintln!("[read] {}", path.display());
        let mut file_seen: u64 = 0;
        let sample_limit = cli.sample;

        'file: loop {
            // チャンク読み込み（レコード境界に揃える）
            let mut filled = 0usize;
            while filled < chunk_bytes {
                let n = f.read(&mut buf[filled..])?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            let n_rec = filled / REC;
            if n_rec == 0 {
                break;
            }

            // 並列判定
            let decisions: Vec<Decision> = buf[..n_rec * REC]
                .par_chunks_exact(REC)
                .map(|rec| {
                    decide(
                        rec,
                        cli.mate_threshold,
                        cli.igyo_min_ply,
                        cli.keep_igyo_loss,
                        cli.eval_band_lo,
                        cli.eval_band_hi,
                        cli.cp_lo,
                        cli.cp_hi,
                        cli.central_king_drop,
                        cli.central_king_keep_score,
                    )
                })
                .collect();

            // 逐次: dedup + keep 判定 + 書き込み（決定的）
            for (i, d) in decisions.iter().enumerate() {
                stats.seen += 1;
                file_seen += 1;
                if !d.survives {
                    stats.mate_or_igyo_dropped += 1;
                } else if bloom.insert_or_check(d.h1, d.h2) {
                    // true = 既出（重複）
                    stats.dedup_dropped += 1;
                } else {
                    let p = d.phase as usize;
                    stats.phase_survivor[p] += 1;
                    if d.igyo_loss {
                        // 居玉敗北: phase-keep をバイパスし oversample 回だけ必ず書く
                        stats.igyo_loss_survivor += 1;
                        let n = cli.igyo_loss_oversample.max(1);
                        for _ in 0..n {
                            stats.phase_written[p] += 1;
                            stats.igyo_loss_written += 1;
                            if let Some(w) = writer.as_mut() {
                                w.write_all(&buf[i * REC..i * REC + REC])?;
                            }
                        }
                    } else if d.in_band && cli.eval_band_oversample > 1 {
                        // eval帯 重点: 中盤の帯内局面は phase-keep をバイパスし oversample 回書く
                        stats.band_survivor += 1;
                        let n = cli.eval_band_oversample;
                        for _ in 0..n {
                            stats.phase_written[p] += 1;
                            stats.band_written += 1;
                            if let Some(w) = writer.as_mut() {
                                w.write_all(&buf[i * REC..i * REC + REC])?;
                            }
                        }
                    } else {
                        let prob = keep_prob[p].clamp(0.0, 1.0);
                        if keep_fraction(d.h1, cli.seed) < prob {
                            stats.phase_written[p] += 1;
                            if let Some(w) = writer.as_mut() {
                                w.write_all(&buf[i * REC..i * REC + REC])?;
                            }
                        }
                    }
                }

                if sample_limit > 0 && file_seen >= sample_limit {
                    break 'file;
                }
            }

            if stats.seen % 40_000_000 < cli.chunk as u64 {
                eprint!(
                    "\r  seen={:.0}M written={:.0}M   ",
                    stats.seen as f64 / 1e6,
                    total_written(&stats) as f64 / 1e6
                );
                std::io::stderr().flush().ok();
            }

            if filled < chunk_bytes {
                break; // EOF
            }
        }
        eprintln!();
        drop(f);
        if cli.delete_after_read && !cli.measure_only {
            std::fs::remove_file(path).with_context(|| format!("削除失敗: {}", path.display()))?;
            eprintln!("[deleted] {}", path.display());
        }
    }

    if let Some(w) = writer.as_mut() {
        w.flush()?;
    }

    report(&stats);
    Ok(())
}

fn total_written(s: &Stats) -> u64 {
    s.phase_written.iter().sum()
}

fn report(s: &Stats) {
    let surv: u64 = s.phase_survivor.iter().sum();
    let writ: u64 = s.phase_written.iter().sum();
    eprintln!("\n===== build_washiun_train report =====");
    eprintln!("seen                 = {}", s.seen);
    eprintln!(
        "mate/igyo dropped    = {} ({:.2}%)",
        s.mate_or_igyo_dropped,
        pct(s.mate_or_igyo_dropped, s.seen)
    );
    eprintln!(
        "dedup dropped        = {} ({:.2}%)",
        s.dedup_dropped,
        pct(s.dedup_dropped, s.seen)
    );
    eprintln!("survivors (post-dedup) = {}", surv);
    let names = ["opening", "middle", "endgame"];
    eprintln!("-- survivor phase shares --");
    for (name, &cnt) in names.iter().zip(s.phase_survivor.iter()) {
        eprintln!("  {:7} = {:>14} ({:.2}%)", name, cnt, pct(cnt, surv));
    }
    eprintln!("-- written (output) --");
    for (name, &cnt) in names.iter().zip(s.phase_written.iter()) {
        eprintln!("  {:7} = {:>14} ({:.2}%)", name, cnt, pct(cnt, writ));
    }
    eprintln!(
        "igyo-loss kept       = {} survivors -> {} written (oversample)",
        s.igyo_loss_survivor, s.igyo_loss_written
    );
    eprintln!(
        "eval-band oversample = {} survivors -> {} written (重点)",
        s.band_survivor, s.band_written
    );
    eprintln!("written total        = {} ({:.3}B)", writ, writ as f64 / 1e9);
}

fn pct(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}
