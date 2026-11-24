//! eval_bias_miner — 学習なしの「評価補正候補」探索ツール
//!
//! NNUE の静的評価 `static_eval` と、同一局面を固定ノード（または固定深さ）で
//! 探索した結果 `search_score` の系統的なズレ
//! `bias = search_score - static_eval` を、局面クラス別に集計する。
//!
//! ★ 本ツールは**測定専用**であり、評価補正ロジックは一切含まない。
//!   「補正する価値がある bias が存在するか」を可視化するだけで、
//!   実際の補正（correction）は行わない。最終目的は C++ v940 floodgate
//!   エンジン改善の仮説作りである（Rust 本番化目的ではない）。
//!
//! # 使う既存 API
//! - SFEN → Position: `Position::new()` + `Position::set_sfen(&sfen)`
//!   （USI `position sfen ...` と同じ経路）
//! - 静的評価: `evaluate_dispatch(&mut pos, &mut AccumulatorStackVariant, &mut None)`
//!   （USI `eval` コマンドと同じ経路。NNUE 生 eval）
//! - 探索: `Search::new_with_eval_hash(...)` +
//!   `Search::go(&mut pos, LimitsType{ nodes / depth }, on_info)`
//!   （USI `go nodes N` / `go depth D` と同じ経路。単一 worker = Threads 1）
//! - 特徴量: `Position` / `movegen::generate_legal_all` / bitboard API
//!
//! # 詰み (mate) の扱い
//! 探索結果 `SearchResult.score` が詰みスコア（`Value::is_mate_score()`）の場合は
//! `is_mate=true` としてフラグ列に記録する。詰みは cp と桁が違うため
//! aggregate からはデフォルト除外し、`--include-mate` 指定時のみ含める。
//! 静的評価が詰みスコアになることはない。
//!
//! # 出力
//! - `positions.csv` : 1 局面 1 行（raw 値 + 特徴量）。ストリーミング書き込み。
//! - `aggregate.csv` : 局面クラス別の bias 統計。
//! - `summary.md`    : 補正候補ランキングと分類。
//! - `errors.csv`    : SFEN パース等の失敗（処理は止めない）。
//!
//! # 使用例
//! ```bash
//! cargo run --release -p tools --bin eval_bias_miner -- \
//!   --input positions.sfen \
//!   --eval-file /path/to/nn.bin \
//!   --nodes 100000 \
//!   --out-dir out/bias
//! ```

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use rshogi_core::bitboard::{Bitboard, init_bitboard_tables, king_effect};
use rshogi_core::movegen::{MoveList, generate_legal_all};
use rshogi_core::nnue::{
    AccumulatorStackVariant, LayerStackBucketMode, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
    evaluate_dispatch, get_network, init_nnue, set_layer_stack_bucket_mode,
    set_layer_stack_progress_kpabs_weights,
};
use rshogi_core::position::Position;
use rshogi_core::search::{LimitsType, Search, SearchInfo};
use rshogi_core::types::{Color, PieceType, Square, Value};

// ============================================================================
// bucket 境界定義（すべて定数。cp は centipawn、ply は手数）
// ============================================================================

/// game_ply（手数）バケット境界。`[0,20] [21,40] [41,60] [61,80] [81,120] [121,∞)`
const PLY_BUCKETS: &[i32] = &[20, 40, 60, 80, 120];
/// eval_abs（|static_eval| cp）バケット境界。`[0,50] [51,150] [151,300] [301,600] [601,1200] [1201,∞)`
const EVAL_ABS_BUCKETS: &[i32] = &[50, 150, 300, 600, 1200];
/// legal_moves_count バケット境界。`[0,10] [11,25] [26,40] [41,60] [61,∞)`
const LEGAL_MOVES_BUCKETS: &[i32] = &[10, 25, 40, 60];
/// hand_piece_count バケット境界。`[0,0] [1,2] [3,5] [6,∞)`
const HAND_PIECE_BUCKETS: &[i32] = &[0, 2, 5];
/// capture_available_count バケット境界。`[0,0] [1,2] [3,5] [6,∞)`
const CAPTURE_BUCKETS: &[i32] = &[0, 2, 5];
/// check_available_count バケット境界。`[0,0] [1,1] [2,∞)`
const CHECK_BUCKETS: &[i32] = &[0, 1];
/// board material（盤上駒の材料合計、玉除く、両者。phase 近似）バケット境界（cp 相当・NNUE内部scale）
/// 序盤 ≈ 全駒、終盤 ≈ 少。`[0,3000] [3001,6000] [6001,9000] [9001,∞)`
const MATERIAL_PHASE_BUCKETS: &[i32] = &[3000, 6000, 9000];

/// nyugyoku_like 判定に使う相対段しきい値。
///
/// `Rank::relative(color)` を +1 した own_king_rank は、両手番とも
/// **9 = 自陣最奥（home）/ 1 = 敵陣最奥（最深侵入）** となる正規化段である。
/// own_king_rank <= この値なら「玉が敵陣 3 段に侵入」とみなす（入玉の簡易近似）。
const NYUGYOKU_KING_REL_RANK_MAX: usize = 3;

/// 材料評価テーブル（YaneuraOu 準拠 base_piece_value と同値。core 側は pub(crate) のため複製）
fn base_piece_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => 90,
        PieceType::Lance => 315,
        PieceType::Knight => 405,
        PieceType::Silver => 495,
        PieceType::Bishop => 855,
        PieceType::Rook => 990,
        PieceType::Gold => 540,
        PieceType::King => 15000,
        PieceType::ProPawn => 540,
        PieceType::ProLance => 540,
        PieceType::ProKnight => 540,
        PieceType::ProSilver => 540,
        PieceType::Horse => 945,
        PieceType::Dragon => 1395,
    }
}

/// 手駒になる駒種
const HAND_PIECES: [PieceType; 7] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Gold,
    PieceType::Bishop,
    PieceType::Rook,
];

/// 境界配列 `bounds` に対し、value がどの upper-bound バケットに入るかのラベルを返す。
/// 例: bounds=[20,40] → `<=20` / `21-40` / `41+`
fn bucket_label(value: i32, bounds: &[i32]) -> String {
    let mut lo = i32::MIN;
    for (i, &hi) in bounds.iter().enumerate() {
        if value <= hi {
            if i == 0 {
                return format!("<={hi}");
            }
            return format!("{}-{hi}", lo + 1);
        }
        lo = hi;
    }
    format!("{}+", lo + 1)
}

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "eval_bias_miner",
    about = "NNUE static_eval と探索 search_score の系統的ズレを局面クラス別に集計する（測定専用、補正ロジックなし）"
)]
struct Cli {
    /// 入力 SFEN ファイル（1 行 1 局面。空行 / `#` コメントはスキップ）
    #[arg(long)]
    input: PathBuf,

    /// 出力ディレクトリ
    #[arg(long)]
    out_dir: PathBuf,

    /// NNUE 評価ファイル（EvalFile）
    #[arg(long)]
    eval_file: PathBuf,

    /// progress8kpabs bucket 係数ファイル（production の LS_PROGRESS_COEFF 相当）。
    /// 指定時は bucket mode を Progress8KPAbs にし、production と同じ bucket 選択で評価する。
    /// 未指定時は progress 重み全ゼロのまま（bucket 選択が degenerate になり production と乖離）。
    #[arg(long)]
    progress_file: Option<PathBuf>,

    /// 固定探索ノード数（nodes 優先。未指定かつ --depth 未指定なら 100000）
    #[arg(long)]
    nodes: Option<u64>,

    /// 固定探索深さ（--nodes が指定された場合は無視）
    #[arg(long)]
    depth: Option<i32>,

    /// TT サイズ（MB）
    #[arg(long, default_value_t = 16)]
    hash_mb: usize,

    /// aggregate / summary に詰み局面を含める
    #[arg(long, default_value_t = false)]
    include_mate: bool,

    /// summary で「count 十分」とみなす最小サンプル数
    #[arg(long, default_value_t = 30)]
    min_group_count: usize,

    /// 処理する局面数の上限（デバッグ / スモークテスト用）
    #[arg(long)]
    limit: Option<usize>,

    /// 局面ごとの TT / history クリアを行わない（既定はクリアして局面独立・再現性確保）
    #[arg(long, default_value_t = false)]
    no_clear: bool,
}

/// progress8kpabs 係数ファイル（f32 or f64 LE）を読み、progress 重みとして設定する。
///
/// production の `LS_PROGRESS_COEFF` と同じ役割。bucket 選択（8 バケットのどの layerstack を
/// 使うか）を production と一致させるために必要。未設定だと重み全ゼロで bucket が degenerate になる。
fn load_progress_coeff_weights(coeff_path: &Path) -> Result<Box<[f32]>> {
    let data = std::fs::read(coeff_path)
        .with_context(|| format!("failed to read progress coeff: {}", coeff_path.display()))?;

    let expected_f32_bytes = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f32>();
    let expected_f64_bytes = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f64>();

    if data.len() == expected_f32_bytes {
        let weights: Vec<f32> = data
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk size checked")))
            .collect();
        return Ok(weights.into_boxed_slice());
    }
    if data.len() == expected_f64_bytes {
        let weights: Vec<f32> = data
            .chunks_exact(std::mem::size_of::<f64>())
            .map(|c| f64::from_le_bytes(c.try_into().expect("chunk size checked")) as f32)
            .collect();
        return Ok(weights.into_boxed_slice());
    }
    anyhow::bail!(
        "invalid progress coeff size: {} bytes (expected {} for f32 or {} for f64 weights): {}",
        data.len(),
        expected_f32_bytes,
        expected_f64_bytes,
        coeff_path.display()
    );
}

/// 探索制限モード
#[derive(Debug, Clone, Copy)]
enum SearchMode {
    Nodes(u64),
    Depth(i32),
}

impl SearchMode {
    fn resolve(nodes: Option<u64>, depth: Option<i32>) -> Self {
        match (nodes, depth) {
            (Some(n), Some(_)) => {
                eprintln!("info: --nodes と --depth 両方指定。nodes を優先します");
                SearchMode::Nodes(n)
            }
            (Some(n), None) => SearchMode::Nodes(n),
            (None, Some(d)) => SearchMode::Depth(d),
            (None, None) => SearchMode::Nodes(100_000),
        }
    }

    fn apply(&self, limits: &mut LimitsType) {
        match *self {
            SearchMode::Nodes(n) => limits.nodes = n,
            SearchMode::Depth(d) => limits.depth = d,
        }
    }

    fn describe(&self) -> String {
        match *self {
            SearchMode::Nodes(n) => format!("nodes={n}"),
            SearchMode::Depth(d) => format!("depth={d}"),
        }
    }
}

// ============================================================================
// per-position 計測結果
// ============================================================================

struct PositionRecord {
    sfen: String,
    index: usize,
    side_to_move: Color,
    game_ply: i32,
    static_eval_cp: i32,
    search_score_cp: i32,
    bestmove: String,
    nodes: u64,
    depth: i32,
    seldepth: i32,
    elapsed_ms: u64,
    in_check: bool,
    legal_moves_count: i32,
    material_phase: i32,
    hand_piece_count: i32,
    hand_material_count: i32,
    own_king_rank: usize,
    opponent_king_rank: usize,
    nyugyoku_like: bool,
    capture_available_count: i32,
    check_available_count: i32,
    promotion_available_count: i32,
    own_crisis_choice: CrisisChoice,
    opponent_crisis_choice: CrisisChoice,
    is_mate: bool,
}

impl PositionRecord {
    fn bias(&self) -> f64 {
        (self.search_score_cp - self.static_eval_cp) as f64
    }

    fn stm_str(&self) -> &'static str {
        match self.side_to_move {
            Color::Black => "b",
            Color::White => "w",
        }
    }

    fn static_eval_sign(&self) -> &'static str {
        match self.static_eval_cp.signum() {
            1 => "pos",
            -1 => "neg",
            _ => "zero",
        }
    }

    /// このレコードが属する (group_name, group_key) の一覧を返す。
    fn group_keys(&self) -> Vec<(&'static str, String)> {
        vec![
            ("side_to_move", self.stm_str().to_string()),
            ("game_ply", bucket_label(self.game_ply, PLY_BUCKETS)),
            ("eval_abs", bucket_label(self.static_eval_cp.abs(), EVAL_ABS_BUCKETS)),
            ("in_check", self.in_check.to_string()),
            ("legal_moves", bucket_label(self.legal_moves_count, LEGAL_MOVES_BUCKETS)),
            ("hand_pieces", bucket_label(self.hand_piece_count, HAND_PIECE_BUCKETS)),
            ("capture_avail", bucket_label(self.capture_available_count, CAPTURE_BUCKETS)),
            ("check_avail", bucket_label(self.check_available_count, CHECK_BUCKETS)),
            ("own_king_rank", self.own_king_rank.to_string()),
            ("opp_king_rank", self.opponent_king_rank.to_string()),
            ("material_phase", bucket_label(self.material_phase, MATERIAL_PHASE_BUCKETS)),
            ("nyugyoku_like", self.nyugyoku_like.to_string()),
            ("static_eval_sign", self.static_eval_sign().to_string()),
            ("own_king_urgency", self.own_crisis_choice.king_urgency.label().to_string()),
            ("opp_king_urgency", self.opponent_crisis_choice.king_urgency.label().to_string()),
            ("own_major_state", self.own_crisis_choice.major_state.label().to_string()),
            ("opp_major_state", self.opponent_crisis_choice.major_state.label().to_string()),
            ("own_drop_pressure", self.own_crisis_choice.drop_pressure.label().to_string()),
            (
                "opp_drop_pressure",
                self.opponent_crisis_choice.drop_pressure.label().to_string(),
            ),
            ("own_crisis_choice", self.own_crisis_choice.label()),
            ("opp_crisis_choice", self.opponent_crisis_choice.label()),
        ]
    }
}

/// positions.csv ヘッダ
const CSV_HEADER: &str = "index,sfen,side_to_move,game_ply,static_eval,corrected_static_eval,\
search_score,bestmove,nodes,depth,seldepth,elapsed_ms,in_check,legal_moves_count,eval_abs_bucket,\
material_phase,hand_piece_count,hand_material_count,own_king_rank,opponent_king_rank,\
nyugyoku_like,capture_available_count,check_available_count,promotion_available_count,\
own_king_pressure,own_safe_escapes,own_drop_pressure,own_major_state,own_major_bearing,own_crisis_choice,\
opponent_king_pressure,opponent_safe_escapes,opponent_drop_pressure,opponent_major_state,opponent_major_bearing,opponent_crisis_choice,\
bias,abs_bias,is_mate";

fn write_csv_row(w: &mut impl Write, r: &PositionRecord) -> Result<()> {
    let bias = r.bias();
    // corrected_static_eval は standalone では公開 API がないため空欄。
    writeln!(
        w,
        "{index},{sfen},{stm},{ply},{seval},,{sscore},{best},{nodes},{depth},{seldepth},\
{elapsed},{in_check},{legal},{eval_abs},{mphase},{hp},{hm},{okr},{opkr},{nyu},{cap},{chk},{promo},\
{own_pressure},{own_escapes},{own_drop},{own_major},{own_bearing},{own_crisis},\
{opp_pressure},{opp_escapes},{opp_drop},{opp_major},{opp_bearing},{opp_crisis},\
{bias:.0},{abs_bias:.0},{is_mate}",
        index = r.index,
        sfen = r.sfen,
        stm = r.stm_str(),
        ply = r.game_ply,
        seval = r.static_eval_cp,
        sscore = r.search_score_cp,
        best = r.bestmove,
        nodes = r.nodes,
        depth = r.depth,
        seldepth = r.seldepth,
        elapsed = r.elapsed_ms,
        in_check = r.in_check,
        legal = r.legal_moves_count,
        eval_abs = bucket_label(r.static_eval_cp.abs(), EVAL_ABS_BUCKETS),
        mphase = bucket_label(r.material_phase, MATERIAL_PHASE_BUCKETS),
        hp = r.hand_piece_count,
        hm = r.hand_material_count,
        okr = r.own_king_rank,
        opkr = r.opponent_king_rank,
        nyu = r.nyugyoku_like,
        cap = r.capture_available_count,
        chk = r.check_available_count,
        promo = r.promotion_available_count,
        own_pressure = r.own_crisis_choice.king_pressure,
        own_escapes = r.own_crisis_choice.safe_escapes,
        own_drop = r.own_crisis_choice.drop_pressure.label(),
        own_major = r.own_crisis_choice.major_state.label(),
        own_bearing = r.own_crisis_choice.major_bearing,
        own_crisis = r.own_crisis_choice.index(),
        opp_pressure = r.opponent_crisis_choice.king_pressure,
        opp_escapes = r.opponent_crisis_choice.safe_escapes,
        opp_drop = r.opponent_crisis_choice.drop_pressure.label(),
        opp_major = r.opponent_crisis_choice.major_state.label(),
        opp_bearing = r.opponent_crisis_choice.major_bearing,
        opp_crisis = r.opponent_crisis_choice.index(),
        bias = bias,
        abs_bias = bias.abs(),
        is_mate = r.is_mate,
    )?;
    Ok(())
}

// ============================================================================
// aggregate
// ============================================================================

#[derive(Default)]
struct GroupAcc {
    biases: Vec<f64>,
    sum_static: f64,
    sum_search: f64,
}

impl GroupAcc {
    fn push(&mut self, r: &PositionRecord) {
        self.biases.push(r.bias());
        self.sum_static += r.static_eval_cp as f64;
        self.sum_search += r.search_score_cp as f64;
    }
}

struct GroupStats {
    group_name: String,
    group_key: String,
    count: usize,
    mean_static_eval: f64,
    mean_search_score: f64,
    mean_bias: f64,
    median_bias: f64,
    mean_abs_bias: f64,
    p75_abs_bias: f64,
    p90_abs_bias: f64,
    stddev_bias: f64,
    positive_bias_rate: f64,
    negative_bias_rate: f64,
}

/// ソート済み配列から線形補間なしの percentile（0.0..=1.0）を取る簡易実装。
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn compute_stats(name: &str, key: &str, acc: &GroupAcc) -> GroupStats {
    let count = acc.biases.len();
    let n = count as f64;
    let mean_bias = acc.biases.iter().sum::<f64>() / n;
    let variance = acc.biases.iter().map(|b| (b - mean_bias).powi(2)).sum::<f64>() / n;
    let stddev_bias = variance.sqrt();

    let mut sorted = acc.biases.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("bias is finite"));
    let median_bias = percentile(&sorted, 0.5);

    let mut abs_sorted: Vec<f64> = acc.biases.iter().map(|b| b.abs()).collect();
    abs_sorted.sort_by(|a, b| a.partial_cmp(b).expect("abs bias is finite"));
    let mean_abs_bias = abs_sorted.iter().sum::<f64>() / n;
    let p75_abs_bias = percentile(&abs_sorted, 0.75);
    let p90_abs_bias = percentile(&abs_sorted, 0.90);

    let positive = acc.biases.iter().filter(|b| **b > 0.0).count() as f64;
    let negative = acc.biases.iter().filter(|b| **b < 0.0).count() as f64;

    GroupStats {
        group_name: name.to_string(),
        group_key: key.to_string(),
        count,
        mean_static_eval: acc.sum_static / n,
        mean_search_score: acc.sum_search / n,
        mean_bias,
        median_bias,
        mean_abs_bias,
        p75_abs_bias,
        p90_abs_bias,
        stddev_bias,
        positive_bias_rate: positive / n,
        negative_bias_rate: negative / n,
    }
}

fn write_aggregate_csv(path: &Path, stats: &[GroupStats]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(
        w,
        "group_name,group_key,count,mean_static_eval,mean_search_score,mean_bias,median_bias,\
mean_abs_bias,p75_abs_bias,p90_abs_bias,stddev_bias,positive_bias_rate,negative_bias_rate"
    )?;
    for s in stats {
        writeln!(
            w,
            "{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.3},{:.3}",
            s.group_name,
            s.group_key,
            s.count,
            s.mean_static_eval,
            s.mean_search_score,
            s.mean_bias,
            s.median_bias,
            s.mean_abs_bias,
            s.p75_abs_bias,
            s.p90_abs_bias,
            s.stddev_bias,
            s.positive_bias_rate,
            s.negative_bias_rate,
        )?;
    }
    w.flush()?;
    Ok(())
}

// ---- summary 分類（閾値は定数化）----

/// 補正候補とみなす |mean_bias| の下限（cp）
const CORR_MEAN_BIAS_CP: f64 = 60.0;
/// 符号が安定とみなす positive/negative_bias_rate の下限
const STABLE_SIGN_RATE: f64 = 0.65;
/// 「危険そう」とみなす mean_abs_bias の下限（cp）
const RISKY_MEAN_ABS_CP: f64 = 200.0;
/// 符号が不安定とみなす rate 上限（両符号とも これ未満なら分散が大きい）
const UNSTABLE_SIGN_RATE: f64 = 0.60;

#[derive(PartialEq)]
enum Classification {
    CorrectionCandidate,
    Risky,
    NeedsMoreSamples,
    Neutral,
}

impl Classification {
    fn label(&self) -> &'static str {
        match self {
            Classification::CorrectionCandidate => "補正候補",
            Classification::Risky => "危険そう",
            Classification::NeedsMoreSamples => "追加サンプル必要",
            Classification::Neutral => "中立",
        }
    }
}

fn classify(s: &GroupStats, min_count: usize) -> Classification {
    let stable_sign =
        s.positive_bias_rate >= STABLE_SIGN_RATE || s.negative_bias_rate >= STABLE_SIGN_RATE;
    let unstable_sign =
        s.positive_bias_rate < UNSTABLE_SIGN_RATE && s.negative_bias_rate < UNSTABLE_SIGN_RATE;

    if s.count < min_count {
        return Classification::NeedsMoreSamples;
    }
    if s.mean_bias.abs() >= CORR_MEAN_BIAS_CP && stable_sign {
        return Classification::CorrectionCandidate;
    }
    if s.mean_abs_bias >= RISKY_MEAN_ABS_CP && unstable_sign {
        return Classification::Risky;
    }
    Classification::Neutral
}

fn write_summary_md(
    path: &Path,
    stats: &[GroupStats],
    min_count: usize,
    total_positions: usize,
    mate_count: usize,
    mode: &SearchMode,
    include_mate: bool,
) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);

    writeln!(w, "# eval_bias_miner summary\n")?;
    writeln!(w, "- 探索条件: {}", mode.describe())?;
    writeln!(w, "- 総局面数: {total_positions}（うち詰み {mate_count}）")?;
    writeln!(
        w,
        "- aggregate 対象: {}（詰みは {}）",
        if include_mate {
            "詰み含む"
        } else {
            "詰み除外"
        },
        if include_mate { "含む" } else { "除外" }
    )?;
    writeln!(w, "- count 十分の閾値 (min_group_count): {min_count}\n")?;

    writeln!(w, "## 分類基準（閾値）\n")?;
    writeln!(
        w,
        "- **補正候補**: count>={min_count} かつ |mean_bias|>={CORR_MEAN_BIAS_CP:.0}cp \
かつ 符号安定 (positive or negative_bias_rate>={STABLE_SIGN_RATE:.2})。\
static_eval を一定方向へずらせば search と系統的に近づく = C++ 側で補正を試す価値がある候補。"
    )?;
    writeln!(
        w,
        "- **危険そう**: count>={min_count} かつ mean_abs_bias>={RISKY_MEAN_ABS_CP:.0}cp \
かつ 符号不安定 (positive/negative_bias_rate 双方 < {UNSTABLE_SIGN_RATE:.2})。\
ズレは大きいが方向が定まらず、単純補正はかえって悪化させうる。"
    )?;
    writeln!(w, "- **追加サンプル必要**: count<{min_count}。統計的に判断不能。\n")?;

    // count 十分な group のみを対象にランキング
    let eligible: Vec<&GroupStats> = stats.iter().filter(|s| s.count >= min_count).collect();

    // ① mean_bias 絶対値大きい順
    writeln!(w, "## ① mean_bias 絶対値が大きい group（符号付きの系統ズレ）\n")?;
    write_ranking_table(&mut w, &eligible, min_count, |s| s.mean_bias.abs())?;

    // ② mean_abs_bias 大きい順
    writeln!(w, "\n## ② mean_abs_bias が大きい group（ばらつき込みのズレ量）\n")?;
    write_ranking_table(&mut w, &eligible, min_count, |s| s.mean_abs_bias)?;

    // ③ 符号が安定して偏る group
    writeln!(
        w,
        "\n## ③ bias 符号が安定して偏る group（positive/negative_bias_rate の偏り順）\n"
    )?;
    write_ranking_table(&mut w, &eligible, min_count, |s| {
        s.positive_bias_rate.max(s.negative_bias_rate)
    })?;

    w.flush()?;
    Ok(())
}

fn write_ranking_table(
    w: &mut impl Write,
    eligible: &[&GroupStats],
    min_count: usize,
    key_fn: impl Fn(&GroupStats) -> f64,
) -> Result<()> {
    writeln!(
        w,
        "| group | key | count | mean_bias | mean_abs_bias | pos_rate | neg_rate | 分類 |"
    )?;
    writeln!(w, "|---|---|---:|---:|---:|---:|---:|---|")?;

    let mut ranked: Vec<&&GroupStats> = eligible.iter().collect();
    ranked.sort_by(|a, b| key_fn(b).partial_cmp(&key_fn(a)).unwrap_or(std::cmp::Ordering::Equal));

    for s in ranked.iter().take(20) {
        writeln!(
            w,
            "| {} | {} | {} | {:.0} | {:.0} | {:.2} | {:.2} | {} |",
            s.group_name,
            s.group_key,
            s.count,
            s.mean_bias,
            s.mean_abs_bias,
            s.positive_bias_rate,
            s.negative_bias_rate,
            classify(s, min_count).label(),
        )?;
    }
    Ok(())
}

// ============================================================================
// 局面計測本体
// ============================================================================

/// SFEN を Position 化し、静的評価・探索・特徴量を計測して PositionRecord を返す。
fn measure_position(
    sfen: &str,
    index: usize,
    search: &mut Search,
    mode: &SearchMode,
    clear_each: bool,
) -> Result<PositionRecord> {
    let mut pos = Position::new();
    pos.set_sfen(sfen).map_err(|e| anyhow::anyhow!("SFEN parse error: {e:?}"))?;

    // ── 静的評価（NNUE 生 eval）──
    let network = get_network().context("NNUE network not loaded")?;
    let mut stack = AccumulatorStackVariant::from_network(&network);
    let static_val: Value = evaluate_dispatch(&mut pos, &mut stack, &mut None);
    let static_eval_cp = static_val.to_cp();

    // ── 特徴量（探索前の局面から算出）──
    let features = extract_features(&pos);

    // ── 探索（固定 nodes / depth、single worker）──
    if clear_each {
        search.clear_tt();
        search.clear_histories();
    }
    search.reset_flags();

    let mut limits = LimitsType::new();
    mode.apply(&mut limits);
    limits.set_start_time();

    let mut seldepth = 0i32;
    let start = Instant::now();
    let mut search_pos = pos.clone();
    let result = search.go(
        &mut search_pos,
        limits,
        Some(|info: &SearchInfo| {
            seldepth = info.sel_depth;
        }),
    );
    let elapsed_ms = start.elapsed().as_millis() as u64;

    let is_mate = result.score.is_mate_score();
    // 詰みスコアは cp に変換しても桁が異なる（Value::to_cp はそのまま返す）ため、
    // is_mate フラグで判別できるようにしておく。
    let search_score_cp = result.score.to_cp();
    let bestmove = if result.best_move == rshogi_core::types::Move::NONE {
        "none".to_string()
    } else {
        result.best_move.to_usi()
    };

    Ok(PositionRecord {
        sfen: sfen.to_string(),
        index,
        side_to_move: pos.side_to_move(),
        game_ply: pos.game_ply(),
        static_eval_cp,
        search_score_cp,
        bestmove,
        nodes: result.nodes,
        depth: result.depth,
        seldepth,
        elapsed_ms,
        in_check: features.in_check,
        legal_moves_count: features.legal_moves_count,
        material_phase: features.material_phase,
        hand_piece_count: features.hand_piece_count,
        hand_material_count: features.hand_material_count,
        own_king_rank: features.own_king_rank,
        opponent_king_rank: features.opponent_king_rank,
        nyugyoku_like: features.nyugyoku_like,
        capture_available_count: features.capture_available_count,
        check_available_count: features.check_available_count,
        promotion_available_count: features.promotion_available_count,
        own_crisis_choice: features.own_crisis_choice,
        opponent_crisis_choice: features.opponent_crisis_choice,
        is_mate,
    })
}

struct Features {
    in_check: bool,
    legal_moves_count: i32,
    material_phase: i32,
    hand_piece_count: i32,
    hand_material_count: i32,
    own_king_rank: usize,
    opponent_king_rank: usize,
    nyugyoku_like: bool,
    capture_available_count: i32,
    check_available_count: i32,
    promotion_available_count: i32,
    own_crisis_choice: CrisisChoice,
    opponent_crisis_choice: CrisisChoice,
}

/// 安全な退路の量と「まだ低圧のうちに逃げる窓」を分離した 4 class。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KingUrgency {
    Trapped,
    ExitNarrow,
    EarlyExitWindow,
    PressuredOpen,
}

impl KingUrgency {
    fn from_counts(pressure: usize, safe_escapes: u32) -> Self {
        if safe_escapes == 0 {
            Self::Trapped
        } else if safe_escapes <= 2 {
            Self::ExitNarrow
        } else if pressure < 2 {
            Self::EarlyExitWindow
        } else {
            Self::PressuredOpen
        }
    }

    const fn index(self) -> usize {
        self as usize
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Trapped => "trapped",
            Self::ExitNarrow => "exit_narrow",
            Self::EarlyExitWindow => "early_exit_window",
            Self::PressuredOpen => "pressured_open",
        }
    }
}

/// 自軍の飛角馬竜で最も危険なものの状態。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MajorState {
    None,
    Retained,
    Contested,
    Hanging,
}

impl MajorState {
    const fn index(self) -> usize {
        self as usize
    }

    const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Retained => "retained",
            Self::Contested => "contested",
            Self::Hanging => "hanging",
        }
    }
}

/// 既存 DropPressure 8 class を、pilot のサンプル密度を保つため 4 class に圧縮。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DropPressure {
    None,
    Low,
    Medium,
    High,
}

impl DropPressure {
    fn from_raw_class(class: usize) -> Self {
        match class {
            0 => Self::None,
            1 | 2 => Self::Low,
            3 | 4 => Self::Medium,
            _ => Self::High,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }

    const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// CrisisChoice-FT の事前検定用 class。
///
/// class = king urgency 4 × drop pressure 4 × major state 4 = 64。
/// 将来 FT 入力にする場合も、各 perspective で active は常に 1 個となる。
#[derive(Debug, Clone, Copy)]
struct CrisisChoice {
    king_pressure: usize,
    safe_escapes: u32,
    king_urgency: KingUrgency,
    drop_pressure: DropPressure,
    major_state: MajorState,
    major_bearing: bool,
}

impl CrisisChoice {
    const NUM_CLASSES: usize = 4 * 4 * 4;

    const fn index(self) -> usize {
        (self.king_urgency.index() * 4 + self.drop_pressure.index()) * 4 + self.major_state.index()
    }

    fn label(self) -> String {
        format!(
            "{}__drop_{}__{}",
            self.king_urgency.label(),
            self.drop_pressure.label(),
            self.major_state.label()
        )
    }
}

fn is_major(pt: PieceType) -> bool {
    matches!(pt, PieceType::Bishop | PieceType::Rook | PieceType::Horse | PieceType::Dragon)
}

/// `side` の大駒について、最も危険な状態と敵玉 3x3 への照準有無を返す。
///
/// defended は pin を除外しない pseudo attack でよい。ここでは特徴量候補の
/// 残差信号を見ることが目的で、合法応手の完全判定は行わない。
fn major_context(pos: &Position, side: Color) -> (MajorState, bool) {
    let enemy = side.opponent();
    let enemy_king = pos.king_square(enemy);
    let enemy_zone = rshogi_core::bitboard::king_effect(enemy_king)
        | rshogi_core::bitboard::Bitboard::from_square(enemy_king);

    let mut state = MajorState::None;
    let mut bearing = false;
    for sq in pos.pieces_c(side).iter() {
        let pt = pos.piece_on(sq).piece_type();
        if !is_major(pt) {
            continue;
        }

        state = state.max(MajorState::Retained);
        let attackers = pos.attackers_to(sq);
        let attacked = (attackers & pos.pieces_c(enemy)).is_not_empty();
        if attacked {
            let defended = (attackers & pos.pieces_c(side)).is_not_empty();
            state = state.max(if defended {
                MajorState::Contested
            } else {
                MajorState::Hanging
            });
        }

        if enemy_zone.iter().any(|target| pos.attackers_to_c(target, side).contains(sq)) {
            bearing = true;
        }
    }
    (state, bearing)
}

fn crisis_choice(pos: &Position, side: Color) -> CrisisChoice {
    let king_sq = pos.king_square(side);
    let enemy = side.opponent();
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);
    let enemy_pieces = pos.pieces_c(enemy);
    let zone = king_effect(king_sq) | Bitboard::from_square(king_sq);
    let king_pressure = zone
        .iter()
        .map(|sq| (pos.attackers_to_occ(sq, occupied) & enemy_pieces).count() as usize)
        .sum();

    let escape = king_effect(king_sq) & !pos.pieces_c(side);
    let safe_escapes =
        escape.iter().filter(|&sq| pos.attackers_to_c(sq, enemy).is_empty()).count() as u32;
    let enemy_hand = pos.hand(side.opponent());
    let hand_threat = 2
        * (enemy_hand.count(PieceType::Rook) as usize
            + enemy_hand.count(PieceType::Bishop) as usize
            + enemy_hand.count(PieceType::Gold) as usize
            + enemy_hand.count(PieceType::Silver) as usize)
        + enemy_hand.count(PieceType::Lance) as usize
        + enemy_hand.count(PieceType::Knight) as usize
        + enemy_hand.count(PieceType::Pawn) as usize;
    let drop_pressure = DropPressure::from_raw_class(hand_threat);
    let (major_state, major_bearing) = major_context(pos, side);
    let king_urgency = KingUrgency::from_counts(king_pressure, safe_escapes);
    let result = CrisisChoice {
        king_pressure,
        safe_escapes,
        king_urgency,
        drop_pressure,
        major_state,
        major_bearing,
    };
    debug_assert!(result.index() < CrisisChoice::NUM_CLASSES);
    result
}

/// Position から特徴量を算出する。
///
/// - `hand_piece_count` / `hand_material_count`: **手番側 (side_to_move)** の持駒。
/// - `own_king_rank`: 手番側玉の正規化相対段（1..9。9=自陣最奥/home、1=敵陣最奥/最深侵入）。
/// - `opponent_king_rank`: 相手玉の正規化相対段（相手視点。9=相手home、1=相手にとっての敵陣最奥）。
/// - `nyugyoku_like`: 手番側玉が敵陣 3 段（own_king_rank <= NYUGYOKU_KING_REL_RANK_MAX）へ
///   侵入していれば true（入玉の簡易近似。玉段のみで判定し駒数は問わない）。
/// - `material_phase`: 盤上の全駒（玉除く、両者）材料合計。序盤ほど大、終盤ほど小の phase 近似。
/// - `capture/check/promotion_available_count`: 全合法手 (generate_legal_all) のうち
///   捕獲 / 王手 / 成りである手の数。
fn extract_features(pos: &Position) -> Features {
    let stm = pos.side_to_move();

    let mut list = MoveList::new();
    generate_legal_all(pos, &mut list);
    let legal_moves_count = list.len() as i32;

    let mut capture_available_count = 0i32;
    let mut check_available_count = 0i32;
    let mut promotion_available_count = 0i32;
    for mv in list.iter() {
        if pos.is_capture(*mv) {
            capture_available_count += 1;
        }
        if !mv.is_pass() && pos.gives_check(*mv) {
            check_available_count += 1;
        }
        if mv.is_promotion() {
            promotion_available_count += 1;
        }
    }

    // 手番側の持駒
    let hand = pos.hand(stm);
    let mut hand_piece_count = 0i32;
    let mut hand_material_count = 0i32;
    for pt in HAND_PIECES {
        let c = hand.count(pt) as i32;
        hand_piece_count += c;
        hand_material_count += c * base_piece_value(pt);
    }

    // 盤上材料（両者、玉除く）= phase 近似
    let mut material_phase = 0i32;
    for n in 0..Square::NUM as u8 {
        let sq = Square::from_u8(n).expect("0..81 is valid square");
        let pc = pos.piece_on(sq);
        if pc.is_some() {
            let pt = pc.piece_type();
            if pt != PieceType::King {
                material_phase += base_piece_value(pt);
            }
        }
    }

    // 玉段（相対段）
    let own_king_rel = pos.king_square(stm).rank().relative(stm).index() + 1;
    let opp = stm.opponent();
    let opp_king_rel = pos.king_square(opp).rank().relative(opp).index() + 1;
    let nyugyoku_like = own_king_rel <= NYUGYOKU_KING_REL_RANK_MAX;
    let own_crisis_choice = crisis_choice(pos, stm);
    let opponent_crisis_choice = crisis_choice(pos, opp);

    Features {
        in_check: pos.in_check(),
        legal_moves_count,
        material_phase,
        hand_piece_count,
        hand_material_count,
        own_king_rank: own_king_rel,
        opponent_king_rank: opp_king_rel,
        nyugyoku_like,
        capture_available_count,
        check_available_count,
        promotion_available_count,
        own_crisis_choice,
        opponent_crisis_choice,
    }
}

// ============================================================================
// main
// ============================================================================

fn main() -> Result<()> {
    // USI と同じく巨大な SearchWorker をスタックに置くため、専用スレッドで実行する。
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("worker thread panicked"))?
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let mode = SearchMode::resolve(cli.nodes, cli.depth);

    init_bitboard_tables();

    // progress8kpabs bucket 係数（production の LS_PROGRESS_COEFF 相当）を NNUE ロード前に設定。
    if let Some(ref coeff_path) = cli.progress_file {
        let weights = load_progress_coeff_weights(coeff_path)?;
        set_layer_stack_progress_kpabs_weights(weights)
            .map_err(|e| anyhow::anyhow!("failed to set progress weights: {e}"))?;
        set_layer_stack_bucket_mode(LayerStackBucketMode::Progress8KPAbs);
        eprintln!("info: progress bucket 係数をロード: {}", coeff_path.display());
    } else {
        eprintln!(
            "warning: --progress-file 未指定。progress 重み全ゼロで bucket が degenerate になり \
production 評価と乖離します（bias の内部整合は保たれます）"
        );
    }

    init_nnue(&cli.eval_file)
        .with_context(|| format!("failed to load NNUE from {}", cli.eval_file.display()))?;

    fs::create_dir_all(&cli.out_dir)
        .with_context(|| format!("failed to create out-dir {}", cli.out_dir.display()))?;

    let positions_path = cli.out_dir.join("positions.csv");
    let aggregate_path = cli.out_dir.join("aggregate.csv");
    let summary_path = cli.out_dir.join("summary.md");
    let errors_path = cli.out_dir.join("errors.csv");

    let input = File::open(&cli.input)
        .with_context(|| format!("failed to open input {}", cli.input.display()))?;
    let reader = BufReader::new(input);

    let mut pos_writer = BufWriter::new(File::create(&positions_path)?);
    writeln!(pos_writer, "{CSV_HEADER}")?;
    let mut err_writer = BufWriter::new(File::create(&errors_path)?);
    writeln!(err_writer, "index,sfen,error")?;

    // Threads=1（Search 既定）で決定性を確保。TT サイズは CLI 指定。
    let mut search = Search::new_with_eval_hash(cli.hash_mb, 16);

    let mut groups: BTreeMap<(String, String), GroupAcc> = BTreeMap::new();
    let mut total_positions = 0usize;
    let mut mate_count = 0usize;
    let mut error_count = 0usize;

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        let sfen = line.trim();
        if sfen.is_empty() || sfen.starts_with('#') {
            continue;
        }
        if let Some(limit) = cli.limit
            && total_positions >= limit
        {
            break;
        }

        let index = total_positions;
        match measure_position(sfen, index, &mut search, &mode, !cli.no_clear) {
            Ok(record) => {
                write_csv_row(&mut pos_writer, &record)?;

                let count_in_agg = cli.include_mate || !record.is_mate;
                if record.is_mate {
                    mate_count += 1;
                }
                if count_in_agg {
                    for (name, key) in record.group_keys() {
                        groups.entry((name.to_string(), key)).or_default().push(&record);
                    }
                }
                total_positions += 1;
            }
            Err(e) => {
                writeln!(err_writer, "{line_idx},{sfen},{e}")?;
                error_count += 1;
            }
        }
    }

    pos_writer.flush()?;
    err_writer.flush()?;

    // aggregate 統計
    let mut stats: Vec<GroupStats> =
        groups.iter().map(|((name, key), acc)| compute_stats(name, key, acc)).collect();
    stats.sort_by(|a, b| a.group_name.cmp(&b.group_name).then(a.group_key.cmp(&b.group_key)));

    write_aggregate_csv(&aggregate_path, &stats)?;
    write_summary_md(
        &summary_path,
        &stats,
        cli.min_group_count,
        total_positions,
        mate_count,
        &mode,
        cli.include_mate,
    )?;

    println!("eval_bias_miner 完了:");
    println!("  局面数        : {total_positions}（詰み {mate_count} / エラー {error_count}）");
    println!("  探索条件      : {}", mode.describe());
    println!("  group 数      : {}", stats.len());
    println!("  positions.csv : {}", positions_path.display());
    println!("  aggregate.csv : {}", aggregate_path.display());
    println!("  summary.md    : {}", summary_path.display());
    println!("  errors.csv    : {}", errors_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_label_boundaries() {
        assert_eq!(bucket_label(0, PLY_BUCKETS), "<=20");
        assert_eq!(bucket_label(20, PLY_BUCKETS), "<=20");
        assert_eq!(bucket_label(21, PLY_BUCKETS), "21-40");
        assert_eq!(bucket_label(40, PLY_BUCKETS), "21-40");
        assert_eq!(bucket_label(200, PLY_BUCKETS), "121+");
    }

    #[test]
    fn bucket_label_zero_first_bound() {
        // 最初の境界が 0 の場合（capture/check 等）
        assert_eq!(bucket_label(0, CAPTURE_BUCKETS), "<=0");
        assert_eq!(bucket_label(1, CAPTURE_BUCKETS), "1-2");
        assert_eq!(bucket_label(6, CAPTURE_BUCKETS), "6+");
    }

    #[test]
    fn percentile_basic() {
        let sorted = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&sorted, 0.0), 0.0);
        assert_eq!(percentile(&sorted, 0.5), 2.0);
        assert_eq!(percentile(&sorted, 1.0), 4.0);
    }

    #[test]
    fn classify_correction_candidate() {
        let s = GroupStats {
            group_name: "x".into(),
            group_key: "y".into(),
            count: 100,
            mean_static_eval: 0.0,
            mean_search_score: 80.0,
            mean_bias: 80.0,
            median_bias: 80.0,
            mean_abs_bias: 90.0,
            p75_abs_bias: 100.0,
            p90_abs_bias: 120.0,
            stddev_bias: 40.0,
            positive_bias_rate: 0.8,
            negative_bias_rate: 0.1,
        };
        assert!(classify(&s, 30) == Classification::CorrectionCandidate);
    }

    #[test]
    fn classify_needs_more_samples() {
        let s = GroupStats {
            group_name: "x".into(),
            group_key: "y".into(),
            count: 5,
            mean_static_eval: 0.0,
            mean_search_score: 80.0,
            mean_bias: 80.0,
            median_bias: 80.0,
            mean_abs_bias: 90.0,
            p75_abs_bias: 100.0,
            p90_abs_bias: 120.0,
            stddev_bias: 40.0,
            positive_bias_rate: 0.8,
            negative_bias_rate: 0.1,
        };
        assert!(classify(&s, 30) == Classification::NeedsMoreSamples);
    }

    #[test]
    fn king_urgency_separates_early_exit_window() {
        assert_eq!(KingUrgency::from_counts(0, 3), KingUrgency::EarlyExitWindow);
        assert_eq!(KingUrgency::from_counts(0, 0), KingUrgency::Trapped);
        assert_eq!(KingUrgency::from_counts(2, 0), KingUrgency::Trapped);
        assert_eq!(KingUrgency::from_counts(2, 1), KingUrgency::ExitNarrow);
        assert_eq!(KingUrgency::from_counts(3, 4), KingUrgency::PressuredOpen);
    }

    #[test]
    fn crisis_choice_index_is_bounded_and_unique() {
        let mut seen = [false; CrisisChoice::NUM_CLASSES];
        for urgency in [
            KingUrgency::Trapped,
            KingUrgency::ExitNarrow,
            KingUrgency::EarlyExitWindow,
            KingUrgency::PressuredOpen,
        ] {
            for drop_pressure in [
                DropPressure::None,
                DropPressure::Low,
                DropPressure::Medium,
                DropPressure::High,
            ] {
                for state in [
                    MajorState::None,
                    MajorState::Retained,
                    MajorState::Contested,
                    MajorState::Hanging,
                ] {
                    let c = CrisisChoice {
                        king_pressure: 0,
                        safe_escapes: 0,
                        king_urgency: urgency,
                        drop_pressure,
                        major_state: state,
                        major_bearing: false,
                    };
                    let idx = c.index();
                    assert!(idx < CrisisChoice::NUM_CLASSES);
                    assert!(!seen[idx]);
                    seen[idx] = true;
                }
            }
        }
        assert!(seen.into_iter().all(|x| x));
    }

    #[test]
    fn major_context_detects_hanging_and_defended_major() {
        let mut hanging = Position::new();
        hanging.set_sfen("4k4/9/9/9/4p4/4R4/9/9/4K4 b - 1").unwrap();
        assert_eq!(major_context(&hanging, Color::Black).0, MajorState::Hanging);

        let mut defended = Position::new();
        defended.set_sfen("4k4/9/9/9/4p4/4R4/5G3/9/4K4 b - 1").unwrap();
        assert_eq!(major_context(&defended, Color::Black).0, MajorState::Contested);
    }

    #[test]
    fn major_context_detects_enemy_king_zone_bearing() {
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/5R3/9/9/4K4 b - 1").unwrap();
        let (state, bearing) = major_context(&pos, Color::Black);
        assert_eq!(state, MajorState::Retained);
        assert!(bearing);
    }
}
