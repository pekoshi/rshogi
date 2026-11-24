//! CSA 棋譜 → PSV (PackedSfenValue 40 byte) 変換ツール
//!
//! Floodgate / WCSC / 電竜戦 等の CSA 棋譜から各局面を PSV レコードに変換する。
//!
//! スコアの作り方は 2 通り:
//!
//! - 既定 (擬似スコア): 対局結果から生成する
//!   (winner side の手番では +score_scale、 loser side では -score_scale、 draw は 0)。
//! - `--use-comment-eval`: floodgate 棋譜の `'** <cp> <PV>` コメントに入っている
//!   **実際の探索評価値**を使う。 実測 (2026-09-09, floodgate 2026 の 2,051 局) では
//!   全手の 82.4% に評価値があり、 値は**常に先手視点**で記録されている
//!   (全 18 エンジンで 99% 以上一致)。 評価値の無い手は局面を進めるだけでスキップする。
//!
//! # 使用例
//!
//! ```bash
//! # 単一ディレクトリ
//! cargo run --release -p tools --bin csa_to_psv -- \
//!   --input-dir data/training/floodgate/csa/2024 \
//!   --output data/training/floodgate_2024.psv
//!
//! # min/max ply フィルタ (= 序盤 / 終盤 切り捨て)
//! cargo run --release -p tools --bin csa_to_psv -- \
//!   --input-dir data/training/wcsc/wcsc34_kifu \
//!   --output data/training/wcsc34.psv \
//!   --min-ply 8 --max-ply 240
//!
//! # floodgate 棋譜を評価値つき教師にする
//! cargo run --release -p tools --bin csa_to_psv -- \
//!   --input-dir E:/shogi/p2shogi/data/floodgate_2026 \
//!   --output E:/shogi/p2shogi/data/floodgate_2026.psv \
//!   --use-comment-eval --min-rate 4000 --min-ply 8 --skip-tail 2
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rshogi_core::movegen::{MoveList, generate_legal_all};
use rshogi_core::position::Position;
use rshogi_core::types::Move;
use tools::common::dedup::collect_input_paths;
use tools::packed_sfen::{PackedSfenValue, move_to_move16, pack_position};

#[derive(Parser, Debug)]
#[command(name = "csa_to_psv", about = "CSA 棋譜 → PSV (PackedSfenValue) 変換")]
struct Args {
    /// 入力 .csa ファイル（カンマ区切りで複数可）。--input-dir と排他
    #[arg(long)]
    input: Option<String>,

    /// 入力ディレクトリ（再帰探索）。--input と排他
    #[arg(long)]
    input_dir: Option<PathBuf>,

    /// --input-dir 使用時の glob パターン
    #[arg(long, default_value = "*.csa")]
    pattern: String,

    /// 出力ファイルパス（PSV 形式）
    #[arg(long)]
    output: PathBuf,

    /// 対局結果ベースの擬似スコアスケール (cp)
    ///
    /// winner 側の手番 = +score_scale、 loser 側 = -score_scale、 draw = 0
    #[arg(long, default_value_t = 200)]
    score_scale: i16,

    /// `'** <cp>` コメントの実評価値を score に使う（無い手はスキップ）
    #[arg(long, default_value_t = false)]
    use_comment_eval: bool,

    /// --use-comment-eval 時の score 絶対値上限 (cp)。 詰みスコアはここに飽和する
    #[arg(long, default_value_t = 3000)]
    eval_clamp: i16,

    /// 両対局者のレーティングがこの値未満の棋譜をスキップ (0 = 無効)
    #[arg(long, default_value_t = 0.0)]
    min_rate: f64,

    /// この ply 未満の局面はスキップ（序盤定跡カット用）
    #[arg(long, default_value_t = 0)]
    min_ply: u16,

    /// この ply 超過の局面はスキップ（投了直前の異常局面カット用）
    #[arg(long, default_value_t = 512)]
    max_ply: u16,

    /// draw 棋譜をスキップ
    #[arg(long, default_value_t = false)]
    skip_draws: bool,

    /// 投了直前 N 手をスキップ（mate に近すぎる局面を学習データから除外）
    #[arg(long, default_value_t = 0)]
    skip_tail: u16,

    /// 処理する最大ファイル数（テスト用、 0=無制限）
    #[arg(long, default_value_t = 0)]
    max_files: usize,

    /// 並列度（0 = rayon default）
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

#[derive(Default, Debug, Clone, Copy)]
struct FileStats {
    games: u64,
    positions: u64,
    skipped_ply: u64,
    skipped_draws: u64,
    skipped_tail: u64,
    skipped_no_eval: u64,
    skipped_rate: u64,
    move_errors: u64,
    parse_errors: u64,
}

impl FileStats {
    fn merge(&mut self, other: &FileStats) {
        self.games += other.games;
        self.positions += other.positions;
        self.skipped_ply += other.skipped_ply;
        self.skipped_draws += other.skipped_draws;
        self.skipped_tail += other.skipped_tail;
        self.skipped_no_eval += other.skipped_no_eval;
        self.skipped_rate += other.skipped_rate;
        self.move_errors += other.move_errors;
        self.parse_errors += other.parse_errors;
    }
}

/// CSA ファイル 1 個を PSV バイト列に変換する
fn process_csa_file(path: &Path, args: &Args) -> (Vec<u8>, FileStats) {
    let mut stats = FileStats::default();
    let mut output = Vec::new();

    // Shift-JIS / CP932 にも対応するため bytes 読み込み後、
    // ASCII 部分のみ抽出する（CSA の構文 (V2/N/P/+/-/$/T/%/'/数字) は全て ASCII 範囲）
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => {
            stats.parse_errors += 1;
            return (output, stats);
        }
    };
    // 非 ASCII byte (= Shift-JIS のコメント等) を ' ' に置換
    let cleaned: String =
        bytes.into_iter().map(|b| if b < 0x80 { b as char } else { ' ' }).collect();
    let text = cleaned;

    // 複数棋譜が連結されている場合に対応 (V2 で分割)
    for game_text in split_csa_games(&text) {
        match convert_one_game(game_text, args, &mut output, &mut stats) {
            Ok(_) => {}
            Err(_) => stats.parse_errors += 1,
        }
    }

    (output, stats)
}

/// CSA テキストを個別の棋譜に分割
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

/// 1 棋譜分を変換し、 PSV バイト列を output に追記
fn convert_one_game(
    text: &str,
    args: &Args,
    output: &mut Vec<u8>,
    stats: &mut FileStats,
) -> Result<(), String> {
    let (csa_init_pos, moves, info) =
        rshogi_csa::parse_csa_full(text).map_err(|e| format!("parse: {e}"))?;

    if args.min_rate > 0.0 && !info.both_ratings_at_least(args.min_rate) {
        stats.skipped_rate += 1;
        return Ok(());
    }

    // 対局結果判定 (先手視点: 1=先手勝ち, -1=後手勝ち, 0=draw)
    // floodgate 棋譜には `'summary:` 行があり、 %KACHI (入玉宣言勝ち) でも
    // どちらが勝ったか一意に決まるので、 あればそれを優先する。
    let game_result = parse_summary_result(text, info.black_name.as_deref())
        .unwrap_or_else(|| determine_game_result(&moves));

    // 各通常手に対応する評価値 (先手視点)。 --use-comment-eval のときだけ使う
    let move_evals = if args.use_comment_eval {
        extract_move_evals(text)
    } else {
        Vec::new()
    };

    if args.skip_draws && game_result == 0 {
        stats.skipped_draws += 1;
        return Ok(());
    }

    // CSA 初期局面 → SFEN → rshogi-core Position に同期
    let init_sfen = csa_init_pos.to_sfen();
    let mut pos = Position::new();
    pos.set_sfen(&init_sfen).map_err(|e| format!("set_sfen: {e:?}"))?;

    // 全 normal move を CSA → USI 変換しつつ rshogi-csa 側でも進める
    // (CSA 側で apply_csa_move、 USI を rshogi-core 側に Move::from_usi + legal moves で対応)
    let mut csa_pos = csa_init_pos.clone();

    // normal move 列のみ抽出（時間情報は不要）
    let normal_moves: Vec<String> = moves
        .iter()
        .filter_map(|m| match m {
            rshogi_csa::ParsedMove::Normal(cm) => Some(cm.mv.clone()),
            rshogi_csa::ParsedMove::Special(_) => None,
        })
        .collect();

    let total_normal = normal_moves.len();

    // 局面ごとに PSV レコードを書き出す
    for (idx, csa_mv) in normal_moves.iter().enumerate() {
        // USI に変換
        let usi = match rshogi_csa::csa_move_to_usi(csa_mv, &csa_pos) {
            Ok(u) => u,
            Err(_) => {
                stats.move_errors += 1;
                // CSA 側で apply に失敗するとそれ以降不整合になるので break
                break;
            }
        };

        // rshogi-core 側で USI → Move (legal moves 経由で from_piece_type 等を確定)
        let parsed = match Move::from_usi(&usi) {
            Some(m) => m,
            None => {
                stats.move_errors += 1;
                break;
            }
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
            None => {
                stats.move_errors += 1;
                break;
            }
        };

        let game_ply = pos.game_ply() as u16;

        // フィルタ判定
        let should_skip_ply = game_ply < args.min_ply || game_ply > args.max_ply;
        let should_skip_tail =
            args.skip_tail > 0 && (total_normal - idx) <= args.skip_tail as usize;

        // 手番 STM 視点の対局結果
        let stm = pos.side_to_move();
        let is_black = stm == rshogi_core::types::Color::Black;
        let result_for_stm: i8 = if game_result == 0 {
            0
        } else if is_black == (game_result == 1) {
            1
        } else {
            -1
        };

        // score: 実評価値 (先手視点 → 手番視点に反転) か、 対局結果ベースの擬似スコア
        let score: Option<i16> = if args.use_comment_eval {
            move_evals.get(idx).copied().flatten().map(|cp_black| {
                let cp = if is_black { cp_black } else { -cp_black };
                let clamp = args.eval_clamp as i32;
                cp.clamp(-clamp, clamp) as i16
            })
        } else {
            Some((result_for_stm as i16) * args.score_scale)
        };

        if should_skip_ply {
            stats.skipped_ply += 1;
        } else if should_skip_tail {
            stats.skipped_tail += 1;
        } else if let Some(score) = score {
            let psv = PackedSfenValue {
                sfen: pack_position(&pos),
                score,
                move16: move_to_move16(mv),
                game_ply,
                game_result: result_for_stm,
                padding: 0,
            };

            output.extend_from_slice(&psv.to_bytes());
            stats.positions += 1;
        } else {
            stats.skipped_no_eval += 1;
        }

        // 両 Position を進める
        let gives_check = pos.gives_check(mv);
        pos.do_move(mv, gives_check);
        csa_pos.apply_csa_move(csa_mv).map_err(|e| format!("apply_csa: {e}"))?;
    }

    stats.games += 1;
    Ok(())
}

/// 各通常手に対応する `'** <cp> <PV>` 評価値を抽出する（先手視点、 単位は cp）。
///
/// 戻り値は `parse_csa_full` が返す通常手列と同じ順序・同じ長さで、
/// 評価値コメントの無い手は `None`。 コメントは指し手行（と消費時間行）の**後**に
/// 現れ、 直前の指し手に対する探索スコアを表す（PV は指した後の継続手順）。
fn extract_move_evals(text: &str) -> Vec<Option<i32>> {
    let mut evals: Vec<Option<i32>> = Vec::new();

    for line in text.lines() {
        let s = line.trim_end_matches('\r').trim();

        // 指し手行の判定は parse_csa_full と同一条件にする
        // (P+/P- や 手番行 "+"/"-" は先頭文字 or 長さで自然に除外される)
        if (s.starts_with('+') || s.starts_with('-')) && s.len() >= 7 {
            evals.push(None);
            continue;
        }

        // 1 手につき最初の '** 行だけを採用する
        if let Some(rest) = s.strip_prefix("'**")
            && let Some(last) = evals.last_mut()
            && last.is_none()
        {
            *last = rest.split_whitespace().next().and_then(|t| t.parse::<i32>().ok());
        }
    }

    evals
}

/// floodgate の `'summary:<kind>:<name> <result>:<name> <result>` 行から
/// 先手視点の勝敗を得る（1=先手勝ち, -1=後手勝ち, 0=draw）。
///
/// 行が無い / 先手名が見つからない場合は `None`（呼び出し側で投了パリティ判定に fallback）。
fn parse_summary_result(text: &str, black_name: Option<&str>) -> Option<i8> {
    let black_name = black_name?.trim();
    let line = text.lines().map(str::trim).find(|l| l.starts_with("'summary:"))?;

    for entry in line.split(':').skip(2) {
        let (name, result) = entry.trim().rsplit_once(' ')?;
        if name != black_name {
            continue;
        }
        return Some(match result {
            "win" => 1,
            "lose" => -1,
            _ => 0,
        });
    }

    None
}

/// CSA 棋譜から対局結果を判定（先手視点: 1=先手勝ち, -1=後手勝ち, 0=draw）
fn determine_game_result(moves: &[rshogi_csa::ParsedMove]) -> i8 {
    use rshogi_csa::{ParsedMove, SpecialMove};

    for mv in moves.iter().rev() {
        if let ParsedMove::Special(special) = mv {
            return match special {
                SpecialMove::Resign => {
                    let normal_count =
                        moves.iter().filter(|m| matches!(m, ParsedMove::Normal(_))).count();
                    if normal_count.is_multiple_of(2) {
                        -1 // 先手が投了 (= 偶数手後で先手番だった)
                    } else {
                        1 // 後手が投了
                    }
                }
                SpecialMove::Win => 1, // 入玉宣言勝ち（先手と仮定、誤判定もあり得る）
                SpecialMove::Draw | SpecialMove::Sennichite | SpecialMove::Jishogi => 0,
                _ => 0,
            };
        }
    }
    0
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().ok();
    }

    let mut paths =
        collect_input_paths(args.input.as_deref(), args.input_dir.as_ref(), &args.pattern)?;
    if paths.is_empty() {
        eprintln!("入力ファイルが見つかりません");
        return Ok(());
    }

    if args.max_files > 0 && paths.len() > args.max_files {
        paths.truncate(args.max_files);
    }

    eprintln!("Processing {} CSA files...", paths.len());
    eprintln!("Output: {}", args.output.display());
    if args.use_comment_eval {
        eprintln!("Score source:     '** comment eval (clamp +-{} cp)", args.eval_clamp);
    } else {
        eprintln!("Score source:     game result (scale {} cp)", args.score_scale);
    }
    eprintln!(
        "Filters: min_ply={}, max_ply={}, skip_tail={}, skip_draws={}, min_rate={}",
        args.min_ply, args.max_ply, args.skip_tail, args.skip_draws, args.min_rate
    );

    let start = std::time::Instant::now();

    let out_file = File::create(&args.output)?;
    let writer = Mutex::new(BufWriter::with_capacity(16 * 1024 * 1024, out_file));

    let progress = ProgressBar::new(paths.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} files {msg}")
            .expect("valid template"),
    );

    let total_stats = Mutex::new(FileStats::default());

    paths.par_iter().for_each(|path| {
        let (bytes, stats) = process_csa_file(path, &args);
        if !bytes.is_empty() {
            let mut w = writer.lock().expect("writer lock");
            // 失敗時は eprintln し、 当 batch を捨てる
            if let Err(e) = w.write_all(&bytes) {
                eprintln!("write_all error: {e}");
            }
        }
        total_stats.lock().expect("stats lock").merge(&stats);
        progress.inc(1);
    });
    progress.finish();

    {
        let mut w = writer.into_inner().expect("writer inner");
        w.flush()?;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let s = total_stats.into_inner().expect("stats inner");

    println!("=== CSA → PSV Summary ===");
    println!("Input files:       {}", paths.len());
    println!("Games processed:   {}", s.games);
    println!("Positions written: {}", s.positions);
    println!("Skipped (ply):     {}", s.skipped_ply);
    println!("Skipped (tail):    {}", s.skipped_tail);
    println!("Skipped (draws):   {}", s.skipped_draws);
    if s.skipped_no_eval > 0 {
        println!("Skipped (no eval): {}", s.skipped_no_eval);
    }
    if s.skipped_rate > 0 {
        println!("Skipped (rate):    {}", s.skipped_rate);
    }
    if s.move_errors > 0 {
        println!("Move errors:       {}", s.move_errors);
    }
    if s.parse_errors > 0 {
        println!("Parse errors:      {}", s.parse_errors);
    }
    println!("Output file:       {}", args.output.display());
    println!(
        "Output size:       {:.2} MB",
        (s.positions * PackedSfenValue::SIZE as u64) as f64 / (1024.0 * 1024.0)
    );
    println!("Elapsed:           {:.1} sec", elapsed);
    if elapsed > 0.0 {
        println!("Throughput:        {:.0} pos/sec", s.positions as f64 / elapsed);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// floodgate 棋譜の抜粋。 評価値は指し手行 + 消費時間行の後に現れる
    const SAMPLE: &str = r#"V2
N+AobaNNUE_N150
N-Ataraxia_2026
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P+00FU00FU
+
'black_rate:AobaNNUE_N150+deadbeef:4134.0
'white_rate:Ataraxia_2026+cafebabe:4082.0
+2726FU
T21
'** 55 -8384FU +6978KI
-1314FU
T1
'** 74 +2625FU
+2625FU
T10
-8384FU
T0
'** -72 +1716FU
%TORYO
'summary:toryo:AobaNNUE_N150 win:Ataraxia_2026 lose
"#;

    #[test]
    fn extract_move_evals_aligns_with_normal_moves() {
        let evals = extract_move_evals(SAMPLE);

        // 通常手 4 手。 3 手目には評価値コメントが無い
        assert_eq!(evals, vec![Some(55), Some(74), None, Some(-72)]);
    }

    #[test]
    fn extract_move_evals_matches_parse_csa_full_move_count() {
        let (_pos, moves, _info) = rshogi_csa::parse_csa_full(SAMPLE).expect("parse");
        let normal =
            moves.iter().filter(|m| matches!(m, rshogi_csa::ParsedMove::Normal(_))).count();

        assert_eq!(extract_move_evals(SAMPLE).len(), normal);
    }

    #[test]
    fn summary_result_is_black_relative() {
        assert_eq!(parse_summary_result(SAMPLE, Some("AobaNNUE_N150")), Some(1));

        let gote_win = "'summary:toryo:A win:B lose";
        assert_eq!(parse_summary_result(gote_win, Some("B")), Some(-1));

        let draw = "'summary:sennichite:A draw:B draw";
        assert_eq!(parse_summary_result(draw, Some("A")), Some(0));

        // 先手名が summary に無ければ None (投了パリティ判定に fallback)
        assert_eq!(parse_summary_result(draw, Some("C")), None);
        assert_eq!(parse_summary_result("V2", Some("A")), None);
    }
}
