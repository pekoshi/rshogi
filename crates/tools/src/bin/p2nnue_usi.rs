//! P2-NNUE USI エンジン
//!
//! PuctEngine (MCTS) + P2-NNUE 評価で USI プロトコルに応答する。

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use p2_nnue::evaluator::CpuEvaluator;
use p2_nnue::network::P2Network;
use p2_search::mcts::config::PuctConfig;
use p2_search::mcts::engine::PuctEngine;
use p2_search::phase::SearchMode;
use p2_search::puct::{PuctBackend, PuctLimits, PuctResult};
use rshogi_core::position::Position;
use rshogi_core::types::Move;

/// USI エンジン本体
struct UsiEngine {
    position: Position,
    /// P2-NNUE モデルパス
    p2_eval_file: Option<PathBuf>,
    /// αβ用 NNUE 評価関数パス
    ab_eval_file: Option<PathBuf>,
    /// PuctEngine（isready 後に初期化）
    engine: Option<PuctEngine>,
    /// 探索中断用フラグ
    stop_flag: Arc<AtomicBool>,
    /// 探索スレッド
    search_thread: Option<thread::JoinHandle<PuctResult>>,
}

impl UsiEngine {
    fn new() -> Self {
        Self {
            position: Position::new(),
            p2_eval_file: None,
            ab_eval_file: None,
            engine: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            search_thread: None,
        }
    }

    fn run(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            match tokens[0] {
                "usi" => self.cmd_usi(),
                "isready" => self.cmd_isready(),
                "setoption" => self.cmd_setoption(&tokens),
                "usinewgame" => {}
                "position" => self.cmd_position(&tokens),
                "go" => self.cmd_go(&tokens),
                "stop" => self.cmd_stop(),
                "quit" => break,
                "gameover" => {}
                _ => {}
            }
        }
    }

    fn cmd_usi(&self) {
        println!("id name P2NNUE-USI");
        println!("id author rshogi");
        println!("option name P2EvalFile type string default model.p2nn");
        println!("option name EvalFile type string default <empty>");
        println!("usiok");
        io::stdout().flush().ok();
    }

    fn cmd_isready(&mut self) {
        // P2-NNUE ロード
        let p2_path = self.p2_eval_file.clone().unwrap_or_else(|| PathBuf::from("model.p2nn"));

        match std::fs::File::open(&p2_path) {
            Ok(file) => {
                let mut reader = io::BufReader::new(file);
                match P2Network::load(&mut reader) {
                    Ok(net) => {
                        let evaluator = Arc::new(CpuEvaluator::new(Arc::new(net)));
                        let engine = PuctEngine::new_direct(PuctConfig::default(), evaluator);
                        self.stop_flag = engine.stop_flag();
                        self.engine = Some(engine);
                        eprintln!("info string P2-NNUE loaded: {:?}", p2_path);
                    }
                    Err(e) => {
                        eprintln!("info string P2-NNUE load error: {e}");
                    }
                }
            }
            Err(e) => {
                eprintln!("info string P2-NNUE file open error: {:?}: {e}", p2_path);
            }
        }

        // αβ用 NNUE ロード（指定されている場合）
        if let Some(ref ab_path) = self.ab_eval_file {
            match rshogi_core::nnue::init_nnue(ab_path) {
                Ok(()) => {
                    eprintln!("info string AB-NNUE loaded: {:?}", ab_path);
                }
                Err(e) => {
                    eprintln!("info string AB-NNUE load error: {:?}: {e}", ab_path);
                }
            }
        }

        println!("readyok");
        io::stdout().flush().ok();
    }

    fn cmd_setoption(&mut self, tokens: &[&str]) {
        // setoption name <name> value <value>
        let name_idx = tokens.iter().position(|&t| t == "name");
        let value_idx = tokens.iter().position(|&t| t == "value");

        if let (Some(ni), Some(vi)) = (name_idx, value_idx) {
            let name = tokens[ni + 1..vi].join(" ");
            let value = tokens[vi + 1..].join(" ");

            match name.as_str() {
                "P2EvalFile" => {
                    self.p2_eval_file = Some(PathBuf::from(&value));
                    eprintln!("info string P2EvalFile set to: {value}");
                }
                "EvalFile" => {
                    self.ab_eval_file = Some(PathBuf::from(&value));
                    eprintln!("info string EvalFile set to: {value}");
                }
                _ => {}
            }
        }
    }

    fn cmd_position(&mut self, tokens: &[&str]) {
        let mut idx = 1;
        if idx >= tokens.len() {
            return;
        }

        if tokens[idx] == "startpos" {
            self.position.set_hirate();
            idx += 1;
        } else if tokens[idx] == "sfen" {
            idx += 1;
            let mut sfen_parts = Vec::new();
            while idx < tokens.len() && tokens[idx] != "moves" {
                sfen_parts.push(tokens[idx]);
                idx += 1;
            }
            let sfen = sfen_parts.join(" ");
            if let Err(e) = self.position.set_sfen(&sfen) {
                eprintln!("info string SFEN parse error: {e}");
                return;
            }
        }

        if idx < tokens.len() && tokens[idx] == "moves" {
            idx += 1;
            while idx < tokens.len() {
                if let Some(mv) = Move::from_usi(tokens[idx]) {
                    let gives_check = if mv.is_pass() {
                        false
                    } else {
                        self.position.gives_check(mv)
                    };
                    self.position.do_move(mv, gives_check);
                }
                idx += 1;
            }
        }
    }

    fn cmd_go(&mut self, tokens: &[&str]) {
        // 前回の探索スレッドを回収
        if let Some(handle) = self.search_thread.take() {
            let _ = handle.join();
        }

        let engine = match self.engine.take() {
            Some(e) => e,
            None => {
                eprintln!("info string Engine not initialized");
                println!("bestmove resign");
                io::stdout().flush().ok();
                return;
            }
        };

        let limits = self.parse_go_options(tokens);
        let pos = self.position.clone();

        let handle = thread::spawn(move || {
            let mut engine = engine;
            let result = engine.search(&pos, &limits);

            let bestmove = if result.bestmove.is_none() {
                "resign".to_string()
            } else {
                result.bestmove.to_usi()
            };

            let pv_str: String = result.pv.iter().map(|m| m.to_usi()).collect::<Vec<_>>().join(" ");

            // winrate [0,1] → cp 変換（簡易）
            let cp = (result.value * 600.0) as i32;
            println!("info nodes {} score cp {cp} pv {pv_str}", result.nodes,);
            println!("bestmove {bestmove}");
            io::stdout().flush().ok();

            // engine を返す（PuctResult は消費）
            // ここでは PuctResult を返して engine は捨てる
            result
        });

        self.search_thread = Some(handle);
    }

    fn cmd_stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }

    fn parse_go_options(&self, tokens: &[&str]) -> PuctLimits {
        let mut max_time_ms = 0u64;
        let mut max_nodes = 0u64;
        let stm = self.position.side_to_move();

        let mut i = 1;
        while i < tokens.len() {
            match tokens[i] {
                "byoyomi" if i + 1 < tokens.len() => {
                    max_time_ms = tokens[i + 1].parse().unwrap_or(0);
                    i += 2;
                }
                "movetime" if i + 1 < tokens.len() => {
                    max_time_ms = tokens[i + 1].parse().unwrap_or(0);
                    i += 2;
                }
                "btime" if i + 1 < tokens.len() => {
                    if stm == rshogi_core::types::Color::Black {
                        let time: u64 = tokens[i + 1].parse().unwrap_or(0);
                        // 残り時間の 1/30 を使う（簡易時間管理）
                        max_time_ms = time / 30;
                    }
                    i += 2;
                }
                "wtime" if i + 1 < tokens.len() => {
                    if stm == rshogi_core::types::Color::White {
                        let time: u64 = tokens[i + 1].parse().unwrap_or(0);
                        max_time_ms = time / 30;
                    }
                    i += 2;
                }
                "nodes" if i + 1 < tokens.len() => {
                    max_nodes = tokens[i + 1].parse().unwrap_or(0);
                    i += 2;
                }
                "infinite" => {
                    max_time_ms = u64::MAX;
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            }
        }

        // byoyomi が指定されていない場合のフォールバック
        if max_time_ms == 0 && max_nodes == 0 {
            max_time_ms = 1000; // デフォルト 1 秒
        }

        PuctLimits {
            max_nodes,
            max_time_ms,
            search_mode: SearchMode::Normal,
        }
    }
}

fn main() {
    // コマンドライン引数でモデルパスを指定可能
    let args: Vec<String> = std::env::args().collect();
    let mut engine = UsiEngine::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "--p2-eval" if i + 1 < args.len() => {
                engine.p2_eval_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--eval" | "--ab-eval" if i + 1 < args.len() => {
                engine.ab_eval_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    engine.run();
}
