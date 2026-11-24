//! Policy-Value PUCT USI エンジン
//!
//! PuctEngine (MCTS) + PolicyValueNet (HalfKAv2_hm + SwiGLU、 bullet raw.bin) で
//! USI プロトコルに応答する。
//!
//! USI option:
//!   PuctEvalFile    = bullet raw.bin path (Compact=v11、 CompactRms=v12、 Wide=v13)
//!   PuctEvalVariant = compact | compact_rms | wide (省略時は raw.bin サイズから自動)

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

#[cfg(feature = "pv-cuda")]
use p2_nnue::policy_value_net::{
    PolicyValueCudaContext, PolicyValueEvaluatorGpu, PolicyValueNetCuda,
};
use p2_nnue::policy_value_net::{
    PolicyValueEvaluator, PolicyValueNet, PolicyValueVariant, infer_variant_from_size, load_raw_bin,
};
use p2_search::mcts::config::PuctConfig;
use p2_search::mcts::engine::PuctEngine;
use p2_search::phase::SearchMode;
use p2_search::puct::{PuctBackend, PuctLimits, PuctResult};
use rshogi_core::position::Position;
use rshogi_core::types::Move;

struct UsiEngine {
    position: Position,
    puct_eval_file: Option<PathBuf>,
    puct_variant: Option<PolicyValueVariant>,
    puct_device: String,
    puct_batch_size: u16,
    engine: Option<PuctEngine>,
    stop_flag: Arc<AtomicBool>,
    search_thread: Option<thread::JoinHandle<PuctResult>>,
}

impl UsiEngine {
    fn new() -> Self {
        Self {
            position: Position::new(),
            puct_eval_file: None,
            puct_variant: None,
            puct_device: "cpu".to_string(),
            puct_batch_size: 1,
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
        println!("id name PV-PUCT-USI");
        println!("id author rshogi");
        println!("option name PuctEvalFile type string default <empty>");
        println!("option name PuctEvalVariant type string default auto");
        println!("option name PuctEvalDevice type string default cpu");
        println!("option name PuctBatchSize type spin default 1 min 1 max 64");
        println!("usiok");
        io::stdout().flush().ok();
    }

    fn cmd_isready(&mut self) {
        let path = match &self.puct_eval_file {
            Some(p) => p.clone(),
            None => {
                eprintln!("info string PuctEvalFile not set");
                println!("readyok");
                io::stdout().flush().ok();
                return;
            }
        };

        // variant 自動推定 (size 基準)、 明示指定があれば優先
        let variant = if let Some(v) = self.puct_variant {
            v
        } else {
            match std::fs::metadata(&path) {
                Ok(meta) => {
                    let n_floats = (meta.len() / 4) as usize;
                    match infer_variant_from_size(n_floats) {
                        Some(v) => {
                            eprintln!(
                                "info string Auto-detected variant: {v:?} ({n_floats} floats)"
                            );
                            v
                        }
                        None => {
                            eprintln!(
                                "info string Variant auto-detect failed (n_floats={n_floats}), default to Compact"
                            );
                            PolicyValueVariant::Compact
                        }
                    }
                }
                Err(e) => {
                    eprintln!("info string PuctEvalFile metadata error: {e}");
                    println!("readyok");
                    io::stdout().flush().ok();
                    return;
                }
            }
        };

        let cfg = variant.config();
        let mut puct_config = PuctConfig::default();
        puct_config.batch_size = self.puct_batch_size;
        // Step 3 未完: PolicyValueGpu モードでは new_policy_value_gpu 内で use_batching=false 強制。
        // PuctBatchSize > 1 を指定しても deferred batch eval は無効 (info string で警告済)。
        match load_raw_bin(&path, cfg) {
            Ok(weights) => {
                let engine_result: Result<PuctEngine, String> = match self.puct_device.as_str() {
                    #[cfg(feature = "pv-cuda")]
                    "cuda" | "gpu" => match PolicyValueCudaContext::new(0) {
                        Ok(cuda_ctx) => {
                            let cuda_ctx = Arc::new(cuda_ctx);
                            match PolicyValueNetCuda::new(cuda_ctx, &weights) {
                                Ok(gpu_net) => {
                                    let evaluator = Arc::new(PolicyValueEvaluatorGpu::new(gpu_net));
                                    eprintln!(
                                        "info string PolicyValueNet GPU loaded: {:?} variant={:?}",
                                        path, variant
                                    );
                                    Ok(PuctEngine::new_policy_value_gpu(
                                        puct_config.clone(),
                                        evaluator,
                                    ))
                                }
                                Err(e) => Err(format!("GPU net init failed: {e:?}")),
                            }
                        }
                        Err(e) => Err(format!("CUDA context init failed: {e:?}")),
                    },
                    _ => {
                        let net = PolicyValueNet::new(weights);
                        let evaluator = Arc::new(PolicyValueEvaluator::new(net));
                        eprintln!(
                            "info string PolicyValueNet CPU loaded: {:?} variant={:?}",
                            path, variant
                        );
                        Ok(PuctEngine::new_policy_value(puct_config.clone(), evaluator))
                    }
                };
                match engine_result {
                    Ok(engine) => {
                        self.stop_flag = engine.stop_flag();
                        self.engine = Some(engine);
                    }
                    Err(e) => eprintln!("info string Engine init failed: {e}"),
                }
            }
            Err(e) => {
                eprintln!("info string PolicyValueNet load error: {e}");
            }
        }
        println!("readyok");
        io::stdout().flush().ok();
    }

    fn cmd_setoption(&mut self, tokens: &[&str]) {
        let name_idx = tokens.iter().position(|&t| t == "name");
        let value_idx = tokens.iter().position(|&t| t == "value");
        if let (Some(ni), Some(vi)) = (name_idx, value_idx) {
            let name = tokens[ni + 1..vi].join(" ");
            let value = tokens[vi + 1..].join(" ");
            match name.as_str() {
                "PuctEvalFile" => {
                    self.puct_eval_file = Some(PathBuf::from(&value));
                    eprintln!("info string PuctEvalFile set: {value}");
                }
                "PuctEvalVariant" => {
                    if value.eq_ignore_ascii_case("auto") {
                        self.puct_variant = None;
                    } else if let Some(v) = PolicyValueVariant::parse(&value) {
                        self.puct_variant = Some(v);
                        eprintln!("info string PuctEvalVariant set: {v:?}");
                    } else {
                        eprintln!("info string PuctEvalVariant unknown: {value}");
                    }
                }
                "PuctEvalDevice" => {
                    self.puct_device = value.to_ascii_lowercase();
                    eprintln!("info string PuctEvalDevice set: {}", self.puct_device);
                }
                "PuctBatchSize" => {
                    let parsed: u16 = value.parse().unwrap_or(1).clamp(1, 64);
                    self.puct_batch_size = parsed;
                    eprintln!(
                        "info string PuctBatchSize set: {} (note: deferred batch eval は Layer 2 実装後に有効化、 現状 1 = 逐次 eval)",
                        self.puct_batch_size
                    );
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
        eprintln!("info string [cmd_go] search start (SYNC), limits={:?}", limits);
        let t0 = std::time::Instant::now();
        let mut engine = engine;
        let result = engine.search(&pos, &limits);
        eprintln!(
            "info string [search done] elapsed={:.2}s nodes={} bestmove={:?} value={}",
            t0.elapsed().as_secs_f64(),
            result.nodes,
            result.bestmove,
            result.value
        );
        let bestmove = if result.bestmove.is_none() {
            "resign".to_string()
        } else {
            result.bestmove.to_usi()
        };
        let pv_str: String = result.pv.iter().map(|m| m.to_usi()).collect::<Vec<_>>().join(" ");
        let cp = (result.value * 600.0) as i32;
        println!("info nodes {} score cp {cp} pv {pv_str}", result.nodes);
        println!("bestmove {bestmove}");
        io::stdout().flush().ok();
        self.engine = Some(engine);
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
        if max_time_ms == 0 && max_nodes == 0 {
            max_time_ms = 1000;
        }
        PuctLimits {
            max_nodes,
            max_time_ms,
            search_mode: SearchMode::Normal,
            game_phase: None,
            endgame_score: 0.0,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut engine = UsiEngine::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--eval" | "--puct-eval" if i + 1 < args.len() => {
                engine.puct_eval_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--variant" if i + 1 < args.len() => {
                engine.puct_variant = PolicyValueVariant::parse(&args[i + 1]);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    engine.run();
}
