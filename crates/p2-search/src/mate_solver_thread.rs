//! MateSolverThread — 別 thread で詰み探索を継続走らせる仕組み
//!
//! ## 設計方針
//!
//! v3-mate (= search 内部 integration) は alpha_beta の読み筋を歪め逆効果だった
//! ため撤退済。 本 module は **main search に完全非干渉** の thread を起動し、
//! root position から dfpn 軽量探索を行い、 詰み発見時のみ atomic flag で
//! main thread に通知する。
//!
//! - main search (= alpha_beta) は完全不変
//! - 通知は `AtomicBool mate_found` + `AtomicU32 mate_move` の 2 word のみ
//! - mate_found が true なら main search の bestmove を上書き
//! - Threads >= 2 時のみ起動 (= N-1 が main search、 1 が mate solver)
//! - Threads = 1 では起動しない (= 旧 binary と完全に同じ挙動)
//!
//! ## YO / 水匠 との関連
//!
//! 水匠 / YaneuraOu の MateSolver と同じ「main search に影響せず別 worker で
//! 詰みを掘る」 design パターン。 ただし本実装は USI 「go mate」 専用ではなく、
//! 通常 go 中の bonus 機能として動く。
//!
//! ## 使い方
//!
//! ```ignore
//! let mate = MateSolverThread::start(pos.clone(), max_nodes, max_depth);
//! // ... main search 実行 ...
//! mate.request_stop();
//! if let Some(mv) = mate.take_mate() {
//!     // bestmove を override
//! }
//! mate.join();
//! ```
//!
//! ## 制約
//!
//! - `Position` は clone して thread に渡す (= main search の Position と独立)
//! - `request_stop()` 後 thread は max 1 dfpn iteration 分 (= 数百 ms 程度) 残るが、
//!   `should_stop` callback で `AtomicBool stop` を監視するので速やかに終了する

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::{self, JoinHandle};

use rshogi_core::position::Position;
use rshogi_core::types::Move;

use crate::dfpn::{BasicDfpnEngine, DfpnEngine, DfpnLimits, DfpnResult};

/// MateSolverThread のデフォルト max_nodes (= 1 探索あたり)
pub const DEFAULT_MATE_SOLVER_MAX_NODES: u32 = 200_000;
/// MateSolverThread のデフォルト max_depth (= 1 探索あたり)
pub const DEFAULT_MATE_SOLVER_MAX_DEPTH: u16 = 31;
/// MateSolverThread のデフォルト TT サイズ (MB)
pub const DEFAULT_MATE_SOLVER_TT_MB: usize = 16;

/// 別 thread で詰み探索を行う solver
pub struct MateSolverThread {
    handle: Option<JoinHandle<()>>,
    mate_found: Arc<AtomicBool>,
    mate_move: Arc<AtomicU32>,
    stop: Arc<AtomicBool>,
}

impl MateSolverThread {
    /// 詰み探索 thread を起動
    ///
    /// `pos` を clone して別 thread で dfpn を走らせる。 詰み発見時、
    /// `mate_found` を true、 `mate_move` を `Move::to_u32()` で書き込む。
    pub fn start(pos: Position, max_nodes: u32, max_depth: u16, tt_size_mb: usize) -> Self {
        let mate_found = Arc::new(AtomicBool::new(false));
        let mate_move = Arc::new(AtomicU32::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let mate_found_t = Arc::clone(&mate_found);
        let mate_move_t = Arc::clone(&mate_move);
        let stop_t = Arc::clone(&stop);

        let limits = DfpnLimits {
            max_nodes,
            max_depth,
            tt_size_mb,
        };

        let handle = thread::Builder::new()
            .name("mate-solver".to_string())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let mut engine = BasicDfpnEngine::new(limits.tt_size_mb);
                let mut probe_pos = pos;

                // 反復的に dfpn を呼び出して node budget を増やしていく。
                // 1 iteration で proven mate 発見 → flag set して即終了。
                // unknown 続きなら次 iteration で node budget を倍増 (= max 8x まで)。
                let mut current_nodes = (limits.max_nodes / 8).max(10_000);
                let max_total = limits.max_nodes;

                while !stop_t.load(Ordering::Relaxed) && current_nodes <= max_total {
                    let iter_limits = DfpnLimits {
                        max_nodes: current_nodes,
                        max_depth: limits.max_depth,
                        tt_size_mb: limits.tt_size_mb,
                    };
                    let result = engine.search(&mut probe_pos, &iter_limits);
                    if stop_t.load(Ordering::Relaxed) {
                        return;
                    }
                    match result {
                        DfpnResult::ProvenMate { best_move, .. } => {
                            mate_move_t.store(best_move.to_u32(), Ordering::SeqCst);
                            mate_found_t.store(true, Ordering::Release);
                            return;
                        }
                        DfpnResult::ProvenNoMate { .. } => {
                            // 不詰確定 → これ以上探索しても無意味
                            return;
                        }
                        DfpnResult::Unknown { .. } => {
                            // node budget を倍増 (saturate at max_total)
                            current_nodes = current_nodes.saturating_mul(2).min(max_total);
                            if current_nodes >= max_total {
                                // 最終 iteration の Unknown は確定不詰扱いせず終了
                                return;
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn mate solver thread");

        Self {
            handle: Some(handle),
            mate_found,
            mate_move,
            stop,
        }
    }

    /// 詰み発見済なら Move を返す。 未発見なら None。
    pub fn take_mate(&self) -> Option<Move> {
        Self::take_mate_from(&self.mate_found, &self.mate_move)
    }

    /// 共有 Arc から詰み手を取り出す (= search thread closure で flag のみ Arc 共有して使う用途)
    pub fn take_mate_from(mate_found: &AtomicBool, mate_move: &AtomicU32) -> Option<Move> {
        if mate_found.load(Ordering::Acquire) {
            let raw = mate_move.load(Ordering::SeqCst);
            if raw == 0 {
                return None;
            }
            // Move の下位 16bit が valid な手なので u16 として再構築
            let mv = Move::from_u16((raw & 0xFFFF) as u16);
            if mv == Move::NONE { None } else { Some(mv) }
        } else {
            None
        }
    }

    /// 詰み発見 flag の Arc を取得 (= search thread closure に共有する用途)
    pub fn mate_found_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.mate_found)
    }

    /// 詰み手の Arc を取得 (= search thread closure に共有する用途)
    pub fn mate_move_flag(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.mate_move)
    }

    /// thread に停止要求を出す (= 非ブロッキング)
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    /// thread の終了を待機
    pub fn join(mut self) {
        if let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            // 詰み探索 thread の panic は致命的ではないので log のみ
            eprintln!("info string mate solver thread panicked");
        }
    }
}

impl Drop for MateSolverThread {
    fn drop(&mut self) {
        // 明示的に join されずに drop された場合は stop を立てて join を試みる
        if self.handle.is_some() {
            self.stop.store(true, Ordering::Release);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::position::Position;

    #[test]
    fn test_no_mate_initial_position() {
        // 初期局面は詰みなし → mate_found は false のまま終了する
        let pos = Position::new();
        let mate = MateSolverThread::start(pos, 50_000, 15, 4);
        // 少し待ってから request_stop
        std::thread::sleep(std::time::Duration::from_millis(100));
        mate.request_stop();
        let result = mate.take_mate();
        // 初期局面で詰みが見つかることはない
        assert!(result.is_none(), "initial position should have no mate");
    }

    #[test]
    fn test_stop_immediately() {
        // start 直後に stop しても panic しない
        let pos = Position::new();
        let mate = MateSolverThread::start(pos, 1_000_000, 31, 8);
        mate.request_stop();
        mate.join();
    }

    #[test]
    fn test_drop_without_join() {
        // join を呼ばず drop しても hang しない (= Drop で stop+join 試行)
        let pos = Position::new();
        let _mate = MateSolverThread::start(pos, 100_000, 15, 4);
        // drop はここで暗黙に呼ばれる
    }
}
