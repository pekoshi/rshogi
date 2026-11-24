//! バッチ評価コレクター + queue/scheduler/worker フレームワーク。
//!
//! MCTS メインループで葉ノードの NNUE 評価を一括収集し、 まとめて評価する
//! ことで効率化する。 本モジュールは `BatchCollector` (既存 API、 互換維持)
//! に加えて、 hybrid (CPU + GPU) backend を吸収するための queue/scheduler/
//! worker レイヤを提供する。
//!
//! ## 互換性
//!
//! 既存 import `use crate::batch_eval::{BatchCollector, BatchResult};` は
//! そのまま動く (再 export)。 mcts/engine.rs / tests/batch_tests.rs は
//! 変更不要。
//!
//! ## 新規モジュール
//!
//! - [`queue`]: lock-free MPMC bounded queue (crossbeam-queue ベース)。
//!   cache line align、 push/pop で allocation 0。
//! - [`scheduler`]: queue から item を pop し、 batch を組み立てて backend に
//!   flush する。 `flush_timeout_us` で latency 上限を保証。
//! - [`worker`]: worker thread の共通 trait と最小構造。
//! - [`cpu_worker`] / [`gpu_worker`]: backend ごとの実装雛形。
//!
//! ## 設計理念
//!
//! - hot path で heap allocation を一切行わない (BatchCollector は new() で
//!   固定 buffer を `Box<[T; MAX_BATCH_SIZE]>` 確保)。
//! - false sharing を避けるため、 共有 atomic は `#[repr(align(64))]` で
//!   囲う。
//! - queue は固定容量、 push 失敗時は drop or backoff (用途次第)。
//!
//! 詳細は `docs/transformer_inference_runtime_design.md` を参照。

use std::time::Instant;

use p2_nnue::evaluator::NnueEvaluator;
use p2_nnue::features::P2NnueInput;

use crate::mcts::tree::SelectionPath;

pub mod cpu_worker;
pub mod gpu_worker;
pub mod queue;
pub mod scheduler;
pub mod worker;

pub use queue::{AlignedAtomicU64, BatchQueue, QueuePopError, QueuePushError};
pub use scheduler::{BatchScheduler, SchedulerConfig};
pub use worker::{Worker, WorkerStats};

/// 最大バッチサイズ。
pub const MAX_BATCH_SIZE: usize = 64;

/// バッチ評価に必要な deadline (μs)。
const SOFT_DEADLINE_US: u64 = 500;
const HARD_DEADLINE_US: u64 = 2000;

/// バッチ収集 + 一括評価。
///
/// ホットパスでのヒープ割り当てなし: `new()` で全バッファ確保。
pub struct BatchCollector {
    /// 選択パスバッファ
    paths: Box<[SelectionPath; MAX_BATCH_SIZE]>,
    /// NNUE 入力バッファ
    inputs: Box<[P2NnueInput; MAX_BATCH_SIZE]>,
    /// 局面 zobrist key (eval cache 用、 0 = 未指定)
    pos_keys: [u64; MAX_BATCH_SIZE],
    /// 評価出力バッファ (winrate)
    values: [f32; MAX_BATCH_SIZE],
    /// evaluator 生出力バッファ
    raw_outputs: [i32; MAX_BATCH_SIZE],
    /// 収集済みカウント
    pending_len: u16,
    /// バッチサイズ設定
    batch_size: u16,
    /// 最後の flush 時刻
    last_flush: Instant,
}

/// flush 結果 (値コピー)。
pub struct BatchResult {
    pub path: SelectionPath,
    pub value: f32,
    /// raw evaluator 出力 (eval cache に store する用、 0 = 未指定)
    pub raw_value: i32,
    /// 局面 zobrist key (eval cache に store する用、 0 = 未指定)
    pub pos_key: u64,
}

impl BatchCollector {
    /// バッファを一括確保して初期化。
    pub fn new(batch_size: u16) -> Self {
        let batch_size = batch_size.min(MAX_BATCH_SIZE as u16);

        // SAFETY: SelectionPath は Default を持ち、 全フィールドがゼロ初期化可能。
        // P2NnueInput は repr(C) で全フィールドが数値型。 ゼロ初期化で valid。
        let paths = unsafe {
            let layout = std::alloc::Layout::new::<[SelectionPath; MAX_BATCH_SIZE]>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut [SelectionPath; MAX_BATCH_SIZE];
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr)
        };
        let inputs = unsafe {
            let layout = std::alloc::Layout::new::<[P2NnueInput; MAX_BATCH_SIZE]>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut [P2NnueInput; MAX_BATCH_SIZE];
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr)
        };

        Self {
            paths,
            inputs,
            pos_keys: [0u64; MAX_BATCH_SIZE],
            values: [0.0; MAX_BATCH_SIZE],
            raw_outputs: [0; MAX_BATCH_SIZE],
            pending_len: 0,
            batch_size,
            last_flush: Instant::now(),
        }
    }

    /// パスと入力をバッチに追加 (key 不要時)。
    pub fn submit(&mut self, path: &SelectionPath, input: &P2NnueInput) {
        self.submit_with_key(path, input, 0);
    }

    /// パスと入力 + 局面 zobrist key をバッチに追加 (eval cache 用)。
    pub fn submit_with_key(&mut self, path: &SelectionPath, input: &P2NnueInput, pos_key: u64) {
        let idx = self.pending_len as usize;
        debug_assert!(idx < MAX_BATCH_SIZE);

        // SelectionPath のコピー (スタック配列なので memcpy)
        self.paths[idx] = SelectionPath::new();
        for i in 0..path.len() {
            self.paths[idx].push(path.get(i));
        }

        // P2NnueInput のコピー
        self.inputs[idx].piece_features = input.piece_features.clone();
        self.inputs[idx].side_to_move = input.side_to_move;
        self.inputs[idx].game_phase = input.game_phase;
        self.inputs[idx].king_bucket = input.king_bucket;

        self.pos_keys[idx] = pos_key;

        self.pending_len += 1;
    }

    /// フラッシュが必要か判定。
    pub fn should_flush(&self) -> bool {
        if self.pending_len >= self.batch_size {
            return true;
        }
        if self.pending_len == 0 {
            return false;
        }
        let elapsed_us = self.last_flush.elapsed().as_micros() as u64;
        if elapsed_us >= HARD_DEADLINE_US {
            return true;
        }
        if self.pending_len >= self.batch_size / 2 && elapsed_us >= SOFT_DEADLINE_US {
            return true;
        }
        false
    }

    /// 一括評価を実行し、 結果を返す。
    ///
    /// 返却値は `pending_len` 件の `(パスコピー, winrate)` ペア。
    pub fn flush(&mut self, evaluator: &(dyn NnueEvaluator + Send + Sync)) -> Vec<BatchResult> {
        let n = self.pending_len as usize;
        if n == 0 {
            self.last_flush = Instant::now();
            return Vec::new();
        }

        // 一括評価 (逐次フォールバック)
        for i in 0..n {
            self.raw_outputs[i] = evaluator.evaluate(&self.inputs[i]);
        }

        // winrate 変換
        let value_scale = p2_nnue::types::VALUE_SCALE as f32;
        for i in 0..n {
            self.values[i] = (self.raw_outputs[i] as f32 / value_scale).clamp(-1.0, 1.0);
        }

        // パスと値を値コピーで返す
        let results: Vec<BatchResult> = (0..n)
            .map(|i| {
                let mut path_copy = SelectionPath::new();
                for j in 0..self.paths[i].len() {
                    path_copy.push(self.paths[i].get(j));
                }
                BatchResult {
                    path: path_copy,
                    value: self.values[i],
                    raw_value: self.raw_outputs[i],
                    pos_key: self.pos_keys[i],
                }
            })
            .collect();

        self.pending_len = 0;
        self.last_flush = Instant::now();
        results
    }

    /// 収集済みカウント。
    pub fn pending_count(&self) -> u16 {
        self.pending_len
    }
}
