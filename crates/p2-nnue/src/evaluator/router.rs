//! CPU/GPU 動的切替ルーター
//!
//! batch size に応じて CPU SIMD と GPU CUDA を自動で切り替える。
//! GPU 不可時や連続失敗時は CPU へフォールバックする。

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

use super::{CpuEvaluator, NnueEvaluator};
use crate::features::P2NnueInput;
use crate::network::P2Network;
use crate::queue::types::{EvalRequest, EvalResult};

#[cfg(feature = "cuda")]
use super::gpu::CudaEvaluator;

/// GPU 連続失敗でフォールバックする閾値
#[cfg(feature = "cuda")]
const GPU_CONSECUTIVE_FAIL_LIMIT: u32 = 3;

/// ルーティング統計（モニタリング用）
pub struct RouterStats {
    cpu_batches: AtomicU64,
    gpu_batches: AtomicU64,
    gpu_fallbacks: AtomicU64,
    total_requests: AtomicU64,
}

impl RouterStats {
    fn new() -> Self {
        Self {
            cpu_batches: AtomicU64::new(0),
            gpu_batches: AtomicU64::new(0),
            gpu_fallbacks: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
        }
    }

    /// 現在の統計をスナップショットとして取得
    pub fn snapshot(&self) -> RouterStatsSnapshot {
        RouterStatsSnapshot {
            cpu_batches: self.cpu_batches.load(Relaxed),
            gpu_batches: self.gpu_batches.load(Relaxed),
            gpu_fallbacks: self.gpu_fallbacks.load(Relaxed),
            total_requests: self.total_requests.load(Relaxed),
        }
    }
}

/// 統計スナップショット（非アトミック）
#[derive(Debug, Clone)]
pub struct RouterStatsSnapshot {
    pub cpu_batches: u64,
    pub gpu_batches: u64,
    pub gpu_fallbacks: u64,
    pub total_requests: u64,
}

/// CPU/GPU 動的切替ルーター
///
/// batch size に応じて CPU SIMD と GPU CUDA を自動で切り替える。
/// - `batch_size < threshold` → CPU
/// - `batch_size >= threshold && GPU 可` → GPU（失敗時 CPU fallback）
pub struct HybridRouter {
    cpu: CpuEvaluator,
    #[cfg(feature = "cuda")]
    gpu: Option<CudaEvaluator>,
    threshold: usize,
    stats: RouterStats,
    consecutive_gpu_fails: AtomicU32,
}

impl HybridRouter {
    /// ルーターを初期化
    ///
    /// GPU が利用不可の場合は CPU のみモードになる。
    pub fn new(network: Arc<P2Network>, threshold: usize) -> Self {
        let cpu = CpuEvaluator::new(Arc::clone(&network));

        #[cfg(feature = "cuda")]
        let gpu = CudaEvaluator::new(network).ok();

        Self {
            cpu,
            #[cfg(feature = "cuda")]
            gpu,
            threshold,
            stats: RouterStats::new(),
            consecutive_gpu_fails: AtomicU32::new(0),
        }
    }

    /// GPU が利用可能か（存在し、連続失敗が閾値未満）
    pub fn gpu_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            return self.gpu.is_some()
                && self.consecutive_gpu_fails.load(Relaxed) < GPU_CONSECUTIVE_FAIL_LIMIT;
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// 統計スナップショット取得
    pub fn stats(&self) -> RouterStatsSnapshot {
        self.stats.snapshot()
    }

    /// バッチリクエストを CPU/GPU にルーティングし結果を返す
    ///
    /// - `batch_size < threshold` → CPU
    /// - `batch_size >= threshold && GPU 可` → GPU（失敗時 CPU fallback）
    /// - 結果の順序は入力と 1:1 対応
    pub fn route_batch(&self, requests: &[EvalRequest]) -> Vec<EvalResult> {
        let batch_size = requests.len();
        self.stats.total_requests.fetch_add(batch_size as u64, Relaxed);

        // GPU パス判定
        if batch_size >= self.threshold && self.gpu_available() {
            if let Some(results) = self.try_gpu_batch(requests) {
                self.consecutive_gpu_fails.store(0, Relaxed);
                self.stats.gpu_batches.fetch_add(1, Relaxed);
                return results;
            }
            self.consecutive_gpu_fails.fetch_add(1, Relaxed);
            self.stats.gpu_fallbacks.fetch_add(1, Relaxed);
        }

        // CPU パス
        self.cpu_batch(requests)
    }

    /// GPU バッチ評価。パニックを catch して None を返す。
    #[cfg(feature = "cuda")]
    fn try_gpu_batch(&self, requests: &[EvalRequest]) -> Option<Vec<EvalResult>> {
        let gpu = self.gpu.as_ref()?;
        let n = requests.len();

        // P2NnueInput のフィールドは全て Copy なので、フィールドごとの代入でコピー
        let inputs: Vec<P2NnueInput> = requests
            .iter()
            .map(|r| P2NnueInput {
                piece_features: r.nnue_input.piece_features,
                side_to_move: r.nnue_input.side_to_move,
                game_phase: r.nnue_input.game_phase,
                king_bucket: r.nnue_input.king_bucket,
            })
            .collect();

        let mut values = vec![0i32; n];

        // CudaEvaluator::evaluate_batch が GPU メモリ不足等でパニックする可能性がある。
        // catch_unwind で捕捉し CPU fallback する。
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gpu.evaluate_batch(&inputs, &mut values);
        }));

        match result {
            Ok(()) => Some(build_results(requests, &values)),
            Err(_) => None,
        }
    }

    /// GPU 非対応ビルドでは常に None
    #[cfg(not(feature = "cuda"))]
    fn try_gpu_batch(&self, _requests: &[EvalRequest]) -> Option<Vec<EvalResult>> {
        None
    }

    /// CPU バッチ評価
    fn cpu_batch(&self, requests: &[EvalRequest]) -> Vec<EvalResult> {
        self.stats.cpu_batches.fetch_add(1, Relaxed);
        let values: Vec<i32> = requests.iter().map(|r| self.cpu.evaluate(&r.nnue_input)).collect();
        build_results(requests, &values)
    }
}

/// request_id / node_id を対応付けて EvalResult を組み立てる
fn build_results(requests: &[EvalRequest], values: &[i32]) -> Vec<EvalResult> {
    requests
        .iter()
        .zip(values)
        .map(|(req, &val)| EvalResult {
            request_id: req.request_id,
            node_id: req.node_id,
            value: val,
        })
        .collect()
}

impl NnueEvaluator for HybridRouter {
    /// 単一評価は常に CPU（GPU カーネル起動のオーバーヘッドを回避）
    fn evaluate(&self, input: &P2NnueInput) -> i32 {
        self.cpu.evaluate(input)
    }

    fn backend_name(&self) -> &'static str {
        #[cfg(feature = "cuda")]
        if self.gpu.is_some() {
            return "hybrid";
        }
        "cpu-only"
    }
}
