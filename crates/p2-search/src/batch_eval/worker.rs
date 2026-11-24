//! Worker 共通 trait と最小構造。
//!
//! 実 backend (`CpuTransformerRuntime` / `CudaTransformerRuntime`) は別 crate
//! (`p2-transformer-runtime`) にあるため、 本 trait は `Item` / `Response` を
//! 抽象的に扱う。 上位 (rshogi-usi 統合層) で具体型に bind する。

use std::sync::atomic::{AtomicU64, Ordering};

/// Worker 1 個分の累積統計。 false sharing 回避は呼び出し側で行う。
#[derive(Debug, Default)]
pub struct WorkerStats {
    /// 処理した batch 件数 (= flush 回数)。
    pub batches: AtomicU64,
    /// 処理した item 総数 (= 累積 batch size)。
    pub items: AtomicU64,
    /// flush 失敗 (queue full 等) 件数。
    pub failures: AtomicU64,
}

impl WorkerStats {
    /// `(batches, items, failures)` の snapshot。 Relaxed で十分。
    pub fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.batches.load(Ordering::Relaxed),
            self.items.load(Ordering::Relaxed),
            self.failures.load(Ordering::Relaxed),
        )
    }

    pub fn record_batch(&self, items: usize) {
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.items.fetch_add(items as u64, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }
}

/// Worker 共通 trait。
///
/// `Item` は scheduler から渡される batch 単位の input、 `Response` は worker
/// が生成する output。 具象実装は CPU / GPU で別。
pub trait Worker<Item, Response>: Send {
    /// 一括処理本体。 戻り値の `Vec` は `items` と同じ長さを保証する。
    fn process_batch(&mut self, items: &[Item]) -> Vec<Response>;

    /// 統計 (公開: snapshot 用)。
    fn stats(&self) -> &WorkerStats;

    /// 名前 (ログ用)。
    fn name(&self) -> &'static str;
}
