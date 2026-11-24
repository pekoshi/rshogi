//! CPU worker 雛形。
//!
//! 小 batch (1-3 局面) を低 latency で処理することを優先する worker。
//! `Item` / `Response` の具体型は MCTS scheduler 統合時に bind する想定。
//!
//! ## 現状
//!
//! - 本 worker は **scaffolding のみ**。 process_batch は `Vec::with_capacity`
//!   + 各 item を Default で response 化する placeholder。
//! - 実 forward は `p2-transformer-runtime::CpuTransformerRuntime` を呼ぶ
//!   形に上位レイヤで切替える (将来 PR)。

use std::marker::PhantomData;

use super::worker::{Worker, WorkerStats};

/// 低 latency CPU worker (scaffolding)。
pub struct CpuWorker<Item, Response>
where
    Item: Send + Clone,
    Response: Send + Default,
{
    stats: WorkerStats,
    _phantom: PhantomData<(Item, Response)>,
}

impl<Item, Response> CpuWorker<Item, Response>
where
    Item: Send + Clone,
    Response: Send + Default,
{
    pub fn new() -> Self {
        Self {
            stats: WorkerStats::default(),
            _phantom: PhantomData,
        }
    }
}

impl<Item, Response> Default for CpuWorker<Item, Response>
where
    Item: Send + Clone,
    Response: Send + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Item, Response> Worker<Item, Response> for CpuWorker<Item, Response>
where
    Item: Send + Clone,
    Response: Send + Default,
{
    fn process_batch(&mut self, items: &[Item]) -> Vec<Response> {
        // TODO: 実 backend (CpuTransformerRuntime) に dispatch する
        let mut out = Vec::with_capacity(items.len());
        for _ in items {
            out.push(Response::default());
        }
        self.stats.record_batch(items.len());
        out
    }

    fn stats(&self) -> &WorkerStats {
        &self.stats
    }

    fn name(&self) -> &'static str {
        "cpu_worker"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_worker_process_records_stats() {
        let mut w: CpuWorker<u32, u32> = CpuWorker::new();
        let items = vec![1u32, 2, 3];
        let out = w.process_batch(&items);
        assert_eq!(out.len(), 3);
        let (b, i, f) = w.stats().snapshot();
        assert_eq!(b, 1);
        assert_eq!(i, 3);
        assert_eq!(f, 0);
    }
}
