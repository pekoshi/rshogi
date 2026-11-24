//! GPU worker 雛形 (persistent buffer の所有を抽象化)。
//!
//! 実 backend は `p2-transformer-runtime::CudaTransformerRuntime` (cuda feature
//! 有効時のみ build) で、 そちら側で device tensor を再利用する。 本 file は
//! scaffolding として trait に乗せるための placeholder を提供する。

use std::marker::PhantomData;

use super::worker::{Worker, WorkerStats};

/// 高 throughput GPU worker (scaffolding)。
///
/// 上位レイヤで `CudaTransformerRuntime` (persistent token_tensor / value_tensor
/// / policy_tensor を保持) を実 backend として注入する想定。
pub struct GpuWorker<Item, Response>
where
    Item: Send + Clone,
    Response: Send + Default,
{
    stats: WorkerStats,
    _phantom: PhantomData<(Item, Response)>,
}

impl<Item, Response> GpuWorker<Item, Response>
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

impl<Item, Response> Default for GpuWorker<Item, Response>
where
    Item: Send + Clone,
    Response: Send + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Item, Response> Worker<Item, Response> for GpuWorker<Item, Response>
where
    Item: Send + Clone,
    Response: Send + Default,
{
    fn process_batch(&mut self, items: &[Item]) -> Vec<Response> {
        // TODO: CudaTransformerRuntime に dispatch (cuda feature 有効時)。
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
        "gpu_worker"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_worker_scaffolding_smoke() {
        let mut w: GpuWorker<u32, u32> = GpuWorker::new();
        let items = vec![1u32, 2, 3, 4];
        let out = w.process_batch(&items);
        assert_eq!(out.len(), 4);
        let (b, i, _) = w.stats().snapshot();
        assert_eq!(b, 1);
        assert_eq!(i, 4);
    }
}
