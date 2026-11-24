//! Batch scheduler: queue から item を集約し、 worker に flush する。
//!
//! ## 役割
//!
//! 1. inbound `BatchQueue<Item>` から item を最大 `max_batch` 件 pop。
//! 2. `max_batch` に達するか `flush_timeout_us` 経過したら worker を呼んで
//!    一括評価。
//! 3. 結果を outbound `BatchQueue<Response>` に push。
//!
//! 本ファイルは batch policy のみを担当し、 実際の forward は `worker` 側に
//! 委譲する。 GPU / CPU worker のいずれを使うかは scheduler の上位で決める
//! (`HybridTransformerRuntime` と組み合わせる想定)。
//!
//! ## flush 判定
//!
//! ```text
//! flush if  pending == max_batch
//!       OR  elapsed_since_first_item > flush_timeout_us
//! ```
//!
//! batch wait latency が MCTS を停滞させないよう、 `flush_timeout_us` 上限は
//! 必須。 デフォルト 200 μs (GPU async 推論 1 回の overhead と均衡)。

use std::time::Instant;

use super::queue::{BatchQueue, QueuePopError};

/// Scheduler 設定。
#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    /// この batch サイズに達したら即 flush。
    pub max_batch: usize,
    /// 1 個目の item を pop してから flush までの上限 (μs)。
    pub flush_timeout_us: u64,
    /// queue 空き時の spin 上限 (回)。 これを超えたら yield。
    pub idle_spin_limit: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch: 128,
            flush_timeout_us: 200,
            idle_spin_limit: 32,
        }
    }
}

/// Scheduler 本体。
///
/// `Item` は worker に渡す型 (例: `BatchRequest<TransformerTokens>`)。 ここ
/// では generic に保ち、 上位 (rshogi-usi 等) で具体型を bind する。
pub struct BatchScheduler<Item> {
    cfg: SchedulerConfig,
    /// 集約用 staging buffer (heap alloc は構築時 1 回のみ)。
    staging: Vec<Item>,
    /// 1 個目を pop した時刻。
    first_arrival: Option<Instant>,
}

impl<Item> BatchScheduler<Item> {
    pub fn new(cfg: SchedulerConfig) -> Self {
        let mut staging = Vec::new();
        staging.reserve_exact(cfg.max_batch);
        Self {
            cfg,
            staging,
            first_arrival: None,
        }
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.cfg
    }

    /// 現在の staging サイズ。
    pub fn pending(&self) -> usize {
        self.staging.len()
    }

    /// flush trigger 判定 (公開: 単体テスト用)。
    #[inline]
    pub fn should_flush(&self) -> bool {
        if self.staging.len() >= self.cfg.max_batch {
            return true;
        }
        if let Some(t0) = self.first_arrival {
            let elapsed_us = t0.elapsed().as_micros() as u64;
            if elapsed_us >= self.cfg.flush_timeout_us {
                return true;
            }
        }
        false
    }

    /// `queue` から item を集約。 `should_flush()` が true になったら停止。
    ///
    /// 戻り値: 集約した item 数 (= `self.staging.len()`)。
    pub fn collect(&mut self, queue: &BatchQueue<Item>) -> usize {
        while !self.should_flush() {
            match queue.pop() {
                Ok(item) => {
                    if self.staging.is_empty() {
                        self.first_arrival = Some(Instant::now());
                    }
                    self.staging.push(item);
                }
                Err(QueuePopError::Empty) => {
                    // 空、 かつ既に 1 個以上溜まっているなら timeout を待つために抜ける
                    // (上位で sleep / spin)。 0 件なら待っても無駄なので即 break。
                    return self.staging.len();
                }
            }
        }
        self.staging.len()
    }

    /// staging buffer を呼び出し側に渡す (move out)。 内部 staging は空に reset。
    pub fn take_batch(&mut self) -> Vec<Item> {
        self.first_arrival = None;
        std::mem::replace(&mut self.staging, Vec::with_capacity(self.cfg.max_batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_until_max_batch() {
        let cfg = SchedulerConfig {
            max_batch: 4,
            flush_timeout_us: 100_000_000,
            idle_spin_limit: 0,
        };
        let mut sched = BatchScheduler::<u32>::new(cfg);
        let q = BatchQueue::with_capacity(16);
        for i in 0..10 {
            q.push(i).unwrap();
        }
        let n = sched.collect(&q);
        assert_eq!(n, 4);
        assert!(sched.should_flush());
        let batch = sched.take_batch();
        assert_eq!(batch, vec![0, 1, 2, 3]);
        assert_eq!(sched.pending(), 0);
    }

    #[test]
    fn collect_returns_on_empty() {
        let cfg = SchedulerConfig {
            max_batch: 8,
            flush_timeout_us: 100_000_000,
            idle_spin_limit: 0,
        };
        let mut sched = BatchScheduler::<u32>::new(cfg);
        let q = BatchQueue::with_capacity(16);
        q.push(1).unwrap();
        q.push(2).unwrap();
        let n = sched.collect(&q);
        assert_eq!(n, 2);
        // max に達していないし timeout も来ていない
        assert!(!sched.should_flush());
    }

    #[test]
    fn flush_timeout_triggers() {
        let cfg = SchedulerConfig {
            max_batch: 1024,
            flush_timeout_us: 1, // 即時 timeout
            idle_spin_limit: 0,
        };
        let mut sched = BatchScheduler::<u32>::new(cfg);
        let q = BatchQueue::with_capacity(8);
        q.push(7).unwrap();
        let _ = sched.collect(&q);
        // 1us はほぼ即時超過
        std::thread::sleep(std::time::Duration::from_micros(50));
        assert!(sched.should_flush());
    }
}
