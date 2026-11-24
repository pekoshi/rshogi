//! Lock-free MPMC queue for batch inference items.
//!
//! `crossbeam-queue::ArrayQueue` をベースに、 batch evaluator 用の
//! request/response 型を流す。 push/pop は allocation-free。
//!
//! ## 設計
//!
//! - **bounded** (容量 fixed、 push 失敗時は `QueuePushError::Full` を返す)。
//! - **lock-free MPMC** (multiple producer / multiple consumer)。
//! - **false sharing 回避**: queue 自体は `ArrayQueue` に委譲。 ユーザ側の
//!   counter は [`AlignedAtomicU64`] (64-byte align) で囲む。
//! - **no allocation per push**: `ArrayQueue::push` は内部 ring buffer の
//!   slot を上書きするのみ。

use std::sync::atomic::AtomicU64;

use crossbeam_queue::ArrayQueue;

/// queue push 時のエラー。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueuePushError {
    /// 容量満杯。 呼び出し側で backoff or drop を選ぶ。
    Full,
}

/// queue pop 時のエラー。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueuePopError {
    /// 空。 scheduler 側で flush_timeout で待つか sleep する。
    Empty,
}

/// false sharing 回避用、 64-byte aligned atomic。
///
/// scheduler / worker 間で共有する flag / counter はこの型でラップして
/// cache line 跨ぎを避ける。
#[repr(C, align(64))]
pub struct AlignedAtomicU64 {
    pub value: AtomicU64,
    /// padding は align(64) で自動付与されるが、 明示しておくと意図が明確。
    _pad: [u8; 0],
}

impl AlignedAtomicU64 {
    pub const fn new(v: u64) -> Self {
        Self {
            value: AtomicU64::new(v),
            _pad: [],
        }
    }
}

impl Default for AlignedAtomicU64 {
    fn default() -> Self {
        Self::new(0)
    }
}

/// MPMC bounded queue。 任意の `T: Send` を運べる thin wrapper。
///
/// 内部は `crossbeam_queue::ArrayQueue` (lock-free)。 `T` が `Copy` でない
/// 場合も push/pop で move されるだけで heap alloc は発生しない (`ArrayQueue`
/// が内部 ring slot を保持)。
pub struct BatchQueue<T> {
    inner: ArrayQueue<T>,
}

impl<T> BatchQueue<T> {
    /// 容量 `cap` の queue を確保。 `cap` 個分の slot が事前 allocation される
    /// (これは構築時のみ、 push/pop で allocation は発生しない)。
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: ArrayQueue::new(cap),
        }
    }

    /// 容量。
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// 現在の要素数。
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 空か。
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// push (allocation-free)。 失敗時は `QueuePushError::Full` と一緒に元の値を返す。
    pub fn push(&self, value: T) -> Result<(), (QueuePushError, T)> {
        match self.inner.push(value) {
            Ok(()) => Ok(()),
            Err(v) => Err((QueuePushError::Full, v)),
        }
    }

    /// pop。 空なら `Empty`。
    pub fn pop(&self) -> Result<T, QueuePopError> {
        self.inner.pop().ok_or(QueuePopError::Empty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn push_pop_roundtrip() {
        let q: BatchQueue<u32> = BatchQueue::with_capacity(8);
        assert!(q.is_empty());
        assert_eq!(q.capacity(), 8);
        q.push(42).unwrap();
        q.push(7).unwrap();
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop().unwrap(), 42);
        assert_eq!(q.pop().unwrap(), 7);
        assert_eq!(q.pop().unwrap_err(), QueuePopError::Empty);
    }

    #[test]
    fn push_full_returns_value() {
        let q: BatchQueue<u32> = BatchQueue::with_capacity(2);
        q.push(1).unwrap();
        q.push(2).unwrap();
        let (e, v) = q.push(3).unwrap_err();
        assert_eq!(e, QueuePushError::Full);
        assert_eq!(v, 3);
    }

    #[test]
    fn mpmc_smoke() {
        let q: Arc<BatchQueue<u32>> = Arc::new(BatchQueue::with_capacity(1024));
        let n_producers = 4;
        let n_items = 100usize;
        let mut handles = Vec::new();
        for p in 0..n_producers {
            let q = q.clone();
            handles.push(thread::spawn(move || {
                for i in 0..n_items {
                    while q.push((p * 10_000 + i) as u32).is_err() {
                        std::hint::spin_loop();
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let mut count = 0;
        while q.pop().is_ok() {
            count += 1;
        }
        assert_eq!(count, n_producers * n_items);
    }

    #[test]
    fn aligned_atomic_alignment() {
        let a = AlignedAtomicU64::new(0);
        let addr = &a as *const _ as usize;
        assert_eq!(addr % 64, 0, "AlignedAtomicU64 must be 64-byte aligned");
    }
}
