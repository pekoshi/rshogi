//! Vyukov bounded MPMC ring buffer queue
//!
//! Lock-free な bounded ring buffer。`std::sync::atomic` のみ使用。
//! Mutex/RwLock 不使用、blocking wait なし。

use std::cell::UnsafeCell;
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// エラー型
// ---------------------------------------------------------------------------

/// `try_push` がキュー満杯で失敗した場合に要素を返却する
pub struct QueueFull<T>(pub T);

impl<T> fmt::Debug for QueueFull<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("QueueFull")
    }
}

/// `try_pop` がキュー空で失敗した場合
#[derive(Debug)]
pub struct QueueEmpty;

// ---------------------------------------------------------------------------
// キャッシュライン分離
// ---------------------------------------------------------------------------

/// 64 バイトアラインメントで false sharing を防止
#[repr(C, align(64))]
struct CacheAligned<T> {
    value: T,
}

// ---------------------------------------------------------------------------
// スロット
// ---------------------------------------------------------------------------

/// リングバッファの各スロット
///
/// # Safety
/// `data` は `sequence` によるライフサイクル管理で保護される。
/// push 完了時に sequence = pos+1 (Release) → pop 側が Acquire で読み取ることで
/// データの可視性を保証する。
struct Slot<T> {
    sequence: AtomicUsize,
    data: UnsafeCell<MaybeUninit<T>>,
}

// ---------------------------------------------------------------------------
// MpmcQueue
// ---------------------------------------------------------------------------

/// Lock-free bounded MPMC queue (Vyukov 方式)
///
/// - capacity は 2 の冪乗のみ許可
/// - `try_push` / `try_pop` は non-blocking で即座に `Result` を返す
/// - head と tail は別キャッシュラインに配置し false sharing を防止
pub struct MpmcQueue<T> {
    buffer: Box<[Slot<T>]>,
    mask: usize,
    head: CacheAligned<AtomicUsize>,
    tail: CacheAligned<AtomicUsize>,
}

// SAFETY: Slot 内の UnsafeCell は sequence atomic による排他制御で保護される。
// 同一スロットに同時に書き込む/読み出すスレッドは存在しない。
unsafe impl<T: Send> Send for MpmcQueue<T> {}
// SAFETY: &MpmcQueue 経由の try_push/try_pop は CAS ループにより安全。
unsafe impl<T: Send> Sync for MpmcQueue<T> {}

impl<T> MpmcQueue<T> {
    /// 新しいキューを作成する
    ///
    /// `capacity` は 2 の冪乗でなければならない。
    pub fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err(format!("capacity must be a power of two, got {}", capacity));
        }

        let buffer: Vec<Slot<T>> = (0..capacity)
            .map(|i| Slot {
                sequence: AtomicUsize::new(i),
                data: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect();

        Ok(Self {
            buffer: buffer.into_boxed_slice(),
            mask: capacity - 1,
            head: CacheAligned {
                value: AtomicUsize::new(0),
            },
            tail: CacheAligned {
                value: AtomicUsize::new(0),
            },
        })
    }

    /// 要素をキューに追加する（non-blocking）
    ///
    /// キューが満杯の場合 `QueueFull(value)` で要素を返却する。
    pub fn try_push(&self, value: T) -> Result<(), QueueFull<T>> {
        let mut pos = self.head.value.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq as isize - pos as isize;

            if diff == 0 {
                // スロットが空 — head を進める CAS
                match self.head.value.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: CAS 成功 = このスロットの書き込み権を排他取得済み
                        unsafe {
                            (*slot.data.get()).write(value);
                        }
                        slot.sequence.store(pos.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => {
                        pos = actual;
                    }
                }
            } else if diff < 0 {
                // キュー満杯
                return Err(QueueFull(value));
            } else {
                // 他のプロデューサーが先に進んだ → 再読み込み
                pos = self.head.value.load(Ordering::Relaxed);
            }
        }
    }

    /// 要素をキューから取り出す（non-blocking）
    ///
    /// キューが空の場合 `QueueEmpty` を返す。
    pub fn try_pop(&self) -> Result<T, QueueEmpty> {
        let mut pos = self.tail.value.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq as isize - (pos.wrapping_add(1)) as isize;

            if diff == 0 {
                // データ準備完了 — tail を進める CAS
                match self.tail.value.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: CAS 成功 = このスロットの読み出し権を排他取得済み
                        let value = unsafe { (*slot.data.get()).assume_init_read() };
                        slot.sequence.store(pos.wrapping_add(self.buffer.len()), Ordering::Release);
                        return Ok(value);
                    }
                    Err(actual) => {
                        pos = actual;
                    }
                }
            } else if diff < 0 {
                // キュー空
                return Err(QueueEmpty);
            } else {
                // 他のコンシューマーが先に進んだ → 再読み込み
                pos = self.tail.value.load(Ordering::Relaxed);
            }
        }
    }

    /// 複数要素を一括で取り出す（non-blocking）
    ///
    /// 最大 `out.len()` 個の要素を `out` に書き込み、実際に取り出した個数を返す。
    /// キューが空の場合は 0 を返す。
    pub fn pop_batch(&self, out: &mut [MaybeUninit<T>]) -> usize {
        if out.is_empty() {
            return 0;
        }

        loop {
            let pos = self.tail.value.load(Ordering::Relaxed);

            // 連続する準備完了スロットをスキャン
            let mut count = 0;
            while count < out.len() {
                let slot = &self.buffer[(pos.wrapping_add(count)) & self.mask];
                let seq = slot.sequence.load(Ordering::Acquire);
                if seq != pos.wrapping_add(count).wrapping_add(1) {
                    break;
                }
                count += 1;
            }

            if count == 0 {
                return 0;
            }

            // tail を count 分進める CAS
            match self.tail.value.compare_exchange_weak(
                pos,
                pos.wrapping_add(count),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // CAS 成功 — データ一括読み出し + sequence 更新
                    let cap = self.buffer.len();
                    for (i, dest) in out.iter_mut().enumerate().take(count) {
                        let slot = &self.buffer[(pos.wrapping_add(i)) & self.mask];
                        // SAFETY: CAS 成功 = この区間の読み出し権を排他取得済み
                        unsafe {
                            dest.write((*slot.data.get()).assume_init_read());
                        }
                        slot.sequence
                            .store(pos.wrapping_add(i).wrapping_add(cap), Ordering::Release);
                    }
                    return count;
                }
                Err(_) => {
                    // 他のコンシューマーが割り込んだ → ループ先頭へ
                    continue;
                }
            }
        }
    }

    /// キュー内の要素数の近似値を返す
    ///
    /// マルチスレッド環境では正確ではない場合がある。
    pub fn queue_len(&self) -> usize {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    /// キューの容量を返す
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// キューが空かどうかの近似判定
    pub fn is_empty(&self) -> bool {
        self.queue_len() == 0
    }

    /// キューが満杯かどうかの近似判定
    pub fn is_full(&self) -> bool {
        self.queue_len() >= self.buffer.len()
    }
}

impl<T> Drop for MpmcQueue<T> {
    fn drop(&mut self) {
        // 残存要素を安全に破棄する
        while self.try_pop().is_ok() {}
    }
}
