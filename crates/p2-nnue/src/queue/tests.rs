//! MpmcQueue の単体テスト + マルチスレッドストレステスト

use std::collections::HashSet;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use crate::features::{P2_FEATURE_DIM, P2NnueInput};
use crate::queue::ring_buffer::{MpmcQueue, QueueEmpty, QueueFull};
use crate::queue::types::{EvalRequest, EvalResult};

// ===========================================================================
// 単体テスト (single-thread)
// ===========================================================================

#[test]
fn new_power_of_two_succeeds() {
    for &cap in &[1, 2, 4, 8, 16, 64, 1024] {
        let q = MpmcQueue::<u64>::new(cap);
        assert!(q.is_ok(), "capacity {} should succeed", cap);
        assert_eq!(q.unwrap().capacity(), cap);
    }
}

#[test]
fn new_invalid_size_fails() {
    for &cap in &[0, 3, 5, 6, 7, 9, 10, 100] {
        let q = MpmcQueue::<u64>::new(cap);
        assert!(q.is_err(), "capacity {} should fail", cap);
    }
}

#[test]
fn push_pop_single_element() {
    let q = MpmcQueue::<u64>::new(4).unwrap();
    assert!(q.try_push(42).is_ok());
    assert_eq!(q.try_pop().unwrap(), 42);
}

#[test]
fn fifo_ordering() {
    let q = MpmcQueue::<u64>::new(8).unwrap();
    for i in 0..8 {
        assert!(q.try_push(i).is_ok());
    }
    for i in 0..8 {
        assert_eq!(q.try_pop().unwrap(), i);
    }
}

#[test]
fn push_full_returns_queue_full() {
    let q = MpmcQueue::<u64>::new(4).unwrap();
    for i in 0..4 {
        assert!(q.try_push(i).is_ok());
    }
    match q.try_push(99) {
        Err(QueueFull(v)) => assert_eq!(v, 99),
        Ok(()) => panic!("expected QueueFull"),
    }
}

#[test]
fn pop_empty_returns_queue_empty() {
    let q = MpmcQueue::<u64>::new(4).unwrap();
    assert!(matches!(q.try_pop(), Err(QueueEmpty)));
}

#[test]
fn wraparound_three_times() {
    let cap = 8;
    let q = MpmcQueue::<u64>::new(cap).unwrap();
    let total = cap * 3;
    for i in 0..total {
        // 1 要素ずつ push → pop を繰り返す
        assert!(q.try_push(i as u64).is_ok());
        assert_eq!(q.try_pop().unwrap(), i as u64);
    }
}

#[test]
fn queue_len_accuracy() {
    let q = MpmcQueue::<u64>::new(8).unwrap();
    assert_eq!(q.queue_len(), 0);
    for i in 0..5 {
        q.try_push(i).unwrap();
        assert_eq!(q.queue_len(), (i + 1) as usize);
    }
    for i in 0..3 {
        q.try_pop().unwrap();
        assert_eq!(q.queue_len(), 5 - (i + 1) as usize);
    }
}

#[test]
fn is_empty_is_full_boundary() {
    let q = MpmcQueue::<u64>::new(4).unwrap();
    assert!(q.is_empty());
    assert!(!q.is_full());

    for i in 0..4 {
        q.try_push(i).unwrap();
    }
    assert!(!q.is_empty());
    assert!(q.is_full());

    q.try_pop().unwrap();
    assert!(!q.is_empty());
    assert!(!q.is_full());
}

#[test]
fn pop_batch_basic() {
    let q = MpmcQueue::<u64>::new(8).unwrap();
    for i in 0..5 {
        q.try_push(i).unwrap();
    }

    let mut buf = [MaybeUninit::<u64>::uninit(); 8];
    let n = q.pop_batch(&mut buf);
    assert_eq!(n, 5);
    for (i, item) in buf.iter().enumerate().take(5) {
        assert_eq!(unsafe { item.assume_init() }, i as u64);
    }
}

#[test]
fn pop_batch_partial() {
    let q = MpmcQueue::<u64>::new(8).unwrap();
    for i in 0..3 {
        q.try_push(i).unwrap();
    }

    let mut buf = [MaybeUninit::<u64>::uninit(); 8];
    let n = q.pop_batch(&mut buf);
    assert_eq!(n, 3);
}

#[test]
fn pop_batch_empty_queue() {
    let q = MpmcQueue::<u64>::new(8).unwrap();
    let mut buf = [MaybeUninit::<u64>::uninit(); 4];
    assert_eq!(q.pop_batch(&mut buf), 0);
}

#[test]
fn drop_cleans_remaining_elements() {
    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    struct DropCounter;
    impl Drop for DropCounter {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    }

    DROP_COUNT.store(0, Ordering::Relaxed);
    {
        let q = MpmcQueue::<DropCounter>::new(8).unwrap();
        for _ in 0..5 {
            q.try_push(DropCounter).unwrap();
        }
        // q がスコープを抜けるとき残存要素の Drop が呼ばれる
    }
    assert_eq!(DROP_COUNT.load(Ordering::Relaxed), 5);
}

#[test]
fn eval_request_result_roundtrip() {
    let q_req = MpmcQueue::<EvalRequest>::new(4).unwrap();
    let req = EvalRequest {
        request_id: 12345,
        node_id: 99,
        nnue_input: P2NnueInput {
            piece_features: Box::new([0i32; P2_FEATURE_DIM]),
            side_to_move: 1,
            game_phase: 128,
            king_bucket: 7,
        },
    };
    q_req.try_push(req).unwrap();
    let got = q_req.try_pop().unwrap();
    assert_eq!(got.request_id, 12345);
    assert_eq!(got.node_id, 99);
    assert_eq!(got.nnue_input.side_to_move, 1);
    assert_eq!(got.nnue_input.game_phase, 128);
    assert_eq!(got.nnue_input.king_bucket, 7);

    let q_res = MpmcQueue::<EvalResult>::new(4).unwrap();
    let res = EvalResult {
        request_id: 12345,
        node_id: 99,
        value: -300,
    };
    q_res.try_push(res).unwrap();
    let got = q_res.try_pop().unwrap();
    assert_eq!(got.request_id, 12345);
    assert_eq!(got.value, -300);
}

// ===========================================================================
// マルチスレッドストレステスト
// ===========================================================================

const STRESS_ITEMS: usize = 100_000;

/// 全要素が正確に 1 回受信されたことを検証する
fn verify_all_received(received: &[u64], total: usize) {
    let set: HashSet<u64> = received.iter().copied().collect();
    assert_eq!(
        set.len(),
        total,
        "重複あり: received {} unique out of {} total",
        set.len(),
        total
    );
    assert_eq!(received.len(), total, "要素数不一致");
    for i in 0..total as u64 {
        assert!(set.contains(&i), "要素 {} が欠落", i);
    }
}

#[test]
fn stress_spsc() {
    let q = Arc::new(MpmcQueue::<u64>::new(1024).unwrap());

    let q_tx = Arc::clone(&q);
    let producer = thread::spawn(move || {
        for i in 0..STRESS_ITEMS as u64 {
            while q_tx.try_push(i).is_err() {
                std::hint::spin_loop();
            }
        }
    });

    let q_rx = Arc::clone(&q);
    let consumer = thread::spawn(move || {
        let mut received = Vec::with_capacity(STRESS_ITEMS);
        while received.len() < STRESS_ITEMS {
            if let Ok(v) = q_rx.try_pop() {
                received.push(v);
            } else {
                std::hint::spin_loop();
            }
        }
        received
    });

    producer.join().unwrap();
    let received = consumer.join().unwrap();
    verify_all_received(&received, STRESS_ITEMS);
}

#[test]
fn stress_mpsc() {
    let num_producers = 4;
    let items_per_producer = STRESS_ITEMS / num_producers;
    let q = Arc::new(MpmcQueue::<u64>::new(1024).unwrap());

    let producers: Vec<_> = (0..num_producers)
        .map(|p| {
            let q = Arc::clone(&q);
            thread::spawn(move || {
                let base = (p * items_per_producer) as u64;
                for i in 0..items_per_producer as u64 {
                    while q.try_push(base + i).is_err() {
                        std::hint::spin_loop();
                    }
                }
            })
        })
        .collect();

    let q_rx = Arc::clone(&q);
    let total = num_producers * items_per_producer;
    let consumer = thread::spawn(move || {
        let mut received = Vec::with_capacity(total);
        while received.len() < total {
            if let Ok(v) = q_rx.try_pop() {
                received.push(v);
            } else {
                std::hint::spin_loop();
            }
        }
        received
    });

    for p in producers {
        p.join().unwrap();
    }
    let received = consumer.join().unwrap();
    verify_all_received(&received, total);
}

#[test]
fn stress_spmc() {
    let num_consumers = 4;
    let q = Arc::new(MpmcQueue::<u64>::new(1024).unwrap());
    let done = Arc::new(AtomicUsize::new(0));

    let q_tx = Arc::clone(&q);
    let producer = thread::spawn(move || {
        for i in 0..STRESS_ITEMS as u64 {
            while q_tx.try_push(i).is_err() {
                std::hint::spin_loop();
            }
        }
    });

    let consumers: Vec<_> = (0..num_consumers)
        .map(|_| {
            let q = Arc::clone(&q);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut local = Vec::new();
                loop {
                    match q.try_pop() {
                        Ok(v) => {
                            local.push(v);
                            done.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(QueueEmpty) => {
                            if done.load(Ordering::Relaxed) >= STRESS_ITEMS {
                                break;
                            }
                            std::hint::spin_loop();
                        }
                    }
                }
                local
            })
        })
        .collect();

    producer.join().unwrap();
    let mut all: Vec<u64> = Vec::with_capacity(STRESS_ITEMS);
    for c in consumers {
        all.extend(c.join().unwrap());
    }
    verify_all_received(&all, STRESS_ITEMS);
}

#[test]
fn stress_mpmc() {
    let num_producers = 4;
    let num_consumers = 4;
    let items_per_producer = STRESS_ITEMS / num_producers;
    let total = num_producers * items_per_producer;
    let q = Arc::new(MpmcQueue::<u64>::new(1024).unwrap());
    let done = Arc::new(AtomicUsize::new(0));

    let producers: Vec<_> = (0..num_producers)
        .map(|p| {
            let q = Arc::clone(&q);
            thread::spawn(move || {
                let base = (p * items_per_producer) as u64;
                for i in 0..items_per_producer as u64 {
                    while q.try_push(base + i).is_err() {
                        std::hint::spin_loop();
                    }
                }
            })
        })
        .collect();

    let consumers: Vec<_> = (0..num_consumers)
        .map(|_| {
            let q = Arc::clone(&q);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut local = Vec::new();
                loop {
                    match q.try_pop() {
                        Ok(v) => {
                            local.push(v);
                            done.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(QueueEmpty) => {
                            if done.load(Ordering::Relaxed) >= total {
                                break;
                            }
                            std::hint::spin_loop();
                        }
                    }
                }
                local
            })
        })
        .collect();

    for p in producers {
        p.join().unwrap();
    }
    let mut all: Vec<u64> = Vec::with_capacity(total);
    for c in consumers {
        all.extend(c.join().unwrap());
    }
    verify_all_received(&all, total);
}

#[test]
fn stress_mpmc_batch() {
    let num_producers = 4;
    let num_consumers = 4;
    let items_per_producer = STRESS_ITEMS / num_producers;
    let total = num_producers * items_per_producer;
    let q = Arc::new(MpmcQueue::<u64>::new(1024).unwrap());
    let done = Arc::new(AtomicUsize::new(0));

    let producers: Vec<_> = (0..num_producers)
        .map(|p| {
            let q = Arc::clone(&q);
            thread::spawn(move || {
                let base = (p * items_per_producer) as u64;
                for i in 0..items_per_producer as u64 {
                    while q.try_push(base + i).is_err() {
                        std::hint::spin_loop();
                    }
                }
            })
        })
        .collect();

    let consumers: Vec<_> = (0..num_consumers)
        .map(|_| {
            let q = Arc::clone(&q);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut local = Vec::new();
                let mut buf = [MaybeUninit::<u64>::uninit(); 32];
                loop {
                    let n = q.pop_batch(&mut buf);
                    if n > 0 {
                        for item in &buf[..n] {
                            local.push(unsafe { item.assume_init() });
                        }
                        done.fetch_add(n, Ordering::Relaxed);
                    } else if done.load(Ordering::Relaxed) >= total {
                        break;
                    } else {
                        std::hint::spin_loop();
                    }
                }
                local
            })
        })
        .collect();

    for p in producers {
        p.join().unwrap();
    }
    let mut all: Vec<u64> = Vec::with_capacity(total);
    for c in consumers {
        all.extend(c.join().unwrap());
    }
    verify_all_received(&all, total);
}

#[test]
fn stress_high_contention() {
    let num_producers = 8;
    let num_consumers = 8;
    let items_per_producer = 50_000 / num_producers;
    let total = num_producers * items_per_producer;
    let q = Arc::new(MpmcQueue::<u64>::new(64).unwrap());
    let done = Arc::new(AtomicUsize::new(0));

    let producers: Vec<_> = (0..num_producers)
        .map(|p| {
            let q = Arc::clone(&q);
            thread::spawn(move || {
                let base = (p * items_per_producer) as u64;
                for i in 0..items_per_producer as u64 {
                    while q.try_push(base + i).is_err() {
                        std::hint::spin_loop();
                    }
                }
            })
        })
        .collect();

    let consumers: Vec<_> = (0..num_consumers)
        .map(|_| {
            let q = Arc::clone(&q);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut local = Vec::new();
                loop {
                    match q.try_pop() {
                        Ok(v) => {
                            local.push(v);
                            done.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(QueueEmpty) => {
                            if done.load(Ordering::Relaxed) >= total {
                                break;
                            }
                            std::hint::spin_loop();
                        }
                    }
                }
                local
            })
        })
        .collect();

    for p in producers {
        p.join().unwrap();
    }
    let mut all: Vec<u64> = Vec::with_capacity(total);
    for c in consumers {
        all.extend(c.join().unwrap());
    }
    verify_all_received(&all, total);
}
