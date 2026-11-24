//! CPU inference benchmarks (criterion)。
//!
//! 計測対象:
//! - batch=1   (root / urgent path)
//! - batch=8   (small mid-batch)
//! - batch=32  (hybrid threshold 付近)
//!
//! GPU bench は `routing.rs` 側で routing overhead を測り、 実 GPU 推論は
//! `cuda` feature 有効時のみ別 bench を追加する想定。

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use p2_transformer::TransformerTokens;
use p2_transformer_runtime::{CpuTransformerRuntime, PolicyBuffer, TransformerRuntime};

fn make_runtime(max_batch: usize) -> CpuTransformerRuntime {
    CpuTransformerRuntime::new_random(max_batch).expect("cpu runtime init")
}

fn bench_batch(c: &mut Criterion, label: &str, batch_size: usize) {
    let mut rt = make_runtime(batch_size.max(64));
    let batch = vec![TransformerTokens::empty(); batch_size];
    let mut out_value = vec![0.0f32; batch_size];
    let mut out_policy = vec![PolicyBuffer::zeroed(); batch_size];

    c.bench_function(label, |b| {
        b.iter(|| {
            rt.eval(black_box(&batch), black_box(&mut out_value), black_box(&mut out_policy))
                .expect("eval");
        })
    });
}

fn bench_batch1(c: &mut Criterion) {
    bench_batch(c, "cpu_batch1", 1);
}

fn bench_batch8(c: &mut Criterion) {
    bench_batch(c, "cpu_batch8", 8);
}

fn bench_batch32(c: &mut Criterion) {
    bench_batch(c, "cpu_batch32", 32);
}

criterion_group!(benches, bench_batch1, bench_batch8, bench_batch32);
criterion_main!(benches);
