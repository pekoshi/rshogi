//! Routing overhead benchmark.
//!
//! Hybrid runtime の `should_use_gpu` 判定 + dispatch のオーバーヘッドを測る。
//! GPU 未 build 環境では CPU フォールバックのみが測られる (CPU eval の overhead
//! が支配的なので、 routing overhead 自体は実質 0)。

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use p2_transformer::TransformerTokens;
use p2_transformer_runtime::{
    CpuTransformerRuntime, EvalPriority, HybridConfig, HybridTransformerRuntime, MatryoshkaLevel,
    PolicyBuffer, TransformerRuntime,
};

fn build_hybrid() -> HybridTransformerRuntime {
    let cpu: Box<dyn TransformerRuntime> =
        Box::new(CpuTransformerRuntime::new_random(64).expect("cpu init"));
    HybridTransformerRuntime::cpu_only(cpu, HybridConfig::default())
}

fn bench_routing_decision(c: &mut Criterion) {
    let rt = build_hybrid();
    c.bench_function("routing_decision_only", |b| {
        b.iter(|| {
            // routing 判定 (GPU 未 build なので常に false)
            let _ = black_box(rt.should_use_gpu(black_box(8), black_box(EvalPriority::Normal)));
            let _ = black_box(rt.should_use_gpu(black_box(64), black_box(EvalPriority::Normal)));
            let _ = black_box(rt.should_use_gpu(black_box(1), black_box(EvalPriority::Urgent)));
            let _ = black_box(rt.should_use_gpu(black_box(128), black_box(EvalPriority::Bulk)));
        })
    });
}

fn bench_dispatch_via_hybrid(c: &mut Criterion) {
    let mut rt = build_hybrid();
    let batch = vec![TransformerTokens::empty(); 4];
    let mut out_value = vec![0.0f32; 4];
    let mut out_policy = vec![PolicyBuffer::zeroed(); 4];

    c.bench_function("hybrid_dispatch_batch4", |b| {
        b.iter(|| {
            rt.eval_with_priority(
                black_box(&batch),
                EvalPriority::Normal,
                MatryoshkaLevel::Full,
                black_box(&mut out_value),
                black_box(&mut out_policy),
            )
            .expect("eval");
        })
    });
}

criterion_group!(benches, bench_routing_decision, bench_dispatch_via_hybrid);
criterion_main!(benches);
