//! AVX2 forward 内部の段階別実行時間を計測する。
//!
//! 各段階を個別に何回も呼ぶことで、 全体に対する寄与を観察する。

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::arch::is_x86_feature_detected;

use p2_transformer::TransformerTokens;
use p2_transformer_runtime::{CpuTransformerRuntime, PolicyBuffer, TransformerRuntime};

fn full_forward(c: &mut Criterion) {
    let mut rt = CpuTransformerRuntime::new_random(1).expect("rt");
    let batch = vec![TransformerTokens::empty(); 1];
    let mut v = vec![0f32; 1];
    let mut p = vec![PolicyBuffer::zeroed(); 1];
    c.bench_function("avx2_full_forward", |b| {
        b.iter(|| {
            rt.eval(black_box(&batch), black_box(&mut v), black_box(&mut p)).unwrap();
        })
    });
}

fn gemm_kernel_bench(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    use p2_transformer_runtime::avx2::matmul::{gemm_a_bt, pack_weight_for_gemm};
    // Q+K+V fused
    let m = 116;
    let n = 576; // 3 * 192
    let k = 192;
    let a = vec![0.5f32; m * k];
    let w = vec![0.3f32; n * k];
    let mut wp = vec![0f32; n * k];
    pack_weight_for_gemm(&w, &mut wp, n, k);
    let mut out = vec![0f32; m * n];
    c.bench_function("gemm_qkv_116x192x576", |bb| {
        bb.iter(|| unsafe { gemm_a_bt(&a, &wp, &mut out, None, m, n, k) });
    });

    // FFN1
    let n2 = 768;
    let w2 = vec![0.3f32; n2 * k];
    let mut wp2 = vec![0f32; n2 * k];
    pack_weight_for_gemm(&w2, &mut wp2, n2, k);
    let mut out2 = vec![0f32; m * n2];
    let bias = vec![0.1f32; n2];
    c.bench_function("gemm_ffn1_116x192x768", |bb| {
        bb.iter(|| unsafe { gemm_a_bt(&a, &wp2, &mut out2, Some(&bias), m, n2, k) });
    });

    // FFN2
    let m3 = 116;
    let n3 = 192;
    let k3 = 768;
    let a3 = vec![0.5f32; m3 * k3];
    let w3 = vec![0.3f32; n3 * k3];
    let mut wp3 = vec![0f32; n3 * k3];
    pack_weight_for_gemm(&w3, &mut wp3, n3, k3);
    let mut out3 = vec![0f32; m3 * n3];
    let bias3 = vec![0.1f32; n3];
    c.bench_function("gemm_ffn2_116x768x192", |bb| {
        bb.iter(|| unsafe { gemm_a_bt(&a3, &wp3, &mut out3, Some(&bias3), m3, n3, k3) });
    });

    // Policy head
    let m4 = 1;
    let n4 = 2187;
    let k4 = 192;
    let a4 = vec![0.5f32; m4 * k4];
    let w4 = vec![0.3f32; n4 * k4];
    let mut wp4 = vec![0f32; n4 * k4];
    pack_weight_for_gemm(&w4, &mut wp4, n4, k4);
    let mut out4 = vec![0f32; m4 * n4];
    let bias4 = vec![0.1f32; n4];
    c.bench_function("gemm_policy_1x192x2187", |bb| {
        bb.iter(|| unsafe { gemm_a_bt(&a4, &wp4, &mut out4, Some(&bias4), m4, n4, k4) });
    });
}

fn attention_bench(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    use p2_transformer_runtime::avx2::attention::mha_avx2;
    let t = 116;
    let n_heads = 6;
    let d_head = 32;
    let d_model = n_heads * d_head;
    let q = vec![0.1f32; t * d_model];
    let k = vec![0.1f32; t * d_model];
    let v = vec![0.1f32; t * d_model];
    let mut out = vec![0f32; t * d_model];
    let mut scratch = vec![0f32; n_heads * t * t];
    c.bench_function("attention_mha_116x6x32", |bb| {
        bb.iter(|| unsafe { mha_avx2(&q, &k, &v, &mut out, &mut scratch, t, t, n_heads, d_head) });
    });
}

criterion_group!(benches, full_forward, gemm_kernel_bench, attention_bench);
criterion_main!(benches);
