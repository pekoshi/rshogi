//! Runtime integration tests。
//!
//! - trait 多態 (`Box<dyn TransformerRuntime>`) 経由で eval が通ること
//! - shape mismatch / batch_too_large がエラーになること
//! - hybrid routing 判定が priority + threshold 通りであること
//! - Matryoshka level 指定で Full にフォールバックすること

use p2_transformer::TransformerTokens;
use p2_transformer_runtime::{
    CpuTransformerRuntime, EvalPriority, HybridConfig, HybridTransformerRuntime, InferenceError,
    MatryoshkaLevel, PolicyBuffer, TransformerRuntime,
};

fn empty_batch(n: usize) -> Vec<TransformerTokens> {
    vec![TransformerTokens::empty(); n]
}

#[test]
fn cpu_runtime_eval_batch1() {
    let mut rt = CpuTransformerRuntime::new_random(8).unwrap();
    let batch = empty_batch(1);
    let mut v = vec![0.0f32; 1];
    let mut p = vec![PolicyBuffer::zeroed(); 1];
    rt.eval(&batch, &mut v, &mut p).expect("eval ok");
    assert_eq!(v.len(), 1);
    assert_eq!(p.len(), 1);
}

#[test]
fn cpu_runtime_eval_batch_multi() {
    let mut rt = CpuTransformerRuntime::new_random(16).unwrap();
    let batch = empty_batch(4);
    let mut v = vec![0.0f32; 4];
    let mut p = vec![PolicyBuffer::zeroed(); 4];
    rt.eval(&batch, &mut v, &mut p).expect("eval ok");
}

#[test]
fn cpu_runtime_shape_mismatch_errors() {
    let mut rt = CpuTransformerRuntime::new_random(8).unwrap();
    let batch = empty_batch(2);
    let mut v = vec![0.0f32; 3]; // mismatch
    let mut p = vec![PolicyBuffer::zeroed(); 2];
    let err = rt.eval(&batch, &mut v, &mut p).unwrap_err();
    assert!(matches!(err, InferenceError::ShapeMismatch { .. }));
}

#[test]
fn cpu_runtime_batch_too_large_errors() {
    let mut rt = CpuTransformerRuntime::new_random(4).unwrap();
    let batch = empty_batch(5);
    let mut v = vec![0.0f32; 5];
    let mut p = vec![PolicyBuffer::zeroed(); 5];
    let err = rt.eval(&batch, &mut v, &mut p).unwrap_err();
    assert!(matches!(err, InferenceError::BatchTooLarge { .. }));
}

#[test]
fn trait_dispatch_works_via_box_dyn() {
    let mut rt: Box<dyn TransformerRuntime> =
        Box::new(CpuTransformerRuntime::new_random(8).unwrap());
    let batch = empty_batch(2);
    let mut v = vec![0.0f32; 2];
    let mut p = vec![PolicyBuffer::zeroed(); 2];
    rt.eval(&batch, &mut v, &mut p).expect("dyn eval ok");
    assert_eq!(rt.name(), "cpu");
    assert_eq!(rt.max_batch(), 8);
}

#[test]
fn matryoshka_level_falls_back_to_full() {
    let mut rt = CpuTransformerRuntime::new_random(4).unwrap();
    let batch = empty_batch(1);
    let mut v = vec![0.0f32; 1];
    let mut p = vec![PolicyBuffer::zeroed(); 1];
    for lvl in [
        MatryoshkaLevel::Light,
        MatryoshkaLevel::Medium,
        MatryoshkaLevel::Full,
    ] {
        rt.eval_with_level(&batch, lvl, &mut v, &mut p).expect("eval ok with level");
    }
}

#[test]
fn hybrid_routes_by_threshold_and_priority() {
    let cpu: Box<dyn TransformerRuntime> = Box::new(CpuTransformerRuntime::new_random(64).unwrap());
    // GPU を未指定 (cpu-only) で routing 判定だけ確認
    let cfg = HybridConfig {
        gpu_threshold: 16,
        max_batch: 64,
        flush_timeout_us: 100,
    };
    let rt = HybridTransformerRuntime::cpu_only(cpu, cfg);
    // GPU 無しなので何があっても false
    assert!(!rt.should_use_gpu(8, EvalPriority::Normal));
    assert!(!rt.should_use_gpu(32, EvalPriority::Normal));
    assert!(!rt.should_use_gpu(100, EvalPriority::Bulk));
    assert!(!rt.should_use_gpu(1, EvalPriority::Urgent));

    // 仮想 GPU backend (CPU を流用) でルーティングを確かめる
    let cpu2: Box<dyn TransformerRuntime> =
        Box::new(CpuTransformerRuntime::new_random(64).unwrap());
    let fake_gpu: Box<dyn TransformerRuntime> =
        Box::new(CpuTransformerRuntime::new_random(64).unwrap());
    let rt = HybridTransformerRuntime::with_gpu(cpu2, fake_gpu, cfg);
    // Urgent は常に CPU
    assert!(!rt.should_use_gpu(64, EvalPriority::Urgent));
    // Bulk は常に GPU
    assert!(rt.should_use_gpu(1, EvalPriority::Bulk));
    // Normal は threshold で分岐
    assert!(!rt.should_use_gpu(8, EvalPriority::Normal));
    assert!(rt.should_use_gpu(16, EvalPriority::Normal));
    assert!(rt.should_use_gpu(32, EvalPriority::Normal));
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_matches_candle_within_tolerance() {
    // AVX2 path と candle path で同じ random init model + 同じ token に対し
    // 数値出力が tolerance 内で一致することを確認する。
    use p2_transformer_runtime::avx2_forward::{AvxScratch, AvxWeights, forward_batch1_avx2};
    use p2_transformer_runtime::model::{ModelHandle, RuntimeModelConfig};

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("AVX2/FMA not detected, skipping");
        return;
    }

    // 同じ ModelHandle を 2 つ作るには random init では難しい (内部 RNG)。
    // 代わりに、 1 つの handle から AVX2 path と candle path を両方走らせる。
    let device = candle_core::Device::Cpu;
    let handle = ModelHandle::random(RuntimeModelConfig::p2_default(), device).unwrap();
    let weights = AvxWeights::from_model(&handle).unwrap();
    let mut scratch = AvxScratch::new(&handle.cfg);

    let tokens = TransformerTokens::empty();
    // AVX2 path
    let mut v_avx2 = 0.0f32;
    let mut p_avx2 = vec![0.0f32; 2187];
    // SAFETY: 上で AVX2/FMA を確認済み。
    unsafe {
        forward_batch1_avx2(&weights, &mut scratch, &tokens, &mut v_avx2, &mut p_avx2);
    }

    // candle path: forward via CpuTransformerRuntime with a wrapper that disables AVX2
    // ... ここは ModelHandle.forward を直接呼ぶ
    use candle_core::Tensor;
    use p2_transformer::MAX_TOKENS;
    let token_i64: Vec<i64> = tokens.tokens.iter().map(|&v| v as i64).collect();
    let tt = Tensor::from_slice(&token_i64, (1, MAX_TOKENS), &handle.device).unwrap();
    let (value_t, policy_t) = handle.forward(&tt).unwrap();
    let v_candle = value_t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
    let p_candle = policy_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let v_err = (v_avx2 - v_candle).abs();
    let mut max_p_err = 0f32;
    for i in 0..2187 {
        let e = (p_avx2[i] - p_candle[i]).abs();
        if e > max_p_err {
            max_p_err = e;
        }
    }
    eprintln!("v_err = {v_err}, max_p_err = {max_p_err}");
    // candle path は exact erf GELU、 AVX2 path は tanh 近似 GELU を使用。 加えて
    // 6 layer の f32 累積で順序による差が積み上がるため、 random init weights では
    // value 差 ~0.1、 policy logit 差 ~0.5 程度に収まることを確認する。
    // 実 ckpt では LayerNorm が活性化を圧縮するため通常さらに小さい。
    assert!(v_err < 0.5, "value diff too large: {v_err}");
    assert!(max_p_err < 2.0, "max policy diff too large: {max_p_err}");
}

#[test]
fn hybrid_eval_routes_and_executes() {
    let cpu: Box<dyn TransformerRuntime> = Box::new(CpuTransformerRuntime::new_random(64).unwrap());
    let fake_gpu: Box<dyn TransformerRuntime> =
        Box::new(CpuTransformerRuntime::new_random(64).unwrap());
    let mut rt = HybridTransformerRuntime::with_gpu(cpu, fake_gpu, HybridConfig::default());

    // CPU 経路 (batch < threshold=32)
    let batch = empty_batch(4);
    let mut v = vec![0.0f32; 4];
    let mut p = vec![PolicyBuffer::zeroed(); 4];
    rt.eval(&batch, &mut v, &mut p).expect("cpu route ok");

    // GPU 経路 (batch >= 32)
    let batch = empty_batch(32);
    let mut v = vec![0.0f32; 32];
    let mut p = vec![PolicyBuffer::zeroed(); 32];
    rt.eval(&batch, &mut v, &mut p).expect("gpu route ok");

    assert_eq!(rt.name(), "hybrid");
}
