//! SIMD / バックエンド等価性テスト

use std::sync::Arc;

use crate::align::Aligned;
use crate::evaluator::{CpuEvaluator, NnueEvaluator};
use crate::features::{P2_FEATURE_DIM, P2NnueInput};
use crate::layers::{AffineTransform, ClippedReLU, InputAffineTransform, scalar_reference};
use crate::network::P2Network;
use crate::types::{P2_INPUT_DIM, P2_L1_DIM, P2_L2_DIM, P2_OUTPUT_DIM};

/// 簡易 PRNG (xorshift32)
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// ランダム i32 生成
fn rand_i32(state: &mut u32) -> i32 {
    xorshift32(state) as i32
}

// ─── InputAffineTransform 等価性 ───

#[test]
fn test_input_affine_simd_equivalence() {
    let mut rng = 12345u32;
    let mut layer = InputAffineTransform::<{ P2_INPUT_DIM }, { P2_L1_DIM }>::new_zeroed();

    // ランダム重みとバイアス
    for b in layer.biases.iter_mut() {
        *b = (rand_i32(&mut rng) % 10000) - 5000;
    }
    let padded = InputAffineTransform::<{ P2_INPUT_DIM }, { P2_L1_DIM }>::PADDED_IN;
    for j in 0..P2_L1_DIM {
        for i in 0..P2_INPUT_DIM {
            layer.weights[j * padded + i] = (rand_i32(&mut rng) % 100) as i16;
        }
    }

    // ランダム入力 (0/1 バイナリ主体 + 少数の大きい値)
    let mut input = [0i32; P2_INPUT_DIM];
    for x in input.iter_mut() {
        *x = if xorshift32(&mut rng).is_multiple_of(3) {
            1
        } else {
            0
        };
    }

    // SIMD/スカラーの propagate 結果
    let mut output_simd = Aligned([0i32; P2_L1_DIM]);
    layer.propagate(&input, &mut output_simd.0);

    // スカラー参照結果
    let mut output_ref = [0i32; P2_L1_DIM];
    scalar_reference::input_affine_propagate(
        &*layer.biases,
        &layer.weights,
        padded,
        &input,
        &mut output_ref,
    );

    for (j, (&simd_val, &ref_val)) in output_simd.0.iter().zip(output_ref.iter()).enumerate() {
        assert_eq!(simd_val, ref_val, "InputAffineTransform mismatch at output[{j}]");
    }
}

// ─── AffineTransform (L2: 512→256) 等価性 ───

#[test]
fn test_affine_l2_simd_equivalence() {
    let mut rng = 54321u32;

    // スクランブルされていない行優先レイアウトでスカラー参照用の重みを作成
    let padded = AffineTransform::<{ P2_L1_DIM }, { P2_L2_DIM }>::PADDED_IN;
    let mut biases = [0i32; P2_L2_DIM];
    for b in biases.iter_mut() {
        *b = (rand_i32(&mut rng) % 10000) - 5000;
    }
    let mut weights_raw = vec![0i8; P2_L2_DIM * padded];
    for j in 0..P2_L2_DIM {
        for i in 0..P2_L1_DIM {
            weights_raw[j * padded + i] = (rand_i32(&mut rng) % 50) as i8;
        }
    }

    // AffineTransform を read() 経由で構築（スクランブルが適用される）
    let mut binary = Vec::new();
    for &b in &biases {
        binary.extend_from_slice(&b.to_le_bytes());
    }
    for j in 0..P2_L2_DIM {
        for i in 0..P2_L1_DIM {
            binary.push(weights_raw[j * padded + i] as u8);
        }
    }
    let layer =
        AffineTransform::<{ P2_L1_DIM }, { P2_L2_DIM }>::read(&mut binary.as_slice()).unwrap();

    // ランダム u8 入力
    let mut input = [0u8; P2_L1_DIM];
    for x in input.iter_mut() {
        *x = (xorshift32(&mut rng) % 128) as u8;
    }

    // SIMD/スカラーの propagate 結果
    let mut output_simd = Aligned([0i32; P2_L2_DIM]);
    layer.propagate(&input, &mut output_simd.0);

    // スカラー参照結果 (行優先重みを使用)
    let mut output_ref = [0i32; P2_L2_DIM];
    scalar_reference::affine_propagate(&biases, &weights_raw, padded, &input, &mut output_ref);

    for (j, (&simd_val, &ref_val)) in output_simd.0.iter().zip(output_ref.iter()).enumerate() {
        assert_eq!(simd_val, ref_val, "AffineTransform L2 mismatch at output[{j}]");
    }
}

// ─── AffineTransform (L3: 256→1) 等価性 ───

#[test]
fn test_affine_l3_simd_equivalence() {
    let mut rng = 99999u32;

    let padded = AffineTransform::<{ P2_L2_DIM }, { P2_OUTPUT_DIM }>::PADDED_IN;
    let mut biases = [0i32; P2_OUTPUT_DIM];
    biases[0] = (rand_i32(&mut rng) % 10000) - 5000;
    let mut weights_raw = vec![0i8; P2_OUTPUT_DIM * padded];
    for w in weights_raw.iter_mut().take(P2_L2_DIM) {
        *w = (rand_i32(&mut rng) % 50) as i8;
    }

    // read() 経由で構築
    let mut binary = Vec::new();
    for &b in &biases {
        binary.extend_from_slice(&b.to_le_bytes());
    }
    for &w in weights_raw.iter().take(P2_L2_DIM) {
        binary.push(w as u8);
    }
    let layer =
        AffineTransform::<{ P2_L2_DIM }, { P2_OUTPUT_DIM }>::read(&mut binary.as_slice()).unwrap();

    // ランダム u8 入力
    let mut input = [0u8; P2_L2_DIM];
    for x in input.iter_mut() {
        *x = (xorshift32(&mut rng) % 128) as u8;
    }

    let mut output_simd = [0i32; P2_OUTPUT_DIM];
    layer.propagate(&input, &mut output_simd);

    let mut output_ref = [0i32; P2_OUTPUT_DIM];
    scalar_reference::affine_propagate(&biases, &weights_raw, padded, &input, &mut output_ref);

    assert_eq!(output_simd[0], output_ref[0], "AffineTransform L3 mismatch");
}

// ─── ClippedReLU 等価性 ───

#[test]
fn test_clipped_relu_simd_equivalence() {
    let mut rng = 77777u32;

    // 全範囲テスト (512 要素: L1 出力サイズ)
    let mut input = Aligned([0i32; P2_L1_DIM]);
    for x in input.0.iter_mut() {
        // 負値、ゼロ付近、正の大きい値をまんべんなく
        *x = rand_i32(&mut rng) % 20000 - 5000;
    }

    let mut output_simd = Aligned([0u8; P2_L1_DIM]);
    ClippedReLU::propagate(&input.0, &mut output_simd.0);

    let mut output_ref = [0u8; P2_L1_DIM];
    scalar_reference::clipped_relu(&input.0, &mut output_ref);

    for (i, ((&simd_val, &ref_val), &in_val)) in
        output_simd.0.iter().zip(output_ref.iter()).zip(input.0.iter()).enumerate()
    {
        assert_eq!(
            simd_val, ref_val,
            "ClippedReLU mismatch at [{i}]: input={in_val}, simd={simd_val}, ref={ref_val}"
        );
    }
}

// ─── ネットワーク全体 forward 等価性 ───

#[test]
fn test_network_forward_zero_weights() {
    let net = P2Network::new_zeroed();
    let evaluator = CpuEvaluator::new(Arc::new(net));

    let input = P2NnueInput {
        piece_features: Box::new([0i32; P2_FEATURE_DIM]),
        side_to_move: 1,
        game_phase: 255,
        king_bucket: 0,
    };
    assert_eq!(evaluator.evaluate(&input), 0);
}

#[test]
fn test_network_forward_random_weights() {
    let mut rng = 42u32;
    let mut net = P2Network::new_zeroed();

    // L1: 小さい重みを設定
    for b in net.l1.biases.iter_mut() {
        *b = (rand_i32(&mut rng) % 1000) - 500;
    }
    let padded_l1 = InputAffineTransform::<{ P2_INPUT_DIM }, { P2_L1_DIM }>::PADDED_IN;
    for j in 0..P2_L1_DIM {
        for i in 0..P2_INPUT_DIM {
            net.l1.weights[j * padded_l1 + i] = (rand_i32(&mut rng) % 20) as i16;
        }
    }

    // L2, L3 のバイアスのみ設定 (重みはゼロ)
    for b in net.l2.biases.iter_mut() {
        *b = (rand_i32(&mut rng) % 1000) - 500;
    }
    net.l3.biases[0] = rand_i32(&mut rng) % 1000;

    let evaluator = CpuEvaluator::new(Arc::new(net));

    let mut input = P2NnueInput {
        piece_features: Box::new([0i32; P2_FEATURE_DIM]),
        side_to_move: 1,
        game_phase: 128,
        king_bucket: 0,
    };
    // バイナリ特徴量を設定
    for (i, x) in input.piece_features.iter_mut().enumerate() {
        *x = if i % 7 == 0 { 1 } else { 0 };
    }

    // 2 回呼んで一致確認 (決定性)
    let v1 = evaluator.evaluate(&input);
    let v2 = evaluator.evaluate(&input);
    assert_eq!(v1, v2, "forward の決定性");
}

// ─── 決定性テスト ───

#[test]
fn test_forward_determinism_100() {
    let net = Arc::new(P2Network::new_zeroed());
    let evaluator = CpuEvaluator::new(net);

    let mut input = P2NnueInput {
        piece_features: Box::new([0i32; P2_FEATURE_DIM]),
        side_to_move: 1,
        game_phase: 255,
        king_bucket: 0,
    };
    input.piece_features[0] = 1;
    input.piece_features[100] = 1;

    let first = evaluator.evaluate(&input);
    for _ in 0..100 {
        assert_eq!(evaluator.evaluate(&input), first);
    }
}

// ─── バッチ評価テスト ───

#[test]
fn test_batch_evaluate() {
    let net = Arc::new(P2Network::new_zeroed());
    let evaluator = CpuEvaluator::new(net);

    let inputs: Vec<P2NnueInput> = (0..8)
        .map(|i| {
            let mut input = P2NnueInput {
                piece_features: Box::new([0i32; P2_FEATURE_DIM]),
                side_to_move: 1,
                game_phase: 255,
                king_bucket: 0,
            };
            input.piece_features[i] = 1;
            input
        })
        .collect();

    let mut batch_outputs = vec![0i32; 8];
    evaluator.evaluate_batch(&inputs, &mut batch_outputs);

    // 各バッチ要素が単独評価と一致
    for (i, input) in inputs.iter().enumerate() {
        let single = evaluator.evaluate(input);
        assert_eq!(batch_outputs[i], single, "batch[{i}] != single");
    }
}

// ─── evaluate_full テスト ───

#[test]
fn test_evaluate_full_returns_policy() {
    let net = Arc::new(P2Network::new_zeroed());
    let evaluator = CpuEvaluator::new(net);

    let input = P2NnueInput {
        piece_features: Box::new([0i32; P2_FEATURE_DIM]),
        side_to_move: 1,
        game_phase: 255,
        king_bucket: 0,
    };
    let output = evaluator.evaluate_full(&input);
    assert!(output.policy_hint.is_some(), "evaluate_full は policy_hint を返す");
    assert!(output.uncertainty.is_none(), "v1 では uncertainty は None");
    assert_eq!(output.value, evaluator.evaluate(&input), "value は evaluate() と一致");
}

// ─── backend_name テスト ───

#[test]
fn test_backend_name() {
    let net = Arc::new(P2Network::new_zeroed());
    let evaluator = CpuEvaluator::new(net);
    let name = evaluator.backend_name();
    assert!(name == "cpu-avx2" || name == "cpu-scalar", "unexpected backend: {name}");
}

// ─── HybridRouter テスト ───

mod router_tests {
    use super::*;
    use crate::evaluator::HybridRouter;
    use crate::queue::types::EvalRequest;

    /// テスト用 EvalRequest を生成
    fn make_request(request_id: u64, node_id: u32, feature_idx: usize) -> EvalRequest {
        let mut input = P2NnueInput {
            piece_features: Box::new([0i32; P2_FEATURE_DIM]),
            side_to_move: 1,
            game_phase: 255,
            king_bucket: 0,
        };
        if feature_idx < P2_FEATURE_DIM {
            input.piece_features[feature_idx] = 1;
        }
        EvalRequest {
            request_id,
            node_id,
            nnue_input: input,
        }
    }

    #[test]
    fn test_router_cpu_only() {
        let net = Arc::new(P2Network::new_zeroed());
        let router = HybridRouter::new(net, 32);

        assert!(!router.gpu_available());
        assert_eq!(router.backend_name(), "cpu-only");
    }

    #[test]
    fn test_router_small_batch_uses_cpu() {
        let net = Arc::new(P2Network::new_zeroed());
        let router = HybridRouter::new(net, 32);

        let requests: Vec<EvalRequest> =
            (0..16).map(|i| make_request(i as u64, i, i as usize)).collect();

        let results = router.route_batch(&requests);
        assert_eq!(results.len(), 16);

        let stats = router.stats();
        assert_eq!(stats.cpu_batches, 1);
        assert_eq!(stats.gpu_batches, 0);
        assert_eq!(stats.total_requests, 16);
    }

    #[test]
    fn test_router_large_batch_cpu_fallback() {
        let net = Arc::new(P2Network::new_zeroed());
        let router = HybridRouter::new(net, 32);

        let requests: Vec<EvalRequest> =
            (0..64).map(|i| make_request(i as u64, i, i as usize)).collect();

        let results = router.route_batch(&requests);
        assert_eq!(results.len(), 64);

        // GPU 無しなので CPU にフォールバック
        let stats = router.stats();
        assert_eq!(stats.cpu_batches, 1);
        assert_eq!(stats.gpu_batches, 0);
        assert_eq!(stats.total_requests, 64);
    }

    #[test]
    fn test_router_request_id_integrity() {
        let net = Arc::new(P2Network::new_zeroed());
        let router = HybridRouter::new(net, 32);

        let requests: Vec<EvalRequest> =
            (0..8).map(|i| make_request(100 + i as u64, 200 + i, i as usize)).collect();

        let results = router.route_batch(&requests);

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.request_id, 100 + i as u64, "request_id mismatch at {i}");
            assert_eq!(result.node_id, 200 + i as u32, "node_id mismatch at {i}");
        }
    }

    #[test]
    fn test_router_result_order_preserved() {
        let net = Arc::new(P2Network::new_zeroed());
        let cpu = CpuEvaluator::new(Arc::clone(&net));
        let router = HybridRouter::new(net, 32);

        // 異なる特徴量で各リクエストの評価値が異なるようにする
        let requests: Vec<EvalRequest> =
            (0..4).map(|i| make_request(i as u64, i, i as usize * 100)).collect();

        let results = router.route_batch(&requests);

        for (i, (req, res)) in requests.iter().zip(results.iter()).enumerate() {
            let expected = cpu.evaluate(&req.nnue_input);
            assert_eq!(res.value, expected, "value mismatch at index {i}");
        }
    }

    #[test]
    fn test_router_stats_accumulate() {
        let net = Arc::new(P2Network::new_zeroed());
        let router = HybridRouter::new(net, 32);

        // 3 回バッチ処理
        for batch_id in 0..3u64 {
            let requests: Vec<EvalRequest> =
                (0..8).map(|i| make_request(batch_id * 8 + i as u64, i, i as usize)).collect();
            router.route_batch(&requests);
        }

        let stats = router.stats();
        assert_eq!(stats.cpu_batches, 3);
        assert_eq!(stats.total_requests, 24);
        assert_eq!(stats.gpu_batches, 0);
        assert_eq!(stats.gpu_fallbacks, 0);
    }

    #[test]
    fn test_router_single_evaluate() {
        let net = Arc::new(P2Network::new_zeroed());
        let cpu = CpuEvaluator::new(Arc::clone(&net));
        let router = HybridRouter::new(net, 32);

        let input = P2NnueInput {
            piece_features: Box::new([0i32; P2_FEATURE_DIM]),
            side_to_move: 1,
            game_phase: 255,
            king_bucket: 0,
        };

        // NnueEvaluator trait の evaluate は常に CPU
        assert_eq!(router.evaluate(&input), cpu.evaluate(&input));
    }

    #[test]
    fn test_router_empty_batch() {
        let net = Arc::new(P2Network::new_zeroed());
        let router = HybridRouter::new(net, 32);

        let results = router.route_batch(&[]);
        assert!(results.is_empty());

        let stats = router.stats();
        assert_eq!(stats.cpu_batches, 1);
        assert_eq!(stats.total_requests, 0);
    }
}

// ─── CUDA 等価性テスト ───

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use crate::evaluator::gpu::CudaEvaluator;

    #[test]
    fn test_cuda_zero_weights() {
        let net = Arc::new(P2Network::new_zeroed());
        let cpu = CpuEvaluator::new(Arc::clone(&net));
        let gpu = match CudaEvaluator::new(Arc::clone(&net)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("CUDA unavailable, skipping: {e}");
                return;
            }
        };

        let input = P2NnueInput {
            piece_features: Box::new([0i32; P2_FEATURE_DIM]),
            side_to_move: 1,
            game_phase: 255,
            king_bucket: 0,
        };

        assert_eq!(cpu.evaluate(&input), gpu.evaluate(&input));
    }

    #[test]
    fn test_cuda_random_weights() {
        let mut rng = 31415u32;
        let mut net = P2Network::new_zeroed();

        for b in net.l1.biases.iter_mut() {
            *b = (rand_i32(&mut rng) % 500) - 250;
        }
        for b in net.l2.biases.iter_mut() {
            *b = (rand_i32(&mut rng) % 500) - 250;
        }
        net.l3.biases[0] = rand_i32(&mut rng) % 500;

        let net = Arc::new(net);
        let cpu = CpuEvaluator::new(Arc::clone(&net));
        let gpu = match CudaEvaluator::new(Arc::clone(&net)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("CUDA unavailable, skipping: {e}");
                return;
            }
        };

        let mut input = P2NnueInput {
            piece_features: Box::new([0i32; P2_FEATURE_DIM]),
            side_to_move: 1,
            game_phase: 128,
            king_bucket: 0,
        };
        for (i, x) in input.piece_features.iter_mut().enumerate() {
            *x = if i % 5 == 0 { 1 } else { 0 };
        }

        let cpu_val = cpu.evaluate(&input);
        let gpu_val = gpu.evaluate(&input);
        assert_eq!(cpu_val, gpu_val, "CPU={cpu_val} GPU={gpu_val}");
    }

    #[test]
    fn test_cuda_batch_consistency() {
        let net = Arc::new(P2Network::new_zeroed());
        let gpu = match CudaEvaluator::new(Arc::clone(&net)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("CUDA unavailable, skipping: {e}");
                return;
            }
        };

        let inputs: Vec<P2NnueInput> = (0..8)
            .map(|i| {
                let mut input = P2NnueInput {
                    piece_features: Box::new([0i32; P2_FEATURE_DIM]),
                    side_to_move: 1,
                    game_phase: 255,
                    king_bucket: 0,
                };
                input.piece_features[i * 10] = 1;
                input
            })
            .collect();

        let mut batch_out = vec![0i32; 8];
        gpu.evaluate_batch(&inputs, &mut batch_out);

        for (i, input) in inputs.iter().enumerate() {
            let single = gpu.evaluate(input);
            assert_eq!(batch_out[i], single, "batch[{i}] mismatch");
        }
    }

    #[test]
    fn test_cuda_determinism() {
        let net = Arc::new(P2Network::new_zeroed());
        let gpu = match CudaEvaluator::new(Arc::clone(&net)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("CUDA unavailable, skipping: {e}");
                return;
            }
        };

        let mut input = P2NnueInput {
            piece_features: Box::new([0i32; P2_FEATURE_DIM]),
            side_to_move: 1,
            game_phase: 255,
            king_bucket: 0,
        };
        input.piece_features[42] = 1;

        let first = gpu.evaluate(&input);
        for _ in 0..100 {
            assert_eq!(gpu.evaluate(&input), first);
        }
    }
}
