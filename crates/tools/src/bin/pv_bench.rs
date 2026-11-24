//! PolicyValueNet forward 速度 + value_raw 検証

use std::path::PathBuf;
use std::time::Instant;

use p2_nnue::policy_value_net::{PolicyValueNet, load_raw_bin, position_to_active_indices};
use rshogi_core::position::Position;
use rshogi_core::types::Move;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: pv_bench <raw.bin path> [iterations=100]");
        std::process::exit(1);
    }
    let path = PathBuf::from(&args[1]);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    // variant 自動推定
    let n_floats = (std::fs::metadata(&path).unwrap().len() / 4) as usize;
    let variant =
        p2_nnue::policy_value_net::infer_variant_from_size(n_floats).expect("variant infer failed");
    eprintln!("variant={variant:?}, floats={n_floats}");

    let t0 = Instant::now();
    let weights = load_raw_bin(&path, variant.config()).expect("load failed");
    eprintln!("load: {:.2}s", t0.elapsed().as_secs_f64());

    let net = PolicyValueNet::new(weights);

    let mut pos = Position::new();
    pos.set_hirate();
    let (stm_idx, nstm_idx) = position_to_active_indices(&pos);
    eprintln!("stm_active={}, nstm_active={}", stm_idx.len(), nstm_idx.len());

    // warm up
    let _ = net.forward(&stm_idx, &nstm_idx);

    let t1 = Instant::now();
    let mut last_value_raw = 0.0;
    let mut last_policy_max = 0.0_f32;
    let mut last_policy_min = 0.0_f32;
    for _ in 0..iters {
        let out = net.forward(&stm_idx, &nstm_idx);
        last_value_raw = out.value_raw;
        last_policy_max = out.policy_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        last_policy_min = out.policy_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    }
    eprintln!(
        "[hirate CPU] value_raw={:.4} policy_range=[{:.2}, {:.2}]",
        last_value_raw, last_policy_min, last_policy_max
    );

    // GPU bench (feature pv-cuda)
    #[cfg(feature = "pv-cuda")]
    {
        use p2_nnue::policy_value_net::{PolicyValueCudaContext, PolicyValueNetCuda};
        use std::sync::Arc;
        eprintln!("\n=== GPU forward bench ===");
        match PolicyValueCudaContext::new(0) {
            Ok(ctx) => {
                let ctx = Arc::new(ctx);
                // load cpu weights again (net moved value_raw だけ取り出してる)
                let cpu_weights = p2_nnue::policy_value_net::load_raw_bin(&path, variant.config())
                    .expect("reload");
                let mut net_gpu = PolicyValueNetCuda::new(ctx, &cpu_weights).expect("gpu net new");
                // warmup
                let _ = net_gpu.forward(&stm_idx, &nstm_idx);
                let t_gpu = Instant::now();
                let mut last_v = 0.0_f32;
                for _ in 0..iters {
                    let out = net_gpu.forward(&stm_idx, &nstm_idx).expect("gpu forward");
                    last_v = out.value_raw;
                }
                let el = t_gpu.elapsed().as_secs_f64();
                let per_call = (el / iters as f64) * 1000.0;
                eprintln!(
                    "[GPU batch=1] value_raw={:.4} {} iters: {:.3}s = {:.3} ms/call ({:.0} leaf/sec)",
                    last_v,
                    iters,
                    el,
                    per_call,
                    iters as f64 / el
                );

                // batch bench
                for batch_size in [4usize, 8, 16, 32, 64] {
                    let stm_batch: Vec<Vec<u32>> =
                        (0..batch_size).map(|_| stm_idx.clone()).collect();
                    let nstm_batch: Vec<Vec<u32>> =
                        (0..batch_size).map(|_| nstm_idx.clone()).collect();
                    let _ = net_gpu.forward_batch(&stm_batch, &nstm_batch);
                    let n_iters = iters.max(20);
                    let t_b = Instant::now();
                    for _ in 0..n_iters {
                        let _ = net_gpu.forward_batch(&stm_batch, &nstm_batch).expect("gpu batch");
                    }
                    let el = t_b.elapsed().as_secs_f64();
                    let total_leaves = n_iters * batch_size;
                    let per_call = (el / n_iters as f64) * 1000.0;
                    eprintln!(
                        "[GPU batch={}] {} iters: {:.3}s = {:.3} ms/call ({:.0} leaf/sec)",
                        batch_size,
                        n_iters,
                        el,
                        per_call,
                        total_leaves as f64 / el
                    );
                }
            }
            Err(e) => eprintln!("[GPU] CudaContext init failed: {e:?}"),
        }
    }

    // ★ Value Sanity Check: 複数局面で value_raw 分布を確認
    eprintln!("\n=== Value Sanity Check ===");
    let test_positions: &[(&str, &[&str])] = &[
        ("hirate (初期局面、 五分)", &[]),
        ("ply=2 (1 手目後)", &["7g7f"]),
        ("ply=4 (序盤、 ほぼ五分)", &["7g7f", "3c3d", "2g2f", "8c8d"]),
        ("ply=6 (序盤展開)", &["7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e"]),
        (
            "ply=8 (序盤後半)",
            &[
                "7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e", "6i7h", "4c4d",
            ],
        ),
        (
            "ply=10 (中盤入口)",
            &[
                "7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e", "6i7h", "4c4d", "1g1f", "1c1d",
            ],
        ),
        (
            "ply=12 (中盤)",
            &[
                "7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e", "6i7h", "4c4d", "1g1f", "1c1d",
                "9g9f", "9c9d",
            ],
        ),
        // 角交換等の drop 手は rshogi-core hand.rs:69 overflow panic で別途 skip 必要
    ];
    let mut value_raws = Vec::new();
    for (label, moves) in test_positions {
        let mut p = Position::new();
        p.set_hirate();
        let mut all_ok = true;
        for m in moves.iter() {
            if let Some(mv) = Move::from_usi(m) {
                if !p.is_legal(mv) {
                    all_ok = false;
                    break;
                }
                let gc = p.gives_check(mv);
                p.do_move(mv, gc);
            } else {
                all_ok = false;
                break;
            }
        }
        if !all_ok {
            eprintln!("  {} -> SKIP (move parse)", label);
            continue;
        }
        let (s, n) = position_to_active_indices(&p);
        let out = net.forward(&s, &n);
        let sig = 1.0 / (1.0 + (-out.value_raw).exp());
        let cp_est = (sig - 0.5) * 1200.0; // ±600cp scale
        eprintln!(
            "  {:30}: value_raw={:+.4} sigmoid={:.3} (cp_est ~ {:+5.0})",
            label, out.value_raw, sig, cp_est
        );
        value_raws.push(out.value_raw);
    }
    // 分布統計
    if !value_raws.is_empty() {
        let mean: f32 = value_raws.iter().sum::<f32>() / value_raws.len() as f32;
        let var: f32 =
            value_raws.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / value_raws.len() as f32;
        let std = var.sqrt();
        let max = value_raws.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = value_raws.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("  [stats] mean={:+.4} std={:.4} range=[{:+.4}, {:+.4}]", mean, std, min, max);
        if std < 0.001 {
            eprintln!("  ⚠ WARNING: value_raw ほぼ定数、 value head 学習不足の可能性");
        } else if std > 0.05 {
            eprintln!("  ✅ value head 機能している (局面で大きく変動)");
        } else {
            eprintln!("  △ value 変動はあるが弱い (std={:.4})", std);
        }
    }
    let elapsed = t1.elapsed().as_secs_f64();
    let per_call_ms = (elapsed / iters as f64) * 1000.0;
    eprintln!(
        "forward {} iters: {:.2}s = {:.2} ms/call ({:.0} calls/sec)",
        iters,
        elapsed,
        per_call_ms,
        iters as f64 / elapsed
    );
}
