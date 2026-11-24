//! P3 e2e: PyTorch (model.py) と Rust CpuEvaluatorV3::forward の数値一致確認.
//!
//! Python 側 `compare_pytorch_rust_v3.py` で生成された fixture (.json) と .p2nn v3 を
//! ロードし、 同一 input での出力が一致するかを確認する。
//!
//! 環境変数 `P2NNUE_E2E_FIXTURE` で fixture path を指定。 設定がなければ skip。
//!
//! 実行例:
//!   set P2NNUE_E2E_FIXTURE=E:/shogi/p2shogi/models/acc_report/e2e_v3_fixture.json
//!   cargo test -p p2-nnue --test e2e_v3 -- --ignored --nocapture

use std::env;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;

use p2_nnue::evaluator::CpuEvaluatorV3;
use p2_nnue::network_v3::P2NetworkV3;

#[derive(Debug, serde::Deserialize)]
struct Fixture {
    p2nn_path: String,
    feats_hex: String,
    phase_idx: u32,
    expected: Expected,
    tolerance: Tolerance,
}

#[derive(Debug, serde::Deserialize)]
struct Expected {
    value_i32: i32,
    outlook_i32: i32,
    policy_top10_indices: Vec<i32>,
    policy_logits_first5: Vec<f32>,
}

#[derive(Debug, serde::Deserialize)]
struct Tolerance {
    value_i32_abs_max: i32,
    policy_logit_abs_max: f32,
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in (0..s.len()).step_by(2) {
        let byte = u8::from_str_radix(&s[i..i + 2], 16).unwrap();
        out.push(byte);
    }
    out
}

#[test]
#[ignore = "requires .p2nn fixture - set P2NNUE_E2E_FIXTURE env var"]
fn pytorch_rust_e2e_v3_value_outlook_policy() {
    let fixture_path = match env::var("P2NNUE_E2E_FIXTURE") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("P2NNUE_E2E_FIXTURE not set, skipping");
            return;
        }
    };

    let mut buf = String::new();
    File::open(&fixture_path)
        .expect("open fixture")
        .read_to_string(&mut buf)
        .expect("read fixture");
    let fx: Fixture = serde_json::from_str(&buf).expect("parse fixture json");

    // 1. .p2nn v3 をロード
    let net = P2NetworkV3::load_from_file(&fx.p2nn_path).expect("load .p2nn v3");
    let net = Arc::new(net);
    let evaluator = CpuEvaluatorV3::new(net);

    // 2. features bytes (i8 9728 dim)
    let feats_u8 = hex_to_bytes(&fx.feats_hex);
    assert_eq!(feats_u8.len(), 9728, "feats hex must decode to 9728 bytes");
    // i8 として扱う
    let feats_i8: Vec<i8> = feats_u8.iter().map(|&b| b as i8).collect();

    // 3. forward
    let out = evaluator.forward(&feats_i8, fx.phase_idx as usize);

    // 4. value/outlook 比較
    let v_diff = (out.value - fx.expected.value_i32).abs();
    let o_diff = (out.outlook - fx.expected.outlook_i32).abs();
    println!(
        "[e2e_v3] value: rust={} pytorch={} diff={}",
        out.value, fx.expected.value_i32, v_diff
    );
    println!(
        "[e2e_v3] outlook: rust={} pytorch={} diff={}",
        out.outlook, fx.expected.outlook_i32, o_diff
    );
    assert!(
        v_diff <= fx.tolerance.value_i32_abs_max,
        "value mismatch: rust={} pytorch={} diff={} tol={}",
        out.value,
        fx.expected.value_i32,
        v_diff,
        fx.tolerance.value_i32_abs_max
    );
    assert!(
        o_diff <= fx.tolerance.value_i32_abs_max,
        "outlook mismatch: rust={} pytorch={} diff={} tol={}",
        out.outlook,
        fx.expected.outlook_i32,
        o_diff,
        fx.tolerance.value_i32_abs_max
    );

    // 5. policy logits[:5] 比較
    for (i, &expected) in fx.expected.policy_logits_first5.iter().enumerate() {
        let actual = out.policy_logits[i];
        let diff = (actual - expected).abs();
        println!(
            "[e2e_v3] policy[{}]: rust={:.4} pytorch={:.4} diff={:.4}",
            i, actual, expected, diff
        );
        assert!(
            diff <= fx.tolerance.policy_logit_abs_max,
            "policy[{}] mismatch: rust={} pytorch={} diff={}",
            i,
            actual,
            expected,
            diff
        );
    }

    // 6. argmax 一致 (top-1)
    let mut rust_top: Vec<usize> = (0..out.policy_logits.len()).collect();
    rust_top.sort_by(|&a, &b| out.policy_logits[b].partial_cmp(&out.policy_logits[a]).unwrap());
    println!(
        "[e2e_v3] policy top-1: rust={} pytorch={}",
        rust_top[0], fx.expected.policy_top10_indices[0]
    );
    assert_eq!(
        rust_top[0] as i32, fx.expected.policy_top10_indices[0],
        "policy top-1 index mismatch"
    );
}
