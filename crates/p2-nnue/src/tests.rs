//! P2 NNUE ネットワークの統合テスト

use std::io::Cursor;

use crate::features::{P2_FEATURE_DIM, P2NnueInput, extract_features};
use crate::io::{P2_NNUE_MAGIC, P2_NNUE_VERSION, P2NnueHeader};
use crate::layers::{AffineTransform, ClippedReLU, InputAffineTransform};
use crate::network::P2Network;
use crate::types::{P2_INPUT_DIM, P2_L1_DIM, P2_L2_DIM, P2_OUTPUT_DIM};

use rshogi_core::position::Position;

/// startpos を作成するヘルパー
fn make_startpos() -> Position {
    let mut pos = Position::new();
    pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
        .unwrap();
    pos
}

/// ゼロ重みバイナリデータを生成
fn generate_zeroed_binary() -> Vec<u8> {
    let header = P2NnueHeader::default_valid();
    let mut data = Vec::new();
    data.extend_from_slice(&header.to_bytes());

    // Layer 1: biases (i32 × 512) + weights (i16 × 512 × 2048)
    let l1_biases_bytes = P2_L1_DIM * 4;
    let l1_weights_bytes = P2_L1_DIM * P2_INPUT_DIM * 2;
    data.extend(std::iter::repeat_n(0u8, l1_biases_bytes + l1_weights_bytes));

    // Layer 2: biases (i32 × 256) + weights (i8 × 256 × 512)
    let l2_biases_bytes = P2_L2_DIM * 4;
    let l2_weights_bytes = P2_L2_DIM * P2_L1_DIM;
    data.extend(std::iter::repeat_n(0u8, l2_biases_bytes + l2_weights_bytes));

    // Layer 3: biases (i32 × 1) + weights (i8 × 1 × 256)
    let l3_biases_bytes = P2_OUTPUT_DIM * 4;
    let l3_weights_bytes = P2_OUTPUT_DIM * P2_L2_DIM;
    data.extend(std::iter::repeat_n(0u8, l3_biases_bytes + l3_weights_bytes));

    data
}

// ─── テスト 1: ゼロ重み forward ───

#[test]
fn test_zero_weights_forward() {
    let net = P2Network::new_zeroed();
    let input = P2NnueInput {
        piece_features: Box::new([0i32; P2_FEATURE_DIM]),
        side_to_move: 1,
        game_phase: 255,
        king_bucket: 0,
    };
    let output = net.forward(&input);
    assert_eq!(output.value, 0, "ゼロ重みネットワークの出力は 0");
}

// ─── テスト 2: InputAffineTransform 単体 ───

#[test]
fn test_input_affine_transform_unit() {
    let mut layer = InputAffineTransform::<4, 2>::new_zeroed();
    layer.biases[0] = 5;
    layer.biases[1] = -3;

    let padded = <InputAffineTransform<4, 2>>::PADDED_IN;
    // row 0: [2, 0, -1, 3]
    layer.weights[0] = 2;
    layer.weights[2] = -1;
    layer.weights[3] = 3;
    // row 1: [0, 1, 1, -2]
    layer.weights[padded + 1] = 1;
    layer.weights[padded + 2] = 1;
    layer.weights[padded + 3] = -2;

    let input = [1i32, 2, 3, 4];
    let mut output = [0i32; 2];
    layer.propagate(&input, &mut output);

    // out[0] = 5 + 2*1 + 0*2 + (-1)*3 + 3*4 = 5 + 2 + 0 - 3 + 12 = 16
    assert_eq!(output[0], 16);
    // out[1] = -3 + 0*1 + 1*2 + 1*3 + (-2)*4 = -3 + 0 + 2 + 3 - 8 = -6
    assert_eq!(output[1], -6);
}

// ─── テスト 3: AffineTransform 単体 ───

#[test]
fn test_affine_transform_unit() {
    let mut layer = AffineTransform::<4, 2>::new_zeroed();
    layer.biases[0] = 10;
    layer.biases[1] = -5;

    let padded = <AffineTransform<4, 2>>::PADDED_IN;
    // row 0: [1, -1, 2, 0]
    layer.weights[0] = 1;
    layer.weights[1] = -1;
    layer.weights[2] = 2;
    // row 1: [3, 0, -1, 1]
    layer.weights[padded] = 3;
    layer.weights[padded + 2] = -1;
    layer.weights[padded + 3] = 1;

    let input: [u8; 4] = [10, 20, 30, 40];
    let mut output = [0i32; 2];
    layer.propagate(&input, &mut output);

    // out[0] = 10 + 1*10 + (-1)*20 + 2*30 + 0*40 = 10 + 10 - 20 + 60 = 60
    assert_eq!(output[0], 60);
    // out[1] = -5 + 3*10 + 0*20 + (-1)*30 + 1*40 = -5 + 30 - 30 + 40 = 35
    assert_eq!(output[1], 35);
}

// ─── テスト 4: ClippedReLU 単体 ───

#[test]
fn test_clipped_relu_unit() {
    let input = [
        -1000i32, // 負 → 0
        0,        // ゼロ → 0
        1,        // 1 >> 6 = 0
        63,       // 63 >> 6 = 0
        64,       // 64 >> 6 = 1
        127 * 64, // 127
        128 * 64, // 128 → clamp 127
        i32::MAX, // 大きい → 127
    ];
    let mut output = [0u8; 8];
    ClippedReLU::propagate(&input, &mut output);

    assert_eq!(output[0], 0, "負値");
    assert_eq!(output[1], 0, "ゼロ");
    assert_eq!(output[2], 0, "1 >> 6 = 0");
    assert_eq!(output[3], 0, "63 >> 6 = 0");
    assert_eq!(output[4], 1, "64 >> 6 = 1");
    assert_eq!(output[5], 127, "127*64 >> 6 = 127");
    assert_eq!(output[6], 127, "128 → clamp 127");
    assert_eq!(output[7], 127, "i32::MAX → clamp 127");
}

// ─── テスト 5: バイナリ入力 forward (NaN/overflow なし) ───

#[test]
fn test_binary_input_forward_no_overflow() {
    let mut net = P2Network::new_zeroed();
    // Layer 1 の重みを全て 1 に設定（最悪ケースのオーバーフローテスト）
    // 入力はバイナリ (0/1) 主体で最大 2025 個 + game_ply 最大 512
    // 最悪: 2025 * 32767 + 512 * 32767 ≈ 8.3e7 < i32::MAX
    // ここではゼロ重みで NaN/panic がないことだけ確認
    let pos = make_startpos();
    let input = extract_features(&pos);
    let output = net.forward(&input);
    // ゼロ重みなので 0
    assert_eq!(output.value, 0);

    // 少しだけ重みを入れて正常動作確認
    // L1 bias[0] = 64 → ReLU 後 1、L3 weights[0] = 1, L3 bias = 0
    // → value = 1
    net.l1.biases[0] = 64; // >> 6 = 1
    // L2 は identity-like にはできないので、L2 bias[0] = 64 にして L3 で拾う
    net.l2.biases[0] = 64; // >> 6 = 1
    net.l3.weights[0] = 1;
    let output = net.forward(&input);
    // L1 → bias[0]=64, rest=0 → ReLU → [1, 0, 0, ...] (u8)
    // L2 → bias[0]=64 + w[0][0]*1=0 (weights are zero) → [64, 0, ...]
    //       → ReLU → [1, 0, ...] (u8)
    // L3 → bias=0 + w[0][0]*1 = 1
    assert_eq!(output.value, 1);
}

// ─── テスト 6: ヘッダ検証 ───

#[test]
fn test_header_valid() {
    let header = P2NnueHeader::default_valid();
    assert!(header.validate().is_ok());
}

#[test]
fn test_header_invalid_magic() {
    let mut header = P2NnueHeader::default_valid();
    header.magic = 0xDEADBEEF;
    assert!(header.validate().is_err());
}

#[test]
fn test_header_invalid_version() {
    let mut header = P2NnueHeader::default_valid();
    header.version = 99;
    assert!(header.validate().is_err());
}

#[test]
fn test_header_invalid_dims() {
    let mut header = P2NnueHeader::default_valid();
    header.input_dim = 1024;
    assert!(header.validate().is_err());

    let mut header = P2NnueHeader::default_valid();
    header.l1_dim = 1024;
    assert!(header.validate().is_err());

    let mut header = P2NnueHeader::default_valid();
    header.l2_dim = 128;
    assert!(header.validate().is_err());

    let mut header = P2NnueHeader::default_valid();
    header.output_dim = 2;
    assert!(header.validate().is_err());
}

#[test]
fn test_header_roundtrip() {
    let header = P2NnueHeader::default_valid();
    let bytes = header.to_bytes();
    let mut cursor = Cursor::new(&bytes);
    let loaded = P2NnueHeader::read(&mut cursor).unwrap();
    assert_eq!(loaded.magic, P2_NNUE_MAGIC);
    assert_eq!(loaded.version, P2_NNUE_VERSION);
    assert_eq!(loaded.input_dim, P2_INPUT_DIM as u32);
    assert_eq!(loaded.l1_dim, P2_L1_DIM as u32);
    assert_eq!(loaded.l2_dim, P2_L2_DIM as u32);
    assert_eq!(loaded.output_dim, P2_OUTPUT_DIM as u32);
}

// ─── テスト 7: ファイルロード ───

#[test]
fn test_load_zeroed_binary() {
    let data = generate_zeroed_binary();
    let mut cursor = Cursor::new(&data);
    let net = P2Network::load(&mut cursor).unwrap();

    let input = P2NnueInput {
        piece_features: Box::new([0i32; P2_FEATURE_DIM]),
        side_to_move: 1,
        game_phase: 255,
        king_bucket: 0,
    };
    let output = net.forward(&input);
    assert_eq!(output.value, 0, "ゼロ重みファイルからロードした出力は 0");
}

#[test]
fn test_load_invalid_magic() {
    let mut data = generate_zeroed_binary();
    // マジックナンバーを壊す
    data[0] = 0xFF;
    let mut cursor = Cursor::new(&data);
    assert!(P2Network::load(&mut cursor).is_err());
}

// ─── テスト 8: 決定性 ───

#[test]
fn test_forward_determinism() {
    let net = P2Network::new_zeroed();
    let pos = make_startpos();
    let input = extract_features(&pos);

    let first = net.forward(&input);
    for _ in 0..100 {
        let result = net.forward(&input);
        assert_eq!(result.value, first.value, "forward の決定性");
    }
}

// ─── テスト 9: startpos 統合 ───

#[test]
fn test_startpos_integration() {
    let net = P2Network::new_zeroed();
    let pos = make_startpos();
    let input = extract_features(&pos);
    let output = net.forward(&input);

    // ゼロ重みネットワークなので value = 0
    assert_eq!(output.value, 0);
    // value は i32 型（コンパイル時保証だが明示確認）
    let _: i32 = output.value;
}

// ─── テスト 10: forward で policy_hint が返る ───

#[test]
fn test_forward_returns_policy_hint() {
    let net = P2Network::new_zeroed();
    let input = P2NnueInput {
        piece_features: Box::new([0i32; P2_FEATURE_DIM]),
        side_to_move: 1,
        game_phase: 255,
        king_bucket: 0,
    };
    let output = net.forward(&input);
    let hint = output.policy_hint.expect("forward() は policy_hint を返す");
    assert_eq!(hint.shared_repr.len(), P2_L2_DIM, "shared_repr は 256 要素");
    assert!(output.uncertainty.is_none(), "v1 では uncertainty は None");
}

// ─── テスト 11: forward_value_only と forward の value が一致 ───

#[test]
fn test_forward_value_only_matches() {
    let net = P2Network::new_zeroed();
    let pos = make_startpos();
    let input = extract_features(&pos);

    let full = net.forward(&input);
    let value_only = net.forward_value_only(&input);
    assert_eq!(full.value, value_only, "forward() と forward_value_only() の value は一致");
}

// ─── テスト 12: head_flags ───

#[test]
fn test_head_flags_default() {
    use crate::io::{HEAD_FLAG_POLICY, HEAD_FLAG_VALUE};
    let header = P2NnueHeader::default_valid();
    assert_eq!(header.head_flags(), HEAD_FLAG_VALUE | HEAD_FLAG_POLICY);
    assert!(header.has_policy_hint());
}

#[test]
fn test_head_flags_legacy_zero() {
    use crate::io::HEAD_FLAG_VALUE;
    let mut header = P2NnueHeader::default_valid();
    header.reserved1 = 0;
    assert_eq!(header.head_flags(), HEAD_FLAG_VALUE);
    assert!(!header.has_policy_hint());
    assert!(header.validate().is_ok());
}

#[test]
fn test_head_flags_unknown_bit_rejected() {
    let mut header = P2NnueHeader::default_valid();
    header.reserved1 = 1 << 16; // 未知ビット
    assert!(header.validate().is_err());
}

// ─── テスト 13: write → load ラウンドトリップ ───

#[test]
fn test_network_write_read_roundtrip() {
    // ゼロでないバイナリを作り、write → load → forward で一致を確認
    let data = generate_zeroed_binary();
    let mut cursor = Cursor::new(&data);
    let mut net = P2Network::load(&mut cursor).unwrap();

    // 重みに値を入れる
    net.l1.biases[0] = 128; // >> 6 = 2
    net.l1.biases[1] = -64;
    net.l1.weights[0] = 100;
    net.l1.weights[1] = -200;
    net.l2.biases[0] = 64;
    net.l2.weights[0] = 50;
    net.l2.weights[1] = -30;
    net.l3.biases[0] = 10;
    net.l3.weights[0] = 1;

    // write → load
    let mut buf = Vec::new();
    net.write(&mut buf).unwrap();
    let mut cursor = Cursor::new(&buf);
    let net2 = P2Network::load(&mut cursor).unwrap();

    // forward で一致
    let pos = make_startpos();
    let input = extract_features(&pos);
    let out1 = net.forward(&input);
    let out2 = net2.forward(&input);
    assert_eq!(out1.value, out2.value, "write → load ラウンドトリップ後の forward 値が一致");
}

// ─── テスト 14: weights_to_f32 ラウンドトリップ ───

#[test]
fn test_weights_f32_roundtrip() {
    let data = generate_zeroed_binary();
    let mut cursor = Cursor::new(&data);
    let mut net = P2Network::load(&mut cursor).unwrap();

    // 重みに値を入れる
    net.l1.biases[0] = 100;
    net.l1.weights[0] = 50;
    net.l1.weights[1] = -100;
    net.l2.biases[0] = 200;
    net.l2.weights[0] = 30;
    net.l3.biases[0] = -5;
    net.l3.weights[0] = 3;

    let flat1 = net.weights_to_f32();
    net.set_weights_from_f32(&flat1);
    let flat2 = net.weights_to_f32();

    assert_eq!(flat1.len(), flat2.len(), "f32 ベクトル長が一致");
    for (i, (a, b)) in flat1.iter().zip(flat2.iter()).enumerate() {
        assert_eq!(*a, *b, "f32 roundtrip mismatch at index {i}");
    }
}

// ─── テスト 15: ModelConfig ───

#[test]
fn test_model_config_default() {
    use crate::types::ModelConfig;
    let cfg = ModelConfig::default();
    assert_eq!(cfg.input_dim, P2_INPUT_DIM);
    assert_eq!(cfg.l1_dim, P2_L1_DIM);
    assert_eq!(cfg.l2_dim, P2_L2_DIM);
}
