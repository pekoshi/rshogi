//! ONNX export hook (feature = "onnx")。
//!
//! candle-onnx 0.8 系は **import (read) のみ** で、 graph export は提供されていない。
//! また build に `protoc` を要求するため Windows env では追加セットアップが必要。
//! そこで本 module では `prost::Message` を **derive で直接定義** した
//! minimum ONNX schema (ModelProto / GraphProto / TensorProto / OperatorSetIdProto)
//! を使い、 protoc に依存せず `.onnx` ファイルを書き出す。
//!
//! ## 出力フォーマット
//!
//! - 重みは [`TensorProto`] の `initializer` として packing。
//! - graph の `node` 列は空 (forward 計算は含まない)。
//! - 拡張子 `.onnx` (ファイルとしては valid な ONNX ModelProto)。
//!
//! 単独で ORT 実行はできないが、 Python 側 (`scripts/onnx_export_nnue_b.py` 想定)
//! で Transformer graph を本格組み立てして overwrite する hand-off フォーマットとして
//! 機能する。
//!
//! ## proto3 schema 抜粋
//!
//! ```proto
//! message ModelProto {
//!   optional int64 ir_version = 1;
//!   repeated OperatorSetIdProto opset_import = 8;
//!   optional string producer_name = 2;
//!   optional string producer_version = 3;
//!   optional string domain = 4;
//!   optional int64 model_version = 5;
//!   optional string doc_string = 6;
//!   optional GraphProto graph = 7;
//! }
//! message GraphProto {
//!   repeated NodeProto node = 1;     // 空
//!   optional string name = 2;
//!   repeated TensorProto initializer = 5;
//!   optional string doc_string = 10;
//! }
//! message TensorProto {
//!   repeated int64 dims = 1;
//!   optional int32 data_type = 2;    // 1 = FLOAT
//!   optional string name = 8;
//!   optional bytes raw_data = 9;
//! }
//! message OperatorSetIdProto {
//!   optional string domain = 1;
//!   optional int64 version = 2;
//! }
//! ```

use anyhow::{anyhow, Result};
use candle_core::Tensor;
use prost::Message;
use std::io::Write;
use std::path::Path;

use crate::model::NnueBModel;

// ===========================================================================
// 最小 ONNX schema (prost::Message derive)
// ===========================================================================

#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    #[prost(string, optional, tag = "1")]
    pub domain: Option<String>,
    #[prost(int64, optional, tag = "2")]
    pub version: Option<i64>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, optional, tag = "2")]
    pub data_type: Option<i32>,
    #[prost(string, optional, tag = "8")]
    pub name: Option<String>,
    #[prost(bytes, optional, tag = "9")]
    pub raw_data: Option<Vec<u8>>,
}

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    // node = 1 (空、 forward 計算なし)
    #[prost(string, optional, tag = "2")]
    pub name: Option<String>,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(string, optional, tag = "10")]
    pub doc_string: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, optional, tag = "1")]
    pub ir_version: Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub producer_name: Option<String>,
    #[prost(string, optional, tag = "3")]
    pub producer_version: Option<String>,
    #[prost(string, optional, tag = "4")]
    pub domain: Option<String>,
    #[prost(int64, optional, tag = "5")]
    pub model_version: Option<i64>,
    #[prost(string, optional, tag = "6")]
    pub doc_string: Option<String>,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
}

/// ONNX TensorProto.DataType.FLOAT
const DATA_TYPE_FLOAT: i32 = 1;

// ===========================================================================
// public API
// ===========================================================================

fn ensure_onnx_path(p: &Path) -> Result<()> {
    match p.extension().and_then(|e| e.to_str()) {
        Some("onnx") => Ok(()),
        _ => Err(anyhow!("export path must end with '.onnx': {}", p.display())),
    }
}

fn tensor_to_proto(name: &str, t: &Tensor) -> Result<TensorProto> {
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    let dims: Vec<i64> = t.dims().iter().map(|&d| d as i64).collect();
    let mut raw = Vec::with_capacity(v.len() * 4);
    for f in v.iter() {
        raw.extend_from_slice(&f.to_le_bytes());
    }
    Ok(TensorProto {
        dims,
        data_type: Some(DATA_TYPE_FLOAT),
        name: Some(name.to_string()),
        raw_data: Some(raw),
    })
}

/// 重みのみ含むスタブ ONNX を書き出す。
///
/// graph.node は空、 graph.initializer に全 weight が入る。
pub fn export_onnx_stub(
    model: &NnueBModel,
    weights: &[(&str, &Tensor)],
    out_path: &Path,
) -> Result<()> {
    ensure_onnx_path(out_path)?;

    let mut initializer = Vec::with_capacity(weights.len());
    for (name, t) in weights.iter() {
        initializer.push(tensor_to_proto(name, t)?);
    }

    let cfg = &model.cfg;
    let graph = GraphProto {
        name: Some("nnue_b_stub".to_string()),
        initializer,
        doc_string: Some(format!(
            "NNUE-B stub export (Candle → ONNX initializer only)。 \
             d_model={}, n_layers={}, n_heads={}, d_ff={}, vocab_size={}, \
             max_tokens={}, policy_size={}",
            cfg.d_model,
            cfg.n_layers,
            cfg.n_heads,
            cfg.d_ff,
            cfg.vocab_size,
            cfg.max_tokens,
            cfg.policy_size,
        )),
    };

    let model_proto = ModelProto {
        ir_version: Some(8),
        producer_name: Some("nnue-b".to_string()),
        producer_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        domain: Some("rshogi.nnue-b".to_string()),
        model_version: Some(1),
        doc_string: Some("PUCT hybrid 用 NNUE-B Transformer (stub、 重みのみ)".to_string()),
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: Some("".to_string()),
            version: Some(17),
        }],
    };

    let mut buf = Vec::with_capacity(model_proto.encoded_len());
    model_proto.encode(&mut buf).map_err(|e| anyhow!("ONNX encode failed: {e}"))?;
    let mut f = std::fs::File::create(out_path)?;
    f.write_all(&buf)?;
    Ok(())
}

/// 既存 `.onnx` を読み込んで ModelProto を返す (round-trip 検証用)。
pub fn read_onnx(path: &Path) -> Result<ModelProto> {
    let buf = std::fs::read(path)?;
    let m = ModelProto::decode(buf.as_slice()).map_err(|e| anyhow!("ONNX decode failed: {e}"))?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NnueBConfig, NnueBModel};
    use candle_core::Device;

    #[test]
    fn export_stub_writes_onnx_file() -> Result<()> {
        let device = Device::Cpu;
        let cfg = NnueBConfig::tiny_for_tests();
        let model = NnueBModel::new(cfg, device)?;
        let tmpdir = tempfile::tempdir()?;
        let path = tmpdir.path().join("nnue_b_stub.onnx");

        let all = model.varmap().all_vars();
        let owned: Vec<(String, Tensor)> = all
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("var{i}"), v.as_tensor().clone()))
            .collect();
        let refs: Vec<(&str, &Tensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
        export_onnx_stub(&model, &refs, &path)?;

        // 再読込で valid な ModelProto か確認
        let m = read_onnx(&path)?;
        let g = m.graph.expect("graph");
        assert!(!g.initializer.is_empty(), "no initializers exported");
        Ok(())
    }
}
