//! PSV / JSONL → 学習サンプル変換
//!
//! PSV: PackedSfenValue (40B) レコードから P2-NNUE 学習用の特徴量・ターゲットを生成する。
//! JSONL: dataset pipeline で生成された統一レコードから dual head 用サンプル
//!        (features + value target + best_move label + legal labels) を生成する。

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use p2_nnue::features::{P2_FEATURE_DIM, extract_features};
use rshogi_core::position::Position;
use rshogi_core::types::Move;
use serde::Deserialize;

use crate::dlshogi_features::make_move_label;
use crate::packed_sfen::{PackedSfenValue, unpack_sfen};

/// 学習サンプル
pub struct TrainingSample {
    /// P2NnueInput.piece_features を f32 化した特徴量ベクトル
    pub features: [f32; P2_FEATURE_DIM],
    /// ターゲット勝率: sigmoid(clamp(score, -2000, 2000) / 400)
    pub target_winrate: f32,
}

/// PSV ファイルからバイト列をロード
///
/// 各レコード 40 バイトの Vec として返す。
pub fn load_psv_records(path: &Path) -> io::Result<Vec<[u8; 40]>> {
    let data = fs::read(path)?;
    if data.len() % PackedSfenValue::SIZE != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("PSV ファイルサイズ ({}) が 40 の倍数でない", data.len()),
        ));
    }

    let count = data.len() / PackedSfenValue::SIZE;
    let mut records = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * PackedSfenValue::SIZE;
        let mut rec = [0u8; 40];
        rec.copy_from_slice(&data[offset..offset + 40]);
        records.push(rec);
    }
    Ok(records)
}

/// score → target winrate 変換
///
/// `sigmoid(clamp(score, -2000, 2000) / 400)`
/// score は手番側基準（YaneuraOu convention: >0 = 手番有利）
#[inline]
fn score_to_winrate(score: i16) -> f32 {
    let clamped = (score as f32).clamp(-2000.0, 2000.0);
    1.0 / (1.0 + (-clamped / 400.0).exp())
}

/// PSV レコードを学習サンプルに変換
pub fn decode_sample(record: &[u8; 40]) -> Result<TrainingSample, String> {
    let psv = PackedSfenValue::from_bytes(record).ok_or("PSV decode failed")?;
    let sfen = unpack_sfen(&psv.sfen)?;

    let mut pos = Position::new();
    pos.set_sfen(&sfen).map_err(|e| format!("SFEN parse error: {e}"))?;

    let input = extract_features(&pos);

    let mut features = [0.0f32; P2_FEATURE_DIM];
    for (i, &v) in input.piece_features.iter().enumerate() {
        features[i] = v as f32;
    }

    let target_winrate = score_to_winrate(psv.score);

    Ok(TrainingSample {
        features,
        target_winrate,
    })
}

/// バッチデコード（rayon 並列）
pub fn decode_batch(records: &[[u8; 40]]) -> Vec<TrainingSample> {
    use rayon::prelude::*;
    records.par_iter().filter_map(|rec| decode_sample(rec).ok()).collect()
}

/// 符号検証: score>0 && game_result==1 が過半数であることをチェック
///
/// 問題がある場合は警告メッセージを返す。
pub fn verify_sign_consistency(records: &[[u8; 40]]) -> Result<(), String> {
    let mut positive_score_win = 0u64;
    let mut total_positive = 0u64;

    for rec in records {
        if let Some(psv) = PackedSfenValue::from_bytes(rec)
            && psv.score > 0
        {
            total_positive += 1;
            if psv.game_result == 1 {
                positive_score_win += 1;
            }
        }
    }

    if total_positive == 0 {
        return Err("score > 0 のレコードが存在しない".to_string());
    }

    let ratio = positive_score_win as f64 / total_positive as f64;
    if ratio < 0.5 {
        Err(format!(
            "符号異常: score>0 && result==1 の比率 = {:.1}% (期待: >50%)",
            ratio * 100.0
        ))
    } else {
        Ok(())
    }
}

// ===========================================================================
// JSONL (dual head: value + policy) ローダ
// ===========================================================================

/// JSONL の生レコード（学習に必要なフィールドのみ抽出）
#[derive(Debug, Clone, Deserialize)]
pub struct JsonlRecord {
    pub sfen: String,
    #[serde(rename = "move")]
    pub best_move: String,
    #[serde(default)]
    pub legal_moves: Vec<String>,
    pub value_cp: i32,
    #[serde(default)]
    pub result: i8,
    #[serde(default)]
    pub ply: u32,
    /// "opening" / "middle" / "endgame" / その他
    #[serde(default)]
    pub phase: String,
    /// データソース名 (source filter 用)
    #[serde(default)]
    pub source: String,
}

/// dual head 学習サンプル
pub struct DualSample {
    pub features: [f32; P2_FEATURE_DIM],
    pub target_winrate: f32,
    /// 元の cp スコア（cp_scale を後から変更したい場合に再計算可能）
    pub raw_value_cp: i32,
    /// 対局結果 (1=win, 0=draw, -1=loss; 手番側基準)
    pub game_result: i8,
    pub target_label: usize,
    pub legal_labels: Vec<u32>,
}

/// score(cp) → 勝率: `sigmoid(clamp(cp, -2000, 2000) / 400)`
#[inline]
fn cp_to_winrate(cp: i32) -> f32 {
    let clamped = (cp as f32).clamp(-2000.0, 2000.0);
    1.0 / (1.0 + (-clamped / 400.0).exp())
}

/// JSONL ファイルからレコードをロードする
///
/// `limit` で読み込む最大件数を制限できる（メモリ節約のため）。
pub fn load_jsonl_records<P: AsRef<Path>>(
    path: P,
    limit: Option<usize>,
) -> io::Result<Vec<JsonlRecord>> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let max = limit.unwrap_or(usize::MAX);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        if let Ok(rec) = serde_json::from_str::<JsonlRecord>(&line) {
            records.push(rec);
            if records.len() >= max {
                break;
            }
        }
    }
    Ok(records)
}

/// JSONL レコードを dual sample にデコードする
///
/// 失敗条件: SFEN パース失敗、best_move パース失敗、legal_moves が空、
/// best_move ラベルが legal labels に含まれない。
pub fn decode_dual_sample(rec: &JsonlRecord) -> Option<DualSample> {
    let mut pos = Position::new();
    pos.set_sfen(&rec.sfen).ok()?;
    let color = pos.side_to_move();

    let input = extract_features(&pos);
    let mut features = [0.0f32; P2_FEATURE_DIM];
    for (i, &v) in input.piece_features.iter().enumerate() {
        features[i] = v as f32;
    }

    let best_mv = Move::from_usi(&rec.best_move)?;
    let target_label = make_move_label(best_mv, color);

    if rec.legal_moves.is_empty() {
        return None;
    }
    let legal_labels: Vec<u32> = rec
        .legal_moves
        .iter()
        .filter_map(|s| Move::from_usi(s))
        .map(|mv| make_move_label(mv, color) as u32)
        .collect();

    if legal_labels.is_empty() {
        return None;
    }
    let target_label_u32 = target_label as u32;
    if !legal_labels.contains(&target_label_u32) {
        return None;
    }

    let target_winrate = cp_to_winrate(rec.value_cp);

    Some(DualSample {
        features,
        target_winrate,
        raw_value_cp: rec.value_cp,
        game_result: rec.result,
        target_label,
        legal_labels,
    })
}

/// cp → winrate 変換 (sigmoid, パラメータ化版)
///
/// `target = sigmoid(clamp(cp, ±cp_clamp) / cp_scale)`
#[inline]
pub fn cp_to_winrate_scaled(cp: i32, cp_clamp: i32, cp_scale: f32) -> f32 {
    let clamped = (cp.clamp(-cp_clamp, cp_clamp)) as f32;
    1.0 / (1.0 + (-clamped / cp_scale).exp())
}

/// バッチデコード（rayon 並列）
pub fn decode_batch_dual(records: &[JsonlRecord]) -> Vec<DualSample> {
    use rayon::prelude::*;
    records.par_iter().filter_map(decode_dual_sample).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_to_winrate_zero() {
        let wr = score_to_winrate(0);
        assert!((wr - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_score_to_winrate_positive() {
        let wr = score_to_winrate(400);
        // sigmoid(1) ≈ 0.7311
        assert!((wr - 0.7311).abs() < 0.01);
    }

    #[test]
    fn test_score_to_winrate_clamp() {
        let wr_high = score_to_winrate(10000);
        let wr_2000 = score_to_winrate(2000);
        assert!((wr_high - wr_2000).abs() < 1e-5, "clamp should cap at 2000");
    }
}
