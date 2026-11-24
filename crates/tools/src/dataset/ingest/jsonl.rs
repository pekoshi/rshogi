//! JSONL フォーマットの Ingester（再取り込み用）

use crate::dataset::ingest::Ingester;
use crate::dataset::types::RawRecord;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::BufRead;
use std::path::Path;

/// JSONL Ingester
pub struct JsonlIngester;

/// JSONL の最低限のフィールド
#[derive(Deserialize)]
struct JsonlRecord {
    sfen: String,
    #[serde(rename = "move", default)]
    best_move: Option<String>,
    #[serde(default)]
    value_cp: Option<i32>,
    #[serde(default)]
    score: Option<i32>,
    #[serde(default)]
    result: Option<i8>,
    #[serde(default)]
    ply: Option<u32>,
}

impl Ingester for JsonlIngester {
    fn ingest(
        &self,
        path: &Path,
        source_name: &str,
        callback: &mut dyn FnMut(RawRecord) -> bool,
    ) -> Result<u64> {
        let reader = crate::common::io::open_reader(path)
            .with_context(|| format!("JSONL ファイルオープン失敗: {}", path.display()))?;

        let mut count = 0u64;
        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let rec: JsonlRecord = serde_json::from_str(trimmed)
                .with_context(|| format!("JSONL パースエラー (行 {})", line_num + 1))?;

            let score = rec.value_cp.or(rec.score).unwrap_or(0);
            let ply = rec.ply.unwrap_or(1);

            let record = RawRecord {
                sfen: rec.sfen,
                best_move: rec.best_move.unwrap_or_else(|| "none".to_string()),
                score,
                game_result: rec.result.unwrap_or(0),
                ply,
                source_name: source_name.to_string(),
            };

            count += 1;
            if !callback(record) {
                break;
            }
        }
        Ok(count)
    }
}
