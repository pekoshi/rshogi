//! CSV/TSV フォーマットの Ingester

use crate::dataset::ingest::Ingester;
use crate::dataset::types::RawRecord;
use anyhow::{Context, Result};
use std::io::BufRead;
use std::path::Path;

/// CSV/TSV Ingester
pub struct CsvTsvIngester;

impl Ingester for CsvTsvIngester {
    fn ingest(
        &self,
        path: &Path,
        source_name: &str,
        callback: &mut dyn FnMut(RawRecord) -> bool,
    ) -> Result<u64> {
        let reader = crate::common::io::open_reader(path)
            .with_context(|| format!("CSV/TSV ファイルオープン失敗: {}", path.display()))?;

        let mut lines = reader.lines();

        // ヘッダ行を読み取り、カラムインデックスを確定
        let header = lines.next().context("CSV/TSV ファイルが空です")??;
        let sep = if header.contains('\t') { '\t' } else { ',' };
        let columns: Vec<&str> = header.split(sep).map(|s| s.trim()).collect();

        let col_index = |name: &str| -> Option<usize> {
            columns.iter().position(|&c| c.eq_ignore_ascii_case(name))
        };

        let sfen_idx = col_index("sfen").context("'sfen' カラムが見つかりません")?;
        let move_idx = col_index("move").or_else(|| col_index("best_move"));
        let score_idx = col_index("score").or_else(|| col_index("value_cp"));
        let result_idx = col_index("result");
        let ply_idx = col_index("ply");

        let mut count = 0u64;
        for line in lines {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let fields: Vec<&str> = trimmed.split(sep).collect();
            let sfen = fields.get(sfen_idx).unwrap_or(&"").trim().to_string();
            if sfen.is_empty() {
                continue;
            }

            let best_move = move_idx
                .and_then(|i| fields.get(i))
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "none".to_string());

            let score = score_idx
                .and_then(|i| fields.get(i))
                .and_then(|s| s.trim().parse::<i32>().ok())
                .unwrap_or(0);

            let game_result = result_idx
                .and_then(|i| fields.get(i))
                .and_then(|s| s.trim().parse::<i8>().ok())
                .unwrap_or(0);

            let ply = ply_idx
                .and_then(|i| fields.get(i))
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or(1);

            let record = RawRecord {
                sfen,
                best_move,
                score,
                game_result,
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
