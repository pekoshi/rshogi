//! SFEN テキストフォーマットの Ingester
//!
//! 1行1局面の SFEN テキストファイルを読み込む。
//! フォーマット: `<sfen> moves <bestmove> <score> <result>`
//! または: `<sfen>`（score/result なしの場合はデフォルト値）

use crate::dataset::ingest::Ingester;
use crate::dataset::types::RawRecord;
use anyhow::{Context, Result};
use std::io::BufRead;
use std::path::Path;

/// SFEN テキスト Ingester
pub struct SfenIngester;

impl Ingester for SfenIngester {
    fn ingest(
        &self,
        path: &Path,
        source_name: &str,
        callback: &mut dyn FnMut(RawRecord) -> bool,
    ) -> Result<u64> {
        let reader = crate::common::io::open_reader(path)
            .with_context(|| format!("SFEN ファイルオープン失敗: {}", path.display()))?;

        let mut count = 0u64;
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if let Some(record) = parse_sfen_line(trimmed, source_name) {
                count += 1;
                if !callback(record) {
                    break;
                }
            }
        }
        Ok(count)
    }
}

/// SFEN 行をパースする
///
/// 対応フォーマット:
/// 1. `<sfen>` — SFEN のみ（score=0, result=0, move=none）
/// 2. `<sfen>\t<move>\t<score>\t<result>` — タブ区切り
/// 3. `<sfen> moves <move>` — USI 拡張
fn parse_sfen_line(line: &str, source_name: &str) -> Option<RawRecord> {
    // タブ区切りの場合
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.len() >= 4 {
        let sfen = parts[0].to_string();
        let best_move = parts[1].to_string();
        let score = parts[2].parse::<i32>().unwrap_or(0);
        let result = parts[3].parse::<i8>().unwrap_or(0);
        let ply = extract_ply_from_sfen(&sfen);
        return Some(RawRecord {
            sfen,
            best_move,
            score,
            game_result: result,
            ply,
            source_name: source_name.to_string(),
        });
    }

    // "moves" を含む場合
    if let Some(idx) = line.find(" moves ") {
        let sfen = line[..idx].to_string();
        let rest = &line[idx + 7..];
        let best_move = rest.split_whitespace().next().unwrap_or("none").to_string();
        let ply = extract_ply_from_sfen(&sfen);
        return Some(RawRecord {
            sfen,
            best_move,
            score: 0,
            game_result: 0,
            ply,
            source_name: source_name.to_string(),
        });
    }

    // SFEN のみ
    let sfen = line.to_string();
    let ply = extract_ply_from_sfen(&sfen);
    Some(RawRecord {
        sfen,
        best_move: "none".to_string(),
        score: 0,
        game_result: 0,
        ply,
        source_name: source_name.to_string(),
    })
}

/// SFEN 文字列から手数を抽出する
fn extract_ply_from_sfen(sfen: &str) -> u32 {
    sfen.split_whitespace().nth(3).and_then(|s| s.parse::<u32>().ok()).unwrap_or(1)
}
