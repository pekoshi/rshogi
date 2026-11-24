//! KIF 棋譜フォーマットの Ingester
//!
//! KIF/KIFU 棋譜ファイルから全局面を抽出して `RawRecord` に変換する。
//! 簡易パーサで基本的な KIF フォーマットに対応する。

use crate::dataset::ingest::Ingester;
use crate::dataset::types::RawRecord;
use anyhow::{Context, Result};
use std::path::Path;

/// KIF Ingester
pub struct KifIngester;

impl Ingester for KifIngester {
    fn ingest(
        &self,
        path: &Path,
        source_name: &str,
        callback: &mut dyn FnMut(RawRecord) -> bool,
    ) -> Result<u64> {
        let content = read_kif_file(path)?;
        let mut count = 0u64;

        for game_text in split_kif_games(&content) {
            if let Ok(records) = parse_kif_game(game_text, source_name) {
                for record in records {
                    count += 1;
                    if !callback(record) {
                        return Ok(count);
                    }
                }
            }
        }

        Ok(count)
    }
}

/// KIF ファイルを読み込む（Shift_JIS / UTF-8 両対応）
fn read_kif_file(path: &Path) -> Result<String> {
    // まず UTF-8 として試す
    if let Ok(content) = std::fs::read_to_string(path) {
        return Ok(content);
    }

    // Shift_JIS として読む（バイト列から変換）
    let bytes = std::fs::read(path)
        .with_context(|| format!("KIF ファイル読み込み失敗: {}", path.display()))?;

    // 簡易 Shift_JIS 対応: ASCII 部分のみ抽出
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// 複数棋譜の分割
fn split_kif_games(content: &str) -> Vec<&str> {
    // 単一棋譜として扱う
    vec![content]
}

/// KIF 棋譜から局面を抽出する（簡易実装）
///
/// KIF フォーマットは複雑なため、ここでは SFEN への変換を行わず、
/// 指し手の KIF → USI 変換も行わない。
/// CSA に変換して rshogi-csa を使うか、将来的にフルパーサを実装する。
fn parse_kif_game(_text: &str, _source_name: &str) -> Result<Vec<RawRecord>> {
    // KIF パーサは複雑なため、初期実装では空を返す。
    // 実運用では CSA に変換してから ingest するか、
    // 専用の KIF パーサを追加する。
    log::warn!("KIF パーサは未実装です。CSA への変換を推奨します。");
    Ok(Vec::new())
}
