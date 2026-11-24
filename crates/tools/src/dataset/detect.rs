//! 入力フォーマット自動検出
//!
//! 拡張子と先頭バイトから入力ファイルのフォーマットを判定する。

use crate::dataset::types::InputFormat;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// ファイルパスからフォーマットを自動検出する
pub fn detect_format(path: &Path) -> Result<InputFormat> {
    // まず拡張子で判定（.gz の場合は二重拡張子を確認）
    let ext = effective_extension(path);

    match ext.as_str() {
        "bin" | "psv" => return Ok(InputFormat::Psv),
        "csa" => return Ok(InputFormat::Csa),
        "kif" | "kifu" => return Ok(InputFormat::Kif),
        "jsonl" => return Ok(InputFormat::Jsonl),
        "csv" | "tsv" => return Ok(InputFormat::CsvTsv),
        "sfen" => return Ok(InputFormat::Sfen),
        _ => {}
    }

    // 拡張子で判定できない場合は先頭バイトを確認
    detect_by_content(path)
}

/// 実効的な拡張子を取得する（.gz の場合は二重拡張子）
fn effective_extension(path: &Path) -> String {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();

    if let Some(stem) = name.strip_suffix(".gz")
        && let Some(pos) = stem.rfind('.')
    {
        return stem[pos + 1..].to_ascii_lowercase();
    }

    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

/// ファイル先頭バイトからフォーマットを判定する
fn detect_by_content(path: &Path) -> Result<InputFormat> {
    let mut file =
        File::open(path).with_context(|| format!("ファイルオープン失敗: {}", path.display()))?;
    let mut buf = [0u8; 256];
    let n = file.read(&mut buf)?;
    let header = &buf[..n];

    if header.is_empty() {
        anyhow::bail!("空のファイルです: {}", path.display());
    }

    // JSONL: { で始まる
    if header[0] == b'{' {
        return Ok(InputFormat::Jsonl);
    }

    // CSA: "V2" or "PI" or "P1" で始まる
    if header.starts_with(b"V2") || header.starts_with(b"PI") || header.starts_with(b"P1") {
        return Ok(InputFormat::Csa);
    }

    // KIF: "#KIF" or 日本語ヘッダ
    if header.starts_with(b"#KIF") {
        return Ok(InputFormat::Kif);
    }
    // UTF-8 "手合割" or "手数" のチェック
    let header_str = std::str::from_utf8(header).unwrap_or_default();
    if header_str.starts_with("手合割") || header_str.starts_with("手数") {
        return Ok(InputFormat::Kif);
    }

    // SFEN テキスト: ASCII で SFEN パターンに一致
    if is_likely_sfen_text(header_str) {
        return Ok(InputFormat::Sfen);
    }

    // CSV/TSV: ヘッダに "sfen" を含む
    if let Some(first_line) = header_str.lines().next() {
        let lower = first_line.to_ascii_lowercase();
        if lower.contains("sfen") && (lower.contains(',') || lower.contains('\t')) {
            return Ok(InputFormat::CsvTsv);
        }
    }

    // PSV: 40バイト以上で PackedSfenValue として解釈可能か
    if n >= 40 {
        return Ok(InputFormat::Psv);
    }

    anyhow::bail!("フォーマットを自動検出できませんでした: {}", path.display())
}

/// SFEN テキストとして妥当そうか簡易チェック
fn is_likely_sfen_text(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let first_line = s.lines().next().unwrap_or_default();
    // SFEN は "l" or "+" or 数字で始まり、" b " or " w " を含む
    let trimmed = first_line.trim();
    if trimmed.is_empty() {
        return false;
    }
    trimmed.starts_with(|c: char| "lnsgkrpLNSGKRP123456789+".contains(c))
        && (trimmed.contains(" b ") || trimmed.contains(" w "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_extension_handles_gz() {
        assert_eq!(effective_extension(Path::new("data.jsonl.gz")), "jsonl");
        assert_eq!(effective_extension(Path::new("data.bin")), "bin");
        assert_eq!(effective_extension(Path::new("data.psv.gz")), "psv");
    }

    #[test]
    fn is_likely_sfen_text_positive() {
        assert!(is_likely_sfen_text(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        ));
    }

    #[test]
    fn is_likely_sfen_text_negative() {
        assert!(!is_likely_sfen_text(""));
        assert!(!is_likely_sfen_text("{\"sfen\": \"...\"}"));
    }
}
