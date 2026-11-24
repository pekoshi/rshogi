//! データ取り込み（Ingester）モジュール
//!
//! 各入力フォーマットに対応する Ingester を提供する。

pub mod csa;
pub mod csv_tsv;
pub mod jsonl;
pub mod kif;
pub mod psv;
pub mod sfen;

use crate::dataset::types::{InputFormat, RawRecord};
use anyhow::Result;
use std::path::Path;

/// ストリーミング Ingester trait
///
/// 入力ファイルを1レコードずつ読み込み、`RawRecord` に変換して
/// コールバックに渡す。
pub trait Ingester {
    /// ファイルからレコードをストリーミング読み込みする。
    ///
    /// 各レコードに対して `callback` を呼び出す。
    /// `callback` が `false` を返した場合、読み込みを中断する。
    fn ingest(
        &self,
        path: &Path,
        source_name: &str,
        callback: &mut dyn FnMut(RawRecord) -> bool,
    ) -> Result<u64>;
}

/// フォーマットに応じた Ingester を生成する
pub fn create_ingester(format: InputFormat) -> Box<dyn Ingester> {
    match format {
        InputFormat::Psv => Box::new(psv::PsvIngester),
        InputFormat::Sfen => Box::new(sfen::SfenIngester),
        InputFormat::Csa => Box::new(csa::CsaIngester),
        InputFormat::Kif => Box::new(kif::KifIngester),
        InputFormat::Jsonl => Box::new(jsonl::JsonlIngester),
        InputFormat::CsvTsv => Box::new(csv_tsv::CsvTsvIngester),
    }
}
