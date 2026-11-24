//! PSV (PackedSfenValue) フォーマットの Ingester
//!
//! YaneuraOu の `.bin` / `.psv` ファイルを読み込み、`RawRecord` に変換する。

use crate::dataset::ingest::Ingester;
use crate::dataset::types::RawRecord;
use crate::packed_sfen::{PackedSfenValue, move16_to_usi, unpack_sfen};
use anyhow::{Context, Result};
use std::io::Read;
use std::path::Path;

/// PSV Ingester
pub struct PsvIngester;

impl Ingester for PsvIngester {
    fn ingest(
        &self,
        path: &Path,
        source_name: &str,
        callback: &mut dyn FnMut(RawRecord) -> bool,
    ) -> Result<u64> {
        let reader = crate::common::io::open_reader(path)
            .with_context(|| format!("PSV ファイルオープン失敗: {}", path.display()))?;
        let mut reader = std::io::BufReader::with_capacity(8 << 20, reader);

        let mut buf = [0u8; PackedSfenValue::SIZE];
        let mut count = 0u64;

        loop {
            match reader.read_exact(&mut buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e).context("PSV レコード読み込みエラー"),
            }

            let psv = PackedSfenValue::from_bytes(&buf).context("PackedSfenValue のパース失敗")?;

            let sfen = match unpack_sfen(&psv.sfen) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let best_move = move16_to_usi(psv.move16);

            let record = RawRecord {
                sfen,
                best_move,
                score: psv.score as i32,
                game_result: psv.game_result,
                ply: psv.game_ply as u32,
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
