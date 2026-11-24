//! PSV (Packed SFEN with Value) reader。
//!
//! やねうら王 / Bullet 互換 40-byte record をパースし、 訓練時に必要な
//! `score / move / game_result` 等を取り出す。
//!
//! packed_sfen の unpack (32 byte → Position) は本 crate の責務外。
//! 既存 `crates/p2-transformer/src/token/psv.rs` の同型 struct を再現しているが、
//! 本 crate は p2-transformer 依存を避けるため独立実装にしてある。
//!
//! 実訓練の dataset では「PSV → Position 経由で `encode_position_to_tokens`
//! に渡す」or 事前 tokenize した binary を使う、 のいずれか。
//!
//! 現状の `train_nnue_b` binary は dataset レイヤを抽象化しており、
//! PSV 直接読みは `--data <PSV>` 指定時、 token cache は `--tokens <BIN>` 指定時に
//! それぞれ enable する。 details は `bin/train_nnue_b.rs` を参照。

use anyhow::Result;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// PSV record サイズ (40 byte fixed)。
pub const PSV_RECORD_SIZE: usize = 40;

/// 1 件の PSV 教師局面。
#[derive(Clone, Copy, Debug)]
pub struct PsvRecord {
    pub packed_sfen: [u8; 32],
    /// 評価値 (cp、 Black 視点)。
    pub score: i16,
    /// 着手 (u16、 やねうら王 encoding)。
    pub move_u16: u16,
    /// 手数。
    pub game_ply: u16,
    /// 対局結果 (-1 / 0 / +1、 Black 視点)。
    pub game_result: i8,
    pub padding: u8,
}

impl PsvRecord {
    /// 40 byte slice からパース。 長さ不足は `None`。
    #[inline]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < PSV_RECORD_SIZE {
            return None;
        }
        let mut packed_sfen = [0u8; 32];
        packed_sfen.copy_from_slice(&bytes[0..32]);
        Some(PsvRecord {
            packed_sfen,
            score: i16::from_le_bytes([bytes[32], bytes[33]]),
            move_u16: u16::from_le_bytes([bytes[34], bytes[35]]),
            game_ply: u16::from_le_bytes([bytes[36], bytes[37]]),
            game_result: bytes[38] as i8,
            padding: bytes[39],
        })
    }
}

/// PSV ファイルを fixed-size record 順に streaming する iterator。
pub struct PsvReader<R: Read> {
    inner: BufReader<R>,
    buf: [u8; PSV_RECORD_SIZE],
}

impl PsvReader<File> {
    /// ファイルパスから PSV reader を作る。
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            inner: BufReader::with_capacity(1 << 20, file),
            buf: [0u8; PSV_RECORD_SIZE],
        })
    }
}

impl<R: Read> PsvReader<R> {
    /// 任意 `Read` から構築。
    pub fn from_reader(r: R) -> Self {
        Self {
            inner: BufReader::with_capacity(1 << 20, r),
            buf: [0u8; PSV_RECORD_SIZE],
        }
    }

    /// 次の record を 1 件読む。 EOF で `Ok(None)`。
    pub fn next_record(&mut self) -> Result<Option<PsvRecord>> {
        match self.inner.read_exact(&mut self.buf) {
            Ok(()) => Ok(PsvRecord::parse(&self.buf)),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parse_zero_record() {
        let buf = [0u8; PSV_RECORD_SIZE];
        let rec = PsvRecord::parse(&buf).expect("parse zero record");
        assert_eq!(rec.score, 0);
        assert_eq!(rec.move_u16, 0);
        assert_eq!(rec.game_ply, 0);
        assert_eq!(rec.game_result, 0);
    }

    #[test]
    fn parse_too_short_returns_none() {
        let buf = [0u8; 39];
        assert!(PsvRecord::parse(&buf).is_none());
    }

    #[test]
    fn reader_streams_two_records() -> Result<()> {
        let mut data = vec![0u8; PSV_RECORD_SIZE * 2];
        data[32] = 100;
        data[PSV_RECORD_SIZE + 32] = 200;
        let mut reader = PsvReader::from_reader(Cursor::new(data));
        let r1 = reader.next_record()?.unwrap();
        let r2 = reader.next_record()?.unwrap();
        let r3 = reader.next_record()?;
        assert_eq!(r1.score, 100);
        assert_eq!(r2.score, 200i16.wrapping_neg().wrapping_neg());
        assert!(r3.is_none());
        Ok(())
    }
}
