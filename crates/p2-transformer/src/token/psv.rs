//! PSV (Packed SFEN with Value) 40 byte fixed record → Transformer 訓練 record の
//! 変換 skeleton。
//!
//! ## PSV record layout (やねうら王 / Bullet 互換、 40 bytes)
//!
//! ```text
//! offset  size  field
//! 0       32    packed_sfen   (256bit、 駒配置 + side + 手駒)
//! 32      2     score         (i16、 cp 単位、 black 視点)
//! 34      2     move          (u16、 着手)
//! 36      2     game_ply      (u16)
//! 38      1     game_result   (i8、 -1/0/+1 = Black 負け/引分/勝ち)
//! 39      1     padding       (0x00)
//! ```
//!
//! ## このモジュールの責務
//!
//! - 40 byte の生 record を [`PsvRecord`] にパース (allocation 0)。
//! - PSV から [`Position`] を再構築する処理は別 crate に既存実装がある想定で、
//!   ここでは parse のみ提供する。 実際の Transformer 訓練 record (token + policy
//!   target + value target) への変換は CLI binary 側 (TODO) で組み合わせる。
//!
//! ## TODO
//!
//! - PSV unpack (packed_sfen → Position) は本 task の責務外。 別 binary crate
//!   (例: `tools/build_transformer_dataset`) でこの [`PsvRecord::parse`] + 既存
//!   PSV → Position 関数 + [`super::convert::position_to_tokens`] を合成する。
//! - dlshogi 互換の `make_move_label` (Move → policy index 0..2187) も別 crate で実装。

use crate::token::convert::{TransformerTokens, position_to_tokens};

/// PSV record の固定サイズ。
pub const PSV_RECORD_SIZE: usize = 40;

/// 1 件の PSV 教師局面 (parsed)。
#[derive(Clone, Copy, Debug)]
pub struct PsvRecord {
    /// 32 byte 圧縮局面 (unpack は別所で行う)。
    pub packed_sfen: [u8; 32],
    /// 評価値 (cp、 Black 視点)。
    pub score: i16,
    /// 着手 (u16 encoding、 やねうら王/Bullet 仕様)。
    pub move_u16: u16,
    /// 手数。
    pub game_ply: u16,
    /// 対局結果 (-1 / 0 / +1)。
    pub game_result: i8,
    /// padding byte (通常 0)。
    pub padding: u8,
}

impl PsvRecord {
    /// 40 byte slice からパース。 length 不足は `None`。
    ///
    /// allocation 0、 byte コピー 1 回のみ (`packed_sfen` のみ、 残りは値型コピー)。
    #[inline]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < PSV_RECORD_SIZE {
            return None;
        }
        let mut packed_sfen = [0u8; 32];
        packed_sfen.copy_from_slice(&bytes[0..32]);
        let score = i16::from_le_bytes([bytes[32], bytes[33]]);
        let move_u16 = u16::from_le_bytes([bytes[34], bytes[35]]);
        let game_ply = u16::from_le_bytes([bytes[36], bytes[37]]);
        let game_result = bytes[38] as i8;
        let padding = bytes[39];
        Some(PsvRecord {
            packed_sfen,
            score,
            move_u16,
            game_ply,
            game_result,
            padding,
        })
    }
}

/// Transformer 訓練 1 件分の record。
///
/// 実 dataset では [`PsvRecord`] から構築する。 ここでは `Position` から直接構築
/// する変換器のみを提供する (PSV unpack は別所)。
#[derive(Clone, Copy)]
pub struct TransformerSample {
    pub tokens: TransformerTokens,
    /// 評価値 (cp、 Black 視点)。 loss 計算側で scale する。
    pub value_cp: i16,
    /// 対局結果 (-1 / 0 / +1)。
    pub game_result: i8,
    /// policy target (dlshogi `make_move_label` の出力、 0..2187)。
    ///
    /// PSV → policy_idx 変換は本 crate の責務外。 CLI 側で実装する。
    pub policy_idx: u16,
}

/// `Position` + 教師情報から `TransformerSample` を構築するヘルパ。
///
/// PSV unpack 部分は別 crate に依存するため、 ここでは Position 経由の薄い wrapper のみ。
#[inline]
pub fn make_sample(
    pos: &rshogi_core::position::Position,
    value_cp: i16,
    game_result: i8,
    policy_idx: u16,
) -> TransformerSample {
    TransformerSample {
        tokens: position_to_tokens(pos),
        value_cp,
        game_result,
        policy_idx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_zero_record() {
        let buf = [0u8; PSV_RECORD_SIZE];
        let rec = PsvRecord::parse(&buf).expect("parse zero record");
        assert_eq!(rec.score, 0);
        assert_eq!(rec.move_u16, 0);
        assert_eq!(rec.game_ply, 0);
        assert_eq!(rec.game_result, 0);
        assert_eq!(rec.padding, 0);
        assert_eq!(rec.packed_sfen, [0u8; 32]);
    }

    #[test]
    fn parse_too_short_returns_none() {
        let buf = [0u8; 39];
        assert!(PsvRecord::parse(&buf).is_none());
    }

    #[test]
    fn parse_decodes_little_endian() {
        let mut buf = [0u8; PSV_RECORD_SIZE];
        // score = -1
        buf[32] = 0xFF;
        buf[33] = 0xFF;
        // move = 0x1234
        buf[34] = 0x34;
        buf[35] = 0x12;
        // game_ply = 0xABCD
        buf[36] = 0xCD;
        buf[37] = 0xAB;
        buf[38] = 0xFF; // -1 as i8
        buf[39] = 0xEE;
        let rec = PsvRecord::parse(&buf).unwrap();
        assert_eq!(rec.score, -1);
        assert_eq!(rec.move_u16, 0x1234);
        assert_eq!(rec.game_ply, 0xABCD);
        assert_eq!(rec.game_result, -1);
        assert_eq!(rec.padding, 0xEE);
    }
}
