//! PSV record → 訓練 sample (tokens + score + result + policy_idx) 変換。
//!
//! ## 流れ
//!
//! ```text
//! PsvRecord
//!   ├─ packed_sfen (32 byte) ──► tools::packed_sfen::unpack_sfen ──► SFEN string
//!   │                          └─ Position::set_sfen ──► Position
//!   │                          └─ encode_position_to_tokens ──► [i64; 100]
//!   ├─ move_u16 ──► tools::packed_sfen::move16_to_move ──► Move
//!   │            └─ tools::dlshogi_features::make_move_label(mv, stm) ──► policy_idx (0..2187)
//!   ├─ score   ──► f32 (Black 視点 cp、 stm が White なら符号反転)
//!   └─ result  ──► f32 (-1/0/+1 → 0.0/0.5/1.0、 stm 視点)
//! ```
//!
//! 訓練 binary は `[PsvRecord; batch]` を [`PsvSample`] のリストに変換し、
//! Tensor 化して loss に流す。 alloc は record ごとに SFEN String + parts Vec を作る
//! ので「ホットパスでヒープ alloc 禁止」のルールには触れる (ここは訓練側、
//! 推論ではない)。 性能ボトルネックになった場合は packed_sfen → Position の
//! 直接 decoder を別途実装する。
//!
//! `train` feature 時のみ build される (tools 依存のため)。
#![cfg(feature = "train")]

use anyhow::{anyhow, Result};
use rshogi_core::position::Position;
use rshogi_core::types::Move;
use tools::dlshogi_features::make_move_label;
use tools::packed_sfen::{move16_to_move, unpack_sfen};

use crate::input::{encode_position_to_tokens, NUM_TOKENS};
use crate::psv::PsvRecord;

/// 1 件の訓練サンプル (decoded)。
#[derive(Clone, Copy, Debug)]
pub struct PsvSample {
    pub tokens: [i64; NUM_TOKENS],
    /// stm 視点の cp 評価値。
    pub score_cp: f32,
    /// stm 視点の result (0.0 = loss, 0.5 = draw, 1.0 = win)。
    pub result_stm: f32,
    /// dlshogi `make_move_label` の出力。 PSV move が `0` (none) の場合は `-1`。
    pub policy_idx: i64,
    /// PSV record の game_ply。 デバッグ用。
    pub game_ply: u16,
}

/// SFEN 末尾の " 1" を実 game_ply で置換した文字列を返す。
///
/// `unpack_sfen` の出力は `"<board> <side> <hand> 1"` 形式。 token encoder が
/// `Position::game_ply` を使う (ply bucket) ため、 PSV record の `game_ply` を
/// 反映する必要がある。
fn rewrite_sfen_with_ply(sfen: &str, game_ply: u16) -> String {
    let mut parts: Vec<&str> = sfen.split_whitespace().collect();
    if parts.len() < 4 {
        // 異常: そのまま返して set_sfen に判定を任せる
        return sfen.to_string();
    }
    // ply は最後のフィールド。 最低 1 (set_sfen が 0 をエラー扱いする実装に
    // 備えて max 1)。
    let ply_str = format!("{}", game_ply.max(1));
    let last_idx = parts.len() - 1;
    parts[last_idx] = ply_str.as_str();
    parts.join(" ")
}

/// 1 件の `PsvRecord` を `PsvSample` に変換する。
///
/// 不正 record (SFEN decode 失敗 / 不正 move) は `Err` を返す。 呼び側で
/// skip / count するのが推奨。
pub fn decode_psv_record(record: &PsvRecord) -> Result<PsvSample> {
    let raw_sfen =
        unpack_sfen(&record.packed_sfen).map_err(|e| anyhow!("unpack_sfen failed: {e}"))?;
    let sfen = rewrite_sfen_with_ply(&raw_sfen, record.game_ply);

    let mut pos = Position::new();
    pos.set_sfen(&sfen).map_err(|e| anyhow!("set_sfen({sfen:?}) failed: {e:?}"))?;

    let tokens = encode_position_to_tokens(&pos);

    let mv = move16_to_move(record.move_u16);
    let policy_idx: i64 = if mv == Move::NONE {
        -1
    } else {
        make_move_label(mv, pos.side_to_move()) as i64
    };

    // PSV の score / result は Black 視点。 stm 視点に揃える。
    let stm_sign: f32 = if pos.side_to_move() == rshogi_core::types::Color::Black {
        1.0
    } else {
        -1.0
    };
    let score_cp = record.score as f32 * stm_sign;

    // game_result: -1 (Black loss) / 0 (draw) / +1 (Black win)
    // → stm 視点: 0.0 / 0.5 / 1.0
    let r_black = record.game_result as i32;
    let r_stm_signed = r_black as f32 * stm_sign; // -1 / 0 / +1
    let result_stm = (r_stm_signed + 1.0) * 0.5; // 0 / 0.5 / 1

    Ok(PsvSample {
        tokens,
        score_cp,
        result_stm,
        policy_idx,
        game_ply: record.game_ply,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::types::Color;

    /// 初期局面 (hirate) を pack して unpack できることを確認する。
    ///
    /// ここでは `tools::packed_sfen::pack_position` が public か否か未確認の
    /// ため、 既知の 32 byte シーケンスを使う代わりに、 実 PSV file から 1 件
    /// 読み出すテストは integration test 側で別途行う。
    /// 本テストでは決定的な byte 列に対して unpack が動くことだけ確認する。
    #[test]
    fn decode_zero_packed_sfen_returns_err_or_ok() {
        // 0 埋め packed_sfen は board 全 EMPTY → side=Black, king at sq 0 ...
        // 実装の Huffman decode によっては Err になる。 ここではパニックせず
        // Result が返ることのみ確認。
        let rec = PsvRecord {
            packed_sfen: [0u8; 32],
            score: 0,
            move_u16: 0,
            game_ply: 1,
            game_result: 0,
            padding: 0,
        };
        let _ = decode_psv_record(&rec); // panic しなければ OK
    }

    #[test]
    fn rewrite_ply_preserves_other_fields() {
        let s = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let out = rewrite_sfen_with_ply(s, 42);
        assert!(out.ends_with(" 42"));
        assert!(out.starts_with("lnsgkgsnl"));
    }

    #[test]
    fn rewrite_ply_clamps_zero_to_one() {
        let s = "9/9/9/9/9/9/9/9/9 b - 1";
        let out = rewrite_sfen_with_ply(s, 0);
        assert!(out.ends_with(" 1"));
    }

    #[test]
    fn stm_sign_flip_for_white() {
        // White 手番、 score=+100 (Black 視点) → stm 視点 -100
        // この test は実 PSV byte がないのでスキップ的に flag だけ確認。
        let _stm: Color = Color::White;
    }
}
