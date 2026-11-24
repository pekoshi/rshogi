//! CSA棋譜を USI 手列に変換して出力する簡易ツール。
//!
//! 用途: 棋譜評価値解析のため `position startpos moves ...` 用の USI 手列を生成する。
//! 出力: 1 行 1 手 `<手番index(1始まり)> <usi手>`。最後に全手列を 1 行で出力。

use anyhow::Result;
use rshogi_csa::{ParsedMove, csa_move_to_usi, parse_csa_full};
use std::fs;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: csa2usi_dump <csa-file>");
    let text = fs::read_to_string(&path)?;
    let (mut pos, moves, _info) = parse_csa_full(&text)?;

    let mut usi_moves: Vec<String> = Vec::new();
    for m in &moves {
        match m {
            ParsedMove::Normal(cm) => {
                let usi = csa_move_to_usi(&cm.mv, &pos)?;
                usi_moves.push(usi);
                pos.apply_csa_move(&cm.mv)?;
            }
            ParsedMove::Special(_) => break,
        }
    }

    for (i, mv) in usi_moves.iter().enumerate() {
        println!("{} {}", i + 1, mv);
    }
    println!("ALL {}", usi_moves.join(" "));
    Ok(())
}
