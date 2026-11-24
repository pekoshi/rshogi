//! KZT_HAND active feature 内訳ダンプ (= 2026-07-02、 tail 局面の feature 別解析用)。
//!
//! stdin から 1 行 1 sfen (= `position sfen ...` 接頭辞も可) を読み、 各局面の
//! **stm 側 (= 攻め手)** の KZT_HAND active feature を [`summarize_hand_features`] で
//! 集計し、 tab 区切りで出力する。 weight 非依存 (= 局面が立てる feature の組成のみ)。
//!
//! 出力: `total\tds:<16csv>\thp:<7csv>\tkr:<16csv>` (= drop_state / hand_piece / king_region)。

use rshogi_core::nnue::kzt::summarize_hand_features;
use rshogi_core::position::Position;
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let body = line
            .strip_prefix("position sfen ")
            .or_else(|| line.strip_prefix("sfen "))
            .unwrap_or(line);
        let mut pos = Position::new();
        if pos.set_sfen(body).is_err() {
            writeln!(out, "ERR")?;
            continue;
        }
        let stm = pos.side_to_move();
        let s = summarize_hand_features(&pos, stm, 0);
        let csv = |a: &[usize]| a.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
        writeln!(
            out,
            "{}\tds:{}\thp:{}\tkr:{}",
            s.total,
            csv(&s.by_drop_state),
            csv(&s.by_hand_piece),
            csv(&s.by_king_region)
        )?;
    }
    Ok(())
}
