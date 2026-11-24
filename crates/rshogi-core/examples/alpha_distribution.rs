//! ARH v2 構造化 alpha class 分布計測 (= 2026-06-11)。
//!
//! 中盤 ply24 SFEN 1000 件サンプル × 2 perspective で
//! `(attackers, balance, overlap, supported, alpha)` 分布を計測。
//! 飽和の真因 (= 真の表現上限 / 偽の二極化) を判別する。

use rshogi_core::nnue::attack_relay_v2::{
    ARH_V2_NUM_ALPHA, ARH_V2_NUM_ATTACKERS, ARH_V2_NUM_BALANCE, ARH_V2_NUM_OVERLAP,
    ARH_V2_NUM_SUPPORTED, compute_arh_v2_classes,
};
use rshogi_core::position::Position;
use rshogi_core::types::Color;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/training/start_sfens_ply24.txt".to_string());
    let limit: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let f = File::open(&path)?;
    let reader = BufReader::new(f);

    let mut alpha_hist = [0u64; ARH_V2_NUM_ALPHA];
    let mut attackers_hist = [0u64; ARH_V2_NUM_ATTACKERS];
    let mut balance_hist = [0u64; ARH_V2_NUM_BALANCE];
    let mut overlap_hist = [0u64; ARH_V2_NUM_OVERLAP];
    let mut supported_hist = [0u64; ARH_V2_NUM_SUPPORTED];
    let mut count: u64 = 0;

    for line in reader.lines().take(limit) {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut pos = Position::new();
        let sfen_body = line.strip_prefix("sfen ").unwrap_or(line);
        if pos.set_sfen(sfen_body).is_err() {
            continue;
        }
        for persp in [Color::Black, Color::White] {
            let c = compute_arh_v2_classes(&pos, persp);
            alpha_hist[c.alpha] += 1;
            attackers_hist[c.attackers] += 1;
            balance_hist[c.balance] += 1;
            overlap_hist[c.overlap] += 1;
            supported_hist[c.supported] += 1;
            count += 1;
        }
    }

    println!("=== ARH v2 軸分布 (= {} sample × 2 persp = {} obs) ===", limit, count);
    let pct = |x: u64| (x as f64) * 100.0 / (count as f64);
    println!("\nalpha (0..4):");
    for (i, &v) in alpha_hist.iter().enumerate() {
        println!("  {}: {:6} ({:5.1}%)", i, v, pct(v));
    }
    println!("\nattackers (0..3):");
    for (i, &v) in attackers_hist.iter().enumerate() {
        println!("  {}: {:6} ({:5.1}%)", i, v, pct(v));
    }
    println!("\nbalance (0..8):");
    for (i, &v) in balance_hist.iter().enumerate() {
        let label = i as i32 - 4;
        println!("  {} (=raw {:+2}): {:6} ({:5.1}%)", i, label, v, pct(v));
    }
    println!("\noverlap (0..3):");
    for (i, &v) in overlap_hist.iter().enumerate() {
        println!("  {}: {:6} ({:5.1}%)", i, v, pct(v));
    }
    println!("\nsupported (0..3):");
    for (i, &v) in supported_hist.iter().enumerate() {
        println!("  {}: {:6} ({:5.1}%)", i, v, pct(v));
    }
    Ok(())
}
