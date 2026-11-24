//! ARH v2 head の純コスト microbench (= 2026-06-15)。
//!
//! `evaluate_arh_v2` を固定局面集合に対して大量反復実行し、 1 call あたりの
//! ns を計測する。 探索やシステム負荷から独立して head 計算コストのみを
//! 切り出すための計測道具 (= search ベース NPS 計測は並行 tournament 負荷で
//! 分散が大きいため、 本 microbench で head 純コストを確定する)。
//!
//! 使い方:
//! ```bash
//! cargo run --release -p tools --bin arh_v2_microbench \
//!   --features nnue-attack-relay-v2 -- --iters 20000000
//! ```

#[cfg(not(feature = "nnue-attack-relay-v2"))]
fn main() {
    eprintln!(
        "feature `nnue-attack-relay-v2` が無効です。 --features nnue-attack-relay-v2 を付与してください。"
    );
    std::process::exit(2);
}

#[cfg(feature = "nnue-attack-relay-v2")]
fn main() {
    use rshogi_core::nnue::attack_relay_v2::{
        ArhV2Weights, build_runtime_weights, evaluate_arh_v2,
    };
    use rshogi_core::position::Position;
    use rshogi_core::types::Color;
    use std::time::Instant;

    // --iters N (= 総反復回数、 局面集合を巡回)。
    let mut iters: u64 = 20_000_000;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--iters" && i + 1 < args.len() {
            iters = args[i + 1].parse().unwrap_or(iters);
            i += 2;
        } else {
            i += 1;
        }
    }

    // 計測局面 (= 序盤 / 複雑中盤 / 入玉気味 / 戦術)。
    let sfens = [
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "ln1g5/1ks1g4/1pppp2+Bp/p4pp2/9/2P6/PPSPPPP1P/2G1K2R1/LN1G3NL b SNLPbr 1",
        "1+R3g1n1/3+P2k2/p1p1pp1pp/9/9/2P6/P3PPPPP/9/1N1K1G1NL w GSrb2sn3l3p 1",
        "l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",
    ];
    let mut positions: Vec<Position> = Vec::new();
    for s in &sfens {
        let mut p = Position::new();
        p.set_sfen(s).expect("valid sfen");
        positions.push(p);
    }

    // 非ゼロ weight runtime (= 実際の分岐コストを反映するため全 weight に 1 を入れる)。
    let mut disk = ArhV2Weights::zeroed();
    fill_ones(disk.as_mut());
    let rt = build_runtime_weights(disk.as_ref());

    // warmup。
    let mut acc: i64 = 0;
    for _ in 0..1_000_000u64 {
        for (k, p) in positions.iter().enumerate() {
            let bucket = k % 8;
            let stm = if k % 2 == 0 {
                Color::Black
            } else {
                Color::White
            };
            acc = acc.wrapping_add(evaluate_arh_v2(&rt, p, bucket, stm) as i64);
        }
    }

    let n = positions.len() as u64;
    let loops = iters / n;
    let start = Instant::now();
    for _ in 0..loops {
        for (k, p) in positions.iter().enumerate() {
            let bucket = k % 8;
            let stm = if k % 2 == 0 {
                Color::Black
            } else {
                Color::White
            };
            acc = acc.wrapping_add(evaluate_arh_v2(&rt, p, bucket, stm) as i64);
        }
    }
    let elapsed = start.elapsed();
    let total_calls = loops * n;
    let ns_per_call = elapsed.as_nanos() as f64 / total_calls as f64;

    // acc を print して最適化で消されないようにする。
    println!(
        "evaluate_arh_v2: calls={} elapsed={:.3}s ns/call={:.2} (acc={})",
        total_calls,
        elapsed.as_secs_f64(),
        ns_per_call,
        acc
    );
}

#[cfg(feature = "nnue-attack-relay-v2")]
fn fill_ones(disk: &mut rshogi_core::nnue::attack_relay_v2::ArhV2Weights) {
    // 全 weight テーブルに 1 を入れる (= 分岐・参照コストを実発火させる)。
    for a in disk.w_base.iter_mut() {
        for b in a.iter_mut() {
            for o in b.iter_mut() {
                for s in o.iter_mut() {
                    *s = 1;
                }
            }
        }
    }
    for al in disk.w_alpha.iter_mut() {
        for a in al.iter_mut() {
            for b in a.iter_mut() {
                *b = 1;
            }
        }
    }
    for ph in disk.w_phase.iter_mut() {
        for a in ph.iter_mut() {
            for b in a.iter_mut() {
                *b = 1;
            }
        }
    }
    for al in disk.w_alpha_shape.iter_mut() {
        for o in al.iter_mut() {
            for s in o.iter_mut() {
                *s = 1;
            }
        }
    }
    for h in disk.w_heavy.iter_mut() {
        for b in h.iter_mut() {
            *b = 1;
        }
    }
}
