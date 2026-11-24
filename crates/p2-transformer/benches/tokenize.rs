//! `position_to_tokens` の microbenchmark。
//!
//! 目標: startpos での 1 回呼出しが < 1 μs (i.e. > 1M calls/sec)。
//! complex 局面でも < 2 μs を目安とする。

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use p2_transformer::position_to_tokens;
use rshogi_core::position::Position;

fn startpos() -> Position {
    let mut p = Position::new();
    p.set_hirate();
    p
}

fn complex_position() -> Position {
    let mut p = Position::new();
    // 中盤局面 (3手相振り風の散らばった駒配置 + 手駒あり)。
    // SFEN: 平手から 20 手程度進めたような盤面 (整合性のある駒数)。
    p.set_sfen("ln1g3nl/1ks1g2b1/1pppppppp/pr7/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 21")
        .expect("valid sfen");
    p
}

fn bench_tokenize(c: &mut Criterion) {
    let pos = startpos();
    c.bench_function("position_to_tokens_startpos", |b| {
        b.iter(|| {
            let t = position_to_tokens(black_box(&pos));
            black_box(t);
        });
    });

    let cpos = complex_position();
    c.bench_function("position_to_tokens_complex", |b| {
        b.iter(|| {
            let t = position_to_tokens(black_box(&cpos));
            black_box(t);
        });
    });
}

criterion_group!(benches, bench_tokenize);
criterion_main!(benches);
