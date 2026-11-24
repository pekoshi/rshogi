//! H1 (qsearch 葉と学習分布のズレ) の第一歩: 学習 psv のうち
//! 「教師 best move が capture」「王手中」の割合を数える。
//!
//! 背景: エンジンが static eval を呼ぶのは qsearch を通った**静かな葉**だが、
//! 教師局面は対局で現れた**根**なので、取れる駒が取れる状態や王手中を含む。
//! threat 特徴は「駒 A が駒 B に利いている」を直接見るため、この分布差を最も強く受ける。
//! nnue-pytorch (Stockfish 系) は capture-best と in-check を学習から除外するのが標準。
//!
//! usage: psv_leaf_stats <file.psv> [--max N] [--step S]
use rshogi_core::position::Position;
use rshogi_core::types::Square;
use std::fs::File;
use std::io::{BufReader, Read};
use tools::packed_sfen::{PackedSfenValue, move16_to_move, unpack_sfen};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: psv_leaf_stats <file.psv> [--max N] [--step S]");
        std::process::exit(2);
    }
    let path = &args[1];
    let opt = |name: &str, d: usize| -> usize {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let max_n = opt("--max", 200_000);
    let step = opt("--step", 1).max(1);

    let mut r = BufReader::with_capacity(1 << 20, File::open(path).expect("open"));
    let mut buf = [0u8; PackedSfenValue::SIZE];
    let (mut seen, mut used) = (0usize, 0usize);
    let (mut n_check, mut n_cap, mut n_either, mut n_drop, mut n_promo) = (0, 0, 0, 0, 0);
    let mut n_badsfen = 0usize;
    let mut n_nomove = 0usize;
    let mut pos = Position::new();
    while r.read_exact(&mut buf).is_ok() {
        seen += 1;
        if (seen - 1) % step != 0 {
            continue;
        }
        if used >= max_n {
            break;
        }
        let psv = match PackedSfenValue::from_bytes(&buf) {
            Some(p) => p,
            None => continue,
        };
        let sfen = match unpack_sfen(&psv.sfen) {
            Ok(s) => s,
            Err(_) => {
                n_badsfen += 1;
                continue;
            }
        };
        if pos.set_sfen(&sfen).is_err() {
            n_badsfen += 1;
            continue;
        }
        used += 1;
        let in_chk = pos.in_check();
        let mv = move16_to_move(psv.move16);
        let is_drop = mv.is_drop();
        let is_promo = mv.is_promote();
        // capture 判定: 移動先に駒があるか (打ちは capture ではない)
        // ★2026-09-11: 我々の学習データ (novel_ext3 / dlsuisho / threat6) は
        //   **move16 が全レコード 0** で、 最善手が記録されていない。
        //   したがって capture-best による除外は現データでは実行できない。
        //   有効なのは sfen だけで判定できる「王手中」のみ。
        let has_move = psv.move16 != 0;
        let is_cap = if !has_move || is_drop {
            false
        } else {
            let to: Square = mv.to();
            pos.piece_on(to) != rshogi_core::types::Piece::NONE
        };
        if in_chk {
            n_check += 1;
        }
        if is_cap {
            n_cap += 1;
        }
        if in_chk || is_cap {
            n_either += 1;
        }
        if has_move && is_drop {
            n_drop += 1;
        }
        if has_move && is_promo {
            n_promo += 1;
        }
        if !has_move {
            n_nomove += 1;
        }
    }
    let p = |x: usize| 100.0 * x as f64 / used.max(1) as f64;
    println!("{}", path);
    println!(
        "  走査 {} 件、 判定に使った {} 件 (step={}){}",
        seen,
        used,
        step,
        if n_badsfen > 0 {
            format!("、 sfen 復元失敗 {}", n_badsfen)
        } else {
            String::new()
        }
    );
    println!("  王手中                    {:>9} ({:5.2}%)", n_check, p(n_check));
    println!("  教師 best move が capture {:>9} ({:5.2}%)", n_cap, p(n_cap));
    println!(
        "  ★どちらか (除外対象)      {:>9} ({:5.2}%)  <- nnue-pytorch はここを学習から外す",
        n_either,
        p(n_either)
    );
    println!("  (参考) best move が駒打ち {:>9} ({:5.2}%)", n_drop, p(n_drop));
    println!("  (参考) best move が成り   {:>9} ({:5.2}%)", n_promo, p(n_promo));
    if n_nomove > 0 {
        println!(
            "  ★move16 が 0 (最善手が記録されていない) {:>9} ({:5.2}%)",
            n_nomove,
            p(n_nomove)
        );
        println!(
            "    -> capture-best による除外はこのデータでは**実行できない**。 王手中のみ有効。"
        );
    }
}
