//! PSV (PackedSfenValue) から検証用の sfen + 教師スコアを取り出す。
//!
//!   psv_to_sfen <src.psv> <out_sfen.txt> <out_teacher.tsv> [--n 300] [--min-ply 20]
//!
//! ## なぜ要るか (2026-09-05)
//! 毎 SB の教師相関は `dlsuisho_preext3_residual_v1.psv` から抜いた 300 局面で測っているが、
//! **このファイルは今まさに学習に使っている 2 本のうちの 1 本**なので in-sample。
//! 学習が進めば暗記でも相関が上がるため汎化を測れない。
//! 学習に一切使っていない psv から held-out pack を作るためのツール。
//!
//! 出力:
//!   out_sfen.txt    … 1 行 1 局面 (`sfen ...`)。 eng_eval.sh がそのまま食える形式
//!   out_teacher.tsv … `<score>\t<sfen>` (既存の teacher_*.tsv と同形式)
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use tools::packed_sfen::{PackedSfenValue, unpack_sfen};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!(
            "usage: psv_to_sfen <src.psv> <out_sfen.txt> <out_teacher.tsv> [--n N] [--min-ply P]"
        );
        std::process::exit(2);
    }
    let (src, out_sfen, out_teacher) = (&a[1], &a[2], &a[3]);
    let mut want = 300usize;
    let mut min_ply = 20u16;
    let mut i = 4;
    while i + 1 < a.len() {
        match a[i].as_str() {
            "--n" => want = a[i + 1].parse()?,
            "--min-ply" => min_ply = a[i + 1].parse()?,
            _ => {}
        }
        i += 2;
    }
    let mut f = File::open(src)?;
    let total = f.metadata()?.len() as usize / PackedSfenValue::SIZE;
    // 先頭偏りを避けるため全体に均等散らして走査する
    let step = std::cmp::max(1, total / (want * 4));
    let mut sfw = BufWriter::new(File::create(out_sfen)?);
    let mut tsw = BufWriter::new(File::create(out_teacher)?);
    let mut n = 0usize;
    let mut idx = 0usize;
    while idx < total && n < want {
        f.seek(SeekFrom::Start((idx * PackedSfenValue::SIZE) as u64))?;
        let mut buf = [0u8; PackedSfenValue::SIZE];
        if f.read_exact(&mut buf).is_err() {
            break;
        }
        let Some(rec) = PackedSfenValue::from_bytes(&buf) else {
            idx += step;
            continue;
        };
        // 詰みスコアと序盤すぎる局面は検証に不向きなので落とす
        if rec.score.unsigned_abs() >= 30000 || rec.game_ply < min_ply {
            idx += step;
            continue;
        }
        match unpack_sfen(&rec.sfen) {
            Ok(s) => {
                writeln!(sfw, "sfen {s}")?;
                writeln!(tsw, "{}\tsfen {s}", rec.score)?;
                n += 1;
            }
            Err(_) => {}
        }
        idx += step;
    }
    eprintln!("抽出 {n} 局面 (source {total} 局面, step={step}) -> {out_sfen} / {out_teacher}");
    Ok(())
}
