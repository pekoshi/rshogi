//! `psv-dump` — PSV を `sfen<TAB>score<TAB>gamePly<TAB>result` のテキストに落とす。
//!
//! ## なぜ要るか
//!
//! 選択的 rescore の Gate 0 では「駒割と教師が不一致」な局面を選ぶ必要があるが、
//! 駒割を出すには 32 byte の packed sfen を展開しないといけない。
//! 既存ツールは展開して bucket を出す (`bucket-profile`) か、展開せず生フィールドだけ
//! 見る (`tools/inspect_psv.py`) のどちらかで、 **sfen 文字列を取り出す口が無かった**。
//!
//! 駒割の計算自体は sfen 文字列があれば Python 側で書けるので、 ここは展開だけを担う。

use anyhow::{Context, Result};
use bucket_profiler::{unpack_sfen, PackedSfenValue, PSV_RECORD_SIZE};
use clap::Parser;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "PSV を sfen<TAB>score<TAB>gamePly<TAB>result に展開する")]
struct Args {
    /// 入力 PSV (40 byte/record)。
    #[arg(long)]
    psv: PathBuf,

    /// 出力先 (省略時は stdout)。
    #[arg(long)]
    output: Option<PathBuf>,

    /// 先頭から何件処理するか (0 = 全件)。
    #[arg(long, default_value_t = 0)]
    limit: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let f = File::open(&args.psv).with_context(|| format!("open {:?}", args.psv))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut w: Box<dyn Write> = match &args.output {
        Some(p) => Box::new(BufWriter::with_capacity(
            1 << 20,
            File::create(p).with_context(|| format!("create {p:?}"))?,
        )),
        None => Box::new(BufWriter::with_capacity(1 << 20, std::io::stdout())),
    };

    let mut buf = [0u8; PSV_RECORD_SIZE];
    let (mut n, mut bad) = (0u64, 0u64);
    loop {
        if args.limit != 0 && n >= args.limit {
            break;
        }
        match r.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let p = match PackedSfenValue::from_bytes(&buf) {
            Some(p) => p,
            None => {
                bad += 1;
                continue;
            }
        };
        match unpack_sfen(&p.sfen) {
            Ok(sfen) => {
                writeln!(w, "{}\t{}\t{}\t{}", sfen, p.score, p.game_ply, p.game_result)?;
                n += 1;
            }
            Err(_) => bad += 1,
        }
    }
    w.flush()?;
    eprintln!("[psv-dump] {n} 件出力、 展開失敗 {bad} 件");
    Ok(())
}
