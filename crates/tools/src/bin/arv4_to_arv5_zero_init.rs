//! arv4 ckpt の末尾に ARH (= Attack Relay Head) **zero-init section** を append し
//! arv5 ckpt を作る変換 binary (= 2026-06-09)。
//!
//! 入力 ckpt は byte-exact copy、 末尾に以下を追記する:
//!
//! ```text
//! sub_magic = "P2AREL\0\0"   (= 8 byte)
//! version   = u32 LE         (= 1)
//! count     = u32 LE         (= 1024)
//! weights   = i16 LE × 1024  (= 2,048 byte、 全 0)
//! ```
//!
//! 合計 +2,064 byte。 zero-init なので推論側で ARH 寄与は 0、 arv4 と
//! eval bit-exact 一致する (= §10.1 Phase A parity gate)。
//!
//! ARH feature が無効の rshogi build は本 binary を build 不可 (= 定数参照のため)。
//! `--features nnue-attack-relay` で build する。
//!
//! # 使い方
//!
//! ```bash
//! cargo run --release -p tools --bin arv4_to_arv5_zero_init \
//!   --features nnue-attack-relay -- \
//!   --input  /path/to/arv4.bin \
//!   --output /path/to/arv5_zero.bin
//! ```

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

#[cfg(feature = "nnue-attack-relay")]
use rshogi_core::nnue::attack_relay::{ARH_MAGIC, ARH_VERSION, ARH_WEIGHT_COUNT};

#[derive(Parser, Debug)]
#[command(name = "arv4_to_arv5_zero_init")]
#[command(about = "arv4 ckpt に ARH zero-init section を append して arv5 ckpt を作る")]
struct Args {
    /// 入力 arv4 ckpt path
    #[arg(long)]
    input: PathBuf,

    /// 出力 arv5 ckpt path
    #[arg(long)]
    output: PathBuf,
}

#[cfg(feature = "nnue-attack-relay")]
fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let bytes = std::fs::read(&args.input)?;
    eprintln!(
        "info string arv4_to_arv5_zero_init: read {} bytes from {}",
        bytes.len(),
        args.input.display()
    );

    let mut out = std::fs::File::create(&args.output)?;
    out.write_all(&bytes)?;

    out.write_all(ARH_MAGIC)?;
    out.write_all(&ARH_VERSION.to_le_bytes())?;
    out.write_all(&(ARH_WEIGHT_COUNT as u32).to_le_bytes())?;
    let zero_weights = [0u8; 2 * ARH_WEIGHT_COUNT];
    out.write_all(&zero_weights)?;
    out.flush()?;

    let total = bytes.len() + 8 + 4 + 4 + 2 * ARH_WEIGHT_COUNT;
    eprintln!(
        "info string arv4_to_arv5_zero_init: wrote {} bytes to {} (= input + P2AREL zero-init {} weight)",
        total,
        args.output.display(),
        ARH_WEIGHT_COUNT
    );
    Ok(())
}

#[cfg(not(feature = "nnue-attack-relay"))]
fn main() {
    eprintln!(
        "error: arv4_to_arv5_zero_init requires --features nnue-attack-relay to build with ARH constants."
    );
    std::process::exit(2);
}
