//! ARH (Attack Relay Head) cross-implementation parity tool (= 2026-06-09)。
//!
//! 設計書 `docs/attack_relay_head_design.md` §11.9 の **2 binary + JSON 比較構成** の
//! p2shogi 側 binary。 固定 SFEN 集合に対して rshogi 側 `compute_arh_classes` を
//! 実行して JSON 出力する。 bullet 側 `arh_fixture_dump.rs` の出力 JSON と比較すれば
//! 学習 / 推論側の利き計算 一致を保証できる。
//!
//! # 入出力
//!
//! - `--fixtures` : 固定 SFEN ファイル (= 1 SFEN / line、 先頭 `sfen ` prefix は除去)
//! - `--output`   : rshogi 側 dump JSON 出力 path
//! - `--compare`  : (optional) bullet 側 dump JSON path、 指定時は 3-tuple を
//!   diff し 不一致 1 件でも `exit 1`
//!
//! JSON 形式:
//!
//! ```json
//! {
//!   "arh_format_version": 1,
//!   "fixtures": [
//!     {"sfen": "...", "black": [c, r, re], "white": [c, r, re]},
//!     ...
//!   ]
//! }
//! ```
//!
//! # 使い方
//!
//! ```bash
//! cargo run --release -p tools --bin arh_parity_check \
//!   --features nnue-attack-relay -- \
//!   --fixtures docs/parity/arh_fixtures.sfen \
//!   --output   docs/parity/arh_rshogi_dump.json \
//!   --compare  docs/parity/arh_bullet_dump.json
//! ```

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use serde::{Deserialize, Serialize};

#[cfg(feature = "nnue-attack-relay")]
use rshogi_core::nnue::attack_relay::compute_arh_classes;
#[cfg(feature = "nnue-attack-relay")]
use rshogi_core::position::Position;
#[cfg(feature = "nnue-attack-relay")]
use rshogi_core::types::Color;

const ARH_FORMAT_VERSION: u32 = 1;

#[derive(Parser, Debug)]
#[command(name = "arh_parity_check")]
#[command(about = "ARH cross-implementation parity check (rshogi 側 dump + optional diff)")]
struct Args {
    /// 固定 SFEN fixtures path (= 1 SFEN / line)
    #[arg(long)]
    fixtures: PathBuf,

    /// rshogi 側 dump JSON 出力 path
    #[arg(long)]
    output: PathBuf,

    /// bullet 側 dump JSON との diff 用 path (optional)
    #[arg(long)]
    compare: Option<PathBuf>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ArhFixtureEntry {
    sfen: String,
    black: [u8; 3],
    white: [u8; 3],
}

#[derive(Serialize, Deserialize, Debug)]
struct ArhDump {
    arh_format_version: u32,
    fixtures: Vec<ArhFixtureEntry>,
}

#[cfg(feature = "nnue-attack-relay")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let raw = std::fs::read_to_string(&args.fixtures)?;
    let sfens: Vec<String> = raw
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.strip_prefix("sfen ").unwrap_or(l).to_string())
        .collect();

    eprintln!(
        "info string arh_parity_check: loaded {} SFEN from {}",
        sfens.len(),
        args.fixtures.display()
    );

    let mut entries: Vec<ArhFixtureEntry> = Vec::with_capacity(sfens.len());
    for sfen in &sfens {
        let mut pos = Position::new();
        if let Err(e) = pos.set_sfen(sfen) {
            return Err(format!("set_sfen failed for {sfen:?}: {e:?}").into());
        }
        let (bc, br, bre) = compute_arh_classes(&pos, Color::Black);
        let (wc, wr, wre) = compute_arh_classes(&pos, Color::White);
        entries.push(ArhFixtureEntry {
            sfen: sfen.clone(),
            black: [bc as u8, br as u8, bre as u8],
            white: [wc as u8, wr as u8, wre as u8],
        });
    }

    let dump = ArhDump {
        arh_format_version: ARH_FORMAT_VERSION,
        fixtures: entries.clone(),
    };

    let json = serde_json::to_string_pretty(&dump)?;
    let mut out = std::fs::File::create(&args.output)?;
    out.write_all(json.as_bytes())?;
    out.flush()?;
    eprintln!(
        "info string arh_parity_check: wrote rshogi dump to {} ({} fixtures)",
        args.output.display(),
        entries.len()
    );

    if let Some(cmp_path) = args.compare.as_ref() {
        let cmp_raw = std::fs::read_to_string(cmp_path)?;
        let cmp_dump: ArhDump = serde_json::from_str(&cmp_raw)?;
        if cmp_dump.arh_format_version != ARH_FORMAT_VERSION {
            return Err(format!(
                "arh_format_version mismatch: rshogi={} bullet={}",
                ARH_FORMAT_VERSION, cmp_dump.arh_format_version
            )
            .into());
        }
        // SFEN → bullet entry の HashMap
        let cmp_map: std::collections::HashMap<&str, &ArhFixtureEntry> =
            cmp_dump.fixtures.iter().map(|e| (e.sfen.as_str(), e)).collect();

        let mut mismatches: Vec<String> = Vec::new();
        for our in entries.iter() {
            match cmp_map.get(our.sfen.as_str()) {
                None => mismatches.push(format!("missing in bullet dump: {}", our.sfen)),
                Some(theirs) => {
                    if our.black != theirs.black || our.white != theirs.white {
                        mismatches.push(format!(
                            "mismatch sfen={} rshogi=B{:?}W{:?} bullet=B{:?}W{:?}",
                            our.sfen, our.black, our.white, theirs.black, theirs.white
                        ));
                    }
                }
            }
        }
        if mismatches.is_empty() {
            eprintln!(
                "info string arh_parity_check: PARITY OK ({} fixtures matched)",
                entries.len()
            );
        } else {
            eprintln!("error: arh_parity_check: {} mismatches found", mismatches.len());
            for m in mismatches.iter().take(20) {
                eprintln!("  - {m}");
            }
            if mismatches.len() > 20 {
                eprintln!("  ... and {} more", mismatches.len() - 20);
            }
            std::process::exit(1);
        }
    }

    Ok(())
}

#[cfg(feature = "nnue-attack-relay")]
fn main() {
    if let Err(e) = run() {
        eprintln!("error: arh_parity_check failed: {e}");
        std::process::exit(2);
    }
}

#[cfg(not(feature = "nnue-attack-relay"))]
fn main() {
    eprintln!(
        "error: arh_parity_check requires --features nnue-attack-relay to call compute_arh_classes."
    );
    std::process::exit(2);
}
