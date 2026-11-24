//! HalfKA_hm → HalfKAv2_hm 重み変換 tool (P5、 2026-05-17 実装)
//!
//! ## 概要
//!
//! HalfKA_hm の quantised.bin を HalfKAv2_hm の quantised.bin に変換する。
//!
//! ## HalfKA_hm と HalfKAv2_hm の関係 (rshogi 実装での実態)
//!
//! rshogi/bullet の HalfKA_hm 実装では既に enemy_king は `pp -= 81` で
//! friend_king と同じ dim 帯 `[1548, 1629)` (= `[F_KING, E_KING)`) にパックされている。
//! したがって PIECE_INPUTS = 1629 のまま、 FT weight 行列のサイズも 1629 × FT_dim で
//! 既に最小化されており、 行列の物理的な縮小は発生しない。
//!
//! 変更点は以下のみ:
//! 1. arch_str header: `Features=HalfKA_hm(Friend)` → `Features=HalfKAv2_hm(Friend)`
//! 2. runtime: rshogi-core / bullet 双方で `feature = "halfkav2-hm"` 有効時に
//!    friend_king の active feature を emit しない (`is_friend_king_bp` で skip)
//!
//! ## bit-exact 考察
//!
//! - 同 ckpt + `halfkav2-hm` ON/OFF で eval を比較すると、 friend_king の重み寄与が
//!   ON 側で削除されるため通常は完全一致しない。 ただし friend_king は K bucket と
//!   完全相関で常に同じ index になるため、 「bucket あたり常に同じ定数を出力する成分」を
//!   減算したことに等しく、 (a) 比較 (相対) 評価ベースで使う UCI score は影響軽微、
//!   (b) L1 bias を再学習すれば吸収される。
//! - 本 tool では FT weight を変更しないため、 旧 binary + 旧 ckpt と 新 binary + 新 ckpt
//!   (arch_str のみ変更) の eval は 「friend_king を含むかどうか」 の差分の分だけ
//!   ずれるが、 これは設計通りの挙動である (Stockfish PR #3506 と同じ trade-off)。
//!
//! ## ファイル形式 (HalfKA_hm bullet 出力)
//!
//! ```text
//! [version u32 (=NNUE_VERSION_HALFKA 0x7AF32F20)]
//! [hash u32]
//! [arch_len u32]
//! [arch_str bytes]   ← ここの "HalfKA_hm" → "HalfKAv2_hm" に書換え
//! [ft_hash u32]
//! [FT: LEB128 圧縮ブロック (1 or 2 個)]
//! [bucket 0..N-1: fc_hash(4B) + bucket_data]
//! [trailing section (PIN/DUAL 等): verbatim copy]
//! ```
//!
//! ## 使用例
//!
//! ```text
//! convert_halfka_to_halfkav2 \
//!     --input  E:/shogi/p2shogi/bullet_checkpoints/.../quantised_v5.bin \
//!     --output E:/shogi/p2shogi/bullet_checkpoints/.../quantised_halfkav2_hm.bin
//! ```
//!
//! ## 参照
//!
//! - 設計 doc: `docs/design/halfkav2_hm_migration_plan.md`
//! - Stockfish PR #3506 (HalfKA → HalfKAv2)

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;

const NNUE_VERSION_HALFKA: u32 = 0x7AF3_2F20;

/// HalfKA_hm の arch_str に含まれる Features 文字列
const FEATURES_OLD: &str = "HalfKA_hm(Friend)";
/// HalfKAv2_hm 用の Features 文字列
const FEATURES_NEW: &str = "HalfKAv2_hm(Friend)";

/// HalfKA_hm 表記 (Friend 注記なし版) — 古い ckpt 互換 fallback
const FEATURES_OLD_NOPAREN: &str = "HalfKA_hm";
const FEATURES_NEW_NOPAREN: &str = "HalfKAv2_hm";

#[derive(Parser, Debug)]
#[command(
    name = "convert_halfka_to_halfkav2",
    about = "HalfKA_hm → HalfKAv2_hm 重み変換 (arch_str header の Features 文字列を書換え)"
)]
struct Cli {
    /// 入力 quantised.bin (HalfKA_hm)
    #[arg(long)]
    input: PathBuf,

    /// 出力 quantised.bin (HalfKAv2_hm)
    #[arg(long)]
    output: PathBuf,

    /// dry-run (実際に書き出さず spec dump のみ)
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();

    let raw = fs::read(&cli.input)
        .with_context(|| format!("入力ファイルが読めません: {:?}", cli.input))?;
    println!("[info] 入力 byte 数: {} ({:.2} MB)", raw.len(), raw.len() as f64 / 1_048_576.0);

    let header = read_header(&raw)?;
    println!("[info] version: 0x{:08x}", header.version);
    println!("[info] hash: 0x{:08x}", header.hash);
    println!("[info] arch_len: {}", header.arch_len);
    println!("[info] arch_str (旧): {}", header.arch_str);
    println!("[info] ft_hash: 0x{:08x}", header.ft_hash);
    println!("[info] header 後 (FT block 開始) offset: 0x{:x}", header.body_offset);

    // arch_str 書換え
    let new_arch_str = rewrite_arch_str(&header.arch_str)?;
    println!("[info] arch_str (新): {new_arch_str}");

    if new_arch_str == header.arch_str {
        bail!(
            "arch_str に '{FEATURES_OLD}' / '{FEATURES_OLD_NOPAREN}' が見つかりません。\
             既に HalfKAv2_hm 化済か、 想定外の format。"
        );
    }

    if cli.dry_run {
        println!("[dry-run] 実書出しはスキップ");
        return Ok(());
    }

    // 出力 buffer
    let body = &raw[header.body_offset..];
    let mut out = Vec::with_capacity(raw.len() + 16);

    out.extend_from_slice(&header.version.to_le_bytes());
    out.extend_from_slice(&header.hash.to_le_bytes());
    out.extend_from_slice(&(new_arch_str.len() as u32).to_le_bytes());
    out.extend_from_slice(new_arch_str.as_bytes());
    out.extend_from_slice(&header.ft_hash.to_le_bytes());
    // FT 以降は素通し (HalfKA_hm と HalfKAv2_hm で FT shape は同一)
    out.extend_from_slice(body);

    fs::write(&cli.output, &out)
        .with_context(|| format!("出力ファイルが書けません: {:?}", cli.output))?;

    println!(
        "[done] 書き出し完了: {:?} ({} byte, {:.2} MB)",
        cli.output,
        out.len(),
        out.len() as f64 / 1_048_576.0
    );
    println!(
        "[info] サイズ差: {} byte (arch_str 長変化: {} → {})",
        out.len() as i64 - raw.len() as i64,
        header.arch_str.len(),
        new_arch_str.len()
    );

    Ok(())
}

struct Header {
    version: u32,
    hash: u32,
    arch_len: u32,
    arch_str: String,
    ft_hash: u32,
    /// ft_hash 直後の offset (FT block 開始)
    body_offset: usize,
}

fn read_header(raw: &[u8]) -> Result<Header> {
    let mut cursor = io::Cursor::new(raw);
    let version = read_u32(&mut cursor)?;
    if version != NNUE_VERSION_HALFKA {
        bail!(
            "予期しない version: 0x{version:08x} (期待: 0x{NNUE_VERSION_HALFKA:08x} = NNUE_VERSION_HALFKA)"
        );
    }
    let hash = read_u32(&mut cursor)?;
    let arch_len = read_u32(&mut cursor)?;
    if arch_len == 0 || arch_len > 4096 {
        bail!("arch_len 異常: {arch_len}");
    }
    let mut arch_buf = vec![0u8; arch_len as usize];
    cursor.read_exact(&mut arch_buf)?;
    let arch_str = String::from_utf8(arch_buf).context("arch_str が UTF-8 でない")?;
    let ft_hash = read_u32(&mut cursor)?;
    let body_offset = cursor.position() as usize;
    Ok(Header {
        version,
        hash,
        arch_len,
        arch_str,
        ft_hash,
        body_offset,
    })
}

fn read_u32(cursor: &mut io::Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// arch_str 内の `HalfKA_hm(Friend)` を `HalfKAv2_hm(Friend)` に書換える。
/// `(Friend)` 注記が無い古い ckpt も `HalfKA_hm` → `HalfKAv2_hm` で互換対応する。
fn rewrite_arch_str(src: &str) -> Result<String> {
    // 優先順: `(Friend)` 付き → 注記なし
    if src.contains(FEATURES_OLD) {
        Ok(src.replacen(FEATURES_OLD, FEATURES_NEW, 1))
    } else if src.contains(FEATURES_OLD_NOPAREN) {
        // ただし HalfKAv2_hm が既に含まれていたら無置換
        if src.contains(FEATURES_NEW_NOPAREN) {
            return Ok(src.to_string());
        }
        Ok(src.replacen(FEATURES_OLD_NOPAREN, FEATURES_NEW_NOPAREN, 1))
    } else {
        Ok(src.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_arch_str_with_friend() {
        let src = "Features=HalfKA_hm(Friend)[73305->1536x2],LMAS=v4:flags=0x003f,Network=...";
        let dst = rewrite_arch_str(src).unwrap();
        assert!(dst.contains("Features=HalfKAv2_hm(Friend)[73305->1536x2]"));
        assert!(!dst.contains("Features=HalfKA_hm(Friend)"));
    }

    #[test]
    fn test_rewrite_arch_str_without_friend() {
        // 古い ckpt: HalfKA_hm 注記なし
        let src = "Features=HalfKA_hm[73305->512x2]-SCReLU,fv_scale=13";
        let dst = rewrite_arch_str(src).unwrap();
        assert!(dst.contains("HalfKAv2_hm"));
        assert!(!dst.contains("Features=HalfKA_hm["));
    }

    #[test]
    fn test_rewrite_arch_str_idempotent() {
        // 既に HalfKAv2_hm 化済
        let src = "Features=HalfKAv2_hm(Friend)[73305->1536x2]";
        let dst = rewrite_arch_str(src).unwrap();
        assert_eq!(dst, src);
    }
}
