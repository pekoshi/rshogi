//! LayerStack 1536-32-96 quantised.bin を 1536-16-48 に縮小初期化変換する。
//!
//! ## 目的
//!
//! - 1536-32-96 の評価器が重く探索 depth が伸びないため、 1536-16-48 へ縮小して
//!   NPS と探索深さを優先する。
//! - 既存重みの **先頭 16 row / 48 row** をそのままコピー (truncation 初期化)。
//! - 変換後は必ず再学習 / 蒸留が前提。 そのままでも eval は動作するが、
//!   表現力が削られているため棋力は再学習で取り戻すこと。
//!
//! ## ファイル形式
//!
//! `convert_layerstack_l1_16_to_32.rs` と同じ FT + 8-bucket FC + 末尾 trailing section 構造。
//! 詳細はそのファイル先頭の doc を参照。
//!
//! ## 変換仕様
//!
//! - FT (L0=1536): byte-copy。
//! - L1 weights: \[32 row x 1536 col\] → \[16 row x 1536 col\]
//!   - 旧 row 0..15 (main 0..15) コピー
//!   - 旧 row 31 (skip)         → 新 row 15 (skip) コピー
//!   - 旧 row 16..31 (main 16..31)、 旧 row 15..30 (旧 main の残り) は破棄
//! - L1 bias: \[32\] → \[16\]、 同様に main 0..15 + skip(15←旧31) コピー
//! - L2 weights: \[96 row x padded(62)=64 col\] → \[48 row x padded(30)=32 col\]
//!   - 旧 col layout: sqr_main(0..31) | crelu_main(31..62) | padding(62..64)
//!   - 新 col layout: sqr_main(0..15) | crelu_main(15..30) | padding(30..32)
//!   - col の前半 15 ずつ、 row の先頭 48 をコピー
//! - L2 bias: \[96\] → \[48\] (先頭 48 コピー)
//! - Output weights: \[1 row x padded(96)=96 col\] → \[1 row x padded(48)=64 col\]
//!   - 旧 col 0..48 を先頭にコピー、 col 48..64 は padding (0)
//! - Output bias: \[1\] そのままコピー
//! - 8 bucket すべて同様
//! - arch_str: AffineTransform 表記を 32→16, 62→30, 96→48 に書き換え
//!
//! ## 使用例
//!
//! ```text
//! cargo run --release --bin shrink_layerstack_1536_32_96_to_1536_16_48 -- \
//!     --input  /path/to/quantised_1536x32x96.bin \
//!     --output /path/to/quantised_1536x16x48.bin
//! ```
//!
//! ## 注意
//!
//! 縮小は **lossy** 変換であり、 そのままでは棋力が大きく低下する可能性がある。
//! 変換後の重みは bullet 等で必ず fine-tune してから棋力測定すること。

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;

// NNUE 関連定数 (rshogi-core から再エクスポートされない private 値は手動で複製)
const NNUE_VERSION_HALFKA: u32 = 0x7AF3_2F20;
const HALFKA_HM_DIMENSIONS: usize = 73_305;
const NUM_BUCKETS: usize = 8;
const L1_FT: usize = 1536;

// 旧 (1536-32-96)
const SRC_L1_OUT: usize = 32;
const SRC_L2_IN: usize = 62;
const SRC_L3: usize = 96;

// 新 (1536-16-48)
const DST_L1_OUT: usize = 16;
const DST_L2_IN: usize = 30;
const DST_L3: usize = 48;

#[derive(Parser, Debug)]
#[command(
    name = "shrink_layerstack_1536_32_96_to_1536_16_48",
    about = "LayerStack 1536-32-96 quantised.bin を 1536-16-48 に縮小初期化変換"
)]
struct Cli {
    /// 入力 quantised.bin (1536-32-96)
    #[arg(long)]
    input: PathBuf,

    /// 出力 quantised.bin (1536-16-48)
    #[arg(long)]
    output: PathBuf,

    /// 入力 arch (現状 "1536-32-96" のみ対応、 妥当性チェック用)
    #[arg(long, default_value = "1536-32-96")]
    from: String,

    /// 出力 arch (現状 "1536-16-48" のみ対応)
    #[arg(long, default_value = "1536-16-48")]
    to: String,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    if cli.from != "1536-32-96" {
        bail!(
            "--from must be \"1536-32-96\" (got \"{}\")。 他の入力 arch は未対応。",
            cli.from
        );
    }
    if cli.to != "1536-16-48" {
        bail!("--to must be \"1536-16-48\" (got \"{}\")。 他の出力 arch は未対応。", cli.to);
    }

    let raw = fs::read(&cli.input)
        .with_context(|| format!("入力ファイルが読めません: {:?}", cli.input))?;
    let mut cursor = io::Cursor::new(&raw[..]);

    let header = read_header(&mut cursor)?;
    println!("[info] arch_str: {}", header.arch_str);
    println!("[info] FT block 開始位置: 0x{:x}", header.ft_block_start);

    // arch_str に AffineTransform[32<-3072] (旧 L1=32) が含まれているか確認
    if !header.arch_str.contains("AffineTransform[32<-3072]") {
        bail!(
            "arch_str に 'AffineTransform[32<-3072]' (= L1=32) が見つかりません。\n\
             1536-32-96 モデルではない可能性があります: {}",
            header.arch_str
        );
    }

    let ft_end = scan_ft_blocks(&raw, header.ft_block_start)?;
    println!("[info] FT block 終了位置: 0x{ft_end:x}");

    if header.arch_str.contains("PSQT=") {
        bail!("PSQT 付きモデルの縮小変換は未対応 (YAGNI、 1536-32-96+PSQT は現状未運用)");
    }
    if header.arch_str.contains("Threat=") {
        bail!("Threat 付きモデルの縮小変換は未対応 (YAGNI)");
    }

    // 8 bucket 分の FC データ
    let src_bucket_size = compute_bucket_size(SRC_L1_OUT, SRC_L2_IN, SRC_L3);
    let fc_end = ft_end + src_bucket_size * NUM_BUCKETS;
    println!("[info] src bucket size: {src_bucket_size} bytes, FC end: 0x{fc_end:x}");
    if fc_end > raw.len() {
        bail!(
            "ファイルサイズ不足: ft_end=0x{ft_end:x}, bucket*8=0x{:x}, total=0x{:x}, file_len=0x{:x}",
            src_bucket_size * NUM_BUCKETS,
            fc_end,
            raw.len()
        );
    }

    // LMAS section は 2026-05-25 廃止。 検出時は bail。
    if header.arch_str.contains("LMAS=") {
        bail!(
            "LMAS section は 2026-05-25 に廃止されました。 旧 LMAS 付き ckpt は別途 \
             LMAS section を除去してから再変換してください。"
        );
    }
    // FC 末尾以降は trailing section (PIN/DUAL/将来 section) を verbatim コピー。
    let trailing_bytes: &[u8] = &raw[fc_end..];
    if !trailing_bytes.is_empty() {
        println!("[info] trailing section: 0x{:x} byte (verbatim copy)", trailing_bytes.len());
    }

    let mut out = Vec::with_capacity(raw.len() / 2);

    // 新 arch_str
    let new_arch_str = rewrite_arch_str(&header.arch_str)?;
    println!("[info] new arch_str: {new_arch_str}");

    out.extend_from_slice(&NNUE_VERSION_HALFKA.to_le_bytes());
    out.extend_from_slice(&header.hash.to_le_bytes());
    out.extend_from_slice(&(new_arch_str.len() as u32).to_le_bytes());
    out.extend_from_slice(new_arch_str.as_bytes());
    out.extend_from_slice(&header.ft_hash.to_le_bytes());
    out.extend_from_slice(&raw[header.ft_block_start..ft_end]);

    for bucket_idx in 0..NUM_BUCKETS {
        let src_off = ft_end + bucket_idx * src_bucket_size;
        let src_bucket = &raw[src_off..src_off + src_bucket_size];
        let new_bucket = shrink_bucket(src_bucket)?;
        out.extend_from_slice(&new_bucket);
    }

    out.extend_from_slice(trailing_bytes);

    fs::write(&cli.output, &out)
        .with_context(|| format!("出力ファイルが書けません: {:?}", cli.output))?;
    println!(
        "[done] 書き出し完了: {:?} ({:.2} MB → 縮小後 {:.2} MB)",
        cli.output,
        raw.len() as f64 / (1024.0 * 1024.0),
        out.len() as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

struct Header {
    arch_str: String,
    hash: u32,
    ft_hash: u32,
    ft_block_start: usize,
}

fn read_header(cursor: &mut io::Cursor<&[u8]>) -> Result<Header> {
    let version = read_u32(cursor)?;
    if version != NNUE_VERSION_HALFKA {
        bail!(
            "予期しない version: 0x{version:08x} (期待: 0x{NNUE_VERSION_HALFKA:08x} = NNUE_VERSION_HALFKA)"
        );
    }
    let hash = read_u32(cursor)?;
    let arch_len = read_u32(cursor)? as usize;
    if arch_len == 0 || arch_len > 4096 {
        bail!("arch_len 異常: {arch_len}");
    }
    let mut arch_buf = vec![0u8; arch_len];
    cursor.read_exact(&mut arch_buf)?;
    let arch_str = String::from_utf8(arch_buf).context("arch_str が UTF-8 でない")?;
    let ft_hash = read_u32(cursor)?;
    let ft_block_start = cursor.position() as usize;
    Ok(Header {
        arch_str,
        hash,
        ft_hash,
        ft_block_start,
    })
}

fn read_u32(cursor: &mut io::Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn scan_ft_blocks(raw: &[u8], mut pos: usize) -> Result<usize> {
    const MAGIC: &[u8] = b"COMPRESSED_LEB128";
    let total_size_per_element = L1_FT + HALFKA_HM_DIMENSIONS * L1_FT;
    if raw.len() < pos + MAGIC.len() + 4 {
        bail!("FT block ヘッダが読めません");
    }
    if &raw[pos..pos + MAGIC.len()] != MAGIC {
        bail!("FT block 開始位置に COMPRESSED_LEB128 magic がない (pos=0x{pos:x})");
    }
    pos += MAGIC.len();
    let block1_size = u32::from_le_bytes(raw[pos..pos + 4].try_into()?) as usize;
    pos += 4;
    let block1_elem_count = count_leb128_elements(&raw[pos..pos + block1_size])?;
    pos += block1_size;
    if block1_elem_count == total_size_per_element {
        return Ok(pos);
    }
    if block1_elem_count != L1_FT {
        bail!(
            "FT 1st block 要素数異常: {block1_elem_count} (期待: {L1_FT} biases-only 又は {total_size_per_element} 結合)"
        );
    }
    if raw.len() < pos + MAGIC.len() + 4 {
        bail!("FT 2nd block header が読めません");
    }
    if &raw[pos..pos + MAGIC.len()] != MAGIC {
        bail!("FT 2nd block magic がない (pos=0x{pos:x})");
    }
    pos += MAGIC.len();
    let block2_size = u32::from_le_bytes(raw[pos..pos + 4].try_into()?) as usize;
    pos += 4 + block2_size;
    Ok(pos)
}

fn count_leb128_elements(data: &[u8]) -> Result<usize> {
    let mut count = 0;
    let mut i = 0;
    while i < data.len() {
        loop {
            if i >= data.len() {
                bail!("LEB128 オーバーラン");
            }
            let b = data[i];
            i += 1;
            if b & 0x80 == 0 {
                break;
            }
        }
        count += 1;
    }
    Ok(count)
}

const fn padded_input(n: usize) -> usize {
    n.div_ceil(32) * 32
}

const fn affine_size(input: usize, output: usize) -> usize {
    output * 4 + output * padded_input(input)
}

const fn compute_bucket_size(l1_out: usize, l2_in: usize, l3: usize) -> usize {
    4 // fc_hash
        + affine_size(L1_FT, l1_out)
        + affine_size(l2_in, l3)
        + affine_size(l3, 1)
}

/// 旧 bucket bytes (1536-32-96) を新 bucket bytes (1536-16-48) に縮小する。
fn shrink_bucket(src: &[u8]) -> Result<Vec<u8>> {
    let dst_size = compute_bucket_size(DST_L1_OUT, DST_L2_IN, DST_L3);
    let mut out = Vec::with_capacity(dst_size);

    let mut cur = 0;
    // fc_hash: そのままコピー
    out.extend_from_slice(&src[cur..cur + 4]);
    cur += 4;

    cur = shrink_l1(src, cur, &mut out)?;
    cur = shrink_l2(src, cur, &mut out)?;
    cur = shrink_output(src, cur, &mut out)?;

    if cur != src.len() {
        bail!("bucket parse 不整合: 読み残し {} byte", src.len() - cur);
    }
    if out.len() != dst_size {
        bail!("bucket 出力サイズ不整合: got {}, expected {dst_size}", out.len());
    }
    Ok(out)
}

/// L1 縮小: input=1536 固定、 output 32 → 16。
///
/// 旧 layout (32 row): main 0..31 (row 0..31)、 skip = row 31。
/// 新 layout (16 row): main 0..15 (row 0..15 = 旧 main 0..15)、 skip = row 15 (← 旧 row 31)。
///
/// 旧 main row 15..31 (= 16 行) は破棄 (lossy)。
fn shrink_l1(src: &[u8], mut cur: usize, out: &mut Vec<u8>) -> Result<usize> {
    // bias: 旧 32 i32 LE
    let mut src_bias = [0i32; SRC_L1_OUT];
    for b in src_bias.iter_mut() {
        *b = i32::from_le_bytes(src[cur..cur + 4].try_into()?);
        cur += 4;
    }
    // 新 bias: row 0..15 = 旧 row 0..15、 row 15 = 旧 row 31 (skip)
    let mut dst_bias = [0i32; DST_L1_OUT];
    let src_main = SRC_L1_OUT - 1; // 31
    let dst_main = DST_L1_OUT - 1; // 15
    dst_bias[..dst_main].copy_from_slice(&src_bias[..dst_main]);
    dst_bias[dst_main] = src_bias[src_main]; // skip コピー
    for b in dst_bias.iter() {
        out.extend_from_slice(&b.to_le_bytes());
    }

    // weights: row-major、 padded_input(1536) = 1536
    let row_bytes = padded_input(L1_FT);
    debug_assert_eq!(row_bytes, L1_FT);

    // 新 row 0..15 = 旧 row 0..15 コピー
    for r in 0..dst_main {
        let off = cur + r * row_bytes;
        out.extend_from_slice(&src[off..off + row_bytes]);
    }
    // 新 row 15 = 旧 row 31 (skip) コピー
    let skip_off = cur + src_main * row_bytes;
    out.extend_from_slice(&src[skip_off..skip_off + row_bytes]);

    // 旧 weights 全体を skip (SRC_L1_OUT * row_bytes)
    cur += SRC_L1_OUT * row_bytes;
    Ok(cur)
}

/// L2 縮小: input src=62 (padded 64) → dst=30 (padded 32)、 output 96 → 48。
///
/// 旧 col layout: sqr_main(0..31) | crelu_main(31..62) | padding(62..64)
/// 新 col layout: sqr_main(0..15) | crelu_main(15..30) | padding(30..32)
///
/// 行 (output) も縮小: row 0..48 のみ採用、 row 48..96 破棄。
/// 各 row の中の sqr/crelu 領域は、 sqr 前半 15 と crelu 前半 15 のみ採用。
fn shrink_l2(src: &[u8], mut cur: usize, out: &mut Vec<u8>) -> Result<usize> {
    // bias: 旧 96 i32 → 新 48 i32 (先頭 48)
    for i in 0..DST_L3 {
        let off = cur + i * 4;
        out.extend_from_slice(&src[off..off + 4]);
    }
    cur += SRC_L3 * 4;

    let src_padded = padded_input(SRC_L2_IN); // 64
    let dst_padded = padded_input(DST_L2_IN); // 32
    let row_bytes_src = src_padded;
    let row_bytes_dst = dst_padded;

    let src_main = SRC_L1_OUT - 1; // 31
    let dst_main = DST_L1_OUT - 1; // 15

    for row in 0..DST_L3 {
        let src_row_off = cur + row * row_bytes_src;
        let src_row = &src[src_row_off..src_row_off + row_bytes_src];
        let mut dst_row = vec![0u8; row_bytes_dst];

        // src col 0..15 (sqr 前半 15) → dst col 0..15
        dst_row[0..dst_main].copy_from_slice(&src_row[0..dst_main]);
        // src col 31..46 (crelu 前半 15) → dst col 15..30
        dst_row[dst_main..2 * dst_main].copy_from_slice(&src_row[src_main..src_main + dst_main]);
        // dst col 30..32 は padding (0)
        out.extend_from_slice(&dst_row);
    }
    cur += SRC_L3 * row_bytes_src;
    Ok(cur)
}

/// Output 縮小: input src=96 (padded 96) → dst=48 (padded 64)、 output=1 固定。
fn shrink_output(src: &[u8], mut cur: usize, out: &mut Vec<u8>) -> Result<usize> {
    // bias: 1 i32 そのままコピー
    out.extend_from_slice(&src[cur..cur + 4]);
    cur += 4;

    // weights: row=1、 input src=96 (padded 96)、 dst=48 (padded 64)
    let src_padded = padded_input(SRC_L3); // 96
    let dst_padded = padded_input(DST_L3); // 64
    let mut dst_row = vec![0u8; dst_padded];
    // 旧 col 0..48 → 新 col 0..48
    dst_row[..DST_L3].copy_from_slice(&src[cur..cur + DST_L3]);
    // 新 col 48..64 は padding (0)、 旧 col 48..96 は破棄 (lossy)
    out.extend_from_slice(&dst_row);
    cur += src_padded;
    Ok(cur)
}

/// arch_str 内の L1=32 / L2_in=62 / L3=96 表記を 16/30/48 に書き換える。
///
/// bullet が実際に書く `AffineTransform[32<-3072]` / `AffineTransform[96<-62]` /
/// `SqrClippedReLU[62]` を 16/30/48 形式へ置換する。
fn rewrite_arch_str(src: &str) -> Result<String> {
    let marker_old = "AffineTransform[32<-3072]";
    if !src.contains(marker_old) {
        return Err(anyhow!(
            "arch_str に '{marker_old}' が見つかりません: {src}\n変換は 1536-32-96 → 1536-16-48 のみサポート"
        ));
    }
    let mut result = src.to_string();
    // L1 dim: 32 → 16
    result = result.replace("AffineTransform[32<-3072]", "AffineTransform[16<-3072]");
    // L2 入力 dim: 62 → 30 (sqr_crelu 31+31 → 15+15)、 出力 96 → 48
    result = result.replace("AffineTransform[96<-62]", "AffineTransform[48<-30]");
    result = result.replace("SqrClippedReLU[62]", "SqrClippedReLU[30]");
    // ClippedReLU の L3 dim: 96 → 48
    result = result.replace("ClippedReLU[96]", "ClippedReLU[48]");
    // Output layer の input dim: 96 → 48
    result = result.replace("AffineTransform[1<-96]", "AffineTransform[1<-48]");
    // L3 dim: 96 → 48 (l3 token、 存在する場合のみ)
    result = replace_token(&result, "l3=96", "l3=48");
    Ok(result)
}

/// `token` を `replacement` に置換する (後続が数字なら誤マッチ回避)。
fn replace_token(src: &str, token: &str, replacement: &str) -> String {
    let bytes = src.as_bytes();
    let tok = token.as_bytes();
    let mut out = String::with_capacity(src.len() + 4);
    let mut i = 0;
    while i < bytes.len() {
        if i + tok.len() <= bytes.len() && &bytes[i..i + tok.len()] == tok {
            let next = bytes.get(i + tok.len()).copied().unwrap_or(b',');
            if !next.is_ascii_digit() {
                out.push_str(replacement);
                i += tok.len();
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padded_input() {
        assert_eq!(padded_input(14), 32);
        assert_eq!(padded_input(30), 32);
        assert_eq!(padded_input(48), 64);
        assert_eq!(padded_input(62), 64);
        assert_eq!(padded_input(96), 96);
        assert_eq!(padded_input(1536), 1536);
    }

    #[test]
    fn test_bucket_size() {
        // 1536-32-96 (= 旧)
        // fc_hash 4 + L1(1536→32) [bias 128 + w 32*1536=49152] = 49284
        // + L2(62→96) [bias 384 + w 96*64=6144] = 6528
        // + Out(96→1) [bias 4 + w 1*96=96] = 100
        // total = 4 + 49284 + 6528 + 100 = 55916? — 既存 test では 55848 を期待していた
        // (旧 test は fc_hash を含めないか別の計算。 本 binary では含む方を使用)
        // 改めて計算: 4 + (32*4 + 32*1536) + (96*4 + 96*padded(62)=96*64) + (1*4 + 1*padded(96)=1*96)
        //          = 4 + (128 + 49152) + (384 + 6144) + (4 + 96)
        //          = 4 + 49280 + 6528 + 100 = 55912
        assert_eq!(compute_bucket_size(32, 62, 96), 55912);
        // 1536-16-48 (= 新)
        // fc_hash 4 + L1(1536→16) [bias 64 + w 16*1536=24576] = 24644
        // + L2(30→48) [bias 192 + w 48*padded(30)=48*32=1536] = 1728
        // + Out(48→1) [bias 4 + w 1*padded(48)=1*64=64] = 68
        // total = 4 + 24640 + 1728 + 68 = 26440
        assert_eq!(compute_bucket_size(16, 30, 48), 26440);
    }

    #[test]
    fn test_rewrite_arch_str() {
        let src = "Features=HalfKA_hm^[73305->1536x2]-SCReLU,\
                   AffineTransform[32<-3072],AffineTransform[96<-62],SqrClippedReLU[62],\
                   fv_scale=16,l2=32,l3=96,qa=127,qb=64";
        let dst = rewrite_arch_str(src).unwrap();
        assert!(dst.contains("AffineTransform[16<-3072]"));
        assert!(dst.contains("AffineTransform[48<-30]"));
        assert!(dst.contains("SqrClippedReLU[30]"));
        assert!(dst.contains("l3=48"));
        assert!(!dst.contains("AffineTransform[32<-3072]"));
        assert!(!dst.contains("AffineTransform[96<-62]"));
        assert!(!dst.contains("SqrClippedReLU[62]"));
        assert!(!dst.contains("l3=96"));
    }

    /// bullet が実際に書く Network=... 形式 (ClippedReLU + AffineTransform[1<-96] 含む) で
    /// 全 96 → 48 置換が正しく動くこと。
    #[test]
    fn test_rewrite_arch_str_bullet_network_format() {
        let src = "Features=HalfKA_hm(Friend)[73305->1536x2],LMAS=v4:flags=0x003f,\
                   Network=AffineTransform[1<-96](ClippedReLU[96](AffineTransform[96<-62](SqrClippedReLU[62](AffineTransform[32<-3072](InputSlice[3072(0:3072)]))))),fv_scale=14";
        let dst = rewrite_arch_str(src).unwrap();
        assert!(dst.contains("AffineTransform[16<-3072]"));
        assert!(dst.contains("AffineTransform[48<-30]"));
        assert!(dst.contains("AffineTransform[1<-48]"));
        assert!(dst.contains("ClippedReLU[48]"));
        assert!(dst.contains("SqrClippedReLU[30]"));
        assert!(!dst.contains("AffineTransform[32<-3072]"));
        assert!(!dst.contains("AffineTransform[96<-62]"));
        assert!(!dst.contains("AffineTransform[1<-96]"));
        assert!(!dst.contains("ClippedReLU[96]"));
        assert!(!dst.contains("SqrClippedReLU[62]"));
    }

    #[test]
    fn test_replace_token_no_overmatch() {
        let src = "fv_scale=8,l3=960";
        let dst = replace_token(src, "l3=96", "l3=48");
        assert_eq!(dst, "fv_scale=8,l3=960");
    }

    #[test]
    fn test_shrink_l1_bias_preserves_main_and_skip() {
        // 簡易検証: shrink_l1 の bias 部分のみテストする。
        // 旧 SRC_L1_OUT=32 の bias を作って、 新 DST_L1_OUT=16 へ shrink。
        let mut src = vec![0u8; SRC_L1_OUT * 4 + SRC_L1_OUT * L1_FT];
        // 旧 bias[i] = i (i32)
        for i in 0..SRC_L1_OUT {
            let off = i * 4;
            src[off..off + 4].copy_from_slice(&(i as i32).to_le_bytes());
        }
        // weights は 0 で OK (b 部分のみ確認)
        let mut out = Vec::new();
        shrink_l1(&src, 0, &mut out).unwrap();
        // 新 bias[0..15] = 旧 bias[0..15] = i
        for i in 0..(DST_L1_OUT - 1) {
            let v = i32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(v, i as i32);
        }
        // 新 bias[15] = 旧 bias[31] (skip)
        let last = i32::from_le_bytes(
            out[(DST_L1_OUT - 1) * 4..(DST_L1_OUT - 1) * 4 + 4].try_into().unwrap(),
        );
        assert_eq!(last, (SRC_L1_OUT - 1) as i32);
    }
}
