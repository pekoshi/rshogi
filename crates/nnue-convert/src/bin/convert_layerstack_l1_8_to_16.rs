//! LayerStack 1536-8-96 quantised.bin を 1536-16-96 形式に変換する。
//!
//! ## 目的
//!
//! - 1536-8-96 の L1=8 ボトルネックを 1536-16-96 に拡張するための初期重みを生成する。
//! - 既存重みをコピーし、 拡張分は小乱数 / ゼロ初期化。
//! - 変換後は bullet で `--l1 16 --l2 96` の resume fine-tune に投入できる。
//!
//! ## ファイル形式
//!
//! ```text
//! [version u32 (=NNUE_VERSION_HALFKA 0x7AF32F20)]
//! [hash u32]
//! [arch_len u32]
//! [arch_str bytes]
//! [ft_hash u32]
//! [FT: LEB128 圧縮ブロック (biases) | LEB128 圧縮ブロック (weights)]
//!     -- 旧 bullet-shogi 形式は 1 ブロック
//! [PSQT (arch に "PSQT=" がある場合のみ)]
//! [Threat (arch に "Threat=" がある場合のみ)]
//! [bucket 0..8: fc_hash(4B) + bucket_data]
//!   bucket_data = L1[bias OUTPUT*4 + weights OUTPUT*padded(INPUT)]
//!               + L2[bias 96*4 + weights 96*padded(L2_IN)]
//!               + Output[bias 1*4 + weights 1*padded(96)]
//! [trailing section (PIN/DUAL 等): verbatim copy]
//! ```
//!
//! ## 変換仕様
//!
//! - FT (L0=1536): そのまま byte-copy。
//! - L1 weights: [8 row x 1536 col] → [16 row x 1536 col]
//!   - 行 0..7: コピー
//!   - 行 7..15 (main 拡張): 小乱数 i8 (std ≈ 0.04 で量子化空間で ±5 程度)
//!   - 行 8 (旧 skip) → 行 15 (新 skip): コピー
//!   - 行 14, 15 → 旧 skip (7) のみが skip。 main 拡張は 0..14 まで。
//!
//! ※ LS_L1_OUT = main + 1 (skip)。 8x96 = main 7 + skip 1、 16x96 = main 15 + skip 1。
//!    元の row[7] (skip) を新しい row[15] にコピーし、 新規 row[7..15] (main 拡張) は乱数。
//!
//! - L1 bias: [8] → [16]、 bias[0..7] コピー、 bias[15] = 旧 bias[7] (skip)、 bias[7..15] = 0
//! - L2 weights: [96 row x padded(14)=32 col] → [96 row x padded(30)=32 col]
//!   - col layout: [sqr_main(7) | crelu_main(7)] → [sqr_main(15) | crelu_main(15)]
//!   - col 0..7: コピー (旧 sqr 7)、 col 7..15: 0 init (新 sqr 8..15)
//!   - col 15..22: 旧 crelu (col 7..14 in src) コピー、 col 22..30: 0 init
//!   - padded col 30..32: padding (0)
//! - L2 bias: [96] そのままコピー
//! - Output weights/bias: [1 row x 96]、 [1] そのままコピー
//! - 8 bucket すべて同様
//! - arch_str: "l2=8" を "l2=16" に書き換え (l3=96 は不変)
//!
//! ## 使用例
//!
//! ```text
//! convert_layerstack_l1_8_to_16 \
//!     --input  /path/to/quantised_1536x8x96.bin \
//!     --output /path/to/quantised_1536x16x96.bin \
//!     --seed   20260514
//! ```

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

// NNUE 関連定数 (rshogi-core から再エクスポートされない private 値は手動で複製)
const NNUE_VERSION_HALFKA: u32 = 0x7AF3_2F20;
const HALFKA_HM_DIMENSIONS: usize = 73_305;
const NUM_BUCKETS: usize = 8;
const L1_FT: usize = 1536;

// 旧 (1536-8-96)
const SRC_L1_OUT: usize = 8;
const SRC_L2_IN: usize = 14;

// 新 (1536-16-96)
const DST_L1_OUT: usize = 16;
const DST_L2_IN: usize = 30;

const L3: usize = 96;

#[derive(Parser, Debug)]
#[command(
    name = "convert_layerstack_l1_8_to_16",
    about = "LayerStack 1536-8-96 quantised.bin を 1536-16-96 に変換"
)]
struct Cli {
    /// 入力 quantised.bin (1536-8-96)
    #[arg(long)]
    input: PathBuf,

    /// 出力 quantised.bin (1536-16-96)
    #[arg(long)]
    output: PathBuf,

    /// 乱数シード (新 main 行の小乱数生成用)
    #[arg(long, default_value = "20260514")]
    seed: u64,

    /// 新 main 行の乱数振幅 (i8 値の絶対値上限)。 既定 4 ≒ std 約 0.03 (i8 量子化空間)。
    #[arg(long, default_value = "4")]
    init_amp: i8,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    let raw = fs::read(&cli.input)
        .with_context(|| format!("入力ファイルが読めません: {:?}", cli.input))?;
    let mut cursor = io::Cursor::new(&raw[..]);

    let header = read_header(&mut cursor)?;
    println!("[info] arch_str: {}", header.arch_str);
    println!("[info] FT block 開始位置: 0x{:x}", header.ft_block_start);

    // FT block (2 ブロック想定: biases + weights)。 旧 bullet-shogi 1 ブロック形式も検出。
    let ft_end = scan_ft_blocks(&raw, header.ft_block_start)?;
    println!("[info] FT block 終了位置: 0x{ft_end:x}");

    // PSQT / Threat の存在判定 (LMAS は 2026-05-25 廃止、 検出時は bail)
    if header.arch_str.contains("LMAS=") {
        bail!(
            "LMAS section は 2026-05-25 に廃止されました。 旧 LMAS 付き ckpt は別途 \
             LMAS section を除去してから再変換してください。"
        );
    }
    if header.arch_str.contains("PSQT=") {
        bail!(
            "PSQT 付きモデルの変換は未対応です (1536-8-96 PSQT モデルは現状運用していないため YAGNI)。\
             必要なら本コンバータを拡張してください。"
        );
    }
    if header.arch_str.contains("Threat=") {
        bail!(
            "Threat 付きモデルの変換は未対応です (1536-8-96 Threat モデルは現状運用していないため YAGNI)。\
             必要なら本コンバータを拡張してください。"
        );
    }

    // 8 bucket 分の FC データを読み込み
    let src_bucket_size = compute_bucket_size(SRC_L1_OUT, SRC_L2_IN, L3);
    let fc_end = ft_end + src_bucket_size * NUM_BUCKETS;
    println!("[info] FC end (8 bucket 後): 0x{fc_end:x}");
    if fc_end > raw.len() {
        bail!(
            "ファイルサイズ不足: ft_end=0x{ft_end:x}, bucket*8=0x{:x}, total=0x{:x}, file_len=0x{:x}",
            src_bucket_size * NUM_BUCKETS,
            fc_end,
            raw.len()
        );
    }

    // FC 末尾以降は trailing section (PIN/DUAL/将来 section) を verbatim コピー。
    let trailing_bytes: &[u8] = &raw[fc_end..];
    if !trailing_bytes.is_empty() {
        println!("[info] trailing section: 0x{:x} byte (verbatim copy)", trailing_bytes.len());
    }

    // RNG (i8 量子化空間での小乱数を生成)
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(cli.seed);

    // 出力 buffer
    let mut out = Vec::with_capacity(raw.len() * 2);

    // 新 arch_str (l2=8 → l2=16)
    let new_arch_str = rewrite_arch_str(&header.arch_str)?;
    println!("[info] new arch_str: {new_arch_str}");

    // version + hash
    out.extend_from_slice(&NNUE_VERSION_HALFKA.to_le_bytes());
    out.extend_from_slice(&header.hash.to_le_bytes());
    // arch_len + arch
    out.extend_from_slice(&(new_arch_str.len() as u32).to_le_bytes());
    out.extend_from_slice(new_arch_str.as_bytes());
    // ft_hash
    out.extend_from_slice(&header.ft_hash.to_le_bytes());
    // FT bytes verbatim
    out.extend_from_slice(&raw[header.ft_block_start..ft_end]);

    // 8 bucket の FC を expand して書き出し
    for bucket_idx in 0..NUM_BUCKETS {
        let src_off = ft_end + bucket_idx * src_bucket_size;
        let src_bucket = &raw[src_off..src_off + src_bucket_size];
        let new_bucket = expand_bucket(src_bucket, &mut rng, cli.init_amp)?;
        out.extend_from_slice(&new_bucket);
    }

    // trailing section (PIN/DUAL 等) は verbatim copy
    out.extend_from_slice(trailing_bytes);

    fs::write(&cli.output, &out)
        .with_context(|| format!("出力ファイルが書けません: {:?}", cli.output))?;
    println!(
        "[done] 書き出し完了: {:?} ({:.2} MB)",
        cli.output,
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

/// FT の LEB128 圧縮ブロックを 1 個以上スキャンし終端 offset を返す。
/// 形式は magic "COMPRESSED_LEB128" + size(u32) + body のブロック列。
/// 1 ブロック (旧 bullet-shogi)、 2 ブロック (YO 形式 biases/weights) のどちらにも対応。
fn scan_ft_blocks(raw: &[u8], mut pos: usize) -> Result<usize> {
    const MAGIC: &[u8] = b"COMPRESSED_LEB128";

    // total_size = L1 (biases) + HALFKA_HM_DIMENSIONS * L1 (weights)
    let total_size_per_element = L1_FT + HALFKA_HM_DIMENSIONS * L1_FT;

    // 1 ブロック目を読む
    if raw.len() < pos + MAGIC.len() + 4 {
        bail!("FT block ヘッダが読めません");
    }
    if &raw[pos..pos + MAGIC.len()] != MAGIC {
        bail!("FT block 開始位置に COMPRESSED_LEB128 magic がない (pos=0x{pos:x})");
    }
    pos += MAGIC.len();
    let block1_size = u32::from_le_bytes(raw[pos..pos + 4].try_into()?) as usize;
    pos += 4;

    // 1 ブロックの要素数を decode して 1 か 2 ブロックか判定
    let block1_elem_count = count_leb128_elements(&raw[pos..pos + block1_size])?;
    pos += block1_size;

    if block1_elem_count == total_size_per_element {
        // 旧形式 (biases + weights が結合)
        return Ok(pos);
    }
    if block1_elem_count != L1_FT {
        bail!(
            "FT 1st block 要素数異常: {block1_elem_count} (期待: {L1_FT} biases-only 又は {total_size_per_element} 結合)"
        );
    }

    // 2 ブロック目 (weights)
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
        // 1 つの LEB128 値を skip
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

/// AffineTransform に必要な padded input。 `n.div_ceil(32) * 32`。
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

/// 旧 bucket bytes (1536-8-96) を新 bucket bytes (1536-16-96) に展開する。
fn expand_bucket(src: &[u8], rng: &mut Xoshiro256PlusPlus, amp: i8) -> Result<Vec<u8>> {
    let dst_size = compute_bucket_size(DST_L1_OUT, DST_L2_IN, L3);
    let mut out = Vec::with_capacity(dst_size);

    let mut cur = 0;
    // fc_hash: そのままコピー
    out.extend_from_slice(&src[cur..cur + 4]);
    cur += 4;

    // L1: input=1536, src_output=8, dst_output=16
    cur = expand_l1(src, cur, &mut out, rng, amp)?;
    // L2: input src=14 (padded 32), output=96; dst input=30 (padded 32 — same!)
    cur = expand_l2(src, cur, &mut out, rng, amp)?;
    // Output: input=96, output=1 — verbatim
    let out_size = affine_size(L3, 1);
    out.extend_from_slice(&src[cur..cur + out_size]);
    cur += out_size;

    if cur != src.len() {
        bail!("bucket parse 不整合: 読み残し {} byte", src.len() - cur);
    }
    if out.len() != dst_size {
        bail!("bucket 出力サイズ不整合: got {}, expected {dst_size}", out.len());
    }
    Ok(out)
}

/// L1 層を展開する (input=1536 固定、 output 8→16)。
///
/// ファイル format:
///   bias: i32 LE × output
///   weights: i8 × output × padded_input(1536) = output × 1536 (padded=1536)
///   weights[output][input] の row-major (各 row は 1536 byte)
///
/// 元 layout の意味:
///   row[0..7] = main 7 row
///   row[7]    = skip (l1_skip = l1_out[MAIN_DIM] = l1_out[7])
///
/// 新 layout:
///   row[0..7]   = 元 main コピー (main 0..7)
///   row[7..15]  = 新規 main 拡張 (main 7..15) → 小乱数
///   row[15]     = 元 skip (元 row[7]) コピー
fn expand_l1(
    src: &[u8],
    mut cur: usize,
    out: &mut Vec<u8>,
    rng: &mut Xoshiro256PlusPlus,
    amp: i8,
) -> Result<usize> {
    // bias
    let mut src_bias = [0i32; SRC_L1_OUT];
    for b in src_bias.iter_mut() {
        *b = i32::from_le_bytes(src[cur..cur + 4].try_into()?);
        cur += 4;
    }
    let mut dst_bias = [0i32; DST_L1_OUT];
    // 旧 row 0..7 (main) → 新 row 0..7
    dst_bias[..SRC_L1_OUT - 1].copy_from_slice(&src_bias[..SRC_L1_OUT - 1]);
    // 旧 skip (row 7) → 新 skip (row 15)
    dst_bias[DST_L1_OUT - 1] = src_bias[SRC_L1_OUT - 1];
    // 新 main 拡張 row 7..15 は 0 init
    for b in dst_bias.iter() {
        out.extend_from_slice(&b.to_le_bytes());
    }

    // weights: row-major, padded_input(1536) = 1536
    let row_bytes = padded_input(L1_FT);
    debug_assert_eq!(row_bytes, L1_FT);
    let src_weight_len = SRC_L1_OUT * row_bytes;
    let src_weights = &src[cur..cur + src_weight_len];
    cur += src_weight_len;

    // 新 row 0..7 = src row 0..7 をそのままコピー
    for r in 0..SRC_L1_OUT - 1 {
        out.extend_from_slice(&src_weights[r * row_bytes..(r + 1) * row_bytes]);
    }
    // 新 row 7..15 = 小乱数
    for _ in 0..(DST_L1_OUT - SRC_L1_OUT) {
        for _ in 0..row_bytes {
            out.push(rand_small_i8(rng, amp));
        }
    }
    // 新 row 15 = 旧 row 7 (skip) コピー
    out.extend_from_slice(&src_weights[(SRC_L1_OUT - 1) * row_bytes..SRC_L1_OUT * row_bytes]);

    Ok(cur)
}

/// L2 層を展開する (input src=14、 dst=30、 output=96 固定)。
///
/// L1 出力 (拡張後 16 dim、 main=15 + skip=1) → SqrCReLU/CReLU で main を 2 倍化 → L2 入力 30。
///
/// 旧 L2 input 14 = sqr_main(7) + crelu_main(7)、 padded_input(14) = 32
/// 新 L2 input 30 = sqr_main(15) + crelu_main(15)、 padded_input(30) = 32
///
/// 偶然 padded_input が両者 32 で一致するが、 col 内 layout (sqr 領域と crelu 領域の境界) が
/// 7→15 にシフトするため、 row ごとに re-pack が必要。
///
/// weights[row][col] (row-major、 row=96、 col=padded=32):
///   src col 0..7 = sqr (main 0..7)  → dst col 0..7 = sqr (main 0..7) コピー
///   src col 7..14 = crelu (main 0..7) → dst col 15..22 = crelu (main 0..7) コピー
///   dst col 7..15 = sqr (新 main 7..15)  → 0 init
///   dst col 22..30 = crelu (新 main 7..15) → 0 init
///   dst col 30..32 = padding (0)
fn expand_l2(
    src: &[u8],
    mut cur: usize,
    out: &mut Vec<u8>,
    _rng: &mut Xoshiro256PlusPlus,
    _amp: i8,
) -> Result<usize> {
    // bias: 96 i32 — そのままコピー
    let bias_bytes = L3 * 4;
    out.extend_from_slice(&src[cur..cur + bias_bytes]);
    cur += bias_bytes;

    // weights
    let src_padded = padded_input(SRC_L2_IN); // 32
    let dst_padded = padded_input(DST_L2_IN); // 32
    debug_assert_eq!(src_padded, dst_padded);

    let row_bytes_src = src_padded;
    let row_bytes_dst = dst_padded;

    for row in 0..L3 {
        let src_row_off = cur + row * row_bytes_src;
        let src_row = &src[src_row_off..src_row_off + row_bytes_src];
        let mut dst_row = vec![0u8; row_bytes_dst];

        // src col 0..7 (= SRC_MAIN sqr) → dst col 0..7
        let src_main = SRC_L1_OUT - 1; // 7
        let dst_main = DST_L1_OUT - 1; // 15
        dst_row[0..src_main].copy_from_slice(&src_row[0..src_main]);
        // src col 7..14 (= SRC_MAIN crelu) → dst col 15..22
        dst_row[dst_main..dst_main + src_main].copy_from_slice(&src_row[src_main..2 * src_main]);
        // dst col 7..15 (sqr 拡張) と 22..30 (crelu 拡張) は 0 init (既に 0)
        // dst col 30..32 は padding (既に 0)
        out.extend_from_slice(&dst_row);
    }
    cur += L3 * row_bytes_src;

    Ok(cur)
}

fn rand_small_i8(rng: &mut Xoshiro256PlusPlus, amp: i8) -> u8 {
    use rand::RngCore;
    if amp == 0 {
        return 0;
    }
    let span = (amp as i32 * 2 + 1) as u32; // [-amp..=amp]
    let r = rng.next_u32() % span;
    let v = (r as i32 - amp as i32) as i8;
    v as u8
}

/// arch_str 内の L1=8 表記を 16 に書き換える。 bullet の Rshogi format で
/// 実 arch_str は YO 互換の `AffineTransform[8<-3072]` / `AffineTransform[96<-14]` /
/// `SqrClippedReLU[14]` 形式 (`l2=8` ではない)。
fn rewrite_arch_str(src: &str) -> Result<String> {
    // 旧 1536-8-96 を示すマーカー (AffineTransform[8<-3072] = L1=8 dense from 1536x2)
    let marker_old = "AffineTransform[8<-3072]";
    if !src.contains(marker_old) {
        return Err(anyhow!(
            "arch_str に '{marker_old}' が見つかりません: {src}\n変換は 1536-8-96 → 1536-16-96 のみサポート"
        ));
    }
    let mut result = src.to_string();
    // L1 dim: 8 → 16
    result = result.replace("AffineTransform[8<-3072]", "AffineTransform[16<-3072]");
    // L2 input dim: 14 → 30 (sqr_crelu 7+7 → 15+15)
    result = result.replace("AffineTransform[96<-14]", "AffineTransform[96<-30]");
    result = result.replace("SqrClippedReLU[14]", "SqrClippedReLU[30]");
    Ok(result)
}

/// `token` を `replacement` に置換する。 ただし token 直後が ASCII 数字なら誤マッチとして
/// スキップする (例: `l2=80` を `l2=160` にしない)。
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
        assert_eq!(padded_input(96), 96);
        assert_eq!(padded_input(1536), 1536);
    }

    #[test]
    fn test_bucket_size() {
        // 1536-8-96
        // fc_hash 4 + L1(1536→8) [bias 8*4=32 + w 8*1536=12288] = 12320
        // + L2(14→96) [bias 96*4=384 + w 96*32=3072] = 3456
        // + Out(96→1) [bias 1*4=4 + w 1*96=96] = 100
        // total = 4 + 12320 + 3456 + 100 = 15880
        assert_eq!(compute_bucket_size(8, 14, 96), 15880);
        // 1536-16-96
        // fc_hash 4 + L1(1536→16) [bias 64 + w 16*1536=24576] = 24640
        // + L2(30→96) [bias 384 + w 96*32=3072] = 3456
        // + Out 100
        // total = 4 + 24640 + 3456 + 100 = 28200
        assert_eq!(compute_bucket_size(16, 30, 96), 28200);
    }

    #[test]
    fn test_rewrite_arch_str() {
        // bullet が実際に書く arch_str に合わせて AffineTransform 表記を含める
        let src = "Features=HalfKA_hm^[73305->1536x2]-SCReLU,\
                   AffineTransform[8<-3072],AffineTransform[96<-14],SqrClippedReLU[14],\
                   fv_scale=16,l2=8,l3=96,qa=127,qb=64";
        let dst = rewrite_arch_str(src).unwrap();
        assert!(dst.contains("AffineTransform[16<-3072]"));
        assert!(dst.contains("AffineTransform[96<-30]"));
        assert!(dst.contains("SqrClippedReLU[30]"));
        assert!(!dst.contains("AffineTransform[8<-3072]"));
        assert!(!dst.contains("SqrClippedReLU[14]"));
    }

    #[test]
    fn test_replace_token_no_overmatch() {
        // l2=80 は触らない
        let src = "fv_scale=8,l2=80,l3=96";
        let dst = replace_token(src, "l2=8", "l2=16");
        assert_eq!(dst, "fv_scale=8,l2=80,l3=96");
    }

    #[test]
    fn test_replace_token_basic() {
        let src = "l2=8,l3=96";
        let dst = replace_token(src, "l2=8", "l2=16");
        assert_eq!(dst, "l2=16,l3=96");
    }

    #[test]
    fn test_rand_small_i8_range() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        for _ in 0..1000 {
            let v = rand_small_i8(&mut rng, 5) as i8;
            assert!(v.abs() <= 5, "out of range: {v}");
        }
    }
}
