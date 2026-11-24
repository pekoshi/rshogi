//! LayerStack 1536-16-32 quantised.bin を 1024-8-64 へ次元削減 warm-init 変換する。
//!
//! ## 目的
//!
//! `q_kda10_A.bin` (= L0=1536, L1=16(eff15+skip1), L2=32, l2_in=30, HalfKA_hm,
//! NUM_BUCKETS=8, PSQT + head sections 付き) を、 L0=1024 / L1=8(eff7+skip1) /
//! L2=64 / l2_in=14 の新 arch へ warm-init する。 訓練本体は別途 bullet で
//! `--l0 1024 --l1 8 --l2 64` で resume fine-tune する前提。
//!
//! warm-init 直後の eval 劣化は前提 (fine-tune で回復させる)。
//!
//! ## 触るテンソルと不変テンソル
//!
//! ### 次元削減する 7 テンソル
//! - `l0w` (halfka_dim × L0): FT 出力 ch を 1536→1024 に重要度プルーニング。
//! - `l0b` (L0): 同じ ch 集合 K に絞る。
//! - `l1w` (NUM_BUCKETS × L1 × L0): L0 列を K に絞り、 L1 行を 16→8 に縮小 (+skip 保存)。
//! - `l1b` (NUM_BUCKETS × L1): L1 16→8 縮小。
//! - `l2w` (NUM_BUCKETS × L2 × l2_in): l2_in 30→14、 L2 出力 32→64 拡大。
//! - `l2b` (NUM_BUCKETS × L2): L2 32→64 拡大。
//! - `l3w` (NUM_BUCKETS × L2): L2 32→64 拡大。
//!
//! ### byte verbatim copy (L0 非依存)
//! - PSQT (psqtw/psqtb)
//! - 末尾 head section 群 (P2DUALZ / P2DUAL2 / P2PAIR / P2KDGR / P2ADEV / ...)
//! - l3b (8) は不変 (= bucket ごとの Output bias、 L2 拡大でも値は据置)。
//!
//! ## L0 次元削減 = 重要度プルーニング
//!
//! 各 FT 出力 ch c のスコア = 全 bucket・全 L1 行にわたる l1w (merged i8, scale=QB) の
//! 二乗和 (= downstream 使用度の L2 ノルム²)。 上位 1024 ch を選び、 元の順序を保つ。
//! 選択 index 集合 K を必ずログ出力する。
//!
//! ## L1 16→8 = 行縮小 + skip 保存
//!
//! 旧 L1 layout: main = row 0..15 (eff 15)、 skip = row 15。
//! 新 L1 layout: main = row 0..7 (eff 7)、 skip = row 7。
//! 旧 main 行から重要度 (L1 row の L2 ノルム²、 K に絞った後) 上位 7 本を index 昇順で採用、
//! 旧 skip (row 15) を新 skip (row 7) へ。 l2w の入力列は sqr 部 (前半 15→7) と
//! crelu 部 (後半 15→7) を、 採用した同じ main 行に対応させて縮小する。
//!
//! ## L2 32→64 = 拡大
//!
//! L2 出力 ch を 32 コピー + 残り 32 を 0 init (i8 wrap 回避)。 l2b/l3w/l2w の出力 ch を拡大。
//!
//! ## 使用例
//!
//! ```text
//! cargo run --release -p nnue-convert --bin shrink_layerstack_1536_16_32_to_1024_8_64 -- \
//!     --input  E:/shogi/p2shogi/data/suisho_ft/q_kda10_A.bin \
//!     --output E:/shogi/p2shogi/data/suisho_ft/q_kda10_A_1024.bin
//! ```

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;

const NNUE_VERSION_HALFKA: u32 = 0x7AF3_2F20;
const HALFKA_HM_DIMENSIONS: usize = 73_305;
const NUM_BUCKETS: usize = 8;

// 旧 (L0=1536, L1=16, L2=32)
const SRC_L0: usize = 1536;
const SRC_L1_OUT: usize = 16;
const SRC_L2_OUT: usize = 32;

// 新 (L0=1024, L1=8, L2=64)
const DST_L0: usize = 1024;
const DST_L1_OUT: usize = 8;
const DST_L2_OUT: usize = 64;

#[derive(Parser, Debug)]
#[command(
    name = "shrink_layerstack_1536_16_32_to_1024_8_64",
    about = "LayerStack 1536-16-32 quantised.bin を 1024-8-64 へ次元削減 warm-init"
)]
struct Cli {
    /// 入力 quantised.bin (1536-16-32, PSQT + heads)
    #[arg(long)]
    input: PathBuf,

    /// 出力 quantised.bin (1024-8-64, PSQT + heads)
    #[arg(long)]
    output: PathBuf,

    /// L2 拡大部 (新 ch 32..64) の初期値振幅 (i8 量子化空間)。 既定 0。
    #[arg(long, default_value = "0")]
    l2_expand_amp: i8,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();

    let raw = fs::read(&cli.input)
        .with_context(|| format!("入力ファイルが読めません: {:?}", cli.input))?;
    let mut cursor = io::Cursor::new(&raw[..]);
    let header = read_header(&mut cursor)?;
    println!("[info] 入力 arch_str: {}", header.arch_str);

    // 入力 arch の妥当性チェック (1536-16-32 マーカー)。
    if !header.arch_str.contains("AffineTransform[16<-3072]") {
        bail!(
            "arch_str に 'AffineTransform[16<-3072]' (= L1=16, L0=1536) が見つかりません。\n\
             1536-16-32 モデルではない可能性: {}",
            header.arch_str
        );
    }
    if !header.arch_str.contains("PSQT=") {
        bail!("PSQT 付きモデルを想定しています (arch に PSQT= がない): {}", header.arch_str);
    }
    if header.arch_str.contains("LMAS=") {
        bail!("LMAS section は廃止済。 LMAS 付き ckpt は非対応。");
    }
    if header.arch_str.contains("Threat=") {
        bail!("Threat (l0w 拡張) 付き ckpt は非対応 (FT プルーニングが threat 部に未対応)。");
    }

    // ---- FT block (biases + weights) を decode ----
    let ft_block_start = header.ft_block_start;
    let (ft_biases_i16, ft_weights_i16, ft_end) = decode_ft(&raw, ft_block_start)?;
    if ft_biases_i16.len() != SRC_L0 {
        bail!("FT biases 要素数異常: {} (期待 {})", ft_biases_i16.len(), SRC_L0);
    }
    if ft_weights_i16.len() != HALFKA_HM_DIMENSIONS * SRC_L0 {
        bail!(
            "FT weights 要素数異常: {} (期待 {})",
            ft_weights_i16.len(),
            HALFKA_HM_DIMENSIONS * SRC_L0
        );
    }
    println!("[info] FT block 終了位置: 0x{ft_end:x}");

    // ---- PSQT bytes (byte copy 対象) ----
    let psqt_bytes_len = NUM_BUCKETS * 4 + HALFKA_HM_DIMENSIONS * NUM_BUCKETS * 4;
    let psqt_start = ft_end;
    let psqt_end = psqt_start + psqt_bytes_len;
    if psqt_end > raw.len() {
        bail!("PSQT 領域がファイルをはみ出します: end=0x{psqt_end:x} len=0x{:x}", raw.len());
    }
    let psqt_bytes = &raw[psqt_start..psqt_end];
    println!("[info] PSQT bytes: {} (byte copy)", psqt_bytes.len());

    // ---- 8 bucket 分の FC (src layout) を読み込む ----
    let src_bucket_size = compute_bucket_size(SRC_L0, SRC_L1_OUT, SRC_L2_OUT);
    let buckets_start = psqt_end;
    let buckets_end = buckets_start + src_bucket_size * NUM_BUCKETS;
    if buckets_end > raw.len() {
        bail!(
            "bucket 領域がファイルをはみ出します: end=0x{buckets_end:x} len=0x{:x}",
            raw.len()
        );
    }
    println!("[info] src bucket size: {src_bucket_size}, 8 bucket end: 0x{buckets_end:x}");

    // 8 bucket の merged L1 weights (i8) を decode し、 L0 重要度スコアを計算する。
    let mut src_buckets: Vec<BucketSrc> = Vec::with_capacity(NUM_BUCKETS);
    for b in 0..NUM_BUCKETS {
        let off = buckets_start + b * src_bucket_size;
        src_buckets.push(parse_bucket_src(&raw[off..off + src_bucket_size])?);
    }

    // ---- 末尾 trailing section (head 群) は verbatim copy ----
    let trailing = &raw[buckets_end..];
    println!("[info] trailing section: {} byte (verbatim copy)", trailing.len());

    // ---- L0 重要度プルーニング: 上位 1024 ch を選ぶ (元順序保持) ----
    let l0_keep = select_l0_indices(&src_buckets);
    println!("[info] L0 prune: {} → {} (selected {} channels)", SRC_L0, DST_L0, l0_keep.len());
    println!("[info] L0 selected index 集合 K (昇順): {:?}", l0_keep);

    // ---- L1 main 行選択 (全 bucket 共通) ----
    let l1_main_rows = select_l1_main_rows(&src_buckets, &l0_keep);
    println!(
        "[info] L1 main 行選択: {} → {} (採用 row 昇順) {:?}; skip row {} → {}",
        SRC_L1_OUT - 1,
        DST_L1_OUT - 1,
        l1_main_rows,
        SRC_L1_OUT - 1,
        DST_L1_OUT - 1
    );

    // ---- 新 arch_str / hash 群を再生成 ----
    let new_arch_str = build_arch_str(&header.arch_str)?;
    println!("[info] 新 arch_str: {new_arch_str}");
    let dst_l1_eff = DST_L1_OUT - 1; // 7
    let dst_l2_in = dst_l1_eff * 2; // 14
    let new_fc_hash = compute_layerstack_fc_hash(DST_L0, dst_l2_in, DST_L2_OUT);
    // FEATURE_HASH_HM_V2 を入力 ft_hash から逆算 (定数ハードコードを避ける):
    //   ft_hash = FEATURE_HASH_HM_V2 ^ (ft_out*2)。
    let feature_hash_hm_v2 = header.ft_hash ^ ((SRC_L0 * 2) as u32);
    let new_ft_hash = feature_hash_hm_v2 ^ ((DST_L0 * 2) as u32);
    let new_net_hash = new_fc_hash ^ new_ft_hash;
    println!(
        "[info] 新 hash: net=0x{new_net_hash:08x} ft=0x{new_ft_hash:08x} fc=0x{new_fc_hash:08x}"
    );

    // ---- FT プルーニング ----
    let new_ft_biases: Vec<i16> = l0_keep.iter().map(|&c| ft_biases_i16[c]).collect();
    // weights: index = feat * SRC_L0 + c → 新 feat * DST_L0 + new_c
    let mut new_ft_weights: Vec<i16> = vec![0; HALFKA_HM_DIMENSIONS * DST_L0];
    for feat in 0..HALFKA_HM_DIMENSIONS {
        let src_row = feat * SRC_L0;
        let dst_row = feat * DST_L0;
        for (new_c, &old_c) in l0_keep.iter().enumerate() {
            new_ft_weights[dst_row + new_c] = ft_weights_i16[src_row + old_c];
        }
    }
    println!(
        "[shape] l0b: {} → {}; l0w: {} → {}",
        SRC_L0,
        new_ft_biases.len(),
        ft_weights_i16.len(),
        new_ft_weights.len()
    );

    // ---- 出力 buffer 組み立て ----
    let mut out = Vec::with_capacity(raw.len() / 2);
    let arch_bytes = new_arch_str.as_bytes();
    out.extend_from_slice(&NNUE_VERSION_HALFKA.to_le_bytes());
    out.extend_from_slice(&new_net_hash.to_le_bytes());
    out.extend_from_slice(&(arch_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(arch_bytes);
    out.extend_from_slice(&new_ft_hash.to_le_bytes());
    // FT biases + weights (2 ブロック LEB128)
    out.extend_from_slice(&encode_leb128_tensor_i16(&new_ft_biases));
    out.extend_from_slice(&encode_leb128_tensor_i16(&new_ft_weights));
    // PSQT byte copy
    out.extend_from_slice(psqt_bytes);

    // 8 bucket 変換
    for (b, src) in src_buckets.iter().enumerate() {
        let new_bucket = shrink_bucket_with_rows(src, &l0_keep, &l1_main_rows, cli.l2_expand_amp)?;
        if b == 0 {
            println!(
                "[shape] (bucket0) l1b: {} → {}; l1w: {}×{} → {}×{}; l2b: {} → {}; l2w: {}×{} → {}×{}; l3w: {} → {}",
                SRC_L1_OUT,
                DST_L1_OUT,
                SRC_L1_OUT,
                SRC_L0,
                DST_L1_OUT,
                DST_L0,
                SRC_L2_OUT,
                DST_L2_OUT,
                SRC_L2_OUT,
                SRC_L1_OUT - 1, // l2_in eff before pad
                DST_L2_OUT,
                DST_L1_OUT - 1,
                SRC_L2_OUT,
                DST_L2_OUT,
            );
        }
        out.extend_from_slice(&new_bucket);
    }

    // trailing (head 群) byte copy
    out.extend_from_slice(trailing);

    fs::write(&cli.output, &out)
        .with_context(|| format!("出力ファイルが書けません: {:?}", cli.output))?;
    println!(
        "[done] 書き出し完了: {:?} ({:.2} MB → {:.2} MB)",
        cli.output,
        raw.len() as f64 / (1024.0 * 1024.0),
        out.len() as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

// ============================================================================
// Header
// ============================================================================

struct Header {
    arch_str: String,
    ft_hash: u32,
    ft_block_start: usize,
}

fn read_header(cursor: &mut io::Cursor<&[u8]>) -> Result<Header> {
    let version = read_u32(cursor)?;
    if version != NNUE_VERSION_HALFKA {
        bail!("予期しない version: 0x{version:08x} (期待 0x{NNUE_VERSION_HALFKA:08x})");
    }
    let _network_hash = read_u32(cursor)?;
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
        ft_hash,
        ft_block_start,
    })
}

fn read_u32(cursor: &mut io::Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

// ============================================================================
// FT LEB128 decode / encode
// ============================================================================

/// FT の biases / weights ブロック (各 COMPRESSED_LEB128) を decode する。
/// 戻り値: (biases_i16, weights_i16, ft_end_offset)。
fn decode_ft(raw: &[u8], mut pos: usize) -> Result<(Vec<i16>, Vec<i16>, usize)> {
    const MAGIC: &[u8] = b"COMPRESSED_LEB128";
    // block 1
    if raw.len() < pos + MAGIC.len() + 4 || &raw[pos..pos + MAGIC.len()] != MAGIC {
        bail!("FT 1st block magic がない (pos=0x{pos:x})");
    }
    pos += MAGIC.len();
    let size1 = u32::from_le_bytes(raw[pos..pos + 4].try_into()?) as usize;
    pos += 4;
    let block1 = decode_leb128_i16(&raw[pos..pos + size1])?;
    pos += size1;

    let total = SRC_L0 + HALFKA_HM_DIMENSIONS * SRC_L0;
    if block1.len() == total {
        // 結合形式 (旧 bullet-shogi)。 biases | weights に分割。
        let (b, w) = block1.split_at(SRC_L0);
        return Ok((b.to_vec(), w.to_vec(), pos));
    }
    if block1.len() != SRC_L0 {
        bail!(
            "FT 1st block 要素数異常: {} (期待 {} または結合 {})",
            block1.len(),
            SRC_L0,
            total
        );
    }
    // block 2 (weights)
    if raw.len() < pos + MAGIC.len() + 4 || &raw[pos..pos + MAGIC.len()] != MAGIC {
        bail!("FT 2nd block magic がない (pos=0x{pos:x})");
    }
    pos += MAGIC.len();
    let size2 = u32::from_le_bytes(raw[pos..pos + 4].try_into()?) as usize;
    pos += 4;
    let block2 = decode_leb128_i16(&raw[pos..pos + size2])?;
    pos += size2;
    Ok((block1, block2, pos))
}

/// signed LEB128 byte 列を i16 配列へ decode する。
fn decode_leb128_i16(data: &[u8]) -> Result<Vec<i16>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let mut result: i64 = 0;
        let mut shift = 0;
        let mut byte;
        loop {
            if i >= data.len() {
                bail!("LEB128 オーバーラン");
            }
            byte = data[i];
            i += 1;
            result |= ((byte & 0x7f) as i64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                break;
            }
        }
        // 符号拡張
        if shift < 64 && (byte & 0x40) != 0 {
            result |= -1i64 << shift;
        }
        if result < i16::MIN as i64 || result > i16::MAX as i64 {
            bail!("LEB128 値が i16 範囲外: {result}");
        }
        out.push(result as i16);
    }
    Ok(out)
}

/// i16 配列を signed LEB128 圧縮し、 magic + size + data を返す (bullet 互換)。
fn encode_leb128_tensor_i16(values: &[i16]) -> Vec<u8> {
    let mut compressed = Vec::new();
    for &val in values {
        encode_signed_leb128(val as i64, &mut compressed);
    }
    let mut result = Vec::with_capacity(17 + 4 + compressed.len());
    result.extend_from_slice(b"COMPRESSED_LEB128");
    result.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    result.extend_from_slice(&compressed);
    result
}

fn encode_signed_leb128(mut value: i64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            out.push(byte);
            break;
        }
        byte |= 0x80;
        out.push(byte);
    }
}

// ============================================================================
// padding / sizes
// ============================================================================

const fn pad32(n: usize) -> usize {
    n.div_ceil(32) * 32
}

const fn affine_size(input: usize, output: usize) -> usize {
    output * 4 + output * pad32(input)
}

/// per-bucket FC bytes (fc_hash + L1 + L2 + Output)。 l2_in = (l1_out-1)*2。
const fn compute_bucket_size(l0: usize, l1_out: usize, l2_out: usize) -> usize {
    let l2_in = (l1_out - 1) * 2;
    4 // fc_hash
        + affine_size(l0, l1_out)
        + affine_size(l2_in, l2_out)
        + affine_size(l2_out, 1)
}

// ============================================================================
// bucket 解析 (src)
// ============================================================================

/// src bucket の dequantize 済み中間表現。
struct BucketSrc {
    fc_hash: u32,
    /// L1 bias (merged), i32 raw, len = SRC_L1_OUT
    l1b: Vec<i32>,
    /// L1 weights i8 raw, row-major [SRC_L1_OUT][pad32(SRC_L0)]、 列は SRC_L0 まで有効。
    /// 保持は [SRC_L1_OUT][SRC_L0] に詰め直したもの。
    l1w: Vec<i8>,
    /// L2 bias i32 raw, len = SRC_L2_OUT
    l2b: Vec<i32>,
    /// L2 weights i8 raw [SRC_L2_OUT][src_l2_in]、 src_l2_in = (SRC_L1_OUT-1)*2 = 30。
    l2w: Vec<i8>,
    /// Output bias i32
    l3b: i32,
    /// Output weights i8 [SRC_L2_OUT]
    l3w: Vec<i8>,
}

fn parse_bucket_src(src: &[u8]) -> Result<BucketSrc> {
    let src_l2_in = (SRC_L1_OUT - 1) * 2; // 30
    let l1_padded = pad32(SRC_L0); // 1536
    let l2_padded = pad32(src_l2_in); // 32
    let out_padded = pad32(SRC_L2_OUT); // 32
    let mut cur = 0usize;

    let fc_hash = u32::from_le_bytes(src[cur..cur + 4].try_into()?);
    cur += 4;

    // L1 bias i32 × SRC_L1_OUT
    let mut l1b = vec![0i32; SRC_L1_OUT];
    for v in l1b.iter_mut() {
        *v = i32::from_le_bytes(src[cur..cur + 4].try_into()?);
        cur += 4;
    }
    // L1 weights i8 [SRC_L1_OUT][l1_padded]、 有効列 0..SRC_L0
    let mut l1w = vec![0i8; SRC_L1_OUT * SRC_L0];
    for row in 0..SRC_L1_OUT {
        for col in 0..l1_padded {
            let b = src[cur] as i8;
            cur += 1;
            if col < SRC_L0 {
                l1w[row * SRC_L0 + col] = b;
            }
        }
    }
    // L2 bias i32 × SRC_L2_OUT
    let mut l2b = vec![0i32; SRC_L2_OUT];
    for v in l2b.iter_mut() {
        *v = i32::from_le_bytes(src[cur..cur + 4].try_into()?);
        cur += 4;
    }
    // L2 weights i8 [SRC_L2_OUT][l2_padded]、 有効列 0..src_l2_in
    let mut l2w = vec![0i8; SRC_L2_OUT * src_l2_in];
    for row in 0..SRC_L2_OUT {
        for col in 0..l2_padded {
            let b = src[cur] as i8;
            cur += 1;
            if col < src_l2_in {
                l2w[row * src_l2_in + col] = b;
            }
        }
    }
    // Output bias i32
    let l3b = i32::from_le_bytes(src[cur..cur + 4].try_into()?);
    cur += 4;
    // Output weights i8 [out_padded]、 有効 0..SRC_L2_OUT、 残りは padding 読み捨て。
    let mut l3w = vec![0i8; SRC_L2_OUT];
    for slot in l3w.iter_mut() {
        *slot = src[cur] as i8;
        cur += 1;
    }
    cur += out_padded - SRC_L2_OUT; // padding 読み捨て

    if cur != src.len() {
        bail!("bucket parse 不整合: 読み残し {} byte", src.len() - cur);
    }
    Ok(BucketSrc {
        fc_hash,
        l1b,
        l1w,
        l2b,
        l2w,
        l3b,
        l3w,
    })
}

// ============================================================================
// L0 重要度プルーニング
// ============================================================================

/// 各 FT 出力 ch c のスコア = 全 bucket・全 L1 行にわたる l1w[:, :, c] の二乗和。
/// 上位 DST_L0 ch を選び、 元の順序 (昇順) を保って返す。
fn select_l0_indices(buckets: &[BucketSrc]) -> Vec<usize> {
    let mut score = vec![0f64; SRC_L0];
    for b in buckets {
        for row in 0..SRC_L1_OUT {
            let base = row * SRC_L0;
            for (c, s) in score.iter_mut().enumerate() {
                let w = b.l1w[base + c] as f64;
                *s += w * w;
            }
        }
    }
    // 上位 DST_L0 を選ぶ。 score の降順、 同点は index 昇順 (安定)。
    let mut idx: Vec<usize> = (0..SRC_L0).collect();
    idx.sort_by(|&a, &b| {
        score[b]
            .partial_cmp(&score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut keep: Vec<usize> = idx.into_iter().take(DST_L0).collect();
    keep.sort_unstable(); // 元順序保持
    keep
}

// ============================================================================
// L1 行選択
// ============================================================================

/// 旧 main 行 (0..SRC_L1_OUT-1 = 0..15) から重要度上位 DST_L1_OUT-1 (=7) 本を index 昇順で選ぶ。
/// 重要度 = K に絞った後の L1 row weight の二乗和 (全 bucket 合算)。
fn select_l1_main_rows(buckets: &[BucketSrc], l0_keep: &[usize]) -> Vec<usize> {
    let src_main = SRC_L1_OUT - 1; // 15
    let dst_main = DST_L1_OUT - 1; // 7
    let mut score = vec![0f64; src_main];
    for b in buckets {
        for (row, s) in score.iter_mut().enumerate() {
            let base = row * SRC_L0;
            let mut acc = 0f64;
            for &c in l0_keep {
                let w = b.l1w[base + c] as f64;
                acc += w * w;
            }
            *s += acc;
        }
    }
    let mut idx: Vec<usize> = (0..src_main).collect();
    idx.sort_by(|&a, &b| {
        score[b]
            .partial_cmp(&score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut rows: Vec<usize> = idx.into_iter().take(dst_main).collect();
    rows.sort_unstable();
    rows
}

// ============================================================================
// bucket 縮小
// ============================================================================

/// 1 bucket を 1024-8-64 layout に変換して bytes を返す。
/// L1 main 行集合 (全 bucket 共通) を明示的に渡す。
fn shrink_bucket_with_rows(
    src: &BucketSrc,
    l0_keep: &[usize],
    l1_main_rows: &[usize],
    l2_expand_amp: i8,
) -> Result<Vec<u8>> {
    let src_skip = SRC_L1_OUT - 1; // 15
    let dst_l2_in = (DST_L1_OUT - 1) * 2; // 14
    let src_l2_in = (SRC_L1_OUT - 1) * 2; // 30
    let src_l1_eff = SRC_L1_OUT - 1; // 15 (= sqr 部 / crelu 部 のサイズ)

    let l1_padded = pad32(DST_L0); // 1024
    let l2_padded = pad32(dst_l2_in); // 32
    let out_padded = pad32(DST_L2_OUT); // 64

    let mut out = Vec::with_capacity(compute_bucket_size(DST_L0, DST_L1_OUT, DST_L2_OUT));

    // fc_hash: rshogi 推論側は読み飛ばすが、 整合のため新 arch の fc_hash を書く
    // (src.fc_hash は旧 arch のものなので破棄)。
    let _ = src.fc_hash;
    let new_fc_hash = compute_layerstack_fc_hash(DST_L0, dst_l2_in, DST_L2_OUT);
    out.extend_from_slice(&new_fc_hash.to_le_bytes());

    // ---- L1 ----
    // 新 row 0..7 = main rows (l1_main_rows)、 新 row 7 = src skip (row 15)。
    // bias
    for &r in l1_main_rows {
        out.extend_from_slice(&src.l1b[r].to_le_bytes());
    }
    out.extend_from_slice(&src.l1b[src_skip].to_le_bytes());
    // weights i8 [DST_L1_OUT][l1_padded]、 列は K に絞る。
    let mut dst_rows: Vec<usize> = l1_main_rows.to_vec();
    dst_rows.push(src_skip);
    debug_assert_eq!(dst_rows.len(), DST_L1_OUT);
    for &r in &dst_rows {
        let base = r * SRC_L0;
        // 有効列 0..DST_L0 は K に絞った重み、 DST_L0..l1_padded は padding。
        for &old_c in l0_keep {
            out.push(src.l1w[base + old_c] as u8);
        }
        out.extend(std::iter::repeat_n(0u8, l1_padded - DST_L0));
    }

    // ---- L2 ----
    // 出力 ch: 0..32 = src コピー、 32..64 = 拡大 (0 or 小値)。
    // 入力列 (dst_l2_in=14): sqr 部 (0..7) と crelu 部 (7..14)。
    //   src l2_in=30: sqr 部 (0..15) | crelu 部 (15..30)。
    //   新 sqr 列 j ← src sqr 列 l1_main_rows[j]、 新 crelu 列 j ← src crelu 列 (src_l1_eff + l1_main_rows[j])。
    // bias i32 × DST_L2_OUT
    for out_idx in 0..DST_L2_OUT {
        if out_idx < SRC_L2_OUT {
            out.extend_from_slice(&src.l2b[out_idx].to_le_bytes());
        } else {
            out.extend_from_slice(&0i32.to_le_bytes());
        }
    }
    // weights i8 [DST_L2_OUT][l2_padded]
    for out_idx in 0..DST_L2_OUT {
        for col in 0..l2_padded {
            if out_idx >= SRC_L2_OUT {
                // 拡大出力 ch
                if col < dst_l2_in {
                    out.push(l2_expand_amp as u8);
                } else {
                    out.push(0);
                }
                continue;
            }
            if col < dst_l2_in {
                // 採用する src 入力列を決定
                let src_col = if col < (DST_L1_OUT - 1) {
                    // sqr 部
                    l1_main_rows[col]
                } else {
                    // crelu 部
                    let k = col - (DST_L1_OUT - 1);
                    src_l1_eff + l1_main_rows[k]
                };
                debug_assert!(src_col < src_l2_in);
                out.push(src.l2w[out_idx * src_l2_in + src_col] as u8);
            } else {
                out.push(0);
            }
        }
    }

    // ---- Output ----
    // bias: src l3b そのまま (L2 拡大しても Output bias は不変)。
    out.extend_from_slice(&src.l3b.to_le_bytes());
    // weights i8 [out_padded]、 0..32 = src、 32..64 = 0、 残り padding。
    for col in 0..out_padded {
        if col < SRC_L2_OUT {
            out.push(src.l3w[col] as u8);
        } else {
            out.push(0);
        }
    }

    let expected = compute_bucket_size(DST_L0, DST_L1_OUT, DST_L2_OUT);
    if out.len() != expected {
        bail!("bucket 出力サイズ不整合: got {}, expected {expected}", out.len());
    }
    Ok(out)
}

// ============================================================================
// hash / arch_str
// ============================================================================

fn compute_layerstack_fc_hash(l1_out: usize, _l2_in: usize, l2_out: usize) -> u32 {
    let mut prev_hash: u32 = 0xEC42_E90D;
    prev_hash ^= (l1_out * 2) as u32;
    let layer_sizes = [(l1_out, true), (l2_out, true), (1usize, false)];
    for (out_features, has_relu) in layer_sizes {
        let mut layer_hash: u32 = 0xCC03_DAE4;
        layer_hash = layer_hash.wrapping_add(out_features as u32);
        layer_hash ^= prev_hash >> 1;
        layer_hash ^= prev_hash << 31;
        if has_relu {
            layer_hash = layer_hash.wrapping_add(0x538D_24C7);
        }
        prev_hash = layer_hash;
    }
    prev_hash
}

/// 入力 arch_str を 1024/8/64/l2_in=14 用に書き換える。
/// bullet の Rshogi format:
///   AffineTransform[1<-32]   (Output input = l2_out)
///   ClippedReLU[32]          (l2_out)
///   AffineTransform[32<-30]  (l2_out <- l2_in)
///   SqrClippedReLU[30]       (l2_in)
///   AffineTransform[16<-3072](l1_out <- ft_out*2)
///   InputSlice[3072(0:3072)] (ft_out*2)
fn build_arch_str(src: &str) -> Result<String> {
    if !src.contains("AffineTransform[16<-3072]") {
        bail!("arch_str に 'AffineTransform[16<-3072]' が見つかりません: {src}");
    }
    let mut r = src.to_string();
    // Output input / l2_out: 32 → 64
    r = r.replace("AffineTransform[1<-32]", "AffineTransform[1<-64]");
    r = r.replace("ClippedReLU[32]", "ClippedReLU[64]");
    // L2: l2_out<-l2_in: 32<-30 → 64<-14
    r = r.replace("AffineTransform[32<-30]", "AffineTransform[64<-14]");
    r = r.replace("SqrClippedReLU[30]", "SqrClippedReLU[14]");
    // L1: l1_out<-ft_out*2: 16<-3072 → 8<-2048
    r = r.replace("AffineTransform[16<-3072]", "AffineTransform[8<-2048]");
    // InputSlice: 3072 → 2048
    r = r.replace("InputSlice[3072(0:3072)]", "InputSlice[2048(0:2048)]");
    // Features FT 出力: 1536x2 → 1024x2
    r = r.replace("1536x2", "1024x2");
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad32() {
        assert_eq!(pad32(14), 32);
        assert_eq!(pad32(30), 32);
        assert_eq!(pad32(32), 32);
        assert_eq!(pad32(64), 64);
        assert_eq!(pad32(1024), 1024);
        assert_eq!(pad32(1536), 1536);
    }

    #[test]
    fn test_bucket_size() {
        // src 1536-16-32: fc4 + L1(1536→16)[16*4 + 16*1536] + L2(30→32)[32*4 + 32*pad(30)=32*32]
        //   + Out(32→1)[4 + pad(32)=32]
        // = 4 + (64 + 24576) + (128 + 1024) + (4 + 32) = 4 + 24640 + 1152 + 36 = 25832
        assert_eq!(compute_bucket_size(1536, 16, 32), 25832);
        // dst 1024-8-64: fc4 + L1(1024→8)[8*4 + 8*1024] + L2(14→64)[64*4 + 64*pad(14)=64*32]
        //   + Out(64→1)[4 + pad(64)=64]
        // = 4 + (32 + 8192) + (256 + 2048) + (4 + 64) = 4 + 8224 + 2304 + 68 = 10600
        assert_eq!(compute_bucket_size(1024, 8, 64), 10600);
    }

    #[test]
    fn test_leb128_roundtrip() {
        let vals: Vec<i16> = vec![
            0,
            1,
            -1,
            127,
            -128,
            255,
            -256,
            12345,
            -12345,
            i16::MAX,
            i16::MIN,
        ];
        let enc = encode_leb128_tensor_i16(&vals);
        assert_eq!(&enc[..17], b"COMPRESSED_LEB128");
        let size = u32::from_le_bytes(enc[17..21].try_into().unwrap()) as usize;
        let dec = decode_leb128_i16(&enc[21..21 + size]).unwrap();
        assert_eq!(dec, vals);
    }

    #[test]
    fn test_select_l0_indices_picks_top() {
        // 1 bucket、 全 row 0、 ただし ch c の weight を c に比例させて、 上位 1024 が
        // 大きい index 側 (512..1536) に集中するように設定。
        let mut l1w = vec![0i8; SRC_L1_OUT * SRC_L0];
        for row in 0..SRC_L1_OUT {
            for c in 0..SRC_L0 {
                // c が大きいほど大きい weight (i8 範囲に収める)
                l1w[row * SRC_L0 + c] = ((c % 100) as i8).wrapping_sub(1);
            }
        }
        // c = 0 のとき weight 0-1 = -1、 c=99 のとき 98... 周期的なので
        // ここでは単純に「上位 DST_L0 個が選ばれ、 昇順で len=1024」を確認。
        let b = BucketSrc {
            fc_hash: 0,
            l1b: vec![0; SRC_L1_OUT],
            l1w,
            l2b: vec![0; SRC_L2_OUT],
            l2w: vec![0; SRC_L2_OUT * 30],
            l3b: 0,
            l3w: vec![0; SRC_L2_OUT],
        };
        let keep = select_l0_indices(&[b]);
        assert_eq!(keep.len(), DST_L0);
        // 昇順
        for w in keep.windows(2) {
            assert!(w[0] < w[1]);
        }
        for &c in &keep {
            assert!(c < SRC_L0);
        }
    }

    #[test]
    fn test_select_l1_main_rows_len_and_order() {
        // row r の weight 振幅を r に比例 → 上位 7 = row 9..15 が選ばれるはず。
        let mut l1w = vec![0i8; SRC_L1_OUT * SRC_L0];
        for row in 0..SRC_L1_OUT {
            for c in 0..SRC_L0 {
                l1w[row * SRC_L0 + c] = (row as i8) % 8;
            }
        }
        let b = BucketSrc {
            fc_hash: 0,
            l1b: vec![0; SRC_L1_OUT],
            l1w,
            l2b: vec![0; SRC_L2_OUT],
            l2w: vec![0; SRC_L2_OUT * 30],
            l3b: 0,
            l3w: vec![0; SRC_L2_OUT],
        };
        let keep: Vec<usize> = (0..SRC_L0).collect();
        let rows = select_l1_main_rows(&[b], &keep);
        assert_eq!(rows.len(), DST_L1_OUT - 1);
        for w in rows.windows(2) {
            assert!(w[0] < w[1]);
        }
        // skip 行 (15) は main 候補に含めない
        for &r in &rows {
            assert!(r < SRC_L1_OUT - 1);
        }
    }

    #[test]
    fn test_shrink_bucket_size() {
        // src bucket を全 0 で作り、 出力サイズが dst bucket size に一致すること。
        let src = BucketSrc {
            fc_hash: 0x1234_5678,
            l1b: vec![0; SRC_L1_OUT],
            l1w: vec![0; SRC_L1_OUT * SRC_L0],
            l2b: vec![0; SRC_L2_OUT],
            l2w: vec![0; SRC_L2_OUT * 30],
            l3b: 0,
            l3w: vec![0; SRC_L2_OUT],
        };
        let keep: Vec<usize> = (0..DST_L0).collect();
        let rows: Vec<usize> = (0..(DST_L1_OUT - 1)).collect();
        let out = shrink_bucket_with_rows(&src, &keep, &rows, 0).unwrap();
        assert_eq!(out.len(), compute_bucket_size(DST_L0, DST_L1_OUT, DST_L2_OUT));
        // fc_hash は新値が書かれる
        let fc = u32::from_le_bytes(out[0..4].try_into().unwrap());
        assert_eq!(fc, compute_layerstack_fc_hash(DST_L0, (DST_L1_OUT - 1) * 2, DST_L2_OUT));
    }

    #[test]
    fn test_shrink_bucket_l1_copies_selected_rows_and_skip() {
        // l1w[row][col] = row*1000 + col に近い識別値 (i8 範囲のため row+col の小さな符号)
        // ここでは row のみ識別: l1w[row][col] = row as i8。
        let mut l1w = vec![0i8; SRC_L1_OUT * SRC_L0];
        for row in 0..SRC_L1_OUT {
            for col in 0..SRC_L0 {
                l1w[row * SRC_L0 + col] = (row as i8) - 8; // -8..7
            }
        }
        let mut l1b = vec![0i32; SRC_L1_OUT];
        for (r, v) in l1b.iter_mut().enumerate() {
            *v = r as i32 * 100;
        }
        let src = BucketSrc {
            fc_hash: 0,
            l1b,
            l1w,
            l2b: vec![0; SRC_L2_OUT],
            l2w: vec![0; SRC_L2_OUT * 30],
            l3b: 0,
            l3w: vec![0; SRC_L2_OUT],
        };
        // K = 先頭 1024、 main rows = [1,3,5,7,9,11,13]
        let keep: Vec<usize> = (0..DST_L0).collect();
        let rows = vec![1usize, 3, 5, 7, 9, 11, 13];
        let out = shrink_bucket_with_rows(&src, &keep, &rows, 0).unwrap();

        // L1 bias 部: offset 4 から DST_L1_OUT 個の i32。
        let mut o = 4;
        for (i, &r) in rows.iter().enumerate() {
            let v = i32::from_le_bytes(out[o..o + 4].try_into().unwrap());
            assert_eq!(v, r as i32 * 100, "bias row {i}");
            o += 4;
        }
        // skip bias = src row 15
        let v = i32::from_le_bytes(out[o..o + 4].try_into().unwrap());
        assert_eq!(v, 15 * 100);
        o += 4;
        // L1 weights: DST_L1_OUT rows × pad32(1024)=1024。 各 row の先頭 byte = (src_row - 8)。
        let l1_padded = pad32(DST_L0);
        let dst_rows: Vec<usize> = rows.iter().copied().chain(std::iter::once(15usize)).collect();
        for (i, &src_row) in dst_rows.iter().enumerate() {
            let b = out[o + i * l1_padded] as i8;
            assert_eq!(b, (src_row as i8) - 8, "l1w row {i} (src {src_row})");
        }
    }

    #[test]
    fn test_build_arch_str() {
        let src = "Features=HalfKA_hm(Friend)[73305->1536x2],PSQT=8,DUAL=1,Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-30](SqrClippedReLU[30](AffineTransform[16<-3072](InputSlice[3072(0:3072)]))))),fv_scale=14";
        let dst = build_arch_str(src).unwrap();
        assert!(dst.contains("1024x2"), "{dst}");
        assert!(dst.contains("AffineTransform[1<-64]"));
        assert!(dst.contains("ClippedReLU[64]"));
        assert!(dst.contains("AffineTransform[64<-14]"));
        assert!(dst.contains("SqrClippedReLU[14]"));
        assert!(dst.contains("AffineTransform[8<-2048]"));
        assert!(dst.contains("InputSlice[2048(0:2048)]"));
        assert!(!dst.contains("1536x2"));
        assert!(!dst.contains("[16<-3072]"));
        assert!(!dst.contains("[32<-30]"));
    }
}
