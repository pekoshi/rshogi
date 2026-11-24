//! NetworkLayerStacks - LayerStacksアーキテクチャのNNUEネットワーク
//!
//! HalfKA_hm^ 特徴量 + LayerStacks 構造の NNUE を実装する。
//! nnue-pytorch で学習したファイルを読み込み、評価を行う。
//!
//! ## アーキテクチャ
//!
//! ```text
//! Feature Transformer (HalfKA_hm^): 73,305 → L1 (各視点)
//! 視点結合: 両視点を連結 → L1*2
//! SqrClippedReLU: L1*2 → L1
//! LayerStacks (両玉の相対段ベースの9バケット選択後):
//!   L1: L1 → LS_L1_OUT
//!   SqrReLU + concat: LS_L2_IN (= 2 * (LS_L1_OUT - 1))
//!   L2: LS_L2_IN → 32
//!   Output: 32 → 1 + skip
//! ```
//!
//! ## バケット選択
//!
//! 両玉の相対段（0-8）に基づいて9個のバケットから1つを選択：
//! - 味方玉の段: 0-2 → 0, 3-5 → 3, 6-8 → 6
//! - 相手玉の段: 0-2 → 0, 3-5 → 1, 6-8 → 2
//! - bucket = f_index + e_index (0-8)

use super::accumulator::Aligned;
use super::accumulator_layer_stacks::{AccumulatorLayerStacks, AccumulatorStackLayerStacks};
use super::constants::{FV_SCALE_HALFKA, MAX_ARCH_LEN};
#[cfg(any(
    feature = "layerstacks-1536x16x32",
    feature = "layerstacks-768x16x32",
    feature = "layerstacks-512x16x32"
))]
use super::constants::{LAYER_STACK_16X32_L1_OUT, LAYER_STACK_16X32_L2_IN};
#[cfg(feature = "layerstacks-1536x32x32")]
use super::constants::{LAYER_STACK_32X32_L1_OUT, LAYER_STACK_32X32_L2_IN};
use super::feature_transformer_layer_stacks::FeatureTransformerLayerStacks;
use super::layer_stacks::{LayerStacks, sqr_clipped_relu_transform};
use super::network::{
    LayerStackBucketMode, compute_layer_stack_progress8kpabs_bucket_index, get_fv_scale_override,
    get_layer_stack_bucket_mode, get_layer_stack_progress_kpabs_weights, parse_fv_scale_from_arch,
};
use crate::position::Position;
use crate::types::{Color, Value};
#[cfg(feature = "diagnostics")]
use log::info;
use std::fs::File;
use std::io::{self, BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

#[inline]
fn compute_layer_stacks_bucket_index(pos: &Position, side_to_move: Color) -> usize {
    match get_layer_stack_bucket_mode() {
        LayerStackBucketMode::Progress8KPAbs => {
            let weights = get_layer_stack_progress_kpabs_weights();
            compute_layer_stack_progress8kpabs_bucket_index(pos, side_to_move, weights)
        }
    }
}

/// i16 配列の要素和: dst[i] = a[i] + b[i] (SIMD 最適化)
#[cfg(any(feature = "nnue-threat", feature = "nnue-kzt-ft"))]
#[inline]
fn add_i16_arrays<const L1: usize>(dst: &mut [i16; L1], a: &[i16; L1], b: &[i16; L1]) {
    // AVX2 ループは `L1 / 16` 回で全要素を処理する前提。L1 が 16 の倍数で
    // ない場合は末端要素が取り残されるため、monomorphization 時に失敗させる。
    const {
        assert!(L1.is_multiple_of(16), "L1 must be a multiple of 16 for AVX2 SIMD loops");
    }
    // AVX2: 256bit = 16 x i16, L1/16 iterations
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY:
        // - `a_ptr` / `b_ptr`: 唯一の呼び出し元は `NetworkLayerStacks::evaluate`
        //   であり、そこから渡される `us_t` / `them_t` は
        //   `AccumulatorLayerStacks::get_threat()` が返す
        //   `&[i16; L1]` (親構造体 `AccumulatorLayerStacks` が
        //   `#[repr(C, align(64))]` で 64 バイトアライン）。
        //   → `_mm256_load_si256` の 32 バイトアライン要件を満たす
        // - `dst_ptr`: 呼び出し元の `sum_t: &mut Aligned<[i16; L1]>`
        //   （`#[repr(C, align(64))]`、64 バイトアライン）→ store 要件を満たす
        // - ループ回数 `L1 / 16` は const generics 由来。`add_i16_arrays` は
        //   `AccumulatorLayerStacks<L1>` で `L1 ∈ {512, 768, 1536}`（全て 16 の倍数）
        //   からのみ呼ばれるため末端要素が取り残されない
        unsafe {
            use std::arch::x86_64::*;
            let dst_ptr = dst.as_mut_ptr();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            for i in 0..(L1 / 16) {
                let va = _mm256_load_si256(a_ptr.add(i * 16) as *const __m256i);
                let vb = _mm256_load_si256(b_ptr.add(i * 16) as *const __m256i);
                let result = _mm256_add_epi16(va, vb);
                _mm256_store_si256(dst_ptr.add(i * 16) as *mut __m256i, result);
            }
        }
    }

    // スカラーフォールバック（AVX2 非対応環境のみコンパイル）
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    for i in 0..L1 {
        dst[i] = a[i].wrapping_add(b[i]);
    }
}

/// Dual PSQT section の **新** sub_magic ("P2DUAL2\0" = 8 bytes、 King-Attacker format)
#[cfg(feature = "nnue-dual-psqt")]
const DUAL_PSQT_SUB_MAGIC: [u8; 8] = *b"P2DUAL2\0";

/// Dual PSQT section の **旧** sub_magic ("P2DUAL\0\0" = 8 bytes、 reserve format)
///
/// 新 binary は旧 magic を見つけたら 36 byte (= 4 version + 32 byte data) を skip
/// して dual head 寄与 = 0 で続行。 警告 1 行のみ出力。
#[cfg(feature = "nnue-dual-psqt")]
const DUAL_PSQT_LEGACY_SUB_MAGIC: [u8; 8] = *b"P2DUAL\0\0";

/// 旧 reserve format の section 残りバイト数 (= version 4 + data 32 = 36)。
#[cfg(feature = "nnue-dual-psqt")]
const DUAL_PSQT_LEGACY_TAIL_BYTES: usize = 36;

/// Dual PSQT section format version (現在 1)
#[cfg(feature = "nnue-dual-psqt")]
const DUAL_PSQT_VERSION: u32 = 1;

/// quantised.bin の Dual PSQT section を読み込む (King-Attacker format)。
///
/// format:
/// ```text
/// sub_magic = "P2DUAL2\0" (8B)
/// version   = u32 LE  // 現在 1
/// count     = u32 LE  // 2 perspective × 18,144 = 36,288 entries
/// weights   = i16[2 persp][2 head][8 bucket][81 sq][14 piece]  // LE
/// ```
///
/// 合計 byte 数: `8 + 4 + 4 + 2 × 18,144 × 2 = 72,592 byte`
/// (= magic 8 + version 4 + count 4 + data 72,576)。
///
/// # 旧 reserve format との互換
///
/// 旧 `P2DUAL\0\0` magic (= 44 byte reserve section) を見つけたら 残り 36 byte を
/// skip して `Ok(None)` を返す (= dual head 寄与 = 0)。 警告 1 行を `info string`
/// 形式で stderr に出す。
///
/// 未知 magic は error。
///
/// **重要**: 本関数は bullet 側 `examples/shogi_layerstack.rs --dual-psqt-king-attacker`
/// の書き出し format と **bit-exact 同値** が前提。 変更時は両側更新必須。
#[cfg(feature = "nnue-dual-psqt")]
fn read_dual_psqt<R: Read>(
    reader: &mut R,
) -> io::Result<Option<Box<crate::nnue::dual_psqt::DualPsqtKingAttackerWeights>>> {
    use crate::nnue::dual_psqt::{
        DUAL_DIMENSIONS_PER_PERSPECTIVE, DUAL_PIECE_TYPES, DUAL_PSQT_COUNT, DUAL_SQUARES,
        DualPsqtKingAttackerWeights, NUM_DUAL_PSQT_BUCKETS, NUM_DUAL_PSQT_HEADS,
    };

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;

    // 旧 reserve magic → skip + 警告
    if magic == DUAL_PSQT_LEGACY_SUB_MAGIC {
        eprintln!(
            "info string dual section magic '{}' (legacy reserve) detected, skipping (dual head contribution = 0)",
            String::from_utf8_lossy(&DUAL_PSQT_LEGACY_SUB_MAGIC).trim_end_matches('\0')
        );
        let mut skip_buf = [0u8; DUAL_PSQT_LEGACY_TAIL_BYTES];
        reader.read_exact(&mut skip_buf)?;
        return Ok(None);
    }

    // 新 magic でなければ error (= 未知 format)。
    if magic != DUAL_PSQT_SUB_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Dual PSQT sub_magic unknown: expected {:?} or {:?}, got {:?}",
                DUAL_PSQT_SUB_MAGIC, DUAL_PSQT_LEGACY_SUB_MAGIC, magic
            ),
        ));
    }

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != DUAL_PSQT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("Dual PSQT version {version} unsupported (expected {DUAL_PSQT_VERSION})"),
        ));
    }

    reader.read_exact(&mut buf4)?;
    let count = u32::from_le_bytes(buf4);
    if count != DUAL_PSQT_COUNT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Dual PSQT count mismatch: expected {DUAL_PSQT_COUNT}, got {count}"),
        ));
    }

    // 145,152 byte = 36,288 i16 = 72,576 byte (= per-perspective 18,144 × 2)。
    // box で heap 確保 (= struct field の inline は ~72 KB と大きい)。
    let mut weights: Box<DualPsqtKingAttackerWeights> = Box::new(
        [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS];
            2],
    );

    let mut buf2 = [0u8; 2];
    for persp in 0..2 {
        for head in 0..NUM_DUAL_PSQT_HEADS {
            for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
                for sq in 0..DUAL_SQUARES {
                    for piece in 0..DUAL_PIECE_TYPES {
                        reader.read_exact(&mut buf2)?;
                        weights[persp][head][bucket][sq][piece] = i16::from_le_bytes(buf2);
                    }
                }
            }
        }
    }

    // 値の sanity: dim 数が固定されているのは const で保証 (`DUAL_DIMENSIONS_PER_PERSPECTIVE`)。
    let _ = DUAL_DIMENSIONS_PER_PERSPECTIVE;

    Ok(Some(weights))
}

/// D2 cross feature section を読み込む (= P2DUALX 旧 format / P2DUALZ 新 format 両対応)。
///
/// ## P2DUALZ (新 format、 Path I で導入、 head 0 のみ)
///
/// ```text
/// sub_magic = "P2DUALZ\0"        (8 byte)
/// version   = u32 LE              (= 1)
/// count     = u32 LE              (= 254,016 = 2 persp × 1 head × 127,008)
/// weights   = i16 LE × 254,016   (= 508,032 byte = ~0.5 MB)
/// ```
///
/// disk 順序: `[persp 2][bucket 8][sq 81][own_p 14][target_p 14]`
///
/// ## P2DUALX (旧 format、 head 0/1 両方の重み、 後方互換)
///
/// ```text
/// sub_magic = "P2DUALX\0"        (8 byte)
/// version   = u32 LE              (= 1)
/// count     = u32 LE              (= 508,032)
/// weights   = i16 LE × 508,032   (= 1,016,064 byte)
/// ```
///
/// disk 順序: `[persp 2][head 2][bucket 8][sq 81][own_p 14][target_p 14]`
///
/// 読み込み時に head 0 weight のみ取得 (= head 1 weight は読み飛ばす)、 memory layout
/// `[persp 2][head 1][sq 81][own_p 14][target_p 14][bucket 8]` (= bucket-inner) に並べ替え。
/// memory footprint は ~0.5 MB に半減。
///
/// 既存 ckpt に section が無い場合 (= D2 cross feature 未使用 ckpt):
/// reader EOF を error として扱い、 caller 側で「marker 無し」 として None を返す。
///
/// 未知 magic は error。
///
/// **注**: 本関数は test 用 stand-alone。 production の読み込みは
/// `NetworkLayerStacks::read_from_le` 内に inline 実装されている (= 既存 ckpt
/// 互換のため probe 動作が必要なため)。
#[cfg(feature = "nnue-dual-psqt")]
#[allow(dead_code)]
fn read_dual_cross<R: Read>(
    reader: &mut R,
) -> io::Result<Option<Box<crate::nnue::dual_cross::DualCrossWeights>>> {
    use crate::nnue::dual_cross::{
        DUAL_CROSS_COUNT, DUAL_CROSS_COUNT_Z, DUAL_CROSS_PIECE_TYPES, DUAL_CROSS_SQUARES,
        DUAL_CROSS_SUB_MAGIC, DUAL_CROSS_SUB_MAGIC_Z, DUAL_CROSS_VERSION, DualCrossWeights,
        NUM_DUAL_CROSS_BUCKETS, NUM_DUAL_CROSS_HEADS,
    };

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;

    let is_z = magic == DUAL_CROSS_SUB_MAGIC_Z;
    let is_x = magic == DUAL_CROSS_SUB_MAGIC;
    if !is_z && !is_x {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Dual Cross sub_magic unknown: expected {:?} or {:?}, got {:?}",
                DUAL_CROSS_SUB_MAGIC_Z, DUAL_CROSS_SUB_MAGIC, magic
            ),
        ));
    }

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != DUAL_CROSS_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("Dual Cross version {version} unsupported (expected {DUAL_CROSS_VERSION})"),
        ));
    }

    reader.read_exact(&mut buf4)?;
    let count = u32::from_le_bytes(buf4);
    let expected_count = if is_z {
        DUAL_CROSS_COUNT_Z
    } else {
        DUAL_CROSS_COUNT
    };
    if count != expected_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Dual Cross count mismatch: expected {expected_count}, got {count}"),
        ));
    }

    // ~0.5 MB heap 確保 (= 254,016 i16 = 508,032 byte)。 memory layout = bucket-inner、 head 0 のみ。
    let mut weights: Box<DualCrossWeights> = Box::new(
        [[[[[[0i16; NUM_DUAL_CROSS_BUCKETS]; DUAL_CROSS_PIECE_TYPES]; DUAL_CROSS_PIECE_TYPES];
            DUAL_CROSS_SQUARES]; NUM_DUAL_CROSS_HEADS]; 2],
    );

    // disk = bucket-outer 順で 1 i16 ずつ読み、 memory 上の bucket-inner slot に書き込み。
    // P2DUALX の場合、 head 1 weight 部分は読み飛ばす (= memory footprint 半減のため)。
    let disk_heads = if is_z { 1 } else { 2 };
    let mut buf2 = [0u8; 2];
    for persp in 0..2 {
        for disk_head in 0..disk_heads {
            let store_to_memory = disk_head == 0;
            for bucket in 0..NUM_DUAL_CROSS_BUCKETS {
                for sq in 0..DUAL_CROSS_SQUARES {
                    for own_p in 0..DUAL_CROSS_PIECE_TYPES {
                        for target_p in 0..DUAL_CROSS_PIECE_TYPES {
                            reader.read_exact(&mut buf2)?;
                            if store_to_memory {
                                weights[persp][0][sq][own_p][target_p][bucket] =
                                    i16::from_le_bytes(buf2);
                            }
                            // disk_head == 1 のときは読むだけで捨てる
                        }
                    }
                }
            }
        }
    }

    Ok(Some(weights))
}

#[cfg(all(test, feature = "nnue-dual-psqt"))]
mod dual_psqt_io_tests {
    use super::*;
    use crate::nnue::dual_psqt::{
        DUAL_PIECE_TYPES, DUAL_PSQT_COUNT, DUAL_SQUARES, DualPsqtKingAttackerWeights,
        NUM_DUAL_PSQT_BUCKETS, NUM_DUAL_PSQT_HEADS,
    };
    use std::io::Cursor;

    /// 旧 reserve magic (P2DUAL\0\0) を見つけたら skip して None を返す。
    /// 警告 stderr 出力は test では non-fatal。
    #[test]
    fn read_dual_legacy_magic_skips_to_none() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&DUAL_PSQT_LEGACY_SUB_MAGIC); // 8 byte
        bytes.extend_from_slice(&1u32.to_le_bytes()); // version 4 byte
        for _ in 0..(NUM_DUAL_PSQT_HEADS * NUM_DUAL_PSQT_BUCKETS) {
            bytes.extend_from_slice(&0i16.to_le_bytes()); // 32 byte data
        }
        assert_eq!(bytes.len(), 44, "legacy section should be 44 byte");

        let mut cursor = Cursor::new(bytes);
        let result = read_dual_psqt(&mut cursor).expect("legacy skip should not error");
        assert!(result.is_none(), "legacy magic should yield None (skip)");
    }

    /// 新 magic (P2DUAL2\0) で zero weight section を round-trip 読み込み。
    #[test]
    fn read_dual_new_magic_round_trip_zero() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&DUAL_PSQT_SUB_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&DUAL_PSQT_COUNT.to_le_bytes());
        // zero data: 2 persp × 18,144 i16
        for _ in
            0..(2 * NUM_DUAL_PSQT_HEADS * NUM_DUAL_PSQT_BUCKETS * DUAL_SQUARES * DUAL_PIECE_TYPES)
        {
            bytes.extend_from_slice(&0i16.to_le_bytes());
        }
        assert_eq!(bytes.len(), 72_592, "new section should be 72_592 byte");

        let mut cursor = Cursor::new(bytes);
        let result = read_dual_psqt(&mut cursor).expect("new format read should succeed");
        let weights = result.expect("new format should yield Some");
        assert_eq!(weights[0][0][0][0][0], 0);
        assert_eq!(weights[1][1][7][80][13], 0);
    }

    /// 新 magic で非 zero weight を書き → 読み戻して値が一致。
    #[test]
    fn read_dual_new_magic_round_trip_pattern() {
        let mut weights_expected: Box<DualPsqtKingAttackerWeights> = Box::new(
            [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS];
                NUM_DUAL_PSQT_HEADS]; 2],
        );
        // パターン: persp=0, head=1, bucket=2, sq=3, piece=4 に 1234 を埋める
        weights_expected[0][1][2][3][4] = 1234;
        weights_expected[1][0][5][6][7] = -567;

        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&DUAL_PSQT_SUB_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&DUAL_PSQT_COUNT.to_le_bytes());
        for persp in 0..2 {
            for head in 0..NUM_DUAL_PSQT_HEADS {
                for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
                    for sq in 0..DUAL_SQUARES {
                        for piece in 0..DUAL_PIECE_TYPES {
                            bytes.extend_from_slice(
                                &weights_expected[persp][head][bucket][sq][piece].to_le_bytes(),
                            );
                        }
                    }
                }
            }
        }

        let mut cursor = Cursor::new(bytes);
        let result = read_dual_psqt(&mut cursor).expect("new format read should succeed");
        let weights = result.expect("new format should yield Some");
        assert_eq!(weights[0][1][2][3][4], 1234);
        assert_eq!(weights[1][0][5][6][7], -567);
        assert_eq!(weights[0][0][0][0][0], 0);
    }

    /// 未知 magic は error を返す。
    #[test]
    fn read_dual_unknown_magic_errors() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"UNKNOWN\0");
        bytes.extend_from_slice(&[0u8; 100]);

        let mut cursor = Cursor::new(bytes);
        let result = read_dual_psqt(&mut cursor);
        assert!(result.is_err(), "unknown magic must error");
    }

    /// version mismatch は error を返す。
    #[test]
    fn read_dual_version_mismatch_errors() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&DUAL_PSQT_SUB_MAGIC);
        bytes.extend_from_slice(&999u32.to_le_bytes()); // 未来 version
        bytes.extend_from_slice(&DUAL_PSQT_COUNT.to_le_bytes());

        let mut cursor = Cursor::new(bytes);
        let result = read_dual_psqt(&mut cursor);
        assert!(result.is_err(), "version mismatch must error");
    }

    /// count mismatch は error を返す。
    #[test]
    fn read_dual_count_mismatch_errors() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&DUAL_PSQT_SUB_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&999u32.to_le_bytes()); // 異常 count

        let mut cursor = Cursor::new(bytes);
        let result = read_dual_psqt(&mut cursor);
        assert!(result.is_err(), "count mismatch must error");
    }

    // =========================================================================
    // P2DUALX (D2 Dual Cross) layout roundtrip test
    //
    // 旧 bucket-outer disk format `[persp][head][bucket][sq][own_p][target_p]` の
    // bytes を構築し、 read_dual_cross で並べ替えて bucket-inner memory layout
    // `[persp][head][sq][own_p][target_p][bucket]` に正しく載るか確認する。
    //
    // 仕様: 任意 (persp, head, bucket, sq, own_p, target_p) に対し、 disk 上の値が
    // memory 上 `weights[persp][head][sq][own_p][target_p][bucket]` で取得できること。
    // =========================================================================

    use crate::nnue::dual_cross::{
        DUAL_CROSS_COUNT, DUAL_CROSS_PIECE_TYPES, DUAL_CROSS_SQUARES, DUAL_CROSS_SUB_MAGIC,
        DUAL_CROSS_VERSION, NUM_DUAL_CROSS_BUCKETS,
    };

    /// 旧 P2DUALX format (= head 2): disk 上 bucket-outer 順で任意 pattern を書き込み、
    /// read で head 0 のみ memory bucket-inner 順に取得できることを確認する。
    /// head 1 weight は load 時に捨てられるため検証対象外。
    #[test]
    fn read_dual_cross_legacy_p2dualx_layout_roundtrip_pattern() {
        // pattern: disk 上 (persp, head, bucket, sq, own_p, target_p) → value
        //   value = (persp * 7 + head * 11 + bucket * 13 + sq * 17 + own_p * 19 + target_p * 23) i32
        //           を clip して i16 (-32..32) に
        let pattern_value = |p: usize, h: usize, b: usize, s: usize, o: usize, t: usize| -> i16 {
            let v = ((p * 7 + h * 11 + b * 13 + s * 17 + o * 19 + t * 23) as i32 % 64) - 32;
            v as i16
        };

        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&DUAL_CROSS_SUB_MAGIC);
        bytes.extend_from_slice(&DUAL_CROSS_VERSION.to_le_bytes());
        bytes.extend_from_slice(&DUAL_CROSS_COUNT.to_le_bytes()); // 508,032
        // disk 順: persp → head (2) → bucket → sq → own_p → target_p
        for persp in 0..2usize {
            for head in 0..2usize {
                for bucket in 0..NUM_DUAL_CROSS_BUCKETS {
                    for sq in 0..DUAL_CROSS_SQUARES {
                        for own_p in 0..DUAL_CROSS_PIECE_TYPES {
                            for target_p in 0..DUAL_CROSS_PIECE_TYPES {
                                let v = pattern_value(persp, head, bucket, sq, own_p, target_p);
                                bytes.extend_from_slice(&v.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
        // 8 magic + 4 version + 4 count + 508_032 × 2 = 1_016_076
        assert_eq!(bytes.len(), 8 + 4 + 4 + 508_032 * 2);

        let mut cursor = Cursor::new(bytes);
        let result = read_dual_cross(&mut cursor).expect("P2DUALX read should succeed");
        let weights = result.expect("P2DUALX should yield Some");

        // verify: memory 順 (persp, sq, own_p, target_p, bucket) で head 0 disk value と一致
        for persp in 0..2usize {
            let mut k = 0usize;
            for sq in 0..DUAL_CROSS_SQUARES {
                for own_p in 0..DUAL_CROSS_PIECE_TYPES {
                    for target_p in 0..DUAL_CROSS_PIECE_TYPES {
                        for bucket in 0..NUM_DUAL_CROSS_BUCKETS {
                            // 全 element の 1/53 を sample (= ~4800 element)
                            if k.is_multiple_of(53) {
                                let expected = pattern_value(persp, 0, bucket, sq, own_p, target_p);
                                let actual = weights[persp][0][sq][own_p][target_p][bucket];
                                assert_eq!(
                                    actual, expected,
                                    "P2DUALX head 0 layout mismatch: persp={persp} bucket={bucket} sq={sq} own_p={own_p} target_p={target_p}"
                                );
                            }
                            k += 1;
                        }
                    }
                }
            }
        }
        // 角での確認 (= 端点)
        assert_eq!(weights[0][0][0][0][0][0], pattern_value(0, 0, 0, 0, 0, 0));
        assert_eq!(
            weights[1][0][DUAL_CROSS_SQUARES - 1][DUAL_CROSS_PIECE_TYPES - 1]
                [DUAL_CROSS_PIECE_TYPES - 1][NUM_DUAL_CROSS_BUCKETS - 1],
            pattern_value(
                1,
                0,
                NUM_DUAL_CROSS_BUCKETS - 1,
                DUAL_CROSS_SQUARES - 1,
                DUAL_CROSS_PIECE_TYPES - 1,
                DUAL_CROSS_PIECE_TYPES - 1,
            )
        );
    }

    /// 新 P2DUALZ format (= head 1): disk 上 bucket-outer 順で任意 pattern を書き込み、
    /// read で memory bucket-inner 順に並べ替えられた値を確認する。
    #[test]
    fn read_dual_cross_new_p2dualz_layout_roundtrip_pattern() {
        use crate::nnue::dual_cross::{DUAL_CROSS_COUNT_Z, DUAL_CROSS_SUB_MAGIC_Z};

        let pattern_value = |p: usize, b: usize, s: usize, o: usize, t: usize| -> i16 {
            let v = ((p * 7 + b * 13 + s * 17 + o * 19 + t * 23) as i32 % 64) - 32;
            v as i16
        };

        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&DUAL_CROSS_SUB_MAGIC_Z);
        bytes.extend_from_slice(&DUAL_CROSS_VERSION.to_le_bytes());
        bytes.extend_from_slice(&DUAL_CROSS_COUNT_Z.to_le_bytes()); // 254,016
        // disk 順 (P2DUALZ): persp → bucket → sq → own_p → target_p (head 次元なし)
        for persp in 0..2usize {
            for bucket in 0..NUM_DUAL_CROSS_BUCKETS {
                for sq in 0..DUAL_CROSS_SQUARES {
                    for own_p in 0..DUAL_CROSS_PIECE_TYPES {
                        for target_p in 0..DUAL_CROSS_PIECE_TYPES {
                            let v = pattern_value(persp, bucket, sq, own_p, target_p);
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                }
            }
        }
        // 8 + 4 + 4 + 254,016 × 2 = 508,048
        assert_eq!(bytes.len(), 8 + 4 + 4 + 254_016 * 2);

        let mut cursor = Cursor::new(bytes);
        let result = read_dual_cross(&mut cursor).expect("P2DUALZ read should succeed");
        let weights = result.expect("P2DUALZ should yield Some");

        for persp in 0..2usize {
            let mut k = 0usize;
            for sq in 0..DUAL_CROSS_SQUARES {
                for own_p in 0..DUAL_CROSS_PIECE_TYPES {
                    for target_p in 0..DUAL_CROSS_PIECE_TYPES {
                        for bucket in 0..NUM_DUAL_CROSS_BUCKETS {
                            if k.is_multiple_of(53) {
                                let expected = pattern_value(persp, bucket, sq, own_p, target_p);
                                let actual = weights[persp][0][sq][own_p][target_p][bucket];
                                assert_eq!(
                                    actual, expected,
                                    "P2DUALZ layout mismatch: persp={persp} bucket={bucket} sq={sq} own_p={own_p} target_p={target_p}"
                                );
                            }
                            k += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(weights[0][0][0][0][0][0], pattern_value(0, 0, 0, 0, 0));
    }
}

/// Pin-aware section の sub_magic ("P2PIN\0\0\0" = 8 bytes)
#[cfg(feature = "nnue-pin-aware")]
const PIN_SUB_MAGIC: [u8; 8] = *b"P2PIN\0\0\0";

/// Pin-aware section format version (現在 1)
#[cfg(feature = "nnue-pin-aware")]
const PIN_VERSION: u32 = 1;

/// quantised.bin の Pin-aware section を読み込む。
///
/// format:
/// ```text
/// sub_magic = "P2PIN\0\0\0" (8B)
/// version   = u32 LE  // 現在 1
/// for bucket in 0..NUM_PIN_BUCKETS {
///   for persp in 0..2 {
///     i16[PIN_DIMENSIONS_PER_PERSPECTIVE]  // LE
///   }
/// }
/// ```
///
/// ファイル全体サイズ: 8 + 4 + 8 × 2 × 2268 × 2 = 12 + 72,576 = **72,588 byte**。
///
/// **重要**: 本関数は bullet 側 `examples/shogi_layerstack.rs --pin-aware` の
/// 書き出し format と **bit-exact 同値** が前提。 変更時は両側更新必須。
#[cfg(feature = "nnue-pin-aware")]
fn read_pin_weights<R: Read>(
    reader: &mut R,
) -> io::Result<Box<crate::nnue::pin_features::PinWeights>> {
    use crate::nnue::pin_features::{NUM_PIN_BUCKETS, PIN_DIMENSIONS_PER_PERSPECTIVE, PinWeights};

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if magic != PIN_SUB_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Pin sub_magic mismatch: expected {PIN_SUB_MAGIC:?}, got {magic:?}"),
        ));
    }

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != PIN_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("Pin version {version} unsupported (expected {PIN_VERSION})"),
        ));
    }

    // PinWeights は ~70 KB なので Box で確保。
    // stack 上に配列 literal を作って Box::new(...) すると一旦 stack に乗るが、
    // 70 KB は test 含めて許容範囲 (default thread stack 2-8 MB)。
    let mut weights: Box<PinWeights> =
        Box::new([[[0i16; PIN_DIMENSIONS_PER_PERSPECTIVE]; 2]; NUM_PIN_BUCKETS]);

    let mut buf2 = [0u8; 2];
    for bucket in 0..NUM_PIN_BUCKETS {
        for persp in 0..2 {
            for dim in 0..PIN_DIMENSIONS_PER_PERSPECTIVE {
                reader.read_exact(&mut buf2)?;
                weights[bucket][persp][dim] = i16::from_le_bytes(buf2);
            }
        }
    }

    Ok(weights)
}

/// 各 head の eval 寄与を分解した診断用 breakdown (= 2026-06-09 追加)。
///
/// `evaluate_with_breakdown` で取得、 bucket 境界 jump 計測等の診断用途。
/// hot path での使用は想定外、 production binary では使わない。
///
/// `total_cp` は `evaluate_with_bucket` と同じ最終 cp 値。 内訳:
/// - `raw_score` = L3 main path output (= L0/L1/L2/L3 forward の生 scalar、 fv_scale 前)
/// - 各 head value = fv_scale 適用前 cp (= 各 head 内部の OUTPUT_SCALE 適用済)
/// - `total_cp` = `(raw_score + sum(head values)) / fv_scale` (= bit-exact `evaluate_with_bucket` 一致)
///
/// 未使用 head (= 訓練 args で OFF、 weight `None`) は 0。
///
/// arv4 default (= main path + 5 head): raw_score / psqt_value / dual_cross_value /
/// pairshape_value / activity_risk_value / king_escape_value のみ非零、 残 3 (=
/// pin_value / dual_psqt_value / kingring_value) は 0。
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadBreakdown {
    /// 推論時 bucket index (= 0..=NUM_BUCKETS-1)
    pub bucket_index: usize,
    /// 推論時 fv_scale (= USI option override 込み)
    pub fv_scale: i32,
    /// L3 main path output (= L0/L1/L2/L3 forward、 fv_scale 適用前)
    pub raw_score: i32,
    /// PSQT scalar head 寄与 (= fv_scale 適用前)
    pub psqt_value: i32,
    /// Pin-aware head 寄与 (= arv4 では 0)
    pub pin_value: i32,
    /// DualPSQT (King-Attacker) head 寄与 (= arv4 では 0)
    pub dual_psqt_value: i32,
    /// D2 cross threat (= dual_w_threat) 寄与 (= fv_scale 適用前、 OUTPUT_SCALE 適用済)
    pub dual_cross_value: i32,
    /// KingRing scalar head 寄与 (= arv4 では 0)
    pub kingring_value: i32,
    /// PairShape scalar head 寄与 (= fv_scale 適用前、 OUTPUT_SCALE 適用済)
    pub pairshape_value: i32,
    /// ActivityRisk v3 寄与 (= fv_scale 適用前、 base + tactical 合算)
    pub activity_risk_value: i32,
    /// KED 寄与 (= fv_scale 適用前、 OUTPUT_SCALE 適用済)
    pub king_escape_value: i32,
    /// ARH (Attack Relay Head) 寄与 (= 2026-06-09、 fv_scale 適用前、 OUTPUT_SCALE 適用済)。
    /// arv4 では 0 (= ARH section 不在 / weight zero)、 arv5 (= ARH 訓練済) で非零。
    pub attack_relay_value: i32,
    /// ARH v2 (= A5 仕様、 2026-06-10) 寄与 (= fv_scale 適用前、 OUTPUT_SCALE 適用済)。
    /// v1 と並列 section、 P2AREL2 magic 不在 ckpt では 0。
    pub attack_relay_v2_value: i32,
    /// King Danger (KDGR) 寄与 (= 2026-06-15、 fv_scale 適用前、 OUTPUT_SCALE 適用済)。
    /// P2KDGR section 不在 ckpt では 0。
    pub king_danger_value: i32,
    /// 解析的 king_danger 補正 (= 2026-06-16、 fv_scale 適用前 raw 単位)。
    /// 訓練重み非依存、 `KING_DANGER_ANALYTIC_COEF` (= default 0) で off。
    pub king_danger_analytic_value: i32,
    /// 解析的 king_danger slider 補正 (= 2026-06-29、 fv_scale 適用前 raw 単位)。
    /// 玉 3x3 に届く敵 slider の block-aware 利き危険度。 訓練重み非依存、
    /// `KING_DANGER_SLIDER_ANALYTIC_COEF` (= default 0) で off。
    pub king_danger_slider_analytic_value: i32,
    /// 解析的 king_danger slider **守備のみ** 補正 (= 2026-06-30、 fv_scale 適用前 raw 単位)。
    /// 自玉 3x3 に届く敵 slider の危険度のみ (= 非対称、 攻めでブーストしない)。
    /// 訓練重み非依存、 `KING_DANGER_SLIDER_DEF_ANALYTIC_COEF` (= default 0) で off。
    pub king_danger_slider_def_analytic_value: i32,
    /// Attack Development (ADEV) 寄与 (= 2026-06-16、 fv_scale 適用前、 OUTPUT_SCALE 適用済)。
    /// P2ADEV section 不在 ckpt では 0。
    pub attack_dev_value: i32,
    /// King Mobility (KMOB) 寄与 (= 2026-06-16、 fv_scale 適用前、 OUTPUT_SCALE 適用済)。
    /// 自玉の安全な逃げ場マス数の class 補正。 P2KMOB section 不在 ckpt では 0。
    pub king_mobility_value: i32,
    /// Drop Pressure (DROP) 寄与 (= 2026-06-16、 fv_scale 適用前、 OUTPUT_SCALE 適用済)。
    /// 相手持駒を弾とした自玉近傍投下脅威の class 補正。 P2DROP section 不在 ckpt では 0。
    pub drop_pressure_value: i32,
    /// Slider X-ray (SXRY) 寄与 (= 2026-06-28、 fv_scale 適用前、 OUTPUT_SCALE 適用済)。
    /// 敵 slider の潜在照準 (= between 1 枚) の class 補正。 P2SXRY section 不在 ckpt では 0。
    pub slider_xray_value: i32,
    /// King Zone Threat / 持駒打ち込み (KZT_HAND) 寄与 (= 2026-06-30、 fv_scale 適用前、
    /// OUTPUT_SCALE 適用済)。 敵玉 5x5 zone への持駒打ち脅威の sparse head。
    /// P2KZT section 不在 ckpt では 0。
    pub kzt_hand_value: i32,
    /// ODH (Optimism Damp Head) 寄与 (= 2026-07-02、 fv_scale 適用前 raw 単位)。
    /// 優勢側 eval を縮める減衰 head のため **負の寄与** (= `-odh` 減算量) を格納。
    /// `ODH_OUTPUT_SCALE` (= default 0) で off (= 0)。
    pub odh_value: i32,
    /// Eval Bias Correction (EBC) 寄与 (= 2026-07-03、 fv_scale 適用前 raw 単位)。
    /// class 別 bias 補正の符号込み加算量 (= 優勢下げなら負 / 劣勢上げなら正)。
    /// `EVAL_BIAS_CORR_SCALE` (= default 0) で off (= 0)。
    pub ebc_value: i32,
    /// 最終 cp 値 (= fv_scale 適用後、 `evaluate_with_bucket` 戻り値と一致)
    pub total_cp: i32,
}

impl HeadBreakdown {
    /// 全 head + raw_score の fv_scale 適用前 sum を返す。
    ///
    /// `total_cp == raw_total_pre_scale() / fv_scale` の関係が成立する (= bit-exact、
    /// 中間 saturating_add の order も `evaluate_with_bucket` と同一)。
    #[inline]
    pub fn raw_total_pre_scale(&self) -> i32 {
        self.raw_score
            .saturating_add(self.psqt_value)
            .saturating_add(self.pin_value)
            .saturating_add(self.dual_psqt_value)
            .saturating_add(self.dual_cross_value)
            .saturating_add(self.kingring_value)
            .saturating_add(self.pairshape_value)
            .saturating_add(self.activity_risk_value)
            .saturating_add(self.king_escape_value)
            .saturating_add(self.attack_relay_value)
            .saturating_add(self.attack_relay_v2_value)
            .saturating_add(self.king_danger_value)
            .saturating_add(self.king_danger_analytic_value)
            .saturating_add(self.king_danger_slider_analytic_value)
            .saturating_add(self.king_danger_slider_def_analytic_value)
            .saturating_add(self.attack_dev_value)
            .saturating_add(self.king_mobility_value)
            .saturating_add(self.drop_pressure_value)
            .saturating_add(self.slider_xray_value)
            .saturating_add(self.kzt_hand_value)
            .saturating_add(self.odh_value)
            .saturating_add(self.ebc_value)
    }
}

/// LayerStacksアーキテクチャのNNUEネットワーク
///
/// HalfKA_hm^ 特徴量（73,305次元）+ L1次元 Feature Transformer + 9バケット LayerStacks
pub struct NetworkLayerStacks<
    const L1: usize,
    const LS_L1_OUT: usize,
    const LS_L2_IN: usize,
    const LS_L2_PADDED_INPUT: usize,
    const LS_L3: usize,
> {
    /// Feature Transformer (73,305 → L1)
    pub feature_transformer: FeatureTransformerLayerStacks<L1>,
    /// LayerStacks (9バケット)
    pub layer_stacks: LayerStacks<L1, LS_L1_OUT, LS_L2_IN, LS_L2_PADDED_INPUT, LS_L3>,
    /// 評価値スケーリング係数（アーキテクチャ文字列から取得、USIオプションでオーバーライド可）
    pub fv_scale: i32,
    /// Pin-aware 補助 weight (2026-05-24 追加)。
    /// arch_str に "PIN=" marker がある ckpt 読み込み時のみ `Some(_)` になる。
    /// 詳細は [`crate::nnue::pin_features`] module 参照。
    ///
    /// **Box 配置**: PinWeights は ~70 KB なので heap 配置 (struct field
    /// inline は struct 全体の size を膨らませる)。
    #[cfg(feature = "nnue-pin-aware")]
    pub pin_weights: Option<Box<crate::nnue::pin_features::PinWeights>>,
    /// Dual PSQT 補助 weight (King-Attacker 型、 2026-05-26 更新)。
    /// arch_str に "DUAL=1" marker がある ckpt 読み込み時のみ `Some(_)` になる。
    /// 旧 reserve format (= P2DUAL\0\0 magic) は skip されて `None` のまま。
    ///
    /// **Box 配置**: weight は ~72 KB あり、 struct inline は struct size を膨らませる。
    /// 詳細は [`crate::nnue::dual_psqt`] module 参照。
    #[cfg(feature = "nnue-dual-psqt")]
    pub dual_psqt_weights: Option<Box<crate::nnue::dual_psqt::DualPsqtKingAttackerWeights>>,
    /// D2 Dual Cross 補助 weight (= piece_threat_relation + piece_defended_by)。
    /// 同 ckpt の P2DUAL2 section 直後に P2DUALX section があれば `Some(_)`。
    /// 古い ckpt (= D2 未対応) では `None` (= eval 寄与 0)。
    ///
    /// **Box 配置**: weight は ~1 MB あり、 struct inline NG。
    /// 詳細は [`crate::nnue::dual_cross`] module 参照。
    #[cfg(feature = "nnue-dual-psqt")]
    pub dual_cross_weights: Option<Box<crate::nnue::dual_cross::DualCrossWeights>>,
    /// KingRing Threat PSQT scalar head 用 weight (= arm 11、 2026-05-28 追加)。
    /// dual_cross 読み込みの後で section probe (= magic = "P2KRING\0") に成功したら
    /// `Some(_)`。 旧 ckpt (= kingring 未対応) では `None` (= eval 寄与 0)。
    ///
    /// **disk shape (= bucket-outer i16)** を保持する。 hot path では使わず、 ckpt 互換
    /// 確認 / 旧 API / fallback 用に残す。 hot path は [`kingring_runtime_weights`] を使う。
    ///
    /// **Box 配置**: weight は ~40 KB と軽量だが、 内部固定 array が大きいため heap。
    /// 詳細は [`crate::nnue::kingring`] module 参照。
    #[cfg(feature = "nnue-kingring-psqt")]
    pub kingring_weights: Option<Box<crate::nnue::kingring::KingRingWeights>>,
    /// KingRing **runtime weight** (= bucket-inner i32 row、 hot path 用、 2026-05-29 追加)。
    ///
    /// disk shape を ckpt read 後に転置構築 (= 1 回だけ)。 hot path で `row[bucket]` の
    /// 8 連続 read が可能、 5 次元 index decode と i16→i32 cast を hot path から削除する。
    /// メモリ消費は ~80 KB / persp × 2 = ~160 KB (= disk shape 比 +120 KB)。
    ///
    /// `kingring_weights` が `Some(_)` の時必ず `Some(_)`、 そうでなければ `None`。
    #[cfg(feature = "nnue-kingring-psqt")]
    pub kingring_runtime_weights: Option<Box<crate::nnue::kingring::KingRingRuntimeWeights>>,
    /// PairShape PSQT scalar head 用 weight (= arm 12、 2026-05-29 追加)。
    /// 既存 section の後ろに probe (= magic = "P2PAIR\0\0") で読む。 旧 ckpt (= pairshape
    /// 未対応) では `None` (= eval 寄与 0)。 disk shape は ckpt 互換確認用に残し、
    /// hot path は [`pairshape_runtime_weights`] を使う。
    #[cfg(feature = "nnue-pairshape-psqt")]
    pub pairshape_weights: Option<Box<crate::nnue::pairshape::PairShapeWeights>>,
    /// PairShape **runtime weight** (= bucket-inner i32 row、 hot path 用、 2026-05-29)。
    ///
    /// disk shape を ckpt read 後に転置構築 (= 1 回だけ)。 hot path で `row[bucket]` の
    /// 8 連続 read が可能。 メモリ消費 ~12 KB。
    #[cfg(feature = "nnue-pairshape-psqt")]
    pub pairshape_runtime_weights: Option<Box<crate::nnue::pairshape::PairShapeRuntimeWeights>>,
    /// Activity-Risk Interaction Head 用 weight (= 2026-06-06 追加、 中盤楽観バイアス補正)。
    ///
    /// 既 section (= pairshape まで) の後ろに probe (= magic = `P2ACTR\0\0`) で読む。
    /// 旧 ckpt (= activity_risk 未対応) では `None` (= eval 寄与 0)。
    /// disk layout = `[bucket 8][persp 2][activity 8][pressure 8][shield 4]` row-major、
    /// 全 4,096 i16。 hot path では `lookup_perspective` で 1 entry/persp lookup する。
    #[cfg(feature = "nnue-activity-risk-psqt")]
    pub activity_risk_weights: Option<Box<crate::nnue::activity_risk::ActivityRiskWeightsV3>>,
    /// King Escape Direction (KED) PSQT scalar head 用 weight (= 2026-06-09 追加、 入玉対応)。
    ///
    /// 既 section (= activity_risk まで) の後ろに probe (= magic = `P2KESC\0\0`) で読む。
    /// 旧 ckpt (= KED 未対応) では `None` (= eval 寄与 0)。
    /// disk layout = `[bucket 8][persp 2][class 4]` row-major、 全 64 i16。
    #[cfg(feature = "nnue-king-escape-psqt")]
    pub king_escape_weights: Option<Box<crate::nnue::king_escape::KingEscapeWeights>>,
    /// KED **runtime weight** (= bucket-inner i32 row、 hot path 用、 2026-06-09)。
    ///
    /// disk shape を ckpt read 後に転置構築 (= 1 回だけ)。 メモリ消費 ~256 byte。
    #[cfg(feature = "nnue-king-escape-psqt")]
    pub king_escape_runtime_weights:
        Option<Box<crate::nnue::king_escape::KingEscapeRuntimeWeights>>,
    /// Attack Relay Head (ARH) scalar head 用 weight (= 2026-06-09 追加)。
    ///
    /// 既 section (= KED まで) の後ろに probe (= magic = `P2AREL\0\0`) で読む。
    /// 旧 ckpt (= ARH 未対応) では `None` (= eval 寄与 0)。
    /// disk layout = `[bucket 8][persp 2][contact 4][reserve 4][relay 4]` row-major、 全 1,024 i16。
    #[cfg(feature = "nnue-attack-relay")]
    pub attack_relay_weights: Option<Box<crate::nnue::attack_relay::ArhWeights>>,
    /// ARH **runtime weight** (= bucket-inner i32 row、 hot path 用、 2026-06-09)。
    ///
    /// disk shape を ckpt read 後に転置構築 (= 1 回だけ)。 メモリ消費 ~4 KB。
    #[cfg(feature = "nnue-attack-relay")]
    pub attack_relay_runtime_weights: Option<Box<crate::nnue::attack_relay::ArhRuntimeWeights>>,
    /// ARH v2 (= A5 仕様、 2026-06-10) weight。 既 section の後ろに probe
    /// (= magic = `P2AREL2\0`) で読む。 旧 ckpt (= v2 未対応) では `None`。
    #[cfg(feature = "nnue-attack-relay-v2")]
    pub attack_relay_v2_weights: Option<Box<crate::nnue::attack_relay_v2::ArhV2Weights>>,
    /// ARH v2 runtime weight (= bucket-inner i32 row、 hot path 用)。
    #[cfg(feature = "nnue-attack-relay-v2")]
    pub attack_relay_v2_runtime_weights:
        Option<Box<crate::nnue::attack_relay_v2::ArhV2RuntimeWeights>>,
    /// King Danger (KDGR) PSQT scalar head 用 weight (= 2026-06-15 追加、 自玉危険補正)。
    ///
    /// 既 section (= ... / KESC / AREL / ARS1) の後ろに probe (= magic = `P2KDGR\0\0`) で
    /// 読む。 旧 ckpt (= KDGR 未対応) では `None` (= eval 寄与 0)。
    /// disk layout = `[bucket 8][persp 2][class 8]` row-major、 全 128 i16。
    #[cfg(feature = "nnue-king-danger-psqt")]
    pub king_danger_weights: Option<Box<crate::nnue::king_danger::KingDangerWeights>>,
    /// KDGR **runtime weight** (= bucket-inner i32 row、 hot path 用、 2026-06-15)。
    ///
    /// disk shape を ckpt read 後に転置構築 (= 1 回だけ)。 メモリ消費 ~512 byte。
    #[cfg(feature = "nnue-king-danger-psqt")]
    pub king_danger_runtime_weights:
        Option<Box<crate::nnue::king_danger::KingDangerRuntimeWeights>>,
    /// Attack Development (ADEV) PSQT scalar head 用 weight (= 2026-06-16 追加、
    /// 相手の発展中の攻め過小評価を補正)。
    ///
    /// 既 section (= ... / P2KDGR) の後ろに probe (= magic = `P2ADEV\0\0`) で読む。
    /// 旧 ckpt (= ADEV 未対応) では `None` (= eval 寄与 0)。
    /// disk layout = `[bucket 8][persp 2][class 8]` row-major、 全 128 i16。
    #[cfg(feature = "nnue-attack-dev-psqt")]
    pub attack_dev_weights: Option<Box<crate::nnue::attack_dev::AttackDevWeights>>,
    /// ADEV **runtime weight** (= bucket-inner i32 row、 hot path 用、 2026-06-16)。
    ///
    /// disk shape を ckpt read 後に転置構築 (= 1 回だけ)。 メモリ消費 ~512 byte。
    #[cfg(feature = "nnue-attack-dev-psqt")]
    pub attack_dev_runtime_weights: Option<Box<crate::nnue::attack_dev::AttackDevRuntimeWeights>>,
    /// Slider X-ray (SXRY) PSQT scalar head 用 weight (= 2026-06-28 追加、 敵 slider の
    /// 潜在照準 (= between 1 枚) を補正)。
    ///
    /// 既 section (= ... / P2KDGR / P2ADEV / ...) の **物理末尾** に probe (= magic = `P2SXRY\0\0`)
    /// で読む。 旧 ckpt (= SXRY 未対応) では `None` (= eval 寄与 0)。
    /// disk layout = `[bucket 8][persp 2][class 8]` row-major、 全 128 i16。
    #[cfg(feature = "nnue-slider-xray-psqt")]
    pub slider_xray_weights: Option<Box<crate::nnue::slider_xray::SliderXrayWeights>>,
    /// SXRY **runtime weight** (= bucket-inner i32 row、 hot path 用、 2026-06-28)。
    ///
    /// disk shape を ckpt read 後に転置構築 (= 1 回だけ)。 メモリ消費 ~512 byte。
    #[cfg(feature = "nnue-slider-xray-psqt")]
    pub slider_xray_runtime_weights:
        Option<Box<crate::nnue::slider_xray::SliderXrayRuntimeWeights>>,
    /// King Zone Threat / 持駒打ち込み (KZT_HAND) sparse head 用 weight (= 2026-06-30 追加、
    /// 2026-07-01 factored v1 化)。
    ///
    /// 既存 section (= ... / P2SXRY) の **物理末尾** に probe (= magic = `P2KZT\0\0\0`) で
    /// 読む。 旧 ckpt (= KZT 未対応) では `None` (= eval 寄与 0)。 section version 2 =
    /// factored 5 テーブル (core/region/bucket/persp/bias、 計 10,336 i16)。 dense version 1 は
    /// 廃止 (= load 時 version 不一致で reject)。
    #[cfg(feature = "nnue-kzt-psqt")]
    pub kzt_hand_weights: Option<Box<crate::nnue::kzt::KztFactoredWeights>>,
    /// King Mobility (KMOB) PSQT scalar head 用 weight (= 2026-06-16 追加、
    /// 自玉の安全な逃げ場マス数 = 守備系特徴、 ARS1 / attack_dev と直交)。
    ///
    /// 既 section (= ... / P2KDGR / P2ADEV) の後ろに probe (= magic = `P2KMOB\0\0`) で読む。
    /// 旧 ckpt (= KMOB 未対応) では `None` (= eval 寄与 0)。
    /// disk layout = `[bucket 8][persp 2][class 9]` row-major、 全 144 i16。
    #[cfg(feature = "nnue-king-mobility-psqt")]
    pub king_mobility_weights: Option<Box<crate::nnue::king_mobility::KingMobilityWeights>>,
    /// KMOB **runtime weight** (= bucket-inner i32 row、 hot path 用、 2026-06-16)。
    ///
    /// disk shape を ckpt read 後に転置構築 (= 1 回だけ)。 メモリ消費 ~576 byte。
    #[cfg(feature = "nnue-king-mobility-psqt")]
    pub king_mobility_runtime_weights:
        Option<Box<crate::nnue::king_mobility::KingMobilityRuntimeWeights>>,
    /// Drop Pressure (DROP) PSQT scalar head 用 weight (= 2026-06-16 追加、
    /// 相手持駒を弾とした自玉近傍投下脅威の過小評価を補正)。
    ///
    /// 既 section (= ... / P2KDGR / P2ADEV / P2KMOB) の後ろに probe
    /// (= magic = `P2DROP\0\0`) で読む。 旧 ckpt (= DROP 未対応) では `None`
    /// (= eval 寄与 0)。 disk layout = `[bucket 8][persp 2][class 8]` row-major、
    /// 全 128 i16。
    #[cfg(feature = "nnue-drop-pressure-psqt")]
    pub drop_pressure_weights: Option<Box<crate::nnue::drop_pressure::DropPressureWeights>>,
    /// DROP **runtime weight** (= bucket-inner i32 row、 hot path 用、 2026-06-16)。
    ///
    /// disk shape を ckpt read 後に転置構築 (= 1 回だけ)。 メモリ消費 ~512 byte。
    #[cfg(feature = "nnue-drop-pressure-psqt")]
    pub drop_pressure_runtime_weights:
        Option<Box<crate::nnue::drop_pressure::DropPressureRuntimeWeights>>,
    /// Full Threat scalar residual weights。section不在時は評価寄与0。
    #[cfg(feature = "nnue-scalar-threat")]
    pub scalar_threat_weights: Option<crate::nnue::scalar_threat::ScalarThreatWeights>,
    /// Dual Cross Geometryのゼロ平均残差weight。
    #[cfg(feature = "nnue-dual-cross-geometry")]
    pub dual_cross_geometry_weights:
        Option<crate::nnue::dual_cross_geometry::DualCrossGeometryWeights>,
}

impl<
    const L1: usize,
    const LS_L1_OUT: usize,
    const LS_L2_IN: usize,
    const LS_L2_PADDED_INPUT: usize,
    const LS_L3: usize,
> NetworkLayerStacks<L1, LS_L1_OUT, LS_L2_IN, LS_L2_PADDED_INPUT, LS_L3>
{
    /// ファイルから読み込み
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read(&mut reader)
    }

    /// リーダーから読み込み（PSQT は arch_str から自動検出）
    pub fn read<R: Read + Seek>(reader: &mut R) -> io::Result<Self> {
        Self::read_with_options(reader, None)
    }

    /// リーダーから読み込み（PSQT オーバーライドオプション付き）
    ///
    /// `psqt_override`:
    /// - `None`: arch_str から自動検出（デフォルト）
    /// - `Some(true)`: arch_str を無視して PSQT ブロックを読む
    /// - `Some(false)`: arch_str を無視して PSQT ブロックを読まない
    pub fn read_with_options<R: Read + Seek>(
        reader: &mut R,
        psqt_override: Option<bool>,
    ) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];

        // version（呼び出し元で検証済み）
        reader.read_exact(&mut buf4)?;

        // 構造ハッシュ
        reader.read_exact(&mut buf4)?;

        // アーキテクチャ文字列を読み込み
        reader.read_exact(&mut buf4)?;
        let arch_len = u32::from_le_bytes(buf4) as usize;
        if arch_len == 0 || arch_len > MAX_ARCH_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid arch string length: {arch_len} (max: {MAX_ARCH_LEN})"),
            ));
        }
        let mut arch = vec![0u8; arch_len];
        reader.read_exact(&mut arch)?;

        // アーキテクチャ文字列を解析
        let arch_str = String::from_utf8_lossy(&arch);

        // FV_SCALE 検出
        let fv_scale = parse_fv_scale_from_arch(&arch_str).unwrap_or(FV_SCALE_HALFKA);

        // Factorizedモデルの検出
        if arch_str.contains("Factorizer") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Unsupported model format: factorized (non-coalesced) model detected.\n\
                     This engine only supports coalesced models.\n\n\
                     To fix: Re-export the model using nnue-pytorch serialize.py:\n\
                       python serialize.py model.ckpt output.nnue\n\n\
                     Architecture string: {arch_str}"
                ),
            ));
        }

        // Feature transformer hash を読み飛ばす
        reader.read_exact(&mut buf4)?;
        let _ft_hash = u32::from_le_bytes(buf4);

        // Feature Transformer を読み込み（圧縮形式を自動検出）
        // nnue-psqt/nnue-threat feature 有効時は
        // read_psqt/read_threat_weights で変更するため mut
        #[cfg(any(
            feature = "nnue-psqt",
            feature = "nnue-threat",
            feature = "nnue-kzt-ft"
        ))]
        let mut feature_transformer = FeatureTransformerLayerStacks::read_leb128(reader)?;
        #[cfg(not(any(
            feature = "nnue-psqt",
            feature = "nnue-threat",
            feature = "nnue-kzt-ft"
        )))]
        let feature_transformer = FeatureTransformerLayerStacks::read_leb128(reader)?;

        // PSQT 読み込み:
        // - psqt_override == Some(true): USI オプションで PSQT 強制 ON（arch_str を無視）
        // - psqt_override == Some(false): USI オプションで PSQT 強制 OFF（arch_str を無視）
        // - psqt_override == None: arch_str から自動検出
        #[cfg(feature = "nnue-psqt")]
        {
            let has_psqt = psqt_override.unwrap_or_else(|| arch_str.contains("PSQT="));
            if has_psqt {
                feature_transformer.read_psqt(reader)?;
            }
        }
        #[cfg(not(feature = "nnue-psqt"))]
        if psqt_override.unwrap_or_else(|| arch_str.contains("PSQT=")) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "PSQT model requires nnue-psqt feature",
            ));
        }

        // Threat 読み込み（arch_str に "Threat=" があれば）
        #[cfg(feature = "nnue-threat")]
        {
            let has_threat = arch_str.contains("Threat=");
            if has_threat {
                // ThreatProfile= が arch_str にあれば profile id を読み込み検証
                // なければ旧モデル (v91 以前): profile 0 と見なす
                let has_profile_field = arch_str.contains("ThreatProfile=");
                if has_profile_field {
                    reader.read_exact(&mut buf4)?;
                    let model_profile_id = u32::from_le_bytes(buf4);
                    let engine_profile_id = super::threat_exclusion::THREAT_PROFILE_ID;
                    if model_profile_id != engine_profile_id {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "Threat profile mismatch: model={model_profile_id}, engine={engine_profile_id}"
                            ),
                        ));
                    }
                } else {
                    // 旧モデル: profile id フィールドなし → profile 0 と見なす
                    let engine_profile_id = super::threat_exclusion::THREAT_PROFILE_ID;
                    if engine_profile_id != 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "Old model (no ThreatProfile) requires engine profile 0, \
                                 but engine has profile {engine_profile_id}. \
                                 Use a model trained with the matching exclusion profile."
                            ),
                        ));
                    }
                }
                feature_transformer.read_threat_weights(reader)?;
            }
        }
        #[cfg(not(feature = "nnue-threat"))]
        if arch_str.contains("Threat=") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Threat model requires nnue-threat feature",
            ));
        }

        // LayerStacks を読み込み（FC層は常に非圧縮）
        let layer_stacks = LayerStacks::read(reader)?;

        // LMAS は 2026-05-25 廃止 (= 物理削除)。 旧 ckpt の "LMAS=" marker は
        // サポートしない。 conversion tool で先に剥がして再 export 必要。
        if arch_str.contains("LMAS=") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "LMAS ckpt format is no longer supported (2026-05-25 廃止). \
                 Re-export the model without LMAS section.",
            ));
        }

        // Pin-aware 読み込み（arch_str に "PIN=" があれば LayerStack の後に Pin section）
        // section format:
        //   sub_magic = "P2PIN\0\0\0" (8B)
        //   version   = u32 (LE)、 現在 1
        //   for bucket in 0..NUM_PIN_BUCKETS:
        //     for persp in 0..2:
        //       i16[PIN_DIMENSIONS_PER_PERSPECTIVE] (LE)
        #[cfg(feature = "nnue-pin-aware")]
        let pin_weights = if arch_str.contains("PIN=") {
            Some(read_pin_weights(reader)?)
        } else {
            None
        };
        #[cfg(not(feature = "nnue-pin-aware"))]
        if arch_str.contains("PIN=") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Pin-aware model requires nnue-pin-aware feature",
            ));
        }

        // Dual PSQT 読み込み (arch_str に "DUAL=1" があれば PIN の後に Dual section)。
        // iter 1 では Zeroed init weight が書き出される (= eval 寄与 0)、 iter 2
        // で head-only fine-tune の後に学習済 weight に置換。
        #[cfg(feature = "nnue-dual-psqt")]
        let dual_psqt_weights = if arch_str.contains(crate::nnue::dual_psqt::DUAL_PSQT_ARCH_MARKER)
        {
            // read_dual_psqt は 旧 reserve format を skip した場合 None を返す。
            // 新 King-Attacker format なら Some(Box<...>) を返す。
            read_dual_psqt(reader)?
        } else {
            None
        };
        #[cfg(not(feature = "nnue-dual-psqt"))]
        if arch_str.contains("DUAL=") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Dual PSQT model requires nnue-dual-psqt feature",
            ));
        }

        // Dual Cross (D2) 読み込み: P2DUAL2 section の直後に P2DUALX section があれば読む。
        // 古い ckpt (= D2 未対応) では section 不在 → EOF で None を返す。
        // probe で peek してから読む方が安全だが、 ここでは tentative に試して
        // UnexpectedEof / InvalidData の場合は None として扱う。
        //
        // KingRing section が先に来るケース (= dual_cross なし、 kingring あり) では、
        // probe で読んだ magic を `next_section_magic` に「先送り」 して KingRing 読み込み
        // 側で再利用する。 これにより 8 byte 重複消費を避ける。
        #[cfg(feature = "nnue-dual-psqt")]
        let mut next_section_magic: Option<[u8; 8]> = None;
        #[cfg(feature = "nnue-dual-psqt")]
        let dual_cross_weights = if dual_psqt_weights.is_some() {
            // probe: 8 byte magic を peek
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(0) => None, // EOF = D2 未対応 ckpt
                Ok(n) if n < 8 => {
                    // unexpected partial read (= corrupt or non-D2 ckpt)
                    None
                }
                Ok(_) => {
                    use crate::nnue::dual_cross::{DUAL_CROSS_SUB_MAGIC, DUAL_CROSS_SUB_MAGIC_Z};
                    let is_z = probe_magic == DUAL_CROSS_SUB_MAGIC_Z;
                    let is_x = probe_magic == DUAL_CROSS_SUB_MAGIC;
                    if is_z || is_x {
                        // magic 一致 → 残り (version + count + data) を読む
                        use crate::nnue::dual_cross::{
                            DUAL_CROSS_COUNT, DUAL_CROSS_COUNT_Z, DUAL_CROSS_PIECE_TYPES,
                            DUAL_CROSS_SQUARES, DUAL_CROSS_VERSION, DualCrossWeights,
                            NUM_DUAL_CROSS_BUCKETS, NUM_DUAL_CROSS_HEADS,
                        };
                        let mut buf4 = [0u8; 4];
                        reader.read_exact(&mut buf4)?;
                        let version = u32::from_le_bytes(buf4);
                        if version != DUAL_CROSS_VERSION {
                            return Err(io::Error::new(
                                io::ErrorKind::Unsupported,
                                format!("Dual Cross version {version} unsupported"),
                            ));
                        }
                        reader.read_exact(&mut buf4)?;
                        let count = u32::from_le_bytes(buf4);
                        let expected_count = if is_z {
                            DUAL_CROSS_COUNT_Z
                        } else {
                            DUAL_CROSS_COUNT
                        };
                        if count != expected_count {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "Dual Cross count mismatch: expected {expected_count}, got {count}"
                                ),
                            ));
                        }
                        // memory layout: bucket-inner、 head 0 のみ (= Path I で memory 半減)。
                        // disk format は bucket-outer のまま (= 後方互換、 bullet 側 ckpt 変更不要)。
                        // P2DUALX (旧) の場合、 head 1 weight は読み飛ばす。
                        let mut w: Box<DualCrossWeights> = Box::new(
                            [[[[[[0i16; NUM_DUAL_CROSS_BUCKETS]; DUAL_CROSS_PIECE_TYPES];
                                DUAL_CROSS_PIECE_TYPES];
                                DUAL_CROSS_SQUARES];
                                NUM_DUAL_CROSS_HEADS]; 2],
                        );
                        let disk_heads = if is_z { 1 } else { 2 };
                        let mut buf2 = [0u8; 2];
                        for persp in 0..2 {
                            for disk_head in 0..disk_heads {
                                let store_to_memory = disk_head == 0;
                                for bucket in 0..NUM_DUAL_CROSS_BUCKETS {
                                    for sq in 0..DUAL_CROSS_SQUARES {
                                        for own_p in 0..DUAL_CROSS_PIECE_TYPES {
                                            for target_p in 0..DUAL_CROSS_PIECE_TYPES {
                                                reader.read_exact(&mut buf2)?;
                                                if store_to_memory {
                                                    w[persp][0][sq][own_p][target_p][bucket] =
                                                        i16::from_le_bytes(buf2);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Some(w)
                    } else {
                        // unknown magic: 8 byte 既に消費しているため、 次の section reader
                        // (= KingRing) が再 read しないよう `next_section_magic` に「先送り」 する。
                        // KingRing magic ならそちらで認識される。 それ以外なら final EOF check で
                        // 「余りデータ error」 が起き、 ckpt format 異常として通知される。
                        next_section_magic = Some(probe_magic);
                        None
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => None,
                Err(e) => return Err(e),
            }
        } else {
            None
        };

        // Scalar Threat は Dual Cross の置換head。section自体は互換のため読み切るが、
        // runtime weightは保持せず更新・評価を停止する。feature OFFで元経路へ戻る。
        #[cfg(all(
            feature = "nnue-dual-psqt",
            feature = "nnue-scalar-threat",
            not(feature = "nnue-scalar-threat-coexist")
        ))]
        let dual_cross_weights: Option<Box<crate::nnue::dual_cross::DualCrossWeights>> = {
            drop(dual_cross_weights);
            None
        };

        // Path G: dual_cross weight を load し終えたら important_set を初期化。
        // `DUAL_CROSS_TOP_PCT` env var (= 0..100、 default 100 = filter なし) で control。
        #[cfg(feature = "nnue-dual-psqt")]
        if let Some(ref cw) = dual_cross_weights {
            crate::nnue::dual_cross::init_important_set_from_weights(cw);
        }

        // KingRing 読み込み中に unknown magic として 8 byte 先読みされた場合の保持先。
        // 後段の PairShape probe で再利用する (= 8 byte 重複消費を避ける)。
        // KingRing feature OFF + dual-psqt OFF でも宣言だけ通すため outer scope に置く。
        let mut post_kingring_magic: Option<[u8; 8]> = None;

        // KingRing weight 読み込み (= arm 11): dual_cross section の後で probe (= 8 byte magic)。
        // section 不在 (= EOF / 不明 magic) の場合は None として続行 (= 後方互換)。
        //
        // dual_cross probe で「unknown magic」 として 8 byte 先読みされた場合は
        // `next_section_magic` に保持されており、 そこを再利用する (= 8 byte 重複消費を避ける)。
        //
        // backward compat: feature `nnue-kingring-psqt` が OFF でも、 magic が P2KRING の
        // 場合は section header + data を read & skip する (= eval 寄与なし、 ckpt 互換維持)。
        #[cfg(feature = "nnue-kingring-psqt")]
        let kingring_weights = {
            #[cfg(feature = "nnue-dual-psqt")]
            let prefetched_magic = next_section_magic.take();
            #[cfg(not(feature = "nnue-dual-psqt"))]
            let prefetched_magic: Option<[u8; 8]> = None;

            let probe_result: Option<[u8; 8]> = if let Some(m) = prefetched_magic {
                Some(m)
            } else {
                let mut probe_magic = [0u8; 8];
                match reader.read(&mut probe_magic) {
                    Ok(0) => None,
                    Ok(n) if n < 8 => None,
                    Ok(_) => Some(probe_magic),
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => None,
                    Err(e) => return Err(e),
                }
            };

            match probe_result {
                None => None,
                Some(probe_magic) => {
                    use crate::nnue::kingring::{
                        KINGRING_COUNT, KINGRING_NUM_BUCKETS, KINGRING_NUM_PIECE, KINGRING_NUM_SQ,
                        KINGRING_NUM_STATE, KINGRING_SUB_MAGIC, KINGRING_VERSION, KingRingWeights,
                    };
                    if probe_magic == KINGRING_SUB_MAGIC {
                        let mut buf4 = [0u8; 4];
                        reader.read_exact(&mut buf4)?;
                        let version = u32::from_le_bytes(buf4);
                        if version != KINGRING_VERSION {
                            return Err(io::Error::new(
                                io::ErrorKind::Unsupported,
                                format!(
                                    "KingRing version {version} unsupported (expected {KINGRING_VERSION})"
                                ),
                            ));
                        }
                        reader.read_exact(&mut buf4)?;
                        let count = u32::from_le_bytes(buf4);
                        if count != KINGRING_COUNT {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "KingRing count mismatch: expected {KINGRING_COUNT}, got {count}"
                                ),
                            ));
                        }
                        // disk layout = [bucket][persp][sq][piece][state] (= bucket-outer)、
                        // memory layout も bucket-outer で完全一致 (= 転置なし sequential read)。
                        let mut w: Box<KingRingWeights> = Box::new(
                            [[[[[0i16; KINGRING_NUM_STATE]; KINGRING_NUM_PIECE]; KINGRING_NUM_SQ];
                                2]; KINGRING_NUM_BUCKETS],
                        );
                        let mut buf2 = [0u8; 2];
                        for bucket in 0..KINGRING_NUM_BUCKETS {
                            for persp in 0..2 {
                                for sq in 0..KINGRING_NUM_SQ {
                                    for piece in 0..KINGRING_NUM_PIECE {
                                        for state in 0..KINGRING_NUM_STATE {
                                            reader.read_exact(&mut buf2)?;
                                            w[bucket][persp][sq][piece][state] =
                                                i16::from_le_bytes(buf2);
                                        }
                                    }
                                }
                            }
                        }
                        Some(w)
                    } else {
                        // KingRing magic 不一致: 後段の section probe (= PairShape 等) に
                        // 「先送り」 して 8 byte 重複消費を避ける。
                        post_kingring_magic = Some(probe_magic);
                        None
                    }
                }
            }
        };

        // backward compat: feature `nnue-kingring-psqt` が OFF の場合でも、
        // KingRing section (= "P2KRING\0") が ckpt に含まれていれば section 全体を
        // read & skip する。 これにより、 kingring 入り ckpt を kingring OFF binary
        // でも読込可能 (= eval 寄与なし、 ckpt 互換維持)。
        // unknown magic (= 例えば PairShape section "P2PAIR\0\0") は post_kingring_magic
        // に「先送り」 して 後段の PairShape probe で再利用する。
        #[cfg(all(not(feature = "nnue-kingring-psqt"), feature = "nnue-dual-psqt"))]
        {
            let prefetched_magic = next_section_magic.take();
            let probe_result: Option<[u8; 8]> = if let Some(m) = prefetched_magic {
                Some(m)
            } else {
                let mut probe_magic = [0u8; 8];
                match reader.read(&mut probe_magic) {
                    Ok(0) => None,
                    Ok(n) if n < 8 => None,
                    Ok(_) => Some(probe_magic),
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => None,
                    Err(e) => return Err(e),
                }
            };

            if let Some(probe_magic) = probe_result {
                // magic (= 8 byte) + header (version u32 + count u32) + weight data
                // (= count × 2 byte) を skip する。
                const KINGRING_MAGIC: [u8; 8] = *b"P2KRING\0";
                if probe_magic == KINGRING_MAGIC {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    // count × 2 byte を skip
                    let total_bytes = (count as usize).saturating_mul(2);
                    let mut skip_buf = [0u8; 4096];
                    let mut remaining = total_bytes;
                    while remaining > 0 {
                        let chunk = remaining.min(skip_buf.len());
                        reader.read_exact(&mut skip_buf[..chunk])?;
                        remaining -= chunk;
                    }
                    eprintln!(
                        "info string P2KRING section skipped ({} bytes, kingring feature OFF)",
                        total_bytes + 16
                    );
                } else {
                    // 後段 PairShape probe に「先送り」 (= 8 byte 重複消費を避ける)。
                    post_kingring_magic = Some(probe_magic);
                }
            }
        }

        // PairShape weight 読み込み (= arm 12): KingRing section の後で probe (= 8 byte magic)。
        // section 不在 (= EOF / 不明 magic) の場合は None として続行 (= 後方互換)。
        //
        // KingRing probe で「unknown magic」 として 8 byte 先読みされた場合は
        // `post_kingring_magic` に保持されており、 そこを再利用する (= 8 byte 重複消費を避ける)。
        //
        // backward compat: feature `nnue-pairshape-psqt` が OFF でも、 magic が P2PAIR の
        // 場合は section 全体を read & skip する (= eval 寄与なし、 ckpt 互換維持)。
        #[cfg(feature = "nnue-pairshape-psqt")]
        let pairshape_weights = {
            let prefetched_magic = post_kingring_magic.take();
            // probe_result = Some((magic, consumed))。 consumed は magic 不一致時に
            // 巻き戻すべき物理 byte 数 (= prefetch 時も KingRing probe が 8 byte 消費済み)。
            let probe_result: Option<([u8; 8], usize)> = if let Some(m) = prefetched_magic {
                Some((m, 8))
            } else {
                let mut probe_magic = [0u8; 8];
                match reader.read(&mut probe_magic) {
                    Ok(0) => None,
                    Ok(n) if n < 8 => {
                        // partial read: 読んだ分だけ巻き戻して None。
                        reader.seek(SeekFrom::Current(-(n as i64)))?;
                        None
                    }
                    Ok(_) => Some((probe_magic, 8)),
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => None,
                    Err(e) => return Err(e),
                }
            };

            match probe_result {
                None => None,
                Some((probe_magic, consumed)) => {
                    use crate::nnue::pairshape::{
                        PAIRSHAPE_COUNT, PAIRSHAPE_NUM_BUCKETS, PAIRSHAPE_NUM_PAIR_TYPES,
                        PAIRSHAPE_NUM_SHAPES, PAIRSHAPE_SUB_MAGIC, PAIRSHAPE_VERSION,
                        PairShapeWeights,
                    };
                    if probe_magic == PAIRSHAPE_SUB_MAGIC {
                        let mut buf4 = [0u8; 4];
                        reader.read_exact(&mut buf4)?;
                        let version = u32::from_le_bytes(buf4);
                        if version != PAIRSHAPE_VERSION {
                            return Err(io::Error::new(
                                io::ErrorKind::Unsupported,
                                format!(
                                    "PairShape version {version} unsupported (expected {PAIRSHAPE_VERSION})"
                                ),
                            ));
                        }
                        reader.read_exact(&mut buf4)?;
                        let count = u32::from_le_bytes(buf4);
                        if count != PAIRSHAPE_COUNT {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "PairShape count mismatch: expected {PAIRSHAPE_COUNT}, got {count}"
                                ),
                            ));
                        }
                        // disk layout = [bucket][persp][pair_type][shape] (= bucket-outer)、
                        // memory layout も bucket-outer で完全一致 (= 転置なし sequential read)。
                        let mut w: Box<PairShapeWeights> = Box::new(
                            [[[[0i16; PAIRSHAPE_NUM_SHAPES]; PAIRSHAPE_NUM_PAIR_TYPES]; 2];
                                PAIRSHAPE_NUM_BUCKETS],
                        );
                        let mut buf2 = [0u8; 2];
                        for bucket in 0..PAIRSHAPE_NUM_BUCKETS {
                            for persp in 0..2 {
                                for pt in 0..PAIRSHAPE_NUM_PAIR_TYPES {
                                    for sh in 0..PAIRSHAPE_NUM_SHAPES {
                                        reader.read_exact(&mut buf2)?;
                                        w[bucket][persp][pt][sh] = i16::from_le_bytes(buf2);
                                    }
                                }
                            }
                        }
                        Some(w)
                    } else {
                        // magic 不一致: 消費した byte を巻き戻して後続 probe へ委譲。
                        // (= 不在 section が ACTR/KESC をミスアラインしないように)。
                        if consumed > 0 {
                            reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                        }
                        eprintln!(
                            "info string PairShape section magic unknown ({probe_magic:?}), treating as None"
                        );
                        None
                    }
                }
            }
        };

        // backward compat: feature `nnue-pairshape-psqt` が OFF でも P2PAIR section を skip。
        #[cfg(not(feature = "nnue-pairshape-psqt"))]
        {
            let prefetched_magic = post_kingring_magic.take();
            let probe_result: Option<[u8; 8]> = if let Some(m) = prefetched_magic {
                Some(m)
            } else {
                let mut probe_magic = [0u8; 8];
                match reader.read(&mut probe_magic) {
                    Ok(0) => None,
                    Ok(n) if n < 8 => None,
                    Ok(_) => Some(probe_magic),
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => None,
                    Err(e) => return Err(e),
                }
            };

            if let Some(probe_magic) = probe_result {
                const PAIRSHAPE_MAGIC: [u8; 8] = *b"P2PAIR\0\0";
                if probe_magic == PAIRSHAPE_MAGIC {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    let total_bytes = (count as usize).saturating_mul(2);
                    let mut skip_buf = [0u8; 4096];
                    let mut remaining = total_bytes;
                    while remaining > 0 {
                        let chunk = remaining.min(skip_buf.len());
                        reader.read_exact(&mut skip_buf[..chunk])?;
                        remaining -= chunk;
                    }
                    eprintln!(
                        "info string P2PAIR section skipped ({} bytes, pairshape feature OFF)",
                        total_bytes + 16
                    );
                } else {
                    eprintln!(
                        "info string unknown post-KINGRING section magic ({:?}, ignored)",
                        probe_magic
                    );
                }
            }
        }

        // Activity-Risk Interaction Head section probe (= 2026-06-06、 pairshape 後ろ配置)。
        // 旧 ckpt (= activity_risk 未対応) は EOF or 異 magic で None、 新 ckpt は read。
        #[cfg(feature = "nnue-activity-risk-psqt")]
        let activity_risk_weights = {
            use crate::nnue::activity_risk::{
                ACTIVITY_RISK_COUNT, ACTIVITY_RISK_SUB_MAGIC, ACTIVITY_RISK_V3_WEIGHT_DIM,
                ACTIVITY_RISK_VERSION, ActivityRiskWeightsV3,
            };
            let mut probe_magic = [0u8; 8];
            let read_result = reader.read(&mut probe_magic);
            match read_result {
                Ok(8) if probe_magic == ACTIVITY_RISK_SUB_MAGIC => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let version = u32::from_le_bytes(buf4);
                    if version != ACTIVITY_RISK_VERSION {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!(
                                "ActivityRisk version {version} unsupported (expected {ACTIVITY_RISK_VERSION})"
                            ),
                        ));
                    }
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    if count != ACTIVITY_RISK_COUNT {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "ActivityRisk count mismatch: expected {ACTIVITY_RISK_COUNT}, got {count}"
                            ),
                        ));
                    }
                    // disk layout v3 = base[4096] + tactical[1024] row-major
                    // sequential read で flatten 順に格納する。
                    let mut w: Box<ActivityRiskWeightsV3> =
                        Box::new([0i16; ACTIVITY_RISK_V3_WEIGHT_DIM]);
                    let mut buf2 = [0u8; 2];
                    for slot in w.iter_mut() {
                        reader.read_exact(&mut buf2)?;
                        *slot = i16::from_le_bytes(buf2);
                    }
                    Some(w)
                }
                // magic 不一致 / partial read: 読んだ byte 数分だけ巻き戻して None。
                // これにより不在 section が後続 probe をミスアラインしない。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(_) => None,
            }
        };

        // King Escape Direction (KED) section probe (= 2026-06-09、 activity_risk 後ろ配置)。
        // 旧 ckpt (= KED 未対応) は EOF or 異 magic で None、 新 ckpt は read。
        #[cfg(feature = "nnue-king-escape-psqt")]
        let king_escape_weights = {
            use crate::nnue::king_escape::{
                KING_ESCAPE_MAGIC, KING_ESCAPE_VERSION, KING_ESCAPE_WEIGHT_COUNT, KingEscapeWeights,
            };
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == KING_ESCAPE_MAGIC => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let version = u32::from_le_bytes(buf4);
                    if version != KING_ESCAPE_VERSION {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!(
                                "KingEscape version {version} unsupported (expected {KING_ESCAPE_VERSION})"
                            ),
                        ));
                    }
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    if (count as usize) != KING_ESCAPE_WEIGHT_COUNT {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "KingEscape count mismatch: expected {KING_ESCAPE_WEIGHT_COUNT}, got {count}"
                            ),
                        ));
                    }
                    let mut w: Box<KingEscapeWeights> = Box::new([[[0i16; 4]; 2]; 8]);
                    let mut buf2 = [0u8; 2];
                    for bucket in 0..8 {
                        for persp in 0..2 {
                            for class in 0..4 {
                                reader.read_exact(&mut buf2)?;
                                w[bucket][persp][class] = i16::from_le_bytes(buf2);
                            }
                        }
                    }
                    Some(w)
                }
                // magic 不一致 / partial read: 読んだ byte 数分だけ巻き戻して None。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(_) => None,
            }
        };

        // backward compat: feature `nnue-king-escape-psqt` が OFF でも P2KESC section を skip。
        #[cfg(not(feature = "nnue-king-escape-psqt"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == b"P2KESC\0\0" => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    let total_bytes = (count as usize).saturating_mul(2);
                    let mut skip_buf = [0u8; 4096];
                    let mut remaining = total_bytes;
                    while remaining > 0 {
                        let chunk = remaining.min(skip_buf.len());
                        reader.read_exact(&mut skip_buf[..chunk])?;
                        remaining -= chunk;
                    }
                    eprintln!(
                        "info string P2KESC section skipped ({} bytes, king-escape feature OFF)",
                        total_bytes + 16
                    );
                }
                // magic 不一致 / partial read: 読んだ byte 数分巻き戻して後続 probe へ委譲。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(_) => {}
            }
        }

        // Attack Relay Head (ARH) section probe (= 2026-06-09、 KED 後ろ配置)。
        // 旧 ckpt (= ARH 未対応) は EOF or 異 magic で None、 新 ckpt は read。
        #[cfg(feature = "nnue-attack-relay")]
        let attack_relay_weights = {
            use crate::nnue::attack_relay::{ARH_MAGIC, ARH_VERSION, ARH_WEIGHT_COUNT, ArhWeights};
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == ARH_MAGIC => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let version = u32::from_le_bytes(buf4);
                    if version != ARH_VERSION {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!(
                                "AttackRelay version {version} unsupported (expected {ARH_VERSION})"
                            ),
                        ));
                    }
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    if (count as usize) != ARH_WEIGHT_COUNT {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "AttackRelay count mismatch: expected {ARH_WEIGHT_COUNT}, got {count}"
                            ),
                        ));
                    }
                    let mut w: Box<ArhWeights> = Box::new([[[[[0i16; 4]; 4]; 4]; 2]; 8]);
                    let mut buf2 = [0u8; 2];
                    for bucket in 0..8 {
                        for persp in 0..2 {
                            for contact in 0..4 {
                                for reserve in 0..4 {
                                    for relay in 0..4 {
                                        reader.read_exact(&mut buf2)?;
                                        w[bucket][persp][contact][reserve][relay] =
                                            i16::from_le_bytes(buf2);
                                    }
                                }
                            }
                        }
                    }
                    Some(w)
                }
                // magic 不一致 / partial read: 読んだ byte 数分だけ巻き戻して None。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(_) => None,
            }
        };

        // backward compat: feature `nnue-attack-relay` が OFF でも P2AREL section を skip。
        #[cfg(not(feature = "nnue-attack-relay"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == b"P2AREL\0\0" => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    let total_bytes = (count as usize).saturating_mul(2);
                    let mut skip_buf = [0u8; 4096];
                    let mut remaining = total_bytes;
                    while remaining > 0 {
                        let chunk = remaining.min(skip_buf.len());
                        reader.read_exact(&mut skip_buf[..chunk])?;
                        remaining -= chunk;
                    }
                    eprintln!(
                        "info string P2AREL section skipped ({} bytes, attack-relay feature OFF)",
                        total_bytes + 16
                    );
                }
                // magic 不一致 / partial read: 読んだ byte 数分巻き戻して後続 probe へ委譲。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(_) => {}
            }
        }

        // ARH v2 (= 構造化 980 weight、 magic P2ARS1) section probe。
        // 旧 P2AREL2 (= 1024 cell 直積) には明確エラーで非互換性を示す。
        #[cfg(feature = "nnue-attack-relay-v2")]
        let attack_relay_v2_weights = {
            use crate::nnue::attack_relay_v2::{ARH_V2_MAGIC, read_weights_after_magic};
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == ARH_V2_MAGIC => Some(read_weights_after_magic(reader)?),
                Ok(8) if &probe_magic == b"P2AREL2\0" => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "AttackRelay v2 legacy P2AREL2 (1024-cell direct-product) \
                         ckpt is not compatible with the structured 980-weight P2ARS1 format. \
                         Re-train with the new ARH v2 implementation.",
                    ));
                }
                // magic 不一致 / partial read: 読んだ byte 数分だけ巻き戻して None。
                // (P2AREL2 の明示エラー枝は上で処理済み)。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(_) => None,
            }
        };

        // backward compat: feature `nnue-attack-relay-v2` が OFF でも ARH section を skip。
        // 旧 P2AREL2 / 新 P2ARS1 両方を skip しないと余り data で factorized error になる。
        #[cfg(not(feature = "nnue-attack-relay-v2"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) => {
                    let (label, expected_count) = if &probe_magic == b"P2ARS1\0\0" {
                        // P2ARS1 (= 構造化 head) は 1016 i16。 旧コードは 980 を hardcode して
                        // 不一致 log を出していた (= load 自体は disk の count で skip するため実害なし)。
                        ("P2ARS1", 1016usize)
                    } else if &probe_magic == b"P2AREL2\0" {
                        ("P2AREL2 (legacy)", 1024usize)
                    } else {
                        ("", 0)
                    };
                    if label.is_empty() {
                        // magic 不一致: 8 byte 巻き戻して後続 (= EOF 検証) へ委譲。
                        reader.seek(SeekFrom::Current(-8))?;
                    } else {
                        let mut buf4 = [0u8; 4];
                        reader.read_exact(&mut buf4)?;
                        let _version = u32::from_le_bytes(buf4);
                        reader.read_exact(&mut buf4)?;
                        let count = u32::from_le_bytes(buf4) as usize;
                        let total_bytes = count.saturating_mul(2);
                        let mut skip_buf = [0u8; 4096];
                        let mut remaining = total_bytes;
                        while remaining > 0 {
                            let chunk = remaining.min(skip_buf.len());
                            reader.read_exact(&mut skip_buf[..chunk])?;
                            remaining -= chunk;
                        }
                        eprintln!(
                            "info string {} section skipped ({} bytes, attack-relay-v2 feature OFF, count={}/expected={})",
                            label,
                            total_bytes + 16,
                            count,
                            expected_count
                        );
                    }
                }
                // partial read: 読んだ分巻き戻す。 EOF (0 byte) は巻き戻し不要。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(_) => {}
            }
        }

        // King Danger (KDGR) section probe (= 2026-06-15、 ARH v2 後ろ配置 = 全 section 末尾)。
        // 旧 ckpt (= KDGR 未対応) は EOF or 異 magic で None、 新 ckpt は read。
        #[cfg(feature = "nnue-king-danger-psqt")]
        let king_danger_weights = {
            use crate::nnue::king_danger::{
                KING_DANGER_MAGIC, KING_DANGER_VERSION, KING_DANGER_WEIGHT_COUNT, KingDangerWeights,
            };
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == KING_DANGER_MAGIC => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let version = u32::from_le_bytes(buf4);
                    if version != KING_DANGER_VERSION {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!(
                                "KingDanger version {version} unsupported (expected {KING_DANGER_VERSION})"
                            ),
                        ));
                    }
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    if (count as usize) != KING_DANGER_WEIGHT_COUNT {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "KingDanger count mismatch: expected {KING_DANGER_WEIGHT_COUNT}, got {count}"
                            ),
                        ));
                    }
                    let mut w: Box<KingDangerWeights> = Box::new([[[0i16; 8]; 2]; 8]);
                    let mut buf2 = [0u8; 2];
                    for bucket in 0..8 {
                        for persp in 0..2 {
                            for class in 0..8 {
                                reader.read_exact(&mut buf2)?;
                                w[bucket][persp][class] = i16::from_le_bytes(buf2);
                            }
                        }
                    }
                    Some(w)
                }
                // magic 不一致 / partial read: 読んだ byte 数分だけ巻き戻して None。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(_) => None,
            }
        };

        // backward compat: feature `nnue-king-danger-psqt` が OFF でも P2KDGR section を skip。
        #[cfg(not(feature = "nnue-king-danger-psqt"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == b"P2KDGR\0\0" => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    let total_bytes = (count as usize).saturating_mul(2);
                    let mut skip_buf = [0u8; 4096];
                    let mut remaining = total_bytes;
                    while remaining > 0 {
                        let chunk = remaining.min(skip_buf.len());
                        reader.read_exact(&mut skip_buf[..chunk])?;
                        remaining -= chunk;
                    }
                    eprintln!(
                        "info string P2KDGR section skipped ({} bytes, king-danger feature OFF)",
                        total_bytes + 16
                    );
                }
                // magic 不一致 / partial read: 読んだ byte 数分巻き戻して後続 probe へ委譲。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(_) => {}
            }
        }

        // Attack Development (ADEV) section probe (= 2026-06-16、 KDGR 後ろ配置 = 全 section 末尾)。
        // 旧 ckpt (= ADEV 未対応) は EOF or 異 magic で None、 新 ckpt は read。
        #[cfg(feature = "nnue-attack-dev-psqt")]
        let attack_dev_weights = {
            use crate::nnue::attack_dev::{
                ATTACK_DEV_MAGIC, ATTACK_DEV_VERSION, ATTACK_DEV_WEIGHT_COUNT, AttackDevWeights,
            };
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == ATTACK_DEV_MAGIC => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let version = u32::from_le_bytes(buf4);
                    if version != ATTACK_DEV_VERSION {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!(
                                "AttackDev version {version} unsupported (expected {ATTACK_DEV_VERSION})"
                            ),
                        ));
                    }
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    if (count as usize) != ATTACK_DEV_WEIGHT_COUNT {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "AttackDev count mismatch: expected {ATTACK_DEV_WEIGHT_COUNT}, got {count}"
                            ),
                        ));
                    }
                    let mut w: Box<AttackDevWeights> = Box::new([[[0i16; 8]; 2]; 8]);
                    let mut buf2 = [0u8; 2];
                    for bucket in 0..8 {
                        for persp in 0..2 {
                            for class in 0..8 {
                                reader.read_exact(&mut buf2)?;
                                w[bucket][persp][class] = i16::from_le_bytes(buf2);
                            }
                        }
                    }
                    Some(w)
                }
                // magic 不一致 / partial read: 読んだ byte 数分だけ巻き戻して None。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(_) => None,
            }
        };

        // backward compat: feature `nnue-attack-dev-psqt` が OFF でも P2ADEV section を skip。
        #[cfg(not(feature = "nnue-attack-dev-psqt"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == b"P2ADEV\0\0" => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    let total_bytes = (count as usize).saturating_mul(2);
                    let mut skip_buf = [0u8; 4096];
                    let mut remaining = total_bytes;
                    while remaining > 0 {
                        let chunk = remaining.min(skip_buf.len());
                        reader.read_exact(&mut skip_buf[..chunk])?;
                        remaining -= chunk;
                    }
                    eprintln!(
                        "info string P2ADEV section skipped ({} bytes, attack-dev feature OFF)",
                        total_bytes + 16
                    );
                }
                // magic 不一致 / partial read: 読んだ byte 数分巻き戻して後続 probe へ委譲。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(_) => {}
            }
        }

        // King Mobility (KMOB) section probe (= 2026-06-16、 ADEV 後ろ配置 = 全 section 末尾)。
        // 旧 ckpt (= KMOB 未対応) は EOF or 異 magic で None、 新 ckpt は read。
        // class が 9 (= 0..=8) である点が ADEV / KDGR (= 8) と異なる。
        #[cfg(feature = "nnue-king-mobility-psqt")]
        let king_mobility_weights = {
            use crate::nnue::king_mobility::{
                KING_MOBILITY_MAGIC, KING_MOBILITY_NUM_CLASS, KING_MOBILITY_VERSION,
                KING_MOBILITY_WEIGHT_COUNT, KingMobilityWeights,
            };
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == KING_MOBILITY_MAGIC => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let version = u32::from_le_bytes(buf4);
                    if version != KING_MOBILITY_VERSION {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!(
                                "KingMobility version {version} unsupported (expected {KING_MOBILITY_VERSION})"
                            ),
                        ));
                    }
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    if (count as usize) != KING_MOBILITY_WEIGHT_COUNT {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "KingMobility count mismatch: expected {KING_MOBILITY_WEIGHT_COUNT}, got {count}"
                            ),
                        ));
                    }
                    let mut w: Box<KingMobilityWeights> =
                        Box::new([[[0i16; KING_MOBILITY_NUM_CLASS]; 2]; 8]);
                    let mut buf2 = [0u8; 2];
                    for bucket in 0..8 {
                        for persp in 0..2 {
                            for class in 0..KING_MOBILITY_NUM_CLASS {
                                reader.read_exact(&mut buf2)?;
                                w[bucket][persp][class] = i16::from_le_bytes(buf2);
                            }
                        }
                    }
                    Some(w)
                }
                // magic 不一致 / partial read: 読んだ byte 数分だけ巻き戻して None。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(_) => None,
            }
        };

        // backward compat: feature `nnue-king-mobility-psqt` が OFF でも P2KMOB section を skip。
        #[cfg(not(feature = "nnue-king-mobility-psqt"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == b"P2KMOB\0\0" => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    let total_bytes = (count as usize).saturating_mul(2);
                    let mut skip_buf = [0u8; 4096];
                    let mut remaining = total_bytes;
                    while remaining > 0 {
                        let chunk = remaining.min(skip_buf.len());
                        reader.read_exact(&mut skip_buf[..chunk])?;
                        remaining -= chunk;
                    }
                    eprintln!(
                        "info string P2KMOB section skipped ({} bytes, king-mobility feature OFF)",
                        total_bytes + 16
                    );
                }
                // magic 不一致 / partial read: 読んだ byte 数分巻き戻して後続 probe へ委譲。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(_) => {}
            }
        }

        // Drop Pressure (DROP) section probe (= 2026-06-16、 KMOB 後ろ配置 = 全 section 物理末尾)。
        // 旧 ckpt (= DROP 未対応) は EOF or 異 magic で None、 新 ckpt は read。
        // 相手持駒投下脅威 class table (= 128 i16、 ADEV / KDGR と同形)。
        #[cfg(feature = "nnue-drop-pressure-psqt")]
        let drop_pressure_weights = {
            use crate::nnue::drop_pressure::{
                DROP_PRESSURE_MAGIC, DROP_PRESSURE_NUM_CLASS, DROP_PRESSURE_VERSION,
                DROP_PRESSURE_WEIGHT_COUNT, DropPressureWeights,
            };
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == DROP_PRESSURE_MAGIC => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let version = u32::from_le_bytes(buf4);
                    if version != DROP_PRESSURE_VERSION {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!(
                                "DropPressure version {version} unsupported (expected {DROP_PRESSURE_VERSION})"
                            ),
                        ));
                    }
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    if (count as usize) != DROP_PRESSURE_WEIGHT_COUNT {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "DropPressure count mismatch: expected {DROP_PRESSURE_WEIGHT_COUNT}, got {count}"
                            ),
                        ));
                    }
                    let mut w: Box<DropPressureWeights> =
                        Box::new([[[0i16; DROP_PRESSURE_NUM_CLASS]; 2]; 8]);
                    let mut buf2 = [0u8; 2];
                    for bucket in 0..8 {
                        for persp in 0..2 {
                            for class in 0..DROP_PRESSURE_NUM_CLASS {
                                reader.read_exact(&mut buf2)?;
                                w[bucket][persp][class] = i16::from_le_bytes(buf2);
                            }
                        }
                    }
                    Some(w)
                }
                // magic 不一致 / partial read: 読んだ byte 数分だけ巻き戻して None。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(_) => None,
            }
        };

        // backward compat: feature `nnue-drop-pressure-psqt` が OFF でも P2DROP section を skip。
        #[cfg(not(feature = "nnue-drop-pressure-psqt"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == b"P2DROP\0\0" => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    let total_bytes = (count as usize).saturating_mul(2);
                    let mut skip_buf = [0u8; 4096];
                    let mut remaining = total_bytes;
                    while remaining > 0 {
                        let chunk = remaining.min(skip_buf.len());
                        reader.read_exact(&mut skip_buf[..chunk])?;
                        remaining -= chunk;
                    }
                    eprintln!(
                        "info string P2DROP section skipped ({} bytes, drop-pressure feature OFF)",
                        total_bytes + 16
                    );
                }
                // magic 不一致 / partial read: 読んだ byte 数分巻き戻して後続 probe へ委譲。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(_) => {}
            }
        }

        // Scalar Threat sectionは物理末尾へ追加する。旧ckptはEOFでNone。
        #[cfg(feature = "nnue-scalar-threat")]
        let scalar_threat_weights = {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == crate::nnue::scalar_threat::SCALAR_THREAT_MAGIC => {
                    Some(crate::nnue::scalar_threat::ScalarThreatWeights::read_after_magic(reader)?)
                }
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => None,
                Err(e) => return Err(e),
            }
        };

        // Rollback互換: feature OFFの旧経路でも新sectionを読み飛ばし、同じckptを使える。
        #[cfg(not(feature = "nnue-scalar-threat"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == b"P2STHR\0\0" => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4) as usize;
                    let mut remaining = count.saturating_mul(2);
                    let mut skip = [0u8; 4096];
                    while remaining > 0 {
                        let chunk = remaining.min(skip.len());
                        reader.read_exact(&mut skip[..chunk])?;
                        remaining -= chunk;
                    }
                }
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {}
                Err(e) => return Err(e),
            }
        }

        #[cfg(feature = "nnue-dual-cross-geometry")]
        let dual_cross_geometry_weights = {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == crate::nnue::dual_cross_geometry::MAGIC => Some(
                    crate::nnue::dual_cross_geometry::DualCrossGeometryWeights::read_after_magic(
                        reader,
                    )?,
                ),
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => None,
                Err(e) => return Err(e),
            }
        };

        #[cfg(not(feature = "nnue-dual-cross-geometry"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == b"P2DCG1\0\0" => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    reader.read_exact(&mut buf4)?;
                    let mut remaining = (u32::from_le_bytes(buf4) as usize).saturating_mul(2);
                    let mut skip = [0u8; 4096];
                    while remaining > 0 {
                        let chunk = remaining.min(skip.len());
                        reader.read_exact(&mut skip[..chunk])?;
                        remaining -= chunk;
                    }
                }
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {}
                Err(e) => return Err(e),
            }
        }

        // Slider X-ray (SXRY) section probe (= 2026-06-28、 全 section の物理末尾)。
        // 旧 ckpt (= SXRY 未対応) は EOF or 異 magic で None、 新 ckpt は read。
        #[cfg(feature = "nnue-slider-xray-psqt")]
        let slider_xray_weights = {
            use crate::nnue::slider_xray::{
                SLIDER_XRAY_MAGIC, SLIDER_XRAY_VERSION, SLIDER_XRAY_WEIGHT_COUNT, SliderXrayWeights,
            };
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == SLIDER_XRAY_MAGIC => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let version = u32::from_le_bytes(buf4);
                    if version != SLIDER_XRAY_VERSION {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!(
                                "SliderXray version {version} unsupported (expected {SLIDER_XRAY_VERSION})"
                            ),
                        ));
                    }
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    if (count as usize) != SLIDER_XRAY_WEIGHT_COUNT {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "SliderXray count mismatch: expected {SLIDER_XRAY_WEIGHT_COUNT}, got {count}"
                            ),
                        ));
                    }
                    let mut w: Box<SliderXrayWeights> = Box::new([[[0i16; 8]; 2]; 8]);
                    let mut buf2 = [0u8; 2];
                    for bucket in 0..8 {
                        for persp in 0..2 {
                            for class in 0..8 {
                                reader.read_exact(&mut buf2)?;
                                w[bucket][persp][class] = i16::from_le_bytes(buf2);
                            }
                        }
                    }
                    Some(w)
                }
                // magic 不一致 / partial read: 読んだ byte 数分だけ巻き戻して None。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(_) => None,
            }
        };

        // backward compat: feature `nnue-slider-xray-psqt` が OFF でも P2SXRY section を skip。
        #[cfg(not(feature = "nnue-slider-xray-psqt"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == b"P2SXRY\0\0" => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    let total_bytes = (count as usize).saturating_mul(2);
                    let mut skip_buf = [0u8; 4096];
                    let mut remaining = total_bytes;
                    while remaining > 0 {
                        let chunk = remaining.min(skip_buf.len());
                        reader.read_exact(&mut skip_buf[..chunk])?;
                        remaining -= chunk;
                    }
                    eprintln!(
                        "info string P2SXRY section skipped ({} bytes, slider-xray feature OFF)",
                        total_bytes + 16
                    );
                }
                // magic 不一致 / partial read: 読んだ byte 数分巻き戻して後続 probe へ委譲。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(_) => {}
            }
        }

        // King Zone Threat (KZT_HAND) section probe (= 2026-06-30、 SXRY 直後 = 全 section
        // の物理末尾)。 旧 ckpt (= KZT 未対応) は EOF or 異 magic で None、 新 ckpt は read。
        #[cfg(feature = "nnue-kzt-psqt")]
        let kzt_hand_weights = {
            use crate::nnue::kzt::{
                KZT_FACTORED_COUNT, KZT_MAGIC, KZT_VERSION, KztFactoredWeights,
            };
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == KZT_MAGIC => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let version = u32::from_le_bytes(buf4);
                    if version != KZT_VERSION {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!("Kzt version {version} unsupported (expected {KZT_VERSION})"),
                        ));
                    }
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    if (count as usize) != KZT_FACTORED_COUNT {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "Kzt count mismatch: expected {KZT_FACTORED_COUNT}, got {count}"
                            ),
                        ));
                    }
                    // factored 5 テーブル (= core → region → bucket → persp → bias の固定順)
                    // を flat i16 で bulk read し、 KztFactoredWeights に分解する。
                    let mut bytes = vec![0u8; KZT_FACTORED_COUNT * 2];
                    reader.read_exact(&mut bytes)?;
                    let mut flat = vec![0i16; KZT_FACTORED_COUNT];
                    for (dst, chunk) in flat.iter_mut().zip(bytes.chunks_exact(2)) {
                        *dst = i16::from_le_bytes([chunk[0], chunk[1]]);
                    }
                    Some(KztFactoredWeights::from_flat(&flat))
                }
                // magic 不一致 / partial read: 読んだ byte 数分だけ巻き戻して None。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                    None
                }
                Err(_) => None,
            }
        };

        // backward compat: feature `nnue-kzt-psqt` が OFF でも P2KZT section を skip。
        #[cfg(not(feature = "nnue-kzt-psqt"))]
        {
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == b"P2KZT\0\0\0" => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let _version = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4);
                    let total_bytes = (count as usize).saturating_mul(2);
                    let mut skip_buf = [0u8; 4096];
                    let mut remaining = total_bytes;
                    while remaining > 0 {
                        let chunk = remaining.min(skip_buf.len());
                        reader.read_exact(&mut skip_buf[..chunk])?;
                        remaining -= chunk;
                    }
                    eprintln!(
                        "info string P2KZT section skipped ({} bytes, kzt feature OFF)",
                        total_bytes + 16
                    );
                }
                // magic 不一致 / partial read: 読んだ byte 数分巻き戻して後続 probe へ委譲。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(_) => {}
            }
        }

        // KZT-FT (= P2KZTF) section probe (= 2026-07-03、 全 section の物理末尾)。
        // 旧 ckpt (= KZT-FT 未対応) は EOF or 異 magic で has_kzt_ft=false のまま (= eval 寄与 0)、
        // 新 ckpt は FT weight [KZT_FT_DIMENSIONS × L1] i8 を読む。
        #[cfg(feature = "nnue-kzt-ft")]
        {
            use crate::nnue::kzt::{KZT_FT_DIMENSIONS, KZT_FT_MAGIC, KZT_FT_VERSION};
            let mut probe_magic = [0u8; 8];
            match reader.read(&mut probe_magic) {
                Ok(8) if &probe_magic == KZT_FT_MAGIC => {
                    let mut buf4 = [0u8; 4];
                    reader.read_exact(&mut buf4)?;
                    let version = u32::from_le_bytes(buf4);
                    if version != KZT_FT_VERSION {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!(
                                "KZT-FT version {version} unsupported (expected {KZT_FT_VERSION})"
                            ),
                        ));
                    }
                    reader.read_exact(&mut buf4)?;
                    let count = u32::from_le_bytes(buf4) as usize;
                    let expected = KZT_FT_DIMENSIONS * L1;
                    if count != expected {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("KZT-FT count mismatch: expected {expected}, got {count}"),
                        ));
                    }
                    feature_transformer.read_kzt_ft_weights(reader)?;
                }
                // magic 不一致 / partial read: 読んだ byte 数分巻き戻して後続 (= EOF 検証) へ委譲。
                Ok(consumed) => {
                    if consumed > 0 {
                        reader.seek(SeekFrom::Current(-(consumed as i64)))?;
                    }
                }
                Err(_) => {}
            }
        }

        // EOF検証: 余りデータがないことを確認
        // factorizedモデル（非coalesced）を誤って読んだ場合、
        // 余りデータが発生する可能性がある。
        let mut probe = [0u8; 1];
        match reader.read(&mut probe) {
            Ok(0) => {
                // EOF到達 - 正常（coalesce済みモデル）
            }
            Ok(_) => {
                // 余りデータあり - おそらくfactorizedモデル
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "NNUE file has unexpected trailing data.\n\
                     This likely indicates a factorized (non-coalesced) model.\n\
                     This engine only supports coalesced models.\n\n\
                     To fix: Re-export the model using nnue-pytorch serialize.py:\n\
                       python serialize.py model.ckpt output.nnue\n\n\
                     The serialize.py script automatically coalesces factor weights.",
                ));
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // EOF - 正常
            }
            Err(e) => {
                // その他のIOエラー
                return Err(e);
            }
        }

        // 診断ログを出力
        #[cfg(feature = "diagnostics")]
        {
            Self::log_load_diagnostics(&feature_transformer, &layer_stacks);
        }

        // KingRing runtime weight 構築 (= hot path 用 bucket-inner i32 row、 2026-05-29)。
        // disk shape (= bucket-outer i16) を読んだ直後に転置で 1 回だけ build。
        // disk shape は ckpt 互換のため別 field で保持し、 hot path はこちら経由。
        #[cfg(feature = "nnue-kingring-psqt")]
        let kingring_runtime_weights =
            kingring_weights.as_deref().map(crate::nnue::kingring::build_runtime_weights);

        // PairShape runtime weight 構築 (= hot path 用 bucket-inner i32 row、 2026-05-29)。
        #[cfg(feature = "nnue-pairshape-psqt")]
        let pairshape_runtime_weights =
            pairshape_weights.as_deref().map(crate::nnue::pairshape::build_runtime_weights);

        // KED runtime weight 構築 (= hot path 用 bucket-inner i32 row、 2026-06-09)。
        #[cfg(feature = "nnue-king-escape-psqt")]
        let king_escape_runtime_weights = king_escape_weights
            .as_deref()
            .map(|disk| Box::new(crate::nnue::king_escape::build_runtime_weights(disk)));

        // ARH runtime weight 構築 (= hot path 用 bucket-inner i32 row、 2026-06-09)。
        #[cfg(feature = "nnue-attack-relay")]
        let attack_relay_runtime_weights = attack_relay_weights
            .as_deref()
            .map(crate::nnue::attack_relay::build_runtime_weights);

        // ARH v2 runtime weight 構築 (= 2026-06-10)。
        #[cfg(feature = "nnue-attack-relay-v2")]
        let attack_relay_v2_runtime_weights = attack_relay_v2_weights
            .as_deref()
            .map(crate::nnue::attack_relay_v2::build_runtime_weights);

        // KingDanger runtime weight 構築 (= hot path 用 bucket-inner i32 row、 2026-06-15)。
        #[cfg(feature = "nnue-king-danger-psqt")]
        let king_danger_runtime_weights = king_danger_weights
            .as_deref()
            .map(|disk| Box::new(crate::nnue::king_danger::build_runtime_weights(disk)));

        // AttackDev runtime weight 構築 (= hot path 用 bucket-inner i32 row、 2026-06-16)。
        #[cfg(feature = "nnue-attack-dev-psqt")]
        let attack_dev_runtime_weights = attack_dev_weights
            .as_deref()
            .map(|disk| Box::new(crate::nnue::attack_dev::build_runtime_weights(disk)));

        // SliderXray runtime weight 構築 (= hot path 用 bucket-inner i32 row、 2026-06-28)。
        #[cfg(feature = "nnue-slider-xray-psqt")]
        let slider_xray_runtime_weights = slider_xray_weights
            .as_deref()
            .map(|disk| Box::new(crate::nnue::slider_xray::build_runtime_weights(disk)));

        // KingMobility runtime weight 構築 (= hot path 用 bucket-inner i32 row、 2026-06-16)。
        #[cfg(feature = "nnue-king-mobility-psqt")]
        let king_mobility_runtime_weights = king_mobility_weights
            .as_deref()
            .map(|disk| Box::new(crate::nnue::king_mobility::build_runtime_weights(disk)));

        // DropPressure runtime weight 構築 (= hot path 用 bucket-inner i32 row、 2026-06-16)。
        #[cfg(feature = "nnue-drop-pressure-psqt")]
        let drop_pressure_runtime_weights = drop_pressure_weights
            .as_deref()
            .map(|disk| Box::new(crate::nnue::drop_pressure::build_runtime_weights(disk)));

        Ok(Self {
            feature_transformer,
            layer_stacks,
            fv_scale,
            #[cfg(feature = "nnue-pin-aware")]
            pin_weights,
            #[cfg(feature = "nnue-dual-psqt")]
            dual_psqt_weights,
            #[cfg(feature = "nnue-dual-psqt")]
            dual_cross_weights,
            #[cfg(feature = "nnue-kingring-psqt")]
            kingring_weights,
            #[cfg(feature = "nnue-kingring-psqt")]
            kingring_runtime_weights,
            #[cfg(feature = "nnue-pairshape-psqt")]
            pairshape_weights,
            #[cfg(feature = "nnue-pairshape-psqt")]
            pairshape_runtime_weights,
            #[cfg(feature = "nnue-activity-risk-psqt")]
            activity_risk_weights,
            #[cfg(feature = "nnue-king-escape-psqt")]
            king_escape_weights,
            #[cfg(feature = "nnue-king-escape-psqt")]
            king_escape_runtime_weights,
            #[cfg(feature = "nnue-attack-relay")]
            attack_relay_weights,
            #[cfg(feature = "nnue-attack-relay")]
            attack_relay_runtime_weights,
            #[cfg(feature = "nnue-attack-relay-v2")]
            attack_relay_v2_weights,
            #[cfg(feature = "nnue-attack-relay-v2")]
            attack_relay_v2_runtime_weights,
            #[cfg(feature = "nnue-king-danger-psqt")]
            king_danger_weights,
            #[cfg(feature = "nnue-king-danger-psqt")]
            king_danger_runtime_weights,
            #[cfg(feature = "nnue-attack-dev-psqt")]
            attack_dev_weights,
            #[cfg(feature = "nnue-attack-dev-psqt")]
            attack_dev_runtime_weights,
            #[cfg(feature = "nnue-slider-xray-psqt")]
            slider_xray_weights,
            #[cfg(feature = "nnue-slider-xray-psqt")]
            slider_xray_runtime_weights,
            #[cfg(feature = "nnue-kzt-psqt")]
            kzt_hand_weights,
            #[cfg(feature = "nnue-king-mobility-psqt")]
            king_mobility_weights,
            #[cfg(feature = "nnue-king-mobility-psqt")]
            king_mobility_runtime_weights,
            #[cfg(feature = "nnue-drop-pressure-psqt")]
            drop_pressure_weights,
            #[cfg(feature = "nnue-drop-pressure-psqt")]
            drop_pressure_runtime_weights,
            #[cfg(feature = "nnue-scalar-threat")]
            scalar_threat_weights,
            #[cfg(feature = "nnue-dual-cross-geometry")]
            dual_cross_geometry_weights,
        })
    }

    /// 読み込み時の診断ログを出力
    #[cfg(feature = "diagnostics")]
    fn log_load_diagnostics(
        ft: &FeatureTransformerLayerStacks<L1>,
        ls: &LayerStacks<L1, LS_L1_OUT, LS_L2_IN, LS_L2_PADDED_INPUT, LS_L3>,
    ) {
        // FT統計
        let bias_sum: i64 = ft.biases.0.iter().map(|&x| x as i64).sum();
        let weight_min = ft.weights.iter().copied().min().unwrap_or(0);
        let weight_max = ft.weights.iter().copied().max().unwrap_or(0);
        let weight_nonzero: usize = ft.weights.iter().filter(|&&x| x != 0).count();
        let weight_total = ft.weights.len();

        info!("[NNUE Load] FT bias sum: {bias_sum}");
        info!("[NNUE Load] FT weight: min={weight_min}, max={weight_max}");
        info!(
            "[NNUE Load] FT weight nonzero: {weight_nonzero}/{weight_total} ({:.2}%)",
            weight_nonzero as f64 / weight_total as f64 * 100.0
        );

        // LayerStacks bucket0 の l1_biases
        let l1_biases = &ls.buckets[0].l1.biases;
        info!("[NNUE Load] LayerStacks bucket0 l1_biases: {l1_biases:?}");
    }

    /// バイト列から読み込み
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut cursor = Cursor::new(bytes);
        Self::read(&mut cursor)
    }

    /// 評価値を計算
    ///
    /// 配列はMaybeUninitで確保し、直後のsqr_clipped_relu_transformで全要素が上書きされる。
    pub fn evaluate(&self, pos: &mut Position, acc: &AccumulatorLayerStacks<L1>) -> Value {
        let side_to_move = pos.side_to_move();
        let bucket_index = compute_layer_stacks_bucket_index(pos, side_to_move);
        self.evaluate_with_bucket(pos, acc, bucket_index)
    }

    /// 評価値計算 (= activity_risk accumulator cached path、 2026-06-06、 NPS 改善版)
    ///
    /// `accum.dirty=true` なら `refresh()` で full compute + cache、 false なら cached features
    /// で `lookup_perspective` 2 呼びのみ (= 真 NPS 改善 path、 cached 時 ~0 cost)。
    ///
    /// SearchWorker が AccumulatorStack 経由で 1 ply / 1 evaluate で accum を持ち回し、
    /// 1 ply 内の同じ局面再 evaluate は cache hit。 ただし alpha-beta では ply 毎 1 evaluate
    /// なので、 真の NPS 改善は incremental update (= shield / pressure / activity の差分計算)
    /// 実装後に Step 4-6 で実現。 現状は dirty 受理 + cached lookup の base 形完成。
    #[cfg(feature = "nnue-activity-risk-psqt")]
    pub fn evaluate_with_bucket_and_activity_risk(
        &self,
        pos: &mut Position,
        acc: &AccumulatorLayerStacks<L1>,
        bucket_index: usize,
        accum: &mut crate::nnue::activity_risk::ActivityRiskAccumulator,
    ) -> Value {
        // dirty なら refresh、 cached path 経由で impl 呼び (= 真 cached path)。
        if accum.dirty {
            accum.refresh(pos);
        }
        self.evaluate_with_bucket_impl(pos, acc, bucket_index, Some(accum), None)
    }

    /// 評価値計算（事前計算済み bucket index を使用）
    ///
    /// `pos` を `&mut` で受けるのは旧 LMAS bit 18/19 (mate threat、 廃止済) の
    /// API 互換性のため。 現状は pin_head_eval が `&Position` を読むだけで state 不変。
    ///
    /// 2026-06-06: activity_risk が enable された場合、 per-eval full compute (= per-eval ~10 μs overhead)。
    /// NPS 改善版は `evaluate_with_bucket_and_activity_risk` で SearchWorker が
    /// AccumulatorStack 経由で accum を渡す。
    pub fn evaluate_with_bucket(
        &self,
        pos: &mut Position,
        acc: &AccumulatorLayerStacks<L1>,
        bucket_index: usize,
    ) -> Value {
        self.evaluate_with_bucket_impl(
            pos,
            acc,
            bucket_index,
            #[cfg(feature = "nnue-activity-risk-psqt")]
            None,
            None,
        )
    }

    /// 評価値を head 別に分解して計算する (= 2026-06-09 診断用)。
    ///
    /// `evaluate_with_bucket` と数値 bit-exact 一致、 内部 logic を共有。
    /// hot path での使用は想定外 (= production binary では呼ばない)、 bucket 境界 jump
    /// 計測等の診断 tool 用途。
    ///
    /// 返り値の `total_cp` は `evaluate_with_bucket(pos, acc, bucket_index).0` と一致する。
    pub fn evaluate_with_breakdown(
        &self,
        pos: &mut Position,
        acc: &AccumulatorLayerStacks<L1>,
        bucket_index: usize,
    ) -> HeadBreakdown {
        let mut breakdown = HeadBreakdown::default();
        let _ = self.evaluate_with_bucket_impl(
            pos,
            acc,
            bucket_index,
            #[cfg(feature = "nnue-activity-risk-psqt")]
            None,
            Some(&mut breakdown),
        );
        breakdown
    }

    /// 評価値を計算 (= activity_risk cached path 経由、 2026-06-06)
    ///
    /// `bucket_index` を内部で計算 + `evaluate_with_bucket_and_activity_risk` を呼ぶ wrapper。
    /// SearchWorker / NNUEEvaluator が AccumulatorStack 経由で accum を渡す想定。
    #[cfg(feature = "nnue-activity-risk-psqt")]
    pub fn evaluate_with_activity_risk(
        &self,
        pos: &mut Position,
        acc: &AccumulatorLayerStacks<L1>,
        accum: &mut crate::nnue::activity_risk::ActivityRiskAccumulator,
    ) -> Value {
        let side_to_move = pos.side_to_move();
        let bucket_index = compute_layer_stacks_bucket_index(pos, side_to_move);
        self.evaluate_with_bucket_and_activity_risk(pos, acc, bucket_index, accum)
    }

    /// 評価値計算 impl 本体 (= body 1 つ、 cfg-gated 引数 activity_risk_accum)。
    ///
    /// `evaluate_with_bucket` (= None 固定) と `evaluate_with_bucket_and_activity_risk`
    /// (= Some 渡し) が呼ぶ共通実装。 cached path 効果はここで実現。
    ///
    /// `breakdown_out` に `Some` を渡すと、 各 head の eval 寄与と raw_score を fill する
    /// (= 2026-06-09 診断用、 hot path では `None` 渡しで overhead 0)。
    fn evaluate_with_bucket_impl(
        &self,
        pos: &mut Position,
        acc: &AccumulatorLayerStacks<L1>,
        bucket_index: usize,
        #[cfg(feature = "nnue-activity-risk-psqt")] activity_risk_accum: Option<
            &mut crate::nnue::activity_risk::ActivityRiskAccumulator,
        >,
        breakdown_out: Option<&mut HeadBreakdown>,
    ) -> Value {
        let side_to_move = pos.side_to_move();

        // SqrClippedReLU変換
        let (us_acc, them_acc) = if side_to_move == Color::Black {
            (acc.get(Color::Black as usize), acc.get(Color::White as usize))
        } else {
            (acc.get(Color::White as usize), acc.get(Color::Black as usize))
        };

        // SAFETY: 直後のsqr_clipped_relu_transformで全要素が上書きされる
        let mut transformed: Aligned<[u8; L1]> = unsafe { Aligned::new_uninit() };

        // Threat の寄与を含めて combined accumulator を構築する。
        // 無効なら piece_acc を直接 SCReLU に渡す。
        // PLAN_C_ABLATE=threat で combined 計算を skip (= piece_acc 直接使用)。
        // KZT-FT 有効ビルド: main (+ threat) (+ kzt) を owned buffer に合成して SCReLU。
        // どちらの寄与も不要なら piece_acc を直接 SCReLU (= 従来 path)。
        #[cfg(feature = "nnue-kzt-ft")]
        {
            let need_threat: bool;
            #[cfg(feature = "nnue-threat")]
            {
                need_threat = self.feature_transformer.has_threat
                    && !crate::nnue::ablation::skip_threat_update();
            }
            #[cfg(not(feature = "nnue-threat"))]
            {
                need_threat = false;
            }

            if need_threat || self.feature_transformer.has_kzt_ft {
                let mut us_combined = Aligned([0i16; L1]);
                let mut them_combined = Aligned([0i16; L1]);
                us_combined.0.copy_from_slice(us_acc);
                them_combined.0.copy_from_slice(them_acc);

                #[cfg(feature = "nnue-threat")]
                if need_threat {
                    let (us_t, them_t) = if side_to_move == Color::Black {
                        (
                            acc.get_threat(Color::Black as usize),
                            acc.get_threat(Color::White as usize),
                        )
                    } else {
                        (
                            acc.get_threat(Color::White as usize),
                            acc.get_threat(Color::Black as usize),
                        )
                    };
                    let mut tu = Aligned([0i16; L1]);
                    let mut tt = Aligned([0i16; L1]);
                    add_i16_arrays::<L1>(&mut tu.0, &us_combined.0, us_t);
                    add_i16_arrays::<L1>(&mut tt.0, &them_combined.0, them_t);
                    us_combined = tu;
                    them_combined = tt;
                }

                if self.feature_transformer.has_kzt_ft {
                    let (us_k, them_k) = if side_to_move == Color::Black {
                        (acc.get_kzt(Color::Black as usize), acc.get_kzt(Color::White as usize))
                    } else {
                        (acc.get_kzt(Color::White as usize), acc.get_kzt(Color::Black as usize))
                    };
                    let mut ku = Aligned([0i16; L1]);
                    let mut kt = Aligned([0i16; L1]);
                    add_i16_arrays::<L1>(&mut ku.0, &us_combined.0, us_k);
                    add_i16_arrays::<L1>(&mut kt.0, &them_combined.0, them_k);
                    us_combined = ku;
                    them_combined = kt;
                }

                sqr_clipped_relu_transform(&us_combined.0, &them_combined.0, &mut transformed.0);
            } else {
                sqr_clipped_relu_transform(us_acc, them_acc, &mut transformed.0);
            }
        }
        // KZT-FT 無効ビルド: 従来の threat 経路をそのまま維持 (= production bit-exact)。
        #[cfg(not(feature = "nnue-kzt-ft"))]
        {
            #[cfg(feature = "nnue-threat")]
            {
                if self.feature_transformer.has_threat
                    && !crate::nnue::ablation::skip_threat_update()
                {
                    let mut us_combined = Aligned([0i16; L1]);
                    let mut them_combined = Aligned([0i16; L1]);
                    us_combined.0.copy_from_slice(us_acc);
                    them_combined.0.copy_from_slice(them_acc);

                    let (us_t, them_t) = if side_to_move == Color::Black {
                        (
                            acc.get_threat(Color::Black as usize),
                            acc.get_threat(Color::White as usize),
                        )
                    } else {
                        (
                            acc.get_threat(Color::White as usize),
                            acc.get_threat(Color::Black as usize),
                        )
                    };
                    let mut tmp_us = Aligned([0i16; L1]);
                    let mut tmp_them = Aligned([0i16; L1]);
                    add_i16_arrays::<L1>(&mut tmp_us.0, &us_combined.0, us_t);
                    add_i16_arrays::<L1>(&mut tmp_them.0, &them_combined.0, them_t);
                    us_combined = tmp_us;
                    them_combined = tmp_them;

                    sqr_clipped_relu_transform(
                        &us_combined.0,
                        &them_combined.0,
                        &mut transformed.0,
                    );
                } else {
                    sqr_clipped_relu_transform(us_acc, them_acc, &mut transformed.0);
                }
            }
            #[cfg(not(feature = "nnue-threat"))]
            {
                sqr_clipped_relu_transform(us_acc, them_acc, &mut transformed.0);
            }
        }

        // LayerStacks で評価
        let raw_score = self.layer_stacks.evaluate_raw(bucket_index, &transformed.0);

        // PSQT ショートカット (Stockfish 準拠: (stm - nstm) / 2)
        // 各駒は両視点に逆符号で寄与するため、stm - nstm は正味の配置価値を
        // 約2倍にカウントする。/2 はこの二重カウントを補正する正規化。
        // PLAN_C_ABLATE=psqt で eval 寄与のみ skip (accumulator 更新は維持)。
        #[cfg(feature = "nnue-psqt")]
        let psqt_value = if self.feature_transformer.has_psqt
            && !crate::nnue::ablation::skip_psqt_eval()
        {
            let stm = side_to_move as usize;
            let nstm = (!side_to_move) as usize;
            (acc.psqt_accumulation[stm][bucket_index] - acc.psqt_accumulation[nstm][bucket_index])
                / 2
        } else {
            0
        };
        #[cfg(not(feature = "nnue-psqt"))]
        let psqt_value = 0;

        // Pin-aware head の加算 (fv_scale 適用前 cp 単位)。
        // bucket index は LayerStacks と共有 (NUM_PIN_BUCKETS == NUM_LAYER_STACK_BUCKETS)。
        // 2026-05-25 から accumulator 化: `pos` を再走査せず
        // `acc.pin_accumulation[persp][bucket]` を直接読む (= -20.5% NPS 対策)。
        // 数値は旧 `pin_head_eval` と bit-exact (= i32 add 順序のみ変更)。
        // PLAN_C_ABLATE=pin で acc 更新と eval 加算の両方を skip。
        #[cfg(feature = "nnue-pin-aware")]
        let pin_value: i32 = if self.pin_weights.is_some() {
            if crate::nnue::ablation::skip_pin_eval() {
                0
            } else {
                let black_sum = acc.pin_accumulation[Color::Black as usize][bucket_index];
                let white_sum = acc.pin_accumulation[Color::White as usize][bucket_index];
                let score = match side_to_move {
                    Color::Black => black_sum.saturating_sub(white_sum),
                    Color::White => white_sum.saturating_sub(black_sum),
                };
                score / crate::nnue::pin_features::PIN_OUTPUT_SCALE
            }
        } else {
            0
        };
        #[cfg(not(feature = "nnue-pin-aware"))]
        let pin_value: i32 = 0;

        // Dual PSQT (King-Attacker) head の加算。
        // 各 perspective × 自軍駒の利き ∩ {自/敵 king zone} で attack/defend 判定し、
        // weight lookup + 集計 → (attack - defend) を fv_scale 前 cp として加算。
        //
        // 2026-05-26 から accumulator 化 (Phase 3b): `pos` を再走査せず
        // `acc.dual_accumulation[persp][head][bucket]` を直接読む。 数値は旧
        // `compute_dual_king_attacker` と bit-exact (= i32 add 順序のみ変更)。
        // PLAN_C_ABLATE=dual で acc 更新と eval 加算の両方を skip。
        #[cfg(feature = "nnue-dual-psqt")]
        let dual_psqt_value: i32 = if self.dual_psqt_weights.is_some() {
            if crate::nnue::ablation::skip_dual_eval() {
                0
            } else {
                let attack_b = acc.dual_accumulation[Color::Black as usize][0][bucket_index];
                let defend_b = acc.dual_accumulation[Color::Black as usize][1][bucket_index];
                let attack_w = acc.dual_accumulation[Color::White as usize][0][bucket_index];
                let defend_w = acc.dual_accumulation[Color::White as usize][1][bucket_index];
                let attack_total = attack_b.saturating_add(attack_w);
                let defend_total = defend_b.saturating_add(defend_w);
                attack_total
                    .saturating_sub(defend_total)
                    .saturating_div(crate::nnue::dual_psqt::DUAL_OUTPUT_SCALE)
            }
        } else {
            0
        };
        #[cfg(not(feature = "nnue-dual-psqt"))]
        let dual_psqt_value: i32 = 0;

        // Dual Cross (D2: piece_threat_relation) head 0 のみ加算 (= arm 6 で head 1 廃止)。
        // weight 学習で sign 込み済み (= bullet 側 graph forward は加算合成)。
        //
        // arm 7 (Path J、 2026-05-27): non-PV node 限定で eval 寄与を skip (= PSQT only
        // 相当の高速 eval)。 thread-local `NON_PV_DUAL_SKIP` が true の間、 dual_cross
        // accumulator の read を完全に省く (= 2 i32 add + 1 div を 0 に)。 update 側も
        // 同じフラグで skip するため stale 値の伝播は問題視しない (= 意図的 eval drift)。
        #[cfg(all(
            feature = "nnue-dual-psqt",
            any(
                not(feature = "nnue-scalar-threat"),
                feature = "nnue-scalar-threat-coexist"
            )
        ))]
        let dual_cross_value: i32 = if self.dual_cross_weights.is_some() {
            // arm 6 endgame-only (= 2026-05-28): `DUAL_CROSS_ENDGAME_ONLY=1` (= ply-based)
            // または `DUAL_CROSS_USE_ENDGAME_DETECTOR=1` (= position-based、 USI 側 USI go
            // で `set_current_phase_idx` 経由で書き込まれた phase index 参照) で
            // opening/middle 時は dual_cross 寄与を 0 に gate。 同時に update 側も
            // skip 経路で stale 化されているため、 stale 値を読む path は存在しない
            // (= PSQT only 相当の高速 eval)。
            let endgame_only_skip = (crate::nnue::ablation::dual_cross_endgame_only()
                && !crate::nnue::dual_cross::is_endgame_ply(pos.game_ply()))
                || (crate::nnue::ablation::dual_cross_use_endgame_detector()
                    && !crate::nnue::dual_cross::current_phase_is_endgame());
            if crate::nnue::ablation::skip_dual_eval()
                || crate::nnue::ablation::non_pv_dual_skip()
                || endgame_only_skip
            {
                0
            } else {
                let mut total: i32 = 0;
                for persp in 0..2 {
                    total = total.saturating_add(acc.dual_cross_accumulation[persp][bucket_index]);
                }
                total.saturating_div(crate::nnue::dual_cross::DUAL_CROSS_OUTPUT_SCALE)
            }
        } else {
            0
        };
        #[cfg(any(
            not(feature = "nnue-dual-psqt"),
            all(
                feature = "nnue-scalar-threat",
                not(feature = "nnue-scalar-threat-coexist")
            )
        ))]
        let dual_cross_value: i32 = 0;

        #[cfg(feature = "nnue-scalar-threat")]
        let scalar_threat_value: i32 = if self.scalar_threat_weights.is_some() {
            let stm = side_to_move as usize;
            let nstm = (!side_to_move) as usize;
            acc.scalar_threat_accumulation[stm][bucket_index]
                .saturating_sub(acc.scalar_threat_accumulation[nstm][bucket_index])
                .saturating_div(crate::nnue::scalar_threat::output_divisor())
        } else {
            0
        };
        #[cfg(not(feature = "nnue-scalar-threat"))]
        let scalar_threat_value: i32 = 0;

        #[cfg(feature = "nnue-dual-cross-geometry")]
        let dual_cross_geometry_value: i32 = if self.dual_cross_geometry_weights.is_some() {
            let stm = side_to_move as usize;
            let nstm = (!side_to_move) as usize;
            acc.dual_cross_geometry_accumulation[stm][bucket_index]
                .saturating_sub(acc.dual_cross_geometry_accumulation[nstm][bucket_index])
        } else {
            0
        };
        #[cfg(not(feature = "nnue-dual-cross-geometry"))]
        let dual_cross_geometry_value: i32 = 0;

        // KingRing PSQT scalar head (= arm 11、 2026-05-28)。
        // 両 perspective の per-bucket scalar を取り、 stm 視点で (us - them) / scale を加算。
        // PLAN_C_ABLATE=kingring で eval 寄与のみ skip。
        #[cfg(feature = "nnue-kingring-psqt")]
        let kingring_value: i32 = if self.kingring_weights.is_some() {
            if crate::nnue::ablation::skip_kingring_eval() {
                0
            } else {
                let stm = side_to_move as usize;
                let nstm = (!side_to_move) as usize;
                let us = acc.kingring_accumulation[stm][bucket_index];
                let them = acc.kingring_accumulation[nstm][bucket_index];
                us.saturating_sub(them)
                    .saturating_div(crate::nnue::kingring::KINGRING_OUTPUT_SCALE)
            }
        } else {
            0
        };
        #[cfg(not(feature = "nnue-kingring-psqt"))]
        let kingring_value: i32 = 0;

        // PairShape PSQT scalar head (= arm 12、 2026-05-29)。
        // stm 視点で (us - them) / scale を加算 (= PSQT 同様の符号化)。
        // PLAN_C_ABLATE=pairshape で eval 寄与のみ skip。
        #[cfg(feature = "nnue-pairshape-psqt")]
        let pairshape_value: i32 = if self.pairshape_weights.is_some() {
            if crate::nnue::ablation::skip_pairshape_eval() {
                0
            } else {
                let stm = side_to_move as usize;
                let nstm = (!side_to_move) as usize;
                let us = acc.pairshape_accumulation[stm][bucket_index];
                let them = acc.pairshape_accumulation[nstm][bucket_index];
                us.saturating_sub(them)
                    .saturating_mul(crate::nnue::pairshape::PAIRSHAPE_OUTPUT_SCALE)
            }
        } else {
            0
        };
        #[cfg(not(feature = "nnue-pairshape-psqt"))]
        let pairshape_value: i32 = 0;

        // Activity-Risk Interaction Head (= 2026-06-06、 中盤楽観バイアス補正)。
        // 各 perspective で activity / pressure / shield を full compute、
        // (black - white) × OUTPUT_SCALE を fv_scale 適用前 cp として加算。
        // initial OUTPUT_SCALE = 1、 weight 全 0 で eval bit-exact 一致 (= ckpt 互換)。
        // 注意: 初版 per-eval full compute (= NPS 影響を要計測、 PairShape 比 97% 未満なら
        // 差分更新案を検討する)。
        #[cfg(feature = "nnue-activity-risk-psqt")]
        let activity_risk_value: i32 = if let Some(w) = self.activity_risk_weights.as_ref() {
            // v3: per-eval full compute path のみ採用 (= cached path は base 4096 weight のみ対応、
            // v3 tactical 1024 weight の incremental update は別 step)。 activity_risk_accum は
            // v3 では一旦使用しない (= dirty=true で full compute、 既存 cached path 経由しない)。
            let _ = &activity_risk_accum;
            crate::nnue::activity_risk::evaluate_activity_risk_v3(pos, w.as_ref(), bucket_index)
        } else {
            0
        };
        #[cfg(not(feature = "nnue-activity-risk-psqt"))]
        let activity_risk_value: i32 = 0;

        // King Escape Direction (KED) PSQT scalar head (= 2026-06-09、 入玉対応)。
        // 各 perspective で 玉 forward 同一 file 1-3 マス先の自軍利き 0..=3 を
        // class 化し、 (us - them) × OUTPUT_SCALE を fv_scale 適用前 cp として加算。
        // weight 全 0 で eval bit-exact (= 旧 ckpt 互換)。
        #[cfg(feature = "nnue-king-escape-psqt")]
        let king_escape_value: i32 = if let Some(rt) = self.king_escape_runtime_weights.as_deref() {
            crate::nnue::king_escape::evaluate_king_escape(rt, pos, bucket_index, side_to_move)
        } else {
            0
        };
        #[cfg(not(feature = "nnue-king-escape-psqt"))]
        let king_escape_value: i32 = 0;

        // Attack Relay Head (ARH) scalar head (= 2026-06-09)。
        // 各 perspective で 敵玉中心 5x5 を inner / outer に分け、
        // (contact, reserve, relay) 3 軸 4×4×4 = 64 cell × 8 bucket × 2 persp = 1024 weight。
        // stm 視点 `(us - them) × OUTPUT_SCALE` を fv_scale 適用前 cp として加算。
        #[cfg(feature = "nnue-attack-relay")]
        let attack_relay_value: i32 = if let Some(rt) = self.attack_relay_runtime_weights.as_deref()
        {
            crate::nnue::attack_relay::evaluate_arh(rt, pos, bucket_index, side_to_move)
        } else {
            0
        };
        #[cfg(not(feature = "nnue-attack-relay"))]
        let attack_relay_value: i32 = 0;

        // ARH v2 (= A5 仕様、 2026-06-10) scalar head。
        #[cfg(feature = "nnue-attack-relay-v2")]
        let attack_relay_v2_value: i32 =
            if let Some(rt) = self.attack_relay_v2_runtime_weights.as_deref() {
                crate::nnue::attack_relay_v2::evaluate_arh_v2(rt, pos, bucket_index, side_to_move)
            } else {
                0
            };
        #[cfg(not(feature = "nnue-attack-relay-v2"))]
        let attack_relay_v2_value: i32 = 0;

        // King Danger (KDGR) PSQT scalar head (= 2026-06-15、 自玉危険補正)。
        // 各 perspective で 自玉 3x3 内の敵駒重み付き数を 0..=7 class 化し、
        // (us - them) × OUTPUT_SCALE を fv_scale 適用前 cp として加算。
        // weight 全 0 で eval bit-exact (= 旧 ckpt 互換、 無害)。
        #[cfg(feature = "nnue-king-danger-psqt")]
        let king_danger_value: i32 = if let Some(rt) = self.king_danger_runtime_weights.as_deref() {
            crate::nnue::king_danger::evaluate_king_danger(rt, pos, bucket_index, side_to_move)
        } else {
            0
        };
        #[cfg(not(feature = "nnue-king-danger-psqt"))]
        let king_danger_value: i32 = 0;

        let fv_scale = get_fv_scale_override().unwrap_or(self.fv_scale);

        // 解析的 king_danger 補正 (= 2026-06-16、 訓練重み非依存)。
        // 自玉 3x3 の生重み付き危険度差 × 固定係数を **fv_scale 適用前 raw 単位**で加算。
        // `KING_DANGER_ANALYTIC_COEF` (= default 0) が 0 の間は常に 0 で bit-exact 互換。
        // cp 寄与 = -COEF × king_danger3_diff (= 残差 gap を方向保証で埋める)。
        #[cfg(feature = "nnue-king-danger-psqt")]
        let king_danger_analytic_value: i32 =
            crate::nnue::king_danger::analytic_king_danger_raw(pos, side_to_move, fv_scale);
        #[cfg(not(feature = "nnue-king-danger-psqt"))]
        let king_danger_analytic_value: i32 = 0;

        // slider 拡張: 玉 3x3 に届く敵 slider の block-aware 利きによる危険度差 ×
        // 固定係数を **fv_scale 適用前 raw 単位**で加算。
        // `KING_DANGER_SLIDER_ANALYTIC_COEF` (= default 0) が 0 の間は常に 0 で bit-exact 互換。
        // cp 寄与 = -COEF × king_danger_slider_diff (= 遠隔 slider の自玉危険を方向保証で補正)。
        #[cfg(feature = "nnue-king-danger-psqt")]
        let king_danger_slider_analytic_value: i32 =
            crate::nnue::king_danger::analytic_king_danger_slider_raw(pos, side_to_move, fv_scale);
        #[cfg(not(feature = "nnue-king-danger-psqt"))]
        let king_danger_slider_analytic_value: i32 = 0;

        // slider 守備のみ (= 非対称) 版: 自玉 3x3 に届く敵 slider の危険度 **のみ** ×
        // 固定係数を **fv_scale 適用前 raw 単位**で加算 (= 攻めでブーストしない純悲観)。
        // `KING_DANGER_SLIDER_DEF_ANALYTIC_COEF` (= default 0) が 0 の間は常に 0 で bit-exact 互換。
        // cp 寄与 = -COEF × king_danger_slider_weighted_sum(stm) (= 自玉危険のみ補正)。
        #[cfg(feature = "nnue-king-danger-psqt")]
        let king_danger_slider_def_analytic_value: i32 =
            crate::nnue::king_danger::analytic_king_danger_slider_def_raw(
                pos,
                side_to_move,
                fv_scale,
            );
        #[cfg(not(feature = "nnue-king-danger-psqt"))]
        let king_danger_slider_def_analytic_value: i32 = 0;

        // Attack Development (ADEV) PSQT scalar head (= 2026-06-16、 相手の発展中の攻め補正)。
        // 各 perspective で 自玉に対する敵 slider の照準を 0..=7 class 化し、
        // (us - them) × OUTPUT_SCALE を fv_scale 適用前 cp として加算。
        // weight 全 0 で eval bit-exact (= 旧 ckpt 互換、 無害)。
        #[cfg(feature = "nnue-attack-dev-psqt")]
        let attack_dev_value: i32 = if let Some(rt) = self.attack_dev_runtime_weights.as_deref() {
            crate::nnue::attack_dev::evaluate_attack_dev(rt, pos, bucket_index, side_to_move)
        } else {
            0
        };
        #[cfg(not(feature = "nnue-attack-dev-psqt"))]
        let attack_dev_value: i32 = 0;

        // Slider X-ray (SXRY) PSQT scalar head (= 2026-06-28、 敵 slider 潜在照準補正)。
        // 各 perspective で 自玉への敵 slider X-ray (= between 1 枚) を 0..=7 class 化し、
        // (us - them) × OUTPUT_SCALE を fv_scale 適用前 cp として加算。
        // weight 全 0 で eval bit-exact (= 旧 ckpt 互換、 無害)。
        #[cfg(feature = "nnue-slider-xray-psqt")]
        let slider_xray_value: i32 = if let Some(rt) = self.slider_xray_runtime_weights.as_deref() {
            crate::nnue::slider_xray::evaluate_slider_xray(rt, pos, bucket_index, side_to_move)
        } else {
            0
        };
        #[cfg(not(feature = "nnue-slider-xray-psqt"))]
        let slider_xray_value: i32 = 0;

        // King Zone Threat / 持駒打ち込み (KZT_HAND) sparse head (= 2026-06-30)。
        // 各 perspective で 自分の持駒で 敵玉 5x5 zone を攻める脅威 feature を sparse gather し、
        // (us - them) × OUTPUT_SCALE を fv_scale 適用前 cp として加算。
        // weight 全 0 / scale 0 / section 不在 のいずれでも eval bit-exact (= 旧 ckpt 互換、 無害)。
        #[cfg(feature = "nnue-kzt-psqt")]
        let kzt_hand_value: i32 = if let Some(w) = self.kzt_hand_weights.as_deref() {
            crate::nnue::kzt::evaluate_kzt_hand(w, pos, bucket_index, side_to_move)
        } else {
            0
        };
        #[cfg(not(feature = "nnue-kzt-psqt"))]
        let kzt_hand_value: i32 = 0;

        // King Mobility (KMOB) PSQT scalar head (= 2026-06-16、 自玉の安全逃げ場 = 守備系)。
        // 各 perspective で 自玉の安全な逃げ場マス数を 0..=8 の 9 class 化し、
        // (us - them) × OUTPUT_SCALE を fv_scale 適用前 cp として加算。
        // weight 全 0 で eval bit-exact (= 旧 ckpt 互換、 無害)。
        #[cfg(feature = "nnue-king-mobility-psqt")]
        let king_mobility_value: i32 = if let Some(rt) =
            self.king_mobility_runtime_weights.as_deref()
        {
            crate::nnue::king_mobility::evaluate_king_mobility(rt, pos, bucket_index, side_to_move)
        } else {
            0
        };
        #[cfg(not(feature = "nnue-king-mobility-psqt"))]
        let king_mobility_value: i32 = 0;

        // Drop Pressure (DROP) PSQT scalar head (= 2026-06-16、 相手持駒投下脅威補正)。
        // 各 perspective で 自玉近傍 5x5 への相手持駒投下脅威を 0..=7 class 化し、
        // (us - them) × OUTPUT_SCALE を fv_scale 適用前 cp として加算。
        // weight 全 0 で eval bit-exact (= 旧 ckpt 互換、 無害)。
        #[cfg(feature = "nnue-drop-pressure-psqt")]
        let drop_pressure_value: i32 = if let Some(rt) =
            self.drop_pressure_runtime_weights.as_deref()
        {
            crate::nnue::drop_pressure::evaluate_drop_pressure(rt, pos, bucket_index, side_to_move)
        } else {
            0
        };
        #[cfg(not(feature = "nnue-drop-pressure-psqt"))]
        let drop_pressure_value: i32 = 0;

        // kingring_value 分布計測用 log (= `LOG_KINGRING_VALUE=1` env + USI `eval` 経路内のみ)。
        // hot path では `log_kingring_request_active()` が早期 false で 1 atomic load
        // + 1 thread_local load + branch のみ。 通常 evaluate には影響しない。
        //
        // 出力 1 行 (= info string):
        //   "info string KINGRING_VALUE bucket=B us=U them=T raw=R cp=C \
        //    psqt_raw=P psqt_cp=PC dual_psqt_raw=D dual_cross_raw=X pin_raw=I fv_scale=F"
        //
        // raw = fv_scale 適用前 (= 各 head の score を直接 sum)、 cp = raw / fv_scale。
        // SFEN scan script は grep "KINGRING_VALUE" で抽出 → Python で集計可能。
        if crate::nnue::ablation::log_kingring_request_active() {
            #[cfg(feature = "nnue-kingring-psqt")]
            let (kr_us, kr_them) = if self.kingring_weights.is_some() {
                let stm = side_to_move as usize;
                let nstm = (!side_to_move) as usize;
                (
                    acc.kingring_accumulation[stm][bucket_index],
                    acc.kingring_accumulation[nstm][bucket_index],
                )
            } else {
                (0i32, 0i32)
            };
            #[cfg(not(feature = "nnue-kingring-psqt"))]
            let (kr_us, kr_them) = (0i32, 0i32);

            // cp 換算 (= fv_scale で割る、 0 除算 ガード)。
            let kr_cp = if fv_scale > 0 {
                kingring_value / fv_scale
            } else {
                0
            };
            let psqt_cp = if fv_scale > 0 {
                psqt_value / fv_scale
            } else {
                0
            };
            // pairshape diag (= arm 12 boost で eval scale 破壊調査用、 2026-05-30)
            #[cfg(feature = "nnue-pairshape-psqt")]
            let (ps_us, ps_them) = if self.pairshape_weights.is_some() {
                let stm = side_to_move as usize;
                let nstm = (!side_to_move) as usize;
                (
                    acc.pairshape_accumulation[stm][bucket_index],
                    acc.pairshape_accumulation[nstm][bucket_index],
                )
            } else {
                (0i32, 0i32)
            };
            #[cfg(not(feature = "nnue-pairshape-psqt"))]
            let (ps_us, ps_them) = (0i32, 0i32);

            let ps_cp = if fv_scale > 0 {
                pairshape_value / fv_scale
            } else {
                0
            };

            // raw_score (= L1/L2/L3 main path 出力、 fv_scale 適用前 cp 単位) も同時 dump。
            let raw_cp = if fv_scale > 0 {
                raw_score / fv_scale
            } else {
                0
            };

            println!(
                "info string KINGRING_VALUE bucket={} us={} them={} raw={} cp={} psqt_raw={} psqt_cp={} dual_psqt_raw={} dual_cross_raw={} pin_raw={} pairshape_us={} pairshape_them={} pairshape_raw={} pairshape_cp={} main_raw={} main_cp={} fv_scale={}",
                bucket_index,
                kr_us,
                kr_them,
                kingring_value,
                kr_cp,
                psqt_value,
                psqt_cp,
                dual_psqt_value,
                dual_cross_value,
                pin_value,
                ps_us,
                ps_them,
                pairshape_value,
                ps_cp,
                raw_score,
                raw_cp,
                fv_scale,
            );
        }

        let raw_total = raw_score
            .saturating_add(psqt_value)
            .saturating_add(pin_value)
            .saturating_add(dual_psqt_value)
            .saturating_add(dual_cross_value)
            .saturating_add(scalar_threat_value)
            .saturating_add(dual_cross_geometry_value)
            .saturating_add(kingring_value)
            .saturating_add(pairshape_value)
            .saturating_add(activity_risk_value)
            .saturating_add(king_escape_value)
            .saturating_add(attack_relay_value)
            .saturating_add(attack_relay_v2_value)
            .saturating_add(king_danger_value)
            .saturating_add(king_danger_analytic_value)
            .saturating_add(king_danger_slider_analytic_value)
            .saturating_add(king_danger_slider_def_analytic_value)
            .saturating_add(attack_dev_value)
            .saturating_add(king_mobility_value)
            .saturating_add(drop_pressure_value)
            .saturating_add(slider_xray_value)
            .saturating_add(kzt_hand_value);

        // ODH (Optimism Damp Head) v0 (= 2026-07-02、 解析注入 hand-set 版)。
        // 他 head / psqt を全部足した合算 raw_total (= main_raw) に **最後に** 適用し、
        // 正側 (= 優勢) の eval 絶対値を damp[class] で縮める (= 符号反転しない)。
        // `ODH_OUTPUT_SCALE` (= default 0) が 0 の間は常に 0 で bit-exact 互換。
        #[cfg(feature = "nnue-odh")]
        let odh_value: i32 =
            crate::nnue::odh::apply_odh(raw_total, pos, side_to_move, bucket_index, fv_scale);
        #[cfg(not(feature = "nnue-odh"))]
        let odh_value: i32 = 0;
        let raw_total = raw_total.saturating_sub(odh_value);

        // Eval Bias Correction (EBC) (= 2026-07-03、 correction-history 型静的補正)。
        // ODH 適用後の raw_total (= 補正前 cp 判定に使う) に、 局面 class 別の bias 補正を
        // **符号込み加算** する (= ODH と別機構、 優勢下げ / 劣勢上げ 両方可能)。
        // `EVAL_BIAS_CORR_SCALE` (= default 0) が 0 の間は常に 0 で bit-exact 互換。
        #[cfg(feature = "nnue-eval-bias-corr")]
        let ebc_value: i32 =
            crate::nnue::eval_bias_corr::apply(raw_total, pos.game_ply(), fv_scale);
        #[cfg(not(feature = "nnue-eval-bias-corr"))]
        let ebc_value: i32 = 0;
        let raw_total = raw_total.saturating_add(ebc_value);

        let total_cp = raw_total / fv_scale;

        if let Some(bd) = breakdown_out {
            bd.bucket_index = bucket_index;
            bd.fv_scale = fv_scale;
            bd.raw_score = raw_score;
            bd.psqt_value = psqt_value;
            bd.pin_value = pin_value;
            bd.dual_psqt_value = dual_psqt_value;
            bd.dual_cross_value = dual_cross_value;
            bd.kingring_value = kingring_value;
            bd.pairshape_value = pairshape_value;
            bd.activity_risk_value = activity_risk_value;
            bd.king_escape_value = king_escape_value;
            bd.attack_relay_value = attack_relay_value;
            bd.attack_relay_v2_value = attack_relay_v2_value;
            bd.king_danger_value = king_danger_value;
            bd.king_danger_analytic_value = king_danger_analytic_value;
            bd.king_danger_slider_analytic_value = king_danger_slider_analytic_value;
            bd.king_danger_slider_def_analytic_value = king_danger_slider_def_analytic_value;
            bd.attack_dev_value = attack_dev_value;
            bd.king_mobility_value = king_mobility_value;
            bd.drop_pressure_value = drop_pressure_value;
            bd.slider_xray_value = slider_xray_value;
            bd.kzt_hand_value = kzt_hand_value;
            // 減衰 head のため負の寄与として格納 (= raw_total_pre_scale 整合)。
            bd.odh_value = -odh_value;
            // EBC は符号込み加算量 (= そのまま寄与、 raw_total_pre_scale 整合)。
            bd.ebc_value = ebc_value;
            bd.total_cp = total_cp;
        }

        Value::new(total_cp)
    }

    /// 評価値を計算（詳細診断ログ付き）
    ///
    /// Python (nnue-pytorch) との比較検証用。
    /// 各中間値をログ出力する。
    #[cfg(feature = "diagnostics")]
    pub fn evaluate_with_diagnostics(
        &self,
        pos: &Position,
        acc: &AccumulatorLayerStacks<L1>,
    ) -> Value {
        use log::info;

        let side_to_move = pos.side_to_move();

        // アキュムレータの統計
        let (us_acc, them_acc) = if side_to_move == Color::Black {
            (acc.get(Color::Black as usize), acc.get(Color::White as usize))
        } else {
            (acc.get(Color::White as usize), acc.get(Color::Black as usize))
        };

        // us_acc の統計
        let us_min = us_acc.iter().copied().min().unwrap_or(0);
        let us_max = us_acc.iter().copied().max().unwrap_or(0);
        let us_first_half_positive: usize = us_acc[0..L1 / 2].iter().filter(|&&x| x > 0).count();
        let us_second_half_positive: usize = us_acc[L1 / 2..L1].iter().filter(|&&x| x > 0).count();

        info!("[NNUE Eval] us_acc: min={us_min}, max={us_max}");
        let half = L1 / 2;
        info!(
            "[NNUE Eval] us_acc positive: first_half={us_first_half_positive}/{half}, second_half={us_second_half_positive}/{half}"
        );
        info!("[NNUE Eval] us_acc (piece) first 16: {:?}", &us_acc[0..16]);

        // Threat 結合 (evaluate_with_bucket と同一ロジック)
        let mut transformed: Aligned<[u8; L1]> = Aligned([0u8; L1]);
        #[cfg(feature = "nnue-threat")]
        {
            if self.feature_transformer.has_threat {
                let mut us_combined = [0i16; L1];
                let mut them_combined = [0i16; L1];
                us_combined.copy_from_slice(us_acc);
                them_combined.copy_from_slice(them_acc);

                let (us_t, them_t) = if side_to_move == Color::Black {
                    (acc.get_threat(Color::Black as usize), acc.get_threat(Color::White as usize))
                } else {
                    (acc.get_threat(Color::White as usize), acc.get_threat(Color::Black as usize))
                };
                info!("[NNUE Eval] us_threat first 16: {:?}", &us_t[0..16]);
                for i in 0..L1 {
                    us_combined[i] = us_combined[i].wrapping_add(us_t[i]);
                    them_combined[i] = them_combined[i].wrapping_add(them_t[i]);
                }

                info!("[NNUE Eval] us_combined (piece+threat) first 16: {:?}", &us_combined[0..16]);
                sqr_clipped_relu_transform(&us_combined, &them_combined, &mut transformed.0);
            } else {
                sqr_clipped_relu_transform(us_acc, them_acc, &mut transformed.0);
            }
        }
        #[cfg(not(feature = "nnue-threat"))]
        {
            sqr_clipped_relu_transform(us_acc, them_acc, &mut transformed.0);
        }

        let transformed_nonzero: usize = transformed.0.iter().filter(|&&x| x > 0).count();
        let transformed_sum: u64 = transformed.0.iter().map(|&x| x as u64).sum();
        info!("[NNUE Eval] transformed: nonzero={transformed_nonzero}/{L1}, sum={transformed_sum}");
        info!("[NNUE Eval] transformed first 32: {:?}", &transformed.0[0..32]);

        // バケットインデックスを計算（通常パスと同じ共通関数を使用）
        let bucket_index = compute_layer_stacks_bucket_index(pos, side_to_move);
        info!(
            "[NNUE Eval] bucket_mode={:?}, bucket_index={bucket_index}",
            get_layer_stack_bucket_mode()
        );

        // LayerStacks で評価（詳細ログ付き）
        let (raw_score, l1_out, l1_skip) =
            self.layer_stacks.evaluate_raw_with_diagnostics(bucket_index, &transformed.0);

        info!("[NNUE Eval] l1_out (16 elements): {l1_out:?}");
        info!("[NNUE Eval] l1_skip: {l1_skip}");
        info!("[NNUE Eval] raw_score (with skip): {raw_score}");

        // PSQT ショートカット
        #[cfg(feature = "nnue-psqt")]
        let psqt_value = if self.feature_transformer.has_psqt {
            let stm = side_to_move as usize;
            let nstm = (!side_to_move) as usize;
            let v = (acc.psqt_accumulation[stm][bucket_index]
                - acc.psqt_accumulation[nstm][bucket_index])
                / 2;
            info!(
                "[NNUE Eval] psqt_acc[stm][{bucket_index}]: {}",
                acc.psqt_accumulation[stm][bucket_index]
            );
            info!(
                "[NNUE Eval] psqt_acc[nstm][{bucket_index}]: {}",
                acc.psqt_accumulation[nstm][bucket_index]
            );
            info!("[NNUE Eval] psqt_value: {v}");
            v
        } else {
            info!("[NNUE Eval] PSQT: disabled");
            0
        };
        #[cfg(not(feature = "nnue-psqt"))]
        let psqt_value = {
            info!("[NNUE Eval] PSQT: disabled (feature not enabled)");
            0
        };

        let fv_scale = get_fv_scale_override().unwrap_or(self.fv_scale);
        let combined = raw_score.saturating_add(psqt_value);
        let score = combined / fv_scale;
        let score_float = combined as f64 / fv_scale as f64;
        info!("[NNUE Eval] fv_scale: {fv_scale}");
        info!(
            "[NNUE Eval] score: {score} (raw_score={raw_score} + psqt={psqt_value} = {combined}, float: {score_float:.4})"
        );

        Value::new(score)
    }

    /// 差分計算を使わずにAccumulatorを計算
    pub fn refresh_accumulator(&self, pos: &Position, acc: &mut AccumulatorLayerStacks<L1>) {
        self.feature_transformer.refresh_accumulator(pos, acc);
        self.refresh_scalar_threat_accumulator(pos, acc);
        self.refresh_pin_accumulator(pos, acc);
        self.refresh_dual_accumulator(pos, acc);
        self.refresh_dual_cross_accumulator(pos, acc);
        self.refresh_dual_cross_geometry_accumulator(pos, acc);
        self.refresh_kingring_accumulator(pos, acc);
        self.refresh_pairshape_accumulator(pos, acc);
    }

    /// 差分計算を使わずにAccumulatorを計算（キャッシュ使用版）
    pub fn refresh_accumulator_with_cache(
        &self,
        pos: &Position,
        acc: &mut AccumulatorLayerStacks<L1>,
        cache: &mut super::accumulator_layer_stacks::AccumulatorCacheLayerStacks<L1>,
    ) {
        self.feature_transformer.refresh_accumulator_with_cache(pos, acc, cache);
        self.refresh_scalar_threat_accumulator(pos, acc);
        self.refresh_pin_accumulator(pos, acc);
        self.refresh_dual_accumulator(pos, acc);
        self.refresh_dual_cross_accumulator(pos, acc);
        self.refresh_dual_cross_geometry_accumulator(pos, acc);
        self.refresh_kingring_accumulator(pos, acc);
        self.refresh_pairshape_accumulator(pos, acc);
    }

    /// 差分計算でAccumulatorを更新
    pub fn update_accumulator(
        &self,
        pos: &Position,
        dirty_piece: &super::accumulator::DirtyPiece,
        acc: &mut AccumulatorLayerStacks<L1>,
        prev_acc: &AccumulatorLayerStacks<L1>,
    ) {
        self.feature_transformer.update_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_scalar_threat_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_pin_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_dual_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_dual_cross_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_dual_cross_geometry_accumulator(pos, acc, prev_acc);
        self.update_kingring_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_pairshape_accumulator(pos, dirty_piece, acc, prev_acc);
    }

    /// 差分計算でAccumulatorを更新（キャッシュ使用版）
    pub fn update_accumulator_with_cache(
        &self,
        pos: &Position,
        dirty_piece: &super::accumulator::DirtyPiece,
        acc: &mut AccumulatorLayerStacks<L1>,
        prev_acc: &AccumulatorLayerStacks<L1>,
        cache: &mut super::accumulator_layer_stacks::AccumulatorCacheLayerStacks<L1>,
    ) {
        self.feature_transformer.update_accumulator_with_cache(
            pos,
            dirty_piece,
            acc,
            prev_acc,
            cache,
        );
        self.update_scalar_threat_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_pin_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_dual_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_dual_cross_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_dual_cross_geometry_accumulator(pos, acc, prev_acc);
        self.update_kingring_accumulator(pos, dirty_piece, acc, prev_acc);
        self.update_pairshape_accumulator(pos, dirty_piece, acc, prev_acc);
    }

    /// 複数手分の差分を適用してアキュムレータを更新
    pub fn forward_update_incremental(
        &self,
        pos: &Position,
        stack: &mut AccumulatorStackLayerStacks<L1>,
        source_idx: usize,
    ) -> bool {
        let ok = self.feature_transformer.forward_update_incremental(pos, stack, source_idx);
        if ok {
            // pos は現局面の after-state なので full refresh で正しい結果になる。
            // path 長 1+ で中間 dirty_piece chain を pin/dual diff として個別適用するのは
            // 複雑度に見合わない (= active 列挙が piece bitboard walk のみで安価)。
            self.refresh_pin_accumulator(pos, &mut stack.current_mut().accumulator);
            self.refresh_scalar_threat_accumulator(pos, &mut stack.current_mut().accumulator);
            self.refresh_dual_accumulator(pos, &mut stack.current_mut().accumulator);
            self.refresh_dual_cross_accumulator(pos, &mut stack.current_mut().accumulator);
            self.refresh_dual_cross_geometry_accumulator(pos, &mut stack.current_mut().accumulator);
            self.refresh_kingring_accumulator(pos, &mut stack.current_mut().accumulator);
            self.refresh_pairshape_accumulator(pos, &mut stack.current_mut().accumulator);
        }
        ok
    }

    /// Pin accumulator を `pos` から full 計算する (= 両 perspective 一括)。
    ///
    /// `pin_weights` が `None` (= ckpt に "PIN=" marker なし) の場合は no-op。
    /// `skip_pin_eval` ablation 有効時も update を skip する (= 計測純粋性の確保、
    /// 設計書 § E Phase 2.5)。
    #[inline]
    pub(crate) fn refresh_pin_accumulator(
        &self,
        pos: &Position,
        acc: &mut AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-pin-aware")]
        {
            let Some(pin_w) = self.pin_weights.as_deref() else {
                return;
            };
            if crate::nnue::ablation::skip_pin_eval() {
                return;
            }
            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;
                let mut active = [0u16; crate::nnue::pin_features::MAX_ACTIVE_PIN_FEATURES];
                let mut len = 0u8;
                let sums = crate::nnue::pin_features::refresh_pin_perspective(
                    pos,
                    pin_w,
                    perspective,
                    &mut active,
                    &mut len,
                );
                acc.pin_accumulation[p] = sums;
                acc.pin_active_indices[p] = active;
                acc.pin_active_len[p] = len;
            }
        }
        #[cfg(not(feature = "nnue-pin-aware"))]
        {
            let _ = (pos, acc);
        }
    }

    /// Pin accumulator を 親 `prev_acc` からの差分で更新する。
    ///
    /// king が動いた perspective は full refresh、 そうでない perspective は
    /// 親の sparse active index cache と現在の sparse active 集合を比較して
    /// `removed` / `added` を抽出 → 8 bucket × 加減算を適用する。
    ///
    /// Phase 2b cheap skip: 動いた駒/取られた駒が sniper でなく、 かつ from/to sq が
    /// 両陣営の king_ray ∪ 全敵 sniper ray の union mask から外れていれば pin 状態は
    /// 不変。 親 acc を bit-copy するだけで済む。
    ///
    /// 環境変数 `PIN_PHASE2B_DISABLE=1` で skip path を無効化 (= ablation 用)。
    #[inline]
    pub(crate) fn update_pin_accumulator(
        &self,
        pos: &Position,
        dirty_piece: &super::accumulator::DirtyPiece,
        acc: &mut AccumulatorLayerStacks<L1>,
        prev_acc: &AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-pin-aware")]
        {
            let Some(pin_w) = self.pin_weights.as_deref() else {
                return;
            };
            if crate::nnue::ablation::skip_pin_eval() {
                return;
            }

            let skip_disabled = crate::nnue::pin_features::pin_phase2b_skip_disabled();
            let any_king_moved = dirty_piece.king_moved[0] || dirty_piece.king_moved[1];

            // Phase 2b cheap skip: king 不動 + pin 状態に影響なし → 親 acc bit-copy
            if !skip_disabled
                && !any_king_moved
                && !crate::nnue::pin_features::dirty_piece_affects_pin_state(dirty_piece, pos)
            {
                acc.pin_accumulation = prev_acc.pin_accumulation;
                acc.pin_active_indices = prev_acc.pin_active_indices;
                acc.pin_active_len = prev_acc.pin_active_len;
                return;
            }

            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;
                let king_moved_p = dirty_piece.king_moved[p];

                if king_moved_p {
                    // king 移動した perspective は full refresh
                    let mut active = [0u16; crate::nnue::pin_features::MAX_ACTIVE_PIN_FEATURES];
                    let mut len = 0u8;
                    let sums = crate::nnue::pin_features::refresh_pin_perspective(
                        pos,
                        pin_w,
                        perspective,
                        &mut active,
                        &mut len,
                    );
                    acc.pin_accumulation[p] = sums;
                    acc.pin_active_indices[p] = active;
                    acc.pin_active_len[p] = len;
                } else {
                    // king 不動 perspective は diff path
                    let mut current_active =
                        [0u16; crate::nnue::pin_features::MAX_ACTIVE_PIN_FEATURES];
                    let current_len = crate::nnue::pin_features::feature_pin_sparse_indices(
                        pos,
                        perspective,
                        &mut current_active,
                    );
                    let parent_len = prev_acc.pin_active_len[p] as usize;
                    let parent_slice = &prev_acc.pin_active_indices[p][..parent_len];
                    let current_slice = &current_active[..current_len];

                    let mut removed = [0u16; crate::nnue::pin_features::MAX_ACTIVE_PIN_FEATURES];
                    let mut added = [0u16; crate::nnue::pin_features::MAX_ACTIVE_PIN_FEATURES];
                    let (removed_len, added_len) =
                        crate::nnue::pin_features::diff_pin_active_indices(
                            parent_slice,
                            current_slice,
                            &mut removed,
                            &mut added,
                        );

                    acc.pin_accumulation[p] = prev_acc.pin_accumulation[p];
                    crate::nnue::pin_features::update_pin_perspective_diff(
                        &removed[..removed_len],
                        &added[..added_len],
                        pin_w,
                        perspective,
                        &mut acc.pin_accumulation[p],
                    );
                    acc.pin_active_indices[p] = current_active;
                    acc.pin_active_len[p] = current_len as u8;
                }
            }
        }
        #[cfg(not(feature = "nnue-pin-aware"))]
        {
            let _ = (pos, dirty_piece, acc, prev_acc);
        }
    }

    /// Full Threat scalar accumulatorを局面から再構築する。
    #[inline]
    fn refresh_scalar_threat_accumulator(
        &self,
        pos: &Position,
        acc: &mut AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-scalar-threat")]
        if let Some(weights) = self.scalar_threat_weights.as_ref() {
            for perspective in [Color::Black, Color::White] {
                crate::nnue::scalar_threat::refresh_perspective(
                    pos,
                    weights,
                    perspective,
                    &mut acc.scalar_threat_accumulation[perspective as usize],
                );
            }
        }
        #[cfg(not(feature = "nnue-scalar-threat"))]
        let _ = (pos, acc);
    }

    /// Full Threat scalar accumulatorを直前stack entryから差分更新する。
    #[inline]
    fn update_scalar_threat_accumulator(
        &self,
        pos: &Position,
        dirty_piece: &super::accumulator::DirtyPiece,
        acc: &mut AccumulatorLayerStacks<L1>,
        prev_acc: &AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-scalar-threat")]
        if let Some(weights) = self.scalar_threat_weights.as_ref() {
            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;
                crate::nnue::scalar_threat::update_perspective(
                    pos,
                    dirty_piece,
                    weights,
                    perspective,
                    &prev_acc.scalar_threat_accumulation[p],
                    &mut acc.scalar_threat_accumulation[p],
                );
            }
        }
        #[cfg(not(feature = "nnue-scalar-threat"))]
        let _ = (pos, dirty_piece, acc, prev_acc);
    }

    #[inline]
    fn refresh_dual_cross_geometry_accumulator(
        &self,
        pos: &Position,
        acc: &mut AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-dual-cross-geometry")]
        if let Some(weights) = self.dual_cross_geometry_weights.as_ref() {
            for perspective in [Color::Black, Color::White] {
                let p = perspective.index();
                crate::nnue::dual_cross_geometry::refresh(
                    pos,
                    weights,
                    perspective,
                    &mut acc.dual_cross_geometry_active_indices[p],
                    &mut acc.dual_cross_geometry_active_source_squares[p],
                    &mut acc.dual_cross_geometry_active_len[p],
                    &mut acc.dual_cross_geometry_accumulation[p],
                );
            }
        }
        #[cfg(not(feature = "nnue-dual-cross-geometry"))]
        let _ = (pos, acc);
    }

    #[inline]
    fn update_dual_cross_geometry_accumulator(
        &self,
        pos: &Position,
        acc: &mut AccumulatorLayerStacks<L1>,
        prev_acc: &AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-dual-cross-geometry")]
        if let Some(weights) = self.dual_cross_geometry_weights.as_ref() {
            let Some(delta) = crate::nnue::dual_delta::compute_move_delta(pos) else {
                self.refresh_dual_cross_geometry_accumulator(pos, acc);
                return;
            };
            for perspective in [Color::Black, Color::White] {
                let p = perspective.index();
                acc.dual_cross_geometry_accumulation[p] =
                    prev_acc.dual_cross_geometry_accumulation[p];
                crate::nnue::dual_cross_geometry::update(
                    pos,
                    &delta,
                    weights,
                    perspective,
                    &prev_acc.dual_cross_geometry_active_indices[p],
                    &prev_acc.dual_cross_geometry_active_source_squares[p],
                    prev_acc.dual_cross_geometry_active_len[p],
                    &mut acc.dual_cross_geometry_active_indices[p],
                    &mut acc.dual_cross_geometry_active_source_squares[p],
                    &mut acc.dual_cross_geometry_active_len[p],
                    &mut acc.dual_cross_geometry_accumulation[p],
                );
            }
        }
        #[cfg(not(feature = "nnue-dual-cross-geometry"))]
        let _ = (pos, acc, prev_acc);
    }

    /// Dual PSQT (King-Attacker) accumulator を `pos` から full 計算する (= 両 perspective)。
    ///
    /// `dual_psqt_weights` が `None` (= ckpt に "DUAL=1" marker なし) の場合は no-op。
    /// `skip_dual_eval` ablation 有効時も update を skip する。
    #[inline]
    pub(crate) fn refresh_dual_accumulator(
        &self,
        pos: &Position,
        acc: &mut AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-dual-psqt")]
        {
            let Some(dual_w) = self.dual_psqt_weights.as_deref() else {
                return;
            };
            if crate::nnue::ablation::skip_dual_eval() {
                return;
            }
            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;
                let mut active = [[0u16; crate::nnue::dual_psqt::MAX_ACTIVE_DUAL_FEATURES];
                    crate::nnue::dual_psqt::NUM_DUAL_PSQT_HEADS];
                let mut active_len = [0u8; crate::nnue::dual_psqt::NUM_DUAL_PSQT_HEADS];
                let sums = crate::nnue::dual_psqt::refresh_dual_perspective(
                    pos,
                    dual_w,
                    perspective,
                    &mut active,
                    &mut active_len,
                );
                acc.dual_accumulation[p] = sums;
                acc.dual_active_indices[p] = active;
                acc.dual_active_len[p] = active_len;
            }
        }
        #[cfg(not(feature = "nnue-dual-psqt"))]
        {
            let _ = (pos, acc);
        }
    }

    /// Dual accumulator を 親 `prev_acc` からの差分で更新する。
    ///
    /// king が動いた perspective は full refresh、 そうでない perspective は
    /// 親の sparse active index cache と現在の sparse active 集合を head ごとに
    /// 比較して `removed` / `added` を抽出 → 8 bucket × 加減算を適用する。
    ///
    /// dual head の active 判定は 任意の自軍駒 × 利き × king zone なので、
    /// 駒の出入りや利きが変わる任意の手で active 集合が変わり得る。 pin の Phase 2b
    /// のような cheap skip は導入せず (= acc 更新の overhead は head × persp で
    /// 一定的に発生するが、 各 active が ~10 entry なので bucket 加減算 8 回 × 2 × 10
    /// = 160 i32 add で軽量)。
    #[inline]
    pub(crate) fn update_dual_accumulator(
        &self,
        pos: &Position,
        dirty_piece: &super::accumulator::DirtyPiece,
        acc: &mut AccumulatorLayerStacks<L1>,
        prev_acc: &AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-dual-psqt")]
        {
            let Some(dual_w) = self.dual_psqt_weights.as_deref() else {
                return;
            };
            if crate::nnue::ablation::skip_dual_eval() {
                return;
            }

            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;
                let king_moved_any = dirty_piece.king_moved[0] || dirty_piece.king_moved[1];

                if king_moved_any {
                    // どちらかの king が動いたら、 両 perspective で king zone が
                    // 変わり得る (= 自 king zone / 敵 king zone のどちらかが移動)
                    // ので両 perspective を full refresh する。
                    let mut active = [[0u16; crate::nnue::dual_psqt::MAX_ACTIVE_DUAL_FEATURES];
                        crate::nnue::dual_psqt::NUM_DUAL_PSQT_HEADS];
                    let mut active_len = [0u8; crate::nnue::dual_psqt::NUM_DUAL_PSQT_HEADS];
                    let sums = crate::nnue::dual_psqt::refresh_dual_perspective(
                        pos,
                        dual_w,
                        perspective,
                        &mut active,
                        &mut active_len,
                    );
                    acc.dual_accumulation[p] = sums;
                    acc.dual_active_indices[p] = active;
                    acc.dual_active_len[p] = active_len;
                } else {
                    // king 不動 → 両 head とも diff path
                    for head in 0..crate::nnue::dual_psqt::NUM_DUAL_PSQT_HEADS {
                        let mut current_active =
                            [0u16; crate::nnue::dual_psqt::MAX_ACTIVE_DUAL_FEATURES];
                        let current_len =
                            crate::nnue::dual_psqt::collect_dual_active_per_head_public(
                                pos,
                                perspective,
                                head,
                                &mut current_active,
                            );
                        let parent_len = prev_acc.dual_active_len[p][head] as usize;
                        let parent_slice = &prev_acc.dual_active_indices[p][head][..parent_len];
                        let current_slice = &current_active[..current_len];

                        let mut removed = [0u16; crate::nnue::dual_psqt::MAX_ACTIVE_DUAL_FEATURES];
                        let mut added = [0u16; crate::nnue::dual_psqt::MAX_ACTIVE_DUAL_FEATURES];
                        let (removed_len, added_len) =
                            crate::nnue::dual_psqt::diff_dual_active_indices(
                                parent_slice,
                                current_slice,
                                &mut removed,
                                &mut added,
                            );

                        acc.dual_accumulation[p][head] = prev_acc.dual_accumulation[p][head];
                        crate::nnue::dual_psqt::update_dual_perspective_diff(
                            &removed[..removed_len],
                            &added[..added_len],
                            dual_w,
                            perspective,
                            head,
                            &mut acc.dual_accumulation[p][head],
                        );
                        acc.dual_active_indices[p][head] = current_active;
                        acc.dual_active_len[p][head] = current_len as u8;
                    }
                }
            }
        }
        #[cfg(not(feature = "nnue-dual-psqt"))]
        {
            let _ = (pos, dirty_piece, acc, prev_acc);
        }
    }

    /// Dual Cross (D2) accumulator を full refresh する。
    #[inline]
    #[cfg(any(
        not(feature = "nnue-scalar-threat"),
        feature = "nnue-scalar-threat-coexist"
    ))]
    pub(crate) fn refresh_dual_cross_accumulator(
        &self,
        pos: &Position,
        acc: &mut AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-dual-psqt")]
        {
            let Some(cross_w) = self.dual_cross_weights.as_deref() else {
                return;
            };
            if crate::nnue::ablation::skip_dual_eval() {
                return;
            }
            // Phase 1 profiling: phase 別 counter 用に thread-local の phase index
            // を 現 ply で書き換える (= ProfileGuard drop で参照)。 default OFF 時も
            // 1 cell write のみで overhead negligible。
            crate::nnue::dual_cross::set_current_phase_from_ply(pos.game_ply());
            // Path K1: AttackBitboardCache 廃止 (= 1.3 KB struct copy が NPS bottleneck)。
            // 各 collect 関数で per-sq に `pos.piece_on` + `piece_attacks_from` 直呼び。
            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;
                // Path I + lean API: per-persp 直接 borrow で wrapper 配列 copy 回避。
                let sums = crate::nnue::dual_cross::refresh_dual_cross_persp_head0(
                    pos,
                    cross_w,
                    perspective,
                    &mut acc.dual_cross_active_indices[p],
                    &mut acc.dual_cross_active_def_sq[p],
                    &mut acc.dual_cross_active_len[p],
                );
                acc.dual_cross_accumulation[p] = sums;
            }
            // refresh 完了で最新 state に置き換わったので stale 解消。
            acc.dual_cross_stale = false;
            // every-N counter も refresh で reset (= 次の skip サイクル開始点)。
            acc.dual_cross_update_skip_count = 0;
        }
        #[cfg(not(feature = "nnue-dual-psqt"))]
        {
            let _ = (pos, acc);
        }
    }

    /// Dual Cross (D2) accumulator を **完全 delta** (= 影響範囲限定 scan) で更新する。
    ///
    /// MoveDelta から「影響を受ける board sq の集合」 を構築し、 prev_active のうち
    /// 「own_sq ∈ affected」 (head 0) または「own_sq ∈ affected OR def_sq ∈ affected」
    /// (head 1) な active のみを remove 候補とし、 pos_after の affected_set 内のみで
    /// active を再 enumerate して add 候補とする。 multiset symmetric difference で
    /// 最終 remove/add を確定し、 weight 加減算で accumulator を更新する。
    ///
    /// head 1 (= defended) は active idx に defender_sq を持たないため、 並列 metadata
    /// 配列 `dual_cross_active_def_sq` で defender の生 board sq を保持し、 「defender
    /// ∈ affected」 を filter 可能にしている (= path B 設計)。
    ///
    /// per-move full scan (= `collect_dual_cross_active_per_head` 全 own 駒 scan) を回避
    /// するため、 selfplay hot path で NPS bottleneck を解消する。
    ///
    /// # fallback
    ///
    /// - `dual_cross_weights` 無効: no-op
    /// - `skip_dual_eval` ablation: no-op
    /// - `compute_move_delta` が None (= NULL/PASS/WIN/特殊手): full refresh
    /// - `delta.king_moved_any`: full refresh (= 安全側)
    #[inline]
    #[cfg(any(
        not(feature = "nnue-scalar-threat"),
        feature = "nnue-scalar-threat-coexist"
    ))]
    pub(crate) fn update_dual_cross_accumulator(
        &self,
        pos: &Position,
        dirty_piece: &super::accumulator::DirtyPiece,
        acc: &mut AccumulatorLayerStacks<L1>,
        prev_acc: &AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-dual-psqt")]
        {
            let Some(cross_w) = self.dual_cross_weights.as_deref() else {
                return;
            };
            if crate::nnue::ablation::skip_dual_eval() {
                return;
            }
            // arm 6 endgame-only (= 2026-05-28): `DUAL_CROSS_ENDGAME_ONLY=1` (= ply-based)
            // または `DUAL_CROSS_USE_ENDGAME_DETECTOR=1` (= position-based、 探索側
            // EndgameDetector が `set_current_phase_idx` で thread-local に書き込んだ
            // override 参照) で opening/middle なら update を完全 skip。 prev → acc に
            // dual_cross 関連 state を memcpy し、 stale flag を立てる (= endgame 突入時に
            // PV refresh helper が full refresh を 1 度走らせる)。 eval 側も同 gate で
            // dual_cross_value 加算を 0 に gate するため stale 値が読まれることはない。
            if (crate::nnue::ablation::dual_cross_endgame_only()
                && !crate::nnue::dual_cross::is_endgame_ply(pos.game_ply()))
                || (crate::nnue::ablation::dual_cross_use_endgame_detector()
                    && !crate::nnue::dual_cross::current_phase_is_endgame())
            {
                let _ = dirty_piece;
                let _ = cross_w;
                acc.dual_cross_accumulation = prev_acc.dual_cross_accumulation;
                acc.dual_cross_active_indices = prev_acc.dual_cross_active_indices;
                acc.dual_cross_active_def_sq = prev_acc.dual_cross_active_def_sq;
                acc.dual_cross_active_len = prev_acc.dual_cross_active_len;
                // skip 1 手分 diff が未反映なので stale 確定。 endgame 突入時の
                // 通常 update path で prev_acc.dual_cross_stale == true として
                // 検出され full refresh される。
                acc.dual_cross_stale = true;
                // every-N counter は本経路では触らない (= stale flag で別管理)。
                acc.dual_cross_update_skip_count = prev_acc.dual_cross_update_skip_count;
                return;
            }
            // arm 7 (Path J、 2026-05-27): non-PV node 限定で update を skip。 prev
            // から acc へ dual_cross 関連 state を そのまま memcpy (= identity update
            // 相当)。 これにより 1 手分の diff 計算 (= split / collect / weight add 等
            // 数百 ~ 数千 cycles) を完全に回避する。 eval 側も `non_pv_dual_skip()` で
            // dual_cross_value を 0 に gate するため non-PV chain 上で stale 値が
            // 読まれることはない。
            //
            // arm 7 + Refresh (= 2026-05-27): skip 経路で `dual_cross_stale = true` を
            // 立て、 PV node 復帰時に `update_and_evaluate_layer_stacks_cached` が
            // この flag を見て full refresh を 1 度だけ走らせる (= stale 伝播断ち)。
            if crate::nnue::ablation::non_pv_dual_skip() {
                let _ = dirty_piece;
                let _ = cross_w;
                // Path J: state 全部を memcpy (= 64 i32 + 2 × 96 u16 + 2 × 96 u8 + 2 u16 = ~656 byte)。
                // 各 array は Copy なので blanket assignment が安全。
                acc.dual_cross_accumulation = prev_acc.dual_cross_accumulation;
                acc.dual_cross_active_indices = prev_acc.dual_cross_active_indices;
                acc.dual_cross_active_def_sq = prev_acc.dual_cross_active_def_sq;
                acc.dual_cross_active_len = prev_acc.dual_cross_active_len;
                // skip 1 手分 diff が未反映なので stale 確定 (= prev が既に stale でも
                // OR で広がる)。
                acc.dual_cross_stale = true;
                // every-N counter は non-PV path では触らない (= stale flag で別管理)。
                acc.dual_cross_update_skip_count = prev_acc.dual_cross_update_skip_count;
                return;
            }
            // arm 7 every-N (= 2026-05-28): PV node で更新間隔 N > 1 のとき、 (N-1) 回連続で
            // delta update を skip し prev → acc memcpy で済ませる。 N 回目で full refresh
            // して累積 stale を解消する。 eval は stale 値を読むため 意図的 eval drift と
            // なり、 selfplay で NPS gain と棋力ロスの balance を検証する。
            //
            // - prev_acc.stale が true (= 過去 non-PV skip 由来) ならば まず refresh で
            //   整合を取り、 every-N counter も 0 にリセット (= 共通の refresh 経路 reset)。
            // - prev_skip + 1 < N: memcpy + counter インクリメント、 stale flag は false に
            //   保つ (= eval が PV node 上の stale 値を読むのは「意図的」)。
            // - prev_skip + 1 == N: full refresh + counter = 0 (= refresh が stale も解消)。
            let n_interval = crate::nnue::ablation::dual_cross_update_interval();
            if n_interval > 1 {
                if prev_acc.dual_cross_stale {
                    let _ = dirty_piece;
                    self.refresh_dual_cross_accumulator(pos, acc);
                    acc.dual_cross_update_skip_count = 0;
                    return;
                }
                let next_count = prev_acc.dual_cross_update_skip_count.saturating_add(1);
                if next_count < n_interval {
                    let _ = dirty_piece;
                    let _ = cross_w;
                    acc.dual_cross_accumulation = prev_acc.dual_cross_accumulation;
                    acc.dual_cross_active_indices = prev_acc.dual_cross_active_indices;
                    acc.dual_cross_active_def_sq = prev_acc.dual_cross_active_def_sq;
                    acc.dual_cross_active_len = prev_acc.dual_cross_active_len;
                    // 意図的 eval drift: stale flag は false のまま (= PV refresh を抑止)。
                    acc.dual_cross_stale = false;
                    acc.dual_cross_update_skip_count = next_count;
                    return;
                }
                // next_count >= N: refresh で stale 累積解消、 counter リセット。
                let _ = dirty_piece;
                self.refresh_dual_cross_accumulator(pos, acc);
                acc.dual_cross_update_skip_count = 0;
                return;
            }
            // PV node の通常 update path。 prev が stale (= 過去 non-PV skip による
            // 未反映 diff あり) なら delta-update の base が不正なので full refresh で
            // 整合を取る (= refresh は stale を false に確定する)。
            if prev_acc.dual_cross_stale {
                let _ = dirty_piece;
                self.refresh_dual_cross_accumulator(pos, acc);
                return;
            }
            // Phase 1 profiling: 現 ply から phase を確定 (thread-local cell 1 write)
            crate::nnue::dual_cross::set_current_phase_from_ply(pos.game_ply());

            // 1 手の delta を計算。 None なら full refresh fallback。
            let Some(delta) = crate::nnue::dual_delta::compute_move_delta(pos) else {
                let _ = dirty_piece;
                self.refresh_dual_cross_accumulator(pos, acc);
                return;
            };

            // king moved → 両 perspective full refresh fallback
            if delta.king_moved_any {
                self.refresh_dual_cross_accumulator(pos, acc);
                return;
            }

            // Path K3: affected_sq_set を 1 度だけ計算し 2 perspective で共有
            // (= build_affected_sq_set の king_effect / knight_effect bitboard OR を半減)。
            let affected = crate::nnue::dual_cross::build_dual_cross_affected_sq_set(&delta);
            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;

                let prev_active = &prev_acc.dual_cross_active_indices[p];
                let prev_active_len = prev_acc.dual_cross_active_len[p];

                let mut acc_buckets = prev_acc.dual_cross_accumulation[p];
                let mut current_active =
                    [0u16; crate::nnue::dual_cross::MAX_ACTIVE_DUAL_CROSS_FEATURES];
                let mut current_active_def_sq = [crate::nnue::dual_cross::DUAL_CROSS_DEF_SQ_NONE;
                    crate::nnue::dual_cross::MAX_ACTIVE_DUAL_CROSS_FEATURES];
                let mut current_active_len = 0u16;

                crate::nnue::dual_cross::update_dual_cross_persp_head0_full_delta_with_affected(
                    pos,
                    &delta,
                    affected,
                    cross_w,
                    perspective,
                    prev_active,
                    prev_active_len,
                    &mut acc_buckets,
                    &mut current_active,
                    &mut current_active_def_sq,
                    &mut current_active_len,
                );

                acc.dual_cross_accumulation[p] = acc_buckets;
                acc.dual_cross_active_indices[p] = current_active;
                acc.dual_cross_active_def_sq[p] = current_active_def_sq;
                acc.dual_cross_active_len[p] = current_active_len;
            }
            // delta-update success path: 最新値が反映されたので stale 解消。
            acc.dual_cross_stale = false;
            // every-N counter も最新 update 完了でリセット。
            acc.dual_cross_update_skip_count = 0;
        }
        #[cfg(not(feature = "nnue-dual-psqt"))]
        {
            let _ = (pos, dirty_piece, acc, prev_acc);
        }
    }

    #[cfg(all(
        feature = "nnue-scalar-threat",
        not(feature = "nnue-scalar-threat-coexist")
    ))]
    #[inline]
    pub(crate) fn refresh_dual_cross_accumulator(
        &self,
        pos: &Position,
        acc: &mut AccumulatorLayerStacks<L1>,
    ) {
        let _ = (self, pos, acc);
    }

    #[cfg(all(
        feature = "nnue-scalar-threat",
        not(feature = "nnue-scalar-threat-coexist")
    ))]
    #[inline]
    pub(crate) fn update_dual_cross_accumulator(
        &self,
        pos: &Position,
        dirty_piece: &super::accumulator::DirtyPiece,
        acc: &mut AccumulatorLayerStacks<L1>,
        prev_acc: &AccumulatorLayerStacks<L1>,
    ) {
        let _ = (self, pos, dirty_piece, acc, prev_acc);
    }

    /// KingRing accumulator を `pos` から full 計算する (= 両 perspective 一括)。
    ///
    /// `kingring_weights` が `None` (= ckpt に KingRing section なし) の場合は no-op。
    /// `skip_kingring_eval` ablation 有効時も update を skip する (= 計測純粋性確保)。
    ///
    /// hot path 経路: bucket-inner i32 runtime weight + ring_idx 固定 slot を使う
    /// (= 候補 1+3+4、 2026-05-29)。
    #[inline]
    pub(crate) fn refresh_kingring_accumulator(
        &self,
        pos: &Position,
        acc: &mut AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-kingring-psqt")]
        {
            let Some(rt) = self.kingring_runtime_weights.as_deref() else {
                return;
            };
            if crate::nnue::ablation::skip_kingring_eval() {
                return;
            }
            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;
                let sums = crate::nnue::kingring::refresh_kingring_perspective_runtime(
                    pos,
                    rt,
                    perspective,
                    &mut acc.kingring_active_indices[p],
                );
                acc.kingring_accumulation[p] = sums;
            }
        }
        #[cfg(not(feature = "nnue-kingring-psqt"))]
        {
            let _ = (pos, acc);
        }
    }

    /// KingRing accumulator を 親 `prev_acc` からの差分で更新する。
    ///
    /// king が動いた perspective は full refresh、 そうでない perspective は
    /// **ring_idx 固定 slot** の old/new を 1 対 1 cmp して直接 add/sub する
    /// (= 候補 1、 2026-05-29)。 ring 24 sq は king 位置依存なので king_moved 時は
    /// ring 自体が動く → full refresh が必須。
    #[inline]
    pub(crate) fn update_kingring_accumulator(
        &self,
        pos: &Position,
        dirty_piece: &super::accumulator::DirtyPiece,
        acc: &mut AccumulatorLayerStacks<L1>,
        prev_acc: &AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-kingring-psqt")]
        {
            let Some(rt) = self.kingring_runtime_weights.as_deref() else {
                return;
            };
            if crate::nnue::ablation::skip_kingring_eval() {
                return;
            }

            // incremental reuse 経路 (= 2026-05-29): MoveDelta から affected sq 集合を 1 回計算し、
            // affected に含まれる ring sq のみ classify_sq する。 大半の move では 1-3 sq だけ
            // 影響を受けるので、 `attackers_to_occ` 呼び出しを 48 → 2-6 回/ply に削減できる。
            // affected_sq_set は perspective 非依存なので 1 度だけ計算して 2 persp で共有 (= Path K3 相当)。
            // MoveDelta 計算不可 (= 古い move format / NULL move 等) の場合は従来 path
            // (= collect_active_slots full re-scan) に fallback。
            let move_delta_opt = crate::nnue::dual_delta::compute_move_delta(pos);
            let affected_opt = move_delta_opt
                .as_ref()
                .map(|d| crate::nnue::kingring::build_kingring_affected_sq_set(pos, d));

            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;
                let king_moved_p = dirty_piece.king_moved[p];

                if king_moved_p {
                    // king 移動 → ring 全体が shift、 full refresh が必須。
                    let sums = crate::nnue::kingring::refresh_kingring_perspective_runtime(
                        pos,
                        rt,
                        perspective,
                        &mut acc.kingring_active_indices[p],
                    );
                    acc.kingring_accumulation[p] = sums;
                } else if let (Some(delta), Some(affected)) =
                    (move_delta_opt.as_ref(), affected_opt)
                {
                    // incremental reuse: affected ring sq のみ再 classify、 残りは prev slot 流用。
                    let prev_slots = &prev_acc.kingring_active_indices[p];
                    let mut cur_slots = [crate::nnue::kingring::KINGRING_SLOT_SENTINEL;
                        crate::nnue::kingring::KINGRING_NUM_SQ];
                    let new_acc =
                        crate::nnue::kingring::update_kingring_perspective_incremental_with_affected(
                            pos,
                            rt,
                            perspective,
                            delta,
                            affected,
                            prev_slots,
                            &prev_acc.kingring_accumulation[p],
                            &mut cur_slots,
                        );
                    acc.kingring_accumulation[p] = new_acc;
                    acc.kingring_active_indices[p] = cur_slots;
                } else {
                    // MoveDelta 計算不可 (= NULL move 等)、 従来 path (= full re-scan + slot diff)。
                    let prev_slots = &prev_acc.kingring_active_indices[p];
                    let mut cur_slots = *prev_slots;
                    crate::nnue::kingring::collect_active_slots(pos, perspective, &mut cur_slots);
                    let mut acc_buckets = prev_acc.kingring_accumulation[p];
                    crate::nnue::kingring::apply_slot_diff_runtime(
                        rt,
                        perspective,
                        prev_slots,
                        &cur_slots,
                        &mut acc_buckets,
                    );
                    acc.kingring_accumulation[p] = acc_buckets;
                    acc.kingring_active_indices[p] = cur_slots;
                }
            }
        }
        #[cfg(not(feature = "nnue-kingring-psqt"))]
        {
            let _ = (pos, dirty_piece, acc, prev_acc);
        }
    }

    /// PairShape accumulator を `pos` から full 計算する (= 両 perspective 一括)。
    ///
    /// `pairshape_weights` が `None` (= ckpt に section なし) の場合は no-op。
    /// `skip_pairshape_eval` ablation 有効時も update を skip する (= 計測純粋性)。
    #[inline]
    pub(crate) fn refresh_pairshape_accumulator(
        &self,
        pos: &Position,
        acc: &mut AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-pairshape-psqt")]
        {
            let Some(rt) = self.pairshape_runtime_weights.as_deref() else {
                return;
            };
            if crate::nnue::ablation::skip_pairshape_eval() {
                return;
            }
            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;
                let sums = crate::nnue::pairshape::refresh_pairshape_perspective(
                    pos,
                    rt,
                    perspective,
                    &mut acc.pairshape_active_indices[p],
                    &mut acc.pairshape_active_len[p],
                );
                acc.pairshape_accumulation[p] = sums;
            }
        }
        #[cfg(not(feature = "nnue-pairshape-psqt"))]
        {
            let _ = (pos, acc);
        }
    }

    /// PairShape accumulator を 親 `prev_acc` からの差分で更新する。
    ///
    /// **真 incremental 経路** (= 2026-05-29): `MoveDelta` で算出した「影響 pair_type」
    /// (= 動いた駒 / 取られた駒に該当する piece_type が member の pair_type) のみを
    /// cur 盤面から recompute し、 影響外 pair_type の active entry は prev からそのまま
    /// 引き継ぐ。 旧 full re-collect (= 12 pair × cross product) に対し、 平均 2-5 pair に
    /// 削減することで NPS 退行 -18% → -5% 以内 に改善。
    ///
    /// `delta == None` (= NULL move / PASS) の場合は `update_pairshape_perspective` 内部で
    /// full refresh fallback。 詳細は `pairshape.rs::update_pairshape_perspective` 参照。
    #[inline]
    pub(crate) fn update_pairshape_accumulator(
        &self,
        pos: &Position,
        _dirty_piece: &super::accumulator::DirtyPiece,
        acc: &mut AccumulatorLayerStacks<L1>,
        prev_acc: &AccumulatorLayerStacks<L1>,
    ) {
        #[cfg(feature = "nnue-pairshape-psqt")]
        {
            let Some(rt) = self.pairshape_runtime_weights.as_deref() else {
                return;
            };
            if crate::nnue::ablation::skip_pairshape_eval() {
                return;
            }
            // MoveDelta は真 incremental update path で「影響 pair_type マスク」 計算に
            // 使用される (= 旧 full re-collect では未使用だったが、 2026-05-29 真
            // incremental 化で本格使用に切り替わった)。
            let delta = crate::nnue::dual_delta::compute_move_delta(pos);
            for perspective in [Color::Black, Color::White] {
                let p = perspective as usize;
                let prev_len = prev_acc.pairshape_active_len[p] as usize;
                let prev_active = &prev_acc.pairshape_active_indices[p][..prev_len];
                let mut cur_active = [0u16; crate::nnue::pairshape::PAIRSHAPE_MAX_ACTIVE];
                let mut cur_len: u16 = 0;
                let new_acc = crate::nnue::pairshape::update_pairshape_perspective(
                    pos,
                    rt,
                    perspective,
                    delta.as_ref(),
                    prev_active,
                    &prev_acc.pairshape_accumulation[p],
                    &mut cur_active,
                    &mut cur_len,
                );
                acc.pairshape_accumulation[p] = new_acc;
                acc.pairshape_active_indices[p] = cur_active;
                acc.pairshape_active_len[p] = cur_len;
            }
        }
        #[cfg(not(feature = "nnue-pairshape-psqt"))]
        {
            let _ = (pos, _dirty_piece, acc, prev_acc);
        }
    }
}

#[cfg(feature = "layerstacks-1536x16x32")]
pub type NetworkLayerStacks1536x16x32 =
    NetworkLayerStacks<1536, LAYER_STACK_16X32_L1_OUT, LAYER_STACK_16X32_L2_IN, 32, 32>;
#[cfg(feature = "layerstacks-1536x32x32")]
pub type NetworkLayerStacks1536x32x32 =
    NetworkLayerStacks<1536, LAYER_STACK_32X32_L1_OUT, LAYER_STACK_32X32_L2_IN, 64, 32>;
#[cfg(feature = "layerstacks-768x16x32")]
pub type NetworkLayerStacks768x16x32 =
    NetworkLayerStacks<768, LAYER_STACK_16X32_L1_OUT, LAYER_STACK_16X32_L2_IN, 32, 32>;
#[cfg(feature = "layerstacks-768x8x32")]
pub type NetworkLayerStacks768x8x32 = NetworkLayerStacks<
    768,
    { super::constants::LAYER_STACK_8X32_L1_OUT },
    { super::constants::LAYER_STACK_8X32_L2_IN },
    32, // LS_L2_PADDED_INPUT = padded_input(LAYER_STACK_8X32_L2_IN=14) = 32
    32, // LS_L3
>;
#[cfg(feature = "layerstacks-512x16x32")]
pub type NetworkLayerStacks512x16x32 =
    NetworkLayerStacks<512, LAYER_STACK_16X32_L1_OUT, LAYER_STACK_16X32_L2_IN, 32, 32>;
#[cfg(feature = "layerstacks-1536x8x96")]
pub type NetworkLayerStacks1536x8x96 = NetworkLayerStacks<
    1536,
    { super::constants::LAYER_STACK_8X96_L1_OUT },
    { super::constants::LAYER_STACK_8X96_L2_IN },
    32, // LS_L2_PADDED_INPUT = padded_input(LAYER_STACK_8X96_L2_IN=14) = 32
    96, // LS_L3
>;
#[cfg(feature = "layerstacks-1536x16x96")]
pub type NetworkLayerStacks1536x16x96 = NetworkLayerStacks<
    1536,
    { super::constants::LAYER_STACK_16X96_L1_OUT },
    { super::constants::LAYER_STACK_16X96_L2_IN },
    32, // LS_L2_PADDED_INPUT = padded_input(LAYER_STACK_16X96_L2_IN=30) = 32
    96, // LS_L3
>;
#[cfg(feature = "layerstacks-1536x32x96")]
pub type NetworkLayerStacks1536x32x96 = NetworkLayerStacks<
    1536,
    { super::constants::LAYER_STACK_32X96_L1_OUT },
    { super::constants::LAYER_STACK_32X96_L2_IN },
    64, // LS_L2_PADDED_INPUT = padded_input(LAYER_STACK_32X96_L2_IN=62) = 64
    96, // LS_L3
>;
#[cfg(feature = "layerstacks-1536x48x96")]
pub type NetworkLayerStacks1536x48x96 = NetworkLayerStacks<
    1536,
    { super::constants::LAYER_STACK_48X96_L1_OUT },
    { super::constants::LAYER_STACK_48X96_L2_IN },
    96, // LS_L2_PADDED_INPUT = padded_input(LAYER_STACK_48X96_L2_IN=94) = 96
    96, // LS_L3
>;
#[cfg(feature = "layerstacks-1536x64x96")]
pub type NetworkLayerStacks1536x64x96 = NetworkLayerStacks<
    1536,
    { super::constants::LAYER_STACK_64X96_L1_OUT },
    { super::constants::LAYER_STACK_64X96_L2_IN },
    128, // LS_L2_PADDED_INPUT = padded_input(LAYER_STACK_64X96_L2_IN=126) = 128
    96,  // LS_L3
>;
#[cfg(feature = "layerstacks-768x32x96")]
pub type NetworkLayerStacks768x32x96 = NetworkLayerStacks<
    768,
    { super::constants::LAYER_STACK_32X96_L1_OUT },
    { super::constants::LAYER_STACK_32X96_L2_IN },
    64, // LS_L2_PADDED_INPUT = padded_input(LAYER_STACK_32X96_L2_IN=62) = 64
    96, // LS_L3
>;
#[cfg(feature = "layerstacks-1536x16x48")]
pub type NetworkLayerStacks1536x16x48 = NetworkLayerStacks<
    1536,
    { super::constants::LAYER_STACK_16X48_L1_OUT },
    { super::constants::LAYER_STACK_16X48_L2_IN },
    32, // LS_L2_PADDED_INPUT = padded_input(LAYER_STACK_16X48_L2_IN=30) = 32
    { super::constants::LAYER_STACK_16X48_L3 }, // LS_L3 = 48
>;
#[cfg(feature = "layerstacks-1024x16x48")]
pub type NetworkLayerStacks1024x16x48 = NetworkLayerStacks<
    1024,
    { super::constants::LAYER_STACK_16X48_L1_OUT },
    { super::constants::LAYER_STACK_16X48_L2_IN },
    32, // LS_L2_PADDED_INPUT = padded_input(LAYER_STACK_16X48_L2_IN=30) = 32
    { super::constants::LAYER_STACK_16X48_L3 }, // LS_L3 = 48
>;
#[cfg(feature = "layerstacks-1024x8x64")]
pub type NetworkLayerStacks1024x8x64 = NetworkLayerStacks<
    1024,
    { super::constants::LAYER_STACK_8X96_L1_OUT }, // = 8 (1536x8x96 と共通)
    { super::constants::LAYER_STACK_8X96_L2_IN },  // = 14 (=7+7 dual perspective)
    32, // LS_L2_PADDED_INPUT = padded_input(LAYER_STACK_8X96_L2_IN=14) = 32
    64, // LS_L3
>;

// =============================================================================
// LayerStacksNetwork - L1 サイズ dispatch enum
// =============================================================================

/// LayerStacks ネットワークの L1 サイズ dispatch enum
///
/// Cargo feature `layerstacks-1536x16x32` / `layerstacks-1536x32x32`
/// / `layerstacks-768x16x32` / `layerstacks-512x16x32` で
/// 有効なバリアントが制御される。
///
/// **重要**: 大会ビルドでは必ず単一バリアントのみを有効化すること。複数バリアントを
/// 同時有効にすると dispatch match の overhead で約 5% NPS 退行する（実測値）。
pub enum LayerStacksNetwork {
    #[cfg(feature = "layerstacks-1536x16x32")]
    L1536x16x32(Box<NetworkLayerStacks1536x16x32>),
    #[cfg(feature = "layerstacks-1536x32x32")]
    L1536x32x32(Box<NetworkLayerStacks1536x32x32>),
    #[cfg(feature = "layerstacks-1536x8x96")]
    L1536x8x96(Box<NetworkLayerStacks1536x8x96>),
    #[cfg(feature = "layerstacks-1536x16x96")]
    L1536x16x96(Box<NetworkLayerStacks1536x16x96>),
    #[cfg(feature = "layerstacks-1536x32x96")]
    L1536x32x96(Box<NetworkLayerStacks1536x32x96>),
    #[cfg(feature = "layerstacks-1536x48x96")]
    L1536x48x96(Box<NetworkLayerStacks1536x48x96>),
    #[cfg(feature = "layerstacks-1536x64x96")]
    L1536x64x96(Box<NetworkLayerStacks1536x64x96>),
    #[cfg(feature = "layerstacks-1536x16x48")]
    L1536x16x48(Box<NetworkLayerStacks1536x16x48>),
    #[cfg(feature = "layerstacks-1024x16x48")]
    L1024x16x48(Box<NetworkLayerStacks1024x16x48>),
    #[cfg(feature = "layerstacks-1024x8x64")]
    L1024x8x64(Box<NetworkLayerStacks1024x8x64>),
    #[cfg(feature = "layerstacks-768x32x96")]
    L768x32x96(Box<NetworkLayerStacks768x32x96>),
    #[cfg(feature = "layerstacks-768x16x32")]
    L768x16x32(Box<NetworkLayerStacks768x16x32>),
    #[cfg(feature = "layerstacks-768x8x32")]
    L768x8x32(Box<NetworkLayerStacks768x8x32>),
    #[cfg(feature = "layerstacks-512x16x32")]
    L512x16x32(Box<NetworkLayerStacks512x16x32>),
}

impl LayerStacksNetwork {
    /// アーキテクチャ寸法 (L1, L2, L3) を返す
    pub fn architecture_dims(&self) -> (usize, usize, usize) {
        let spec = self.architecture_spec();
        (spec.l1, spec.l2, spec.l3)
    }

    /// L1 サイズを取得
    pub fn l1_size(&self) -> usize {
        match self {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32(_) => 1536,
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32(_) => 1536,
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48(_) => 1536,
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48(_) => 1024,
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64(_) => 1024,
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96(_) => 768,
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32(_) => 768,
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32(_) => 768,
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32(_) => 512,
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    }

    /// アーキテクチャ仕様を取得
    pub fn architecture_spec(&self) -> super::spec::ArchitectureSpec {
        match self {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                1536,
                16,
                32,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                1536,
                32,
                32,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                1536,
                8,
                96,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                1536,
                16,
                96,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                1536,
                32,
                96,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                1536,
                48,
                96,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                1536,
                64,
                96,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                1536,
                16,
                48,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                1024,
                16,
                48,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                1024,
                8,
                64,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                768,
                32,
                96,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                768,
                16,
                32,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                768,
                8,
                32,
                super::spec::Activation::CReLU,
            ),
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32(_) => super::spec::ArchitectureSpec::new(
                super::spec::FeatureSet::LayerStacks,
                512,
                16,
                32,
                super::spec::Activation::CReLU,
            ),
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    }

    /// FV_SCALE を取得
    pub fn fv_scale(&self) -> i32 {
        match self {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32(net) => net.fv_scale,
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32(net) => net.fv_scale,
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    }

    /// ファイルから読み込み（exact architecture で dispatch）
    pub fn read_with_options<R: Read + Seek>(
        reader: &mut R,
        l1: usize,
        l2: usize,
        l3: usize,
        psqt_override: Option<bool>,
    ) -> io::Result<Self> {
        match (l1, l2, l3) {
            #[cfg(feature = "layerstacks-1536x16x32")]
            (1536, 16, 32) => {
                let net = NetworkLayerStacks1536x16x32::read_with_options(reader, psqt_override)?;
                Ok(Self::L1536x16x32(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-1536x32x32")]
            (1536, 32, 32) => {
                let net = NetworkLayerStacks1536x32x32::read_with_options(reader, psqt_override)?;
                Ok(Self::L1536x32x32(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-1536x8x96")]
            (1536, 8, 96) => {
                let net = NetworkLayerStacks1536x8x96::read_with_options(reader, psqt_override)?;
                Ok(Self::L1536x8x96(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-1536x16x96")]
            (1536, 16, 96) => {
                let net = NetworkLayerStacks1536x16x96::read_with_options(reader, psqt_override)?;
                Ok(Self::L1536x16x96(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-1536x32x96")]
            (1536, 32, 96) => {
                let net = NetworkLayerStacks1536x32x96::read_with_options(reader, psqt_override)?;
                Ok(Self::L1536x32x96(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-1536x48x96")]
            (1536, 48, 96) => {
                let net = NetworkLayerStacks1536x48x96::read_with_options(reader, psqt_override)?;
                Ok(Self::L1536x48x96(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-1536x64x96")]
            (1536, 64, 96) => {
                let net = NetworkLayerStacks1536x64x96::read_with_options(reader, psqt_override)?;
                Ok(Self::L1536x64x96(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-1536x16x48")]
            (1536, 16, 48) => {
                let net = NetworkLayerStacks1536x16x48::read_with_options(reader, psqt_override)?;
                Ok(Self::L1536x16x48(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-1024x16x48")]
            (1024, 16, 48) => {
                let net = NetworkLayerStacks1024x16x48::read_with_options(reader, psqt_override)?;
                Ok(Self::L1024x16x48(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-1024x8x64")]
            (1024, 8, 64) => {
                let net = NetworkLayerStacks1024x8x64::read_with_options(reader, psqt_override)?;
                Ok(Self::L1024x8x64(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-768x32x96")]
            (768, 32, 96) => {
                let net = NetworkLayerStacks768x32x96::read_with_options(reader, psqt_override)?;
                Ok(Self::L768x32x96(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-768x16x32")]
            (768, 16, 32) => {
                let net = NetworkLayerStacks768x16x32::read_with_options(reader, psqt_override)?;
                Ok(Self::L768x16x32(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-768x8x32")]
            (768, 8, 32) => {
                let net = NetworkLayerStacks768x8x32::read_with_options(reader, psqt_override)?;
                Ok(Self::L768x8x32(Box::new(net)))
            }
            #[cfg(feature = "layerstacks-512x16x32")]
            (512, 16, 32) => {
                let net = NetworkLayerStacks512x16x32::read_with_options(reader, psqt_override)?;
                Ok(Self::L512x16x32(Box::new(net)))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported LayerStacks architecture: {l1}x{l2}x{l3}"),
            )),
        }
    }

    /// 評価値を計算 (= mutable stack 版、 activity_risk cached path 採用)
    ///
    /// 2026-06-06: activity_risk Head の cached path 経由で eval (= NPS 改善目的)。
    /// L1536x16x32 variant のみ cached path 採用 (= production binary)、 他 variant は既存 path。
    pub fn evaluate_mut(
        &self,
        pos: &mut Position,
        stack: &mut super::accumulator_layer_stacks::LayerStacksAccStack,
    ) -> Value {
        match (self, stack) {
            #[cfg(feature = "layerstacks-1536x16x32")]
            (
                Self::L1536x16x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x32(st),
            ) => {
                #[cfg(feature = "nnue-activity-risk-psqt")]
                {
                    let entry = st.current_mut();
                    let acc_ref = &entry.accumulator;
                    let ar_accum = &mut entry.activity_risk_accum;
                    net.evaluate_with_activity_risk(pos, acc_ref, ar_accum)
                }
                #[cfg(not(feature = "nnue-activity-risk-psqt"))]
                {
                    net.evaluate(pos, &st.current().accumulator)
                }
            }
            // 他 variant: 既存 path に fallback (= LayerStacksAccStack の immutable 参照 経由)
            (this, stack) => Self::evaluate_dispatch_immut(this, pos, &*stack),
        }
    }

    /// 明示 bucket での評価値計算 (= LS_BUCKET_INTERP の隣接 bucket 再評価用)。
    ///
    /// natural eval (= `evaluate`) と同じ immutable accumulator (= L0/L1 features、
    /// bucket 非依存) を再利用し、`bucket_index` で指定した bucket の L1 以降 + heads
    /// のみ再 forward する。補間ロジックが natural bucket と neighbor bucket の eval を
    /// blend するために使う。
    ///
    /// search の cached path (`update_and_evaluate_layer_stacks_cached`) は immutable
    /// `evaluate` (= activity_risk 非 cached) を使うため、bit-exact 整合のため本関数も
    /// 全 variant で immutable `evaluate_with_bucket` (activity_risk 非 cached) を呼ぶ。
    pub fn evaluate_with_bucket_immut(
        &self,
        pos: &mut Position,
        stack: &super::accumulator_layer_stacks::LayerStacksAccStack,
        bucket_index: usize,
    ) -> Value {
        self.evaluate_with_bucket_dispatch_immut(pos, stack, bucket_index)
    }

    /// 既存 evaluate path への dispatch (= L1536x16x32 以外の variant、 immutable stack)。
    fn evaluate_dispatch_immut(
        this: &Self,
        pos: &mut Position,
        stack: &super::accumulator_layer_stacks::LayerStacksAccStack,
    ) -> Value {
        this.evaluate(pos, stack)
    }

    /// 明示 bucket での immutable accumulator 評価 (= LS_BUCKET_INTERP 用、非 cached variant)。
    fn evaluate_with_bucket_dispatch_immut(
        &self,
        pos: &mut Position,
        stack: &super::accumulator_layer_stacks::LayerStacksAccStack,
        bucket_index: usize,
    ) -> Value {
        let (net_l1, net_l2, net_l3) = self.architecture_dims();
        let (stack_l1, stack_l2, stack_l3) = stack.architecture_dims();
        match (self, stack) {
            #[cfg(feature = "layerstacks-1536x16x32")]
            (
                Self::L1536x16x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x32(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-1536x32x32")]
            (
                Self::L1536x32x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x32x32(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-1536x8x96")]
            (
                Self::L1536x8x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x8x96(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-1536x16x96")]
            (
                Self::L1536x16x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x96(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-1536x32x96")]
            (
                Self::L1536x32x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x32x96(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-1536x48x96")]
            (
                Self::L1536x48x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x48x96(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-1536x64x96")]
            (
                Self::L1536x64x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x64x96(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-1536x16x48")]
            (
                Self::L1536x16x48(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x48(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-1024x16x48")]
            (
                Self::L1024x16x48(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1024x16x48(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-1024x8x64")]
            (
                Self::L1024x8x64(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1024x8x64(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-768x32x96")]
            (
                Self::L768x32x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x32x96(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-768x16x32")]
            (
                Self::L768x16x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x16x32(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-768x8x32")]
            (
                Self::L768x8x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x8x32(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[cfg(feature = "layerstacks-512x16x32")]
            (
                Self::L512x16x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L512x16x32(st),
            ) => net.evaluate_with_bucket(pos, &st.current().accumulator, bucket_index),
            #[allow(unreachable_patterns)]
            _ => panic!(
                "LayerStacksNetwork/LayerStacksAccStack architecture mismatch: network={net_l1}x{net_l2}x{net_l3}, stack={stack_l1}x{stack_l2}x{stack_l3}"
            ),
        }
    }

    /// 評価値を計算
    ///
    /// `pos` は LMAS bit 18/19 (mate threat) で `do_move`/`undo_move` を呼ぶため `&mut`。
    /// mate flag 立ってない場合は state 不変。
    pub fn evaluate(
        &self,
        pos: &mut Position,
        stack: &super::accumulator_layer_stacks::LayerStacksAccStack,
    ) -> Value {
        let (net_l1, net_l2, net_l3) = self.architecture_dims();
        let (stack_l1, stack_l2, stack_l3) = stack.architecture_dims();
        match (self, stack) {
            #[cfg(feature = "layerstacks-1536x16x32")]
            (
                Self::L1536x16x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x32(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-1536x32x32")]
            (
                Self::L1536x32x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x32x32(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-1536x8x96")]
            (
                Self::L1536x8x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x8x96(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-1536x16x96")]
            (
                Self::L1536x16x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x96(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-1536x32x96")]
            (
                Self::L1536x32x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x32x96(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-1536x48x96")]
            (
                Self::L1536x48x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x48x96(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-1536x64x96")]
            (
                Self::L1536x64x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x64x96(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-1536x16x48")]
            (
                Self::L1536x16x48(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x48(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-1024x16x48")]
            (
                Self::L1024x16x48(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1024x16x48(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-1024x8x64")]
            (
                Self::L1024x8x64(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1024x8x64(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-768x32x96")]
            (
                Self::L768x32x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x32x96(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-768x16x32")]
            (
                Self::L768x16x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x16x32(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-768x8x32")]
            (
                Self::L768x8x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x8x32(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[cfg(feature = "layerstacks-512x16x32")]
            (
                Self::L512x16x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L512x16x32(st),
            ) => net.evaluate(pos, &st.current().accumulator),
            #[allow(unreachable_patterns)]
            _ => panic!(
                "LayerStacksNetwork/LayerStacksAccStack architecture mismatch: network={net_l1}x{net_l2}x{net_l3}, stack={stack_l1}x{stack_l2}x{stack_l3}"
            ),
        }
    }

    /// アキュムレータを更新（キャッシュ対応）
    pub fn update_accumulator(
        &self,
        pos: &Position,
        stack: &mut super::accumulator_layer_stacks::LayerStacksAccStack,
        cache: &mut Option<super::accumulator_layer_stacks::LayerStacksAccCache>,
    ) {
        let (net_l1, net_l2, net_l3) = self.architecture_dims();
        let (stack_l1, stack_l2, stack_l3) = stack.architecture_dims();
        macro_rules! do_update {
            ($net:expr, $stack:expr, $cache_variant:ident) => {{
                let current_entry = $stack.current();
                if current_entry.accumulator.computed_accumulation {
                    return;
                }

                let mut updated = false;

                // --- Tier 1: 直前局面 (depth=1) で差分更新 ---
                if let Some(prev_idx) = current_entry.previous {
                    let prev_computed = $stack.entry_at(prev_idx).accumulator.computed_accumulation;
                    if prev_computed {
                        let dirty_piece = $stack.current().dirty_piece;
                        let (prev_acc, current_acc) =
                            $stack.get_prev_and_current_accumulators(prev_idx);
                        if let Some(
                            super::accumulator_layer_stacks::LayerStacksAccCache::$cache_variant(c),
                        ) = cache
                        {
                            $net.update_accumulator_with_cache(
                                pos,
                                &dirty_piece,
                                current_acc,
                                prev_acc,
                                c,
                            );
                        } else {
                            $net.update_accumulator(pos, &dirty_piece, current_acc, prev_acc);
                        }
                        updated = true;
                    }
                }

                // --- Tier 2: 祖先探索 + forward_update_incremental ---
                // Tier 1 で失敗しても、MAX_DEPTH=4 以内の computed 祖先があれば
                // そこから forward 方向に dirty_piece を適用して更新できる。
                // HalfKA_hm / HalfKA / HalfKP では既に有効だが LayerStacks では
                // 未使用だった。
                if !updated {
                    if let Some((source_idx, _depth)) = $stack.find_usable_accumulator() {
                        updated = $net.forward_update_incremental(pos, $stack, source_idx);
                    }
                }

                // --- Tier 3: 全計算 (cache 経由) ---
                if !updated {
                    let acc = &mut $stack.current_mut().accumulator;
                    if let Some(
                        super::accumulator_layer_stacks::LayerStacksAccCache::$cache_variant(c),
                    ) = cache
                    {
                        $net.refresh_accumulator_with_cache(pos, acc, c);
                    } else {
                        $net.refresh_accumulator(pos, acc);
                    }
                }
            }};
        }

        match (self, stack) {
            #[cfg(feature = "layerstacks-1536x16x32")]
            (
                Self::L1536x16x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x32(st),
            ) => {
                do_update!(net, st, L1536x16x32);
            }
            #[cfg(feature = "layerstacks-1536x32x32")]
            (
                Self::L1536x32x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x32x32(st),
            ) => {
                do_update!(net, st, L1536x32x32);
            }
            #[cfg(feature = "layerstacks-1536x8x96")]
            (
                Self::L1536x8x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x8x96(st),
            ) => {
                do_update!(net, st, L1536x8x96);
            }
            #[cfg(feature = "layerstacks-1536x16x96")]
            (
                Self::L1536x16x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x96(st),
            ) => {
                do_update!(net, st, L1536x16x96);
            }
            #[cfg(feature = "layerstacks-1536x32x96")]
            (
                Self::L1536x32x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x32x96(st),
            ) => {
                do_update!(net, st, L1536x32x96);
            }
            #[cfg(feature = "layerstacks-1536x48x96")]
            (
                Self::L1536x48x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x48x96(st),
            ) => {
                do_update!(net, st, L1536x48x96);
            }
            #[cfg(feature = "layerstacks-1536x64x96")]
            (
                Self::L1536x64x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x64x96(st),
            ) => {
                do_update!(net, st, L1536x64x96);
            }
            #[cfg(feature = "layerstacks-1536x16x48")]
            (
                Self::L1536x16x48(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x48(st),
            ) => {
                do_update!(net, st, L1536x16x48);
            }
            #[cfg(feature = "layerstacks-1024x16x48")]
            (
                Self::L1024x16x48(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1024x16x48(st),
            ) => {
                do_update!(net, st, L1024x16x48);
            }
            #[cfg(feature = "layerstacks-1024x8x64")]
            (
                Self::L1024x8x64(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L1024x8x64(st),
            ) => {
                do_update!(net, st, L1024x8x64);
            }
            #[cfg(feature = "layerstacks-768x32x96")]
            (
                Self::L768x32x96(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x32x96(st),
            ) => {
                do_update!(net, st, L768x32x96);
            }
            #[cfg(feature = "layerstacks-768x16x32")]
            (
                Self::L768x16x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x16x32(st),
            ) => {
                do_update!(net, st, L768x16x32);
            }
            #[cfg(feature = "layerstacks-768x8x32")]
            (
                Self::L768x8x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x8x32(st),
            ) => {
                do_update!(net, st, L768x8x32);
            }
            #[cfg(feature = "layerstacks-512x16x32")]
            (
                Self::L512x16x32(net),
                super::accumulator_layer_stacks::LayerStacksAccStack::L512x16x32(st),
            ) => {
                do_update!(net, st, L512x16x32);
            }
            #[allow(unreachable_patterns)]
            _ => panic!(
                "LayerStacksNetwork/LayerStacksAccStack architecture mismatch: network={net_l1}x{net_l2}x{net_l3}, stack={stack_l1}x{stack_l2}x{stack_l3}"
            ),
        }
    }

    /// PV node 復帰時に dual_cross accumulator が stale なら full refresh する。
    ///
    /// arm 7 + Refresh (= 2026-05-27): non-PV node で `update_dual_cross_accumulator`
    /// が prev → acc を memcpy (= identity update) すると 1 手分 diff が反映されず
    /// `dual_cross_stale = true` になる。 PV node で eval を呼ぶ前にこの flag を確認し、
    /// true なら現局面から full refresh して stale 値の eval 伝播を断つ。
    ///
    /// non-PV node では `non_pv_dual_skip()` が true で eval 側が dual_cross_value を
    /// 0 に gate するため、 stale 値を読むことはない (= 呼ばない)。
    ///
    /// # 呼び出し位置
    ///
    /// `update_and_evaluate_layer_stacks_cached` の 評価直前 (= PV node 限定)。
    /// non-PV node ではこの method を呼ばないため refresh コストは PV node のみに
    /// 限定される。
    #[inline]
    #[cfg(any(
        not(feature = "nnue-scalar-threat"),
        feature = "nnue-scalar-threat-coexist"
    ))]
    pub fn refresh_dual_cross_if_stale(
        &self,
        pos: &Position,
        stack: &mut super::accumulator_layer_stacks::LayerStacksAccStack,
    ) {
        #[cfg(feature = "nnue-dual-psqt")]
        {
            // arm 6 endgame-only (= 2026-05-28): `DUAL_CROSS_ENDGAME_ONLY=1` (= ply-based)
            // または `DUAL_CROSS_USE_ENDGAME_DETECTOR=1` (= position-based) で
            // opening/middle では eval 側で dual_cross 寄与を 0 に gate するため stale 値
            // が読まれることはない。 refresh は NPS 損失を伴うので skip し、 endgame
            // 突入後の最初の update path で full refresh される (= update 側
            // prev_acc.dual_cross_stale 経路)。
            if (crate::nnue::ablation::dual_cross_endgame_only()
                && !crate::nnue::dual_cross::is_endgame_ply(pos.game_ply()))
                || (crate::nnue::ablation::dual_cross_use_endgame_detector()
                    && !crate::nnue::dual_cross::current_phase_is_endgame())
            {
                return;
            }
            macro_rules! do_refresh {
                ($net:expr, $stack:expr) => {{
                    let acc = &mut $stack.current_mut().accumulator;
                    if acc.dual_cross_stale {
                        $net.refresh_dual_cross_accumulator(pos, acc);
                    }
                }};
            }
            match (self, stack) {
                #[cfg(feature = "layerstacks-1536x16x32")]
                (
                    Self::L1536x16x32(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x32(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-1536x32x32")]
                (
                    Self::L1536x32x32(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L1536x32x32(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-1536x8x96")]
                (
                    Self::L1536x8x96(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L1536x8x96(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-1536x16x96")]
                (
                    Self::L1536x16x96(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x96(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-1536x32x96")]
                (
                    Self::L1536x32x96(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L1536x32x96(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-1536x48x96")]
                (
                    Self::L1536x48x96(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L1536x48x96(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-1536x64x96")]
                (
                    Self::L1536x64x96(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L1536x64x96(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-1536x16x48")]
                (
                    Self::L1536x16x48(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x48(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-1024x16x48")]
                (
                    Self::L1024x16x48(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L1024x16x48(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-1024x8x64")]
                (
                    Self::L1024x8x64(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L1024x8x64(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-768x32x96")]
                (
                    Self::L768x32x96(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L768x32x96(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-768x16x32")]
                (
                    Self::L768x16x32(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L768x16x32(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-768x8x32")]
                (
                    Self::L768x8x32(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L768x8x32(st),
                ) => do_refresh!(net, st),
                #[cfg(feature = "layerstacks-512x16x32")]
                (
                    Self::L512x16x32(net),
                    super::accumulator_layer_stacks::LayerStacksAccStack::L512x16x32(st),
                ) => do_refresh!(net, st),
                #[allow(unreachable_patterns)]
                _ => {}
            }
        }
        #[cfg(not(feature = "nnue-dual-psqt"))]
        {
            let _ = (pos, stack);
        }
    }

    /// Scalar Threat構成ではDual Cross state自体が存在しないためno-op。
    #[cfg(all(
        feature = "nnue-scalar-threat",
        not(feature = "nnue-scalar-threat-coexist")
    ))]
    #[inline]
    pub fn refresh_dual_cross_if_stale(
        &self,
        pos: &Position,
        stack: &mut super::accumulator_layer_stacks::LayerStacksAccStack,
    ) {
        let _ = (self, pos, stack);
    }

    /// 新しい L1 サイズに対応する AccStack を作成
    pub fn new_acc_stack(&self) -> super::accumulator_layer_stacks::LayerStacksAccStack {
        match self {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x32(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x32x32(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x8x96(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x96(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x32x96(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x48x96(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x64x96(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L1536x16x48(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L1024x16x48(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<1024>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L1024x8x64(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<1024>::new(),
                )
            }
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x32x96(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<768>::new(),
                )
            }
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L768x16x32(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<768>::new(),
                )
            }
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32(_) => super::accumulator_layer_stacks::LayerStacksAccStack::L768x8x32(
                super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<768>::new(),
            ),
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32(_) => {
                super::accumulator_layer_stacks::LayerStacksAccStack::L512x16x32(
                    super::accumulator_layer_stacks::AccumulatorStackLayerStacks::<512>::new(),
                )
            }
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    }

    /// 新しい L1 サイズに対応する AccCache を作成
    pub fn new_acc_cache(&self) -> super::accumulator_layer_stacks::LayerStacksAccCache {
        match self {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L1536x16x32(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L1536x32x32(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L1536x8x96(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L1536x16x96(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L1536x32x96(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L1536x48x96(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L1536x64x96(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L1536x16x48(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<1536>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L1024x16x48(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<1024>::new(),
                )
            }
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L1024x8x64(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<1024>::new(),
                )
            }
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L768x32x96(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<768>::new(),
                )
            }
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L768x16x32(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<768>::new(),
                )
            }
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32(_) => super::accumulator_layer_stacks::LayerStacksAccCache::L768x8x32(
                super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<768>::new(),
            ),
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32(_) => {
                super::accumulator_layer_stacks::LayerStacksAccCache::L512x16x32(
                    super::accumulator_layer_stacks::AccumulatorCacheLayerStacks::<512>::new(),
                )
            }
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::nnue::constants::{FV_SCALE_HALFKA, NNUE_PYTORCH_L1};

    const TEST_L1: usize = NNUE_PYTORCH_L1;

    #[test]
    fn test_network_dimensions() {
        assert_eq!(TEST_L1, 1536);
        assert_eq!(FV_SCALE_HALFKA, 16);
    }

    /// LayerStacks NNUEファイルの読み込みと評価テスト
    ///
    /// このテストは外部NNUEファイルが必要なため通常はスキップ。
    /// 実行方法: `cargo test test_load_layer_stacks_file -- --ignored`
    ///
    /// テスト結果 (epoch82.nnue):
    /// - FT bias sum: -1
    /// - FT weight nonzero: 2,143,627
    /// - L1 bias (bucket 0): [-15, 57, -182, -97, -202, -55, 120, 1, 87, -133, -16, 44, -27, -37, -201, -186]
    /// - Initial position score: 0 (epoch82は学習初期のため)
    #[cfg(feature = "layerstacks-1536x32x96")]
    #[test]
    #[ignore]
    fn test_load_layer_stacks_file() {
        use crate::nnue::layer_stacks::{compute_bucket_index, sqr_clipped_relu_transform};

        // テスト用NNUEファイルのパスを設定してください
        let path = std::env::var("NNUE_TEST_FILE")
            .unwrap_or_else(|_| "/path/to/your/layer_stacks.nnue".to_string());

        let network = match NetworkLayerStacks1536x32x96::load(path) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("Skipping test: {e}");
                return;
            }
        };

        // Feature Transformer のバイアスが読み込まれていることを確認
        let bias_sum: i64 = network.feature_transformer.biases.0.iter().map(|&x| x as i64).sum();
        eprintln!("FT bias sum: {bias_sum}");

        // Feature Transformer の重みの一部を確認
        let weight_sample: Vec<i16> = network.feature_transformer.weights[0..10].to_vec();
        eprintln!("FT weight sample (first 10): {weight_sample:?}");

        // 異なるオフセットで重みを確認
        let weight_total = network.feature_transformer.weights.len();
        let weight_nonzero: usize =
            network.feature_transformer.weights.iter().filter(|&&x| x != 0).count();
        eprintln!("FT weight total: {weight_total}, nonzero: {weight_nonzero}");

        // 中間位置の重みをサンプル
        let mid_offset = weight_total / 2;
        let weight_mid_sample: Vec<i16> =
            network.feature_transformer.weights[mid_offset..mid_offset + 10].to_vec();
        eprintln!("FT weight sample (mid): {weight_mid_sample:?}");

        // 最初のnonzero重みの位置を探す
        let first_nonzero_pos = network.feature_transformer.weights.iter().position(|&x| x != 0);
        if let Some(weight_pos) = first_nonzero_pos {
            let sample_end = (weight_pos + 10usize).min(weight_total);
            let first_nonzero_sample: Vec<i16> =
                network.feature_transformer.weights[weight_pos..sample_end].to_vec();
            eprintln!("First nonzero at position {weight_pos}, sample: {first_nonzero_sample:?}");
            // 特徴インデックスを計算 (weight layout: [feature_index][output_dim])
            let feature_idx = weight_pos / TEST_L1;
            eprintln!("  -> Feature index: {feature_idx}");
        }

        // LayerStacks の重みの一部を確認
        let l1_bias_sample: Vec<i32> = network.layer_stacks.buckets[0].l1.biases.to_vec();
        eprintln!("L1 bias (bucket 0): {l1_bias_sample:?}");

        // 初期局面を評価
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();

        // アクティブ特徴量を確認
        use crate::nnue::features::{FeatureSet, HalfKA_hm_FeatureSet};
        use crate::types::Color;
        let active_black = HalfKA_hm_FeatureSet::collect_active_indices(&pos, Color::Black);
        eprintln!("Active features for Black: {} features", active_black.len());
        let first_5: Vec<usize> = active_black.iter().take(5).collect();
        eprintln!("  First 5 indices: {first_5:?}");

        // 最初のアクティブ特徴量の重みを確認
        if let Some(first_idx) = active_black.iter().next() {
            let offset = first_idx * TEST_L1;
            eprintln!("  Weight offset for feature {first_idx}: {offset}");
            if offset + 10 <= weight_total {
                let active_weight_sample: Vec<i16> =
                    network.feature_transformer.weights[offset..offset + 10].to_vec();
                eprintln!("  Weight sample for first active feature: {active_weight_sample:?}");
            }
        }

        let mut acc = AccumulatorLayerStacks::<TEST_L1>::new();
        network.refresh_accumulator(&pos, &mut acc);

        // Accumulatorの値を確認
        let black_acc = acc.get(0);
        let white_acc = acc.get(1);
        let black_acc_sum: i64 = black_acc.iter().map(|&x| x as i64).sum();
        let white_acc_sum: i64 = white_acc.iter().map(|&x| x as i64).sum();
        eprintln!("Black acc sum: {black_acc_sum}, White acc sum: {white_acc_sum}");
        eprintln!("Black acc sample (first 10): {:?}", &black_acc[0..10]);

        // アキュムレータの統計
        let black_min = black_acc.iter().copied().min().unwrap_or(0);
        let black_max = black_acc.iter().copied().max().unwrap_or(0);
        let black_positive: usize = black_acc.iter().filter(|&&x| x > 0).count();
        eprintln!(
            "Black acc: min={black_min}, max={black_max}, positive={black_positive}/{TEST_L1}"
        );

        // 前半と後半の統計（SqrClippedReLUでペア乗算される）
        let half = TEST_L1 / 2;
        let first_half = &black_acc[0..half];
        let second_half = &black_acc[half..TEST_L1];
        let first_positive: usize = first_half.iter().filter(|&&x| x > 0).count();
        let second_positive: usize = second_half.iter().filter(|&&x| x > 0).count();
        eprintln!(
            "First half positive: {first_positive}/{half}, Second half positive: {second_positive}/{half}"
        );

        // ペア乗算で非ゼロになるペアの数
        let mut pairs_both_positive = 0usize;
        for i in 0..half {
            if first_half[i] > 0 && second_half[i] > 0 {
                pairs_both_positive += 1;
            }
        }
        eprintln!("Pairs where both halves > 0: {pairs_both_positive}/{half}");

        // SqrClippedReLU変換後の値を確認
        let mut transformed: Aligned<[u8; TEST_L1]> = Aligned([0u8; TEST_L1]);
        sqr_clipped_relu_transform(black_acc, white_acc, &mut transformed.0);
        let transformed_sum: u64 = transformed.0.iter().map(|&x| x as u64).sum();
        let transformed_nonzero: usize = transformed.0.iter().filter(|&&x| x > 0).count();
        eprintln!("Transformed sum: {transformed_sum}, nonzero count: {transformed_nonzero}");
        eprintln!("Transformed sample (first 20): {:?}", &transformed.0[0..20]);

        // バケットインデックスを計算（玉の段に基づく）
        let side_to_move = pos.side_to_move();
        let f_king = pos.king_square(side_to_move);
        let e_king = pos.king_square(!side_to_move);
        let (f_rank, e_rank) =
            crate::nnue::layer_stacks::compute_king_ranks(side_to_move, f_king, e_king);
        let bucket_index = compute_bucket_index(f_rank, e_rank);
        eprintln!("King ranks: f={f_rank}, e={e_rank}, bucket index: {bucket_index}");

        // LayerStacks の生スコアを計算
        let raw_score = network.layer_stacks.evaluate_raw(bucket_index, &transformed.0);
        eprintln!("Raw score (before /fv_scale): {raw_score}, fv_scale: {}", network.fv_scale);

        // 評価値を計算
        let value = network.evaluate(&mut pos, &acc);
        eprintln!("Initial position score: {}", value.raw());

        // 評価値が妥当な範囲内であることを確認（-1000〜1000）
        assert!(value.raw().abs() < 1000, "Score {} is out of expected range", value.raw());

        // 様々な局面での評価値を確認
        eprintln!("\n=== Various positions ===");
        let test_positions = [
            ("初期局面", "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"),
            ("後手1歩得", "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPP1/1B5R1/LNSGKGSNL b p 1"),
            ("先手1歩得", "lnsgkgsnl/1r5b1/pppppppp1/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b P 1"),
            ("後手飛車落ち", "lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"),
            ("先手角得", "lnsgkgsnl/1r7/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b B 1"),
        ];

        for (name, sfen) in test_positions {
            pos.set_sfen(sfen).unwrap();
            network.refresh_accumulator(&pos, &mut acc);

            // raw score（/600前）を計算
            let (us_acc, them_acc) = (acc.get(0), acc.get(1));
            let mut transformed: Aligned<[u8; TEST_L1]> = Aligned([0u8; TEST_L1]);
            sqr_clipped_relu_transform(us_acc, them_acc, &mut transformed.0);
            let stm = pos.side_to_move();
            let f_k = pos.king_square(stm);
            let e_k = pos.king_square(!stm);
            let (f_r, e_r) = crate::nnue::layer_stacks::compute_king_ranks(stm, f_k, e_k);
            let bucket_idx = compute_bucket_index(f_r, e_r);
            let raw = network.layer_stacks.evaluate_raw(bucket_idx, &transformed.0);

            let val = network.evaluate(&mut pos, &acc);
            eprintln!("{:15}: {:6} (raw: {:6})", name, val.raw(), raw);
        }
    }
}
