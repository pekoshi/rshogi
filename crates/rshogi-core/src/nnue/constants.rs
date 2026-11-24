//! NNUE定数定義
//!
//! YaneuraOu の HalfKP 256x2-32-32 アーキテクチャに基づき、
//! ネットワーク構造とスケーリングに関する定数をまとめる。

/// 評価関数ファイルのバージョン（YaneuraOu互換）
pub const NNUE_VERSION: u32 = 0x7AF32F16;

/// アーキテクチャ文字列の最大長（破損ファイル/DoS対策）
pub const MAX_ARCH_LEN: usize = 4096;

/// 評価値のスケーリング（水匠5用: 24）
///
/// FV_SCALEは評価関数の訓練時に決まるパラメータ。
/// 同じファイル形式でも評価関数によって異なる場合がある。
/// 例: YaneuraOuのデフォルトは16だが、水匠5は24を使用。
pub const FV_SCALE: i32 = 24;

/// 評価値のスケーリング（デフォルト: 16）
///
/// nnue-pytorchでハードコードされている値（kBiasScale = 600 * 16 = 9600）。
/// YaneuraOuのデフォルト値でもある。
/// bullet-shogiで学習したモデル（scale=600）もこの値で動作する。
pub const FV_SCALE_HALFKA: i32 = 16;

/// 重みのスケーリングビット数
pub const WEIGHT_SCALE_BITS: u32 = 6;

/// SCReLU のデフォルト QA 値
///
/// FT SCReLU 出力は QA に依存せず 0〜127 に正規化されるため、
/// L1/L2 の積和スケールは常に `SCRELU_DEFAULT_QA × QB` になる。
/// QA > 127 の場合は bias スケールを調整する必要がある。
pub const SCRELU_DEFAULT_QA: i32 = 127;

/// キャッシュラインサイズ（バイト）
pub const CACHE_LINE_SIZE: usize = 64;

/// 変換後の次元数（片方の視点）
pub const TRANSFORMED_FEATURE_DIMENSIONS: usize = 256;

/// リフレッシュトリガーの数（YO kRefreshTriggers.size() 相当）
/// HalfKP の場合は FriendKingMoved のみで 1
///
/// 注意: この値を変更する場合、以下の箇所も更新が必要:
/// - `Accumulator` の `accumulation` 配列サイズ
/// - `FeatureTransformer` の `refresh_accumulator` / `update_accumulator` / `transform`
/// - `HalfKPFeatureSet::REFRESH_TRIGGERS`
pub const NUM_REFRESH_TRIGGERS: usize = 1;

/// HalfKP特徴量の次元数
/// 81（玉の位置）× FE_END（BonaPiece数）
pub const HALFKP_DIMENSIONS: usize = 81 * super::bona_piece::FE_END;

// =============================================================================
// HalfKA_hm^ アーキテクチャ用定数
// =============================================================================

/// HalfKA_hm^のバージョン（nnue-pytorch互換）
pub const NNUE_VERSION_HALFKA: u32 = 0x7AF32F20;

/// キングバケット数（Half-Mirror: 9段 × 5筋）
pub const KING_BUCKETS: usize = 45;

/// 駒入力数（DISTINGUISH_GOLDS有効時のe_king = 1629）
pub const PIECE_INPUTS_HALFKA: usize = 1629;

/// HalfKA_hm^のベース入力数（キングバケット × 駒入力）
pub const BASE_INPUTS_HALFKA: usize = KING_BUCKETS * PIECE_INPUTS_HALFKA; // 73,305

/// HalfKA_hm^の総入力次元数
///
/// nnue-pytorch標準のcoalesce済みモデル専用。
/// Factorizationの重みはBase側に畳み込み済みのため、推論時はBaseのみで計算する。
/// これにより特徴量数が半減（80→40）し、NPSが約20%向上する。
///
/// 非coalesceモデル（74,934次元）はサポートしない。
/// nnue-pytorch serialize.py でエクスポートすると自動的にcoalesceされる。
pub const HALFKA_HM_DIMENSIONS: usize = BASE_INPUTS_HALFKA; // 73,305

/// HalfKA_hm^のFactorization込み入力次元数（未coalesce）
///
/// 訓練時のみ使用。推論用モデルは serialize.py で自動的に coalesce される。
/// この定数は互換性エラー検出のために定義。
pub const HALFKA_HM_DIMENSIONS_FACTORIZED: usize = BASE_INPUTS_HALFKA + PIECE_INPUTS_HALFKA; // 74,934

// =============================================================================
// HalfKA（非ミラー）アーキテクチャ用定数（Hisui 仕様）
// =============================================================================

/// HalfKA（非ミラー）の入力平面数
///
/// Hisui の学習設定: 1548 + 81 * 2 = 1710
pub const HALFKA_PLANES: usize = 1548 + 81 * 2;

/// HalfKA（非ミラー）の総入力次元数
///
/// 81（玉位置）× 1710（入力平面）
pub const HALFKA_DIMENSIONS: usize = HALFKA_PLANES * 81; // 138,510

/// 隠れ層1の次元数（YaneuraOu classic）
pub const HIDDEN1_DIMENSIONS: usize = 32;

/// 隠れ層2の次元数（YaneuraOu classic）
pub const HIDDEN2_DIMENSIONS: usize = 32;

/// 出力次元数
pub const OUTPUT_DIMENSIONS: usize = 1;

// =============================================================================
// nnue-pytorch LayerStacks アーキテクチャ用定数
// =============================================================================

/// nnue-pytorch の Feature Transformer 出力次元数（片方の視点）
pub const NNUE_PYTORCH_L1: usize = 1536;

/// LayerStacks 16x32 バリアントの L2 直前 main 次元数
pub const LAYER_STACK_16X32_MAIN_DIM: usize = 15;

/// LayerStacks の L2 出力次元数
pub const NNUE_PYTORCH_L3: usize = 32;

/// LayerStacks のバケット数
///
/// nodchip bullet-shogi の `NUM_BUCKETS = 8` と整合 (progress8 = sigmoid を 8 等分)。
/// 以前は 9 で defined されていたが、 PROGRESS_BUCKET_THRESHOLDS は 7 thresholds で
/// 出力範囲は 0..=7 = 8 buckets。 9 は never-used で余分メモリを確保していた。
/// 2026-05-12 修正: 1536-8-96 ckpt (nodchip 出力 8 bucket) と互換にするため 8 へ。
pub const NUM_LAYER_STACK_BUCKETS: usize = 8;

/// LayerStacks 16x32 バリアントの L1層出力次元数（main 15 + skip 1 = 16）
pub const LAYER_STACK_16X32_L1_OUT: usize = LAYER_STACK_16X32_MAIN_DIM + 1; // 16

/// LayerStacks 16x32 バリアントの L2層入力次元数（sqr 15 + crelu 15 = 30）
pub const LAYER_STACK_16X32_L2_IN: usize = LAYER_STACK_16X32_MAIN_DIM * 2; // 30

/// LayerStacks 32x32 バリアントの L1層出力次元数（main 31 + skip 1 = 32）
pub const LAYER_STACK_32X32_L1_OUT: usize = 32;

/// LayerStacks 32x32 バリアントの main 次元数
pub const LAYER_STACK_32X32_MAIN_DIM: usize = LAYER_STACK_32X32_L1_OUT - 1; // 31

/// LayerStacks 32x32 バリアントの L2層入力次元数（sqr 31 + crelu 31 = 62）
pub const LAYER_STACK_32X32_L2_IN: usize = LAYER_STACK_32X32_MAIN_DIM * 2; // 62

/// LayerStacks 8x96 バリアントの L1層出力次元数（main 7 + skip 1 = 8）
/// ユーザー指定 HalfKAv2_hm 1536-8-96 用 (2026-05-12)。
pub const LAYER_STACK_8X96_L1_OUT: usize = 8;

/// LayerStacks 8x96 バリアントの main 次元数
pub const LAYER_STACK_8X96_MAIN_DIM: usize = LAYER_STACK_8X96_L1_OUT - 1; // 7

/// LayerStacks 8x96 バリアントの L2層入力次元数（sqr 7 + crelu 7 = 14）
pub const LAYER_STACK_8X96_L2_IN: usize = LAYER_STACK_8X96_MAIN_DIM * 2; // 14

/// LayerStacks 8x32 バリアントの L1層出力次元数（main 7 + skip 1 = 8）
/// nodchip 2026-06 実験で L1_OUT 16→8 (MAIN_DIM 15→7) でレーティング維持 +
/// NPS 向上、 特に 768_7_32 (FT768/L1_8/L2_32) が最適と判明したため追加。
/// 8X96 との違いは L2 出力次元 (96→32) のみで、 L1_OUT/MAIN_DIM/L2_IN は同値。
pub const LAYER_STACK_8X32_L1_OUT: usize = 8;

/// LayerStacks 8x32 バリアントの main 次元数
pub const LAYER_STACK_8X32_MAIN_DIM: usize = LAYER_STACK_8X32_L1_OUT - 1; // 7

/// LayerStacks 8x32 バリアントの L2層入力次元数（sqr 7 + crelu 7 = 14）
pub const LAYER_STACK_8X32_L2_IN: usize = LAYER_STACK_8X32_MAIN_DIM * 2; // 14

/// LayerStacks 16x96 バリアントの L1層出力次元数（main 15 + skip 1 = 16）
/// 1536-8-96 の L1=8 ボトルネック緩和用 (2026-05-14)。
pub const LAYER_STACK_16X96_L1_OUT: usize = 16;

/// LayerStacks 16x96 バリアントの main 次元数
pub const LAYER_STACK_16X96_MAIN_DIM: usize = LAYER_STACK_16X96_L1_OUT - 1; // 15

/// LayerStacks 16x96 バリアントの L2層入力次元数（sqr 15 + crelu 15 = 30）
pub const LAYER_STACK_16X96_L2_IN: usize = LAYER_STACK_16X96_MAIN_DIM * 2; // 30

/// LayerStacks 32x96 バリアントの L1層出力次元数（main 31 + skip 1 = 32）
/// 1536-16-96 を更に拡張した表現力向上版 (2026-05-15)。
pub const LAYER_STACK_32X96_L1_OUT: usize = 32;

/// LayerStacks 32x96 バリアントの main 次元数
pub const LAYER_STACK_32X96_MAIN_DIM: usize = LAYER_STACK_32X96_L1_OUT - 1; // 31

/// LayerStacks 32x96 バリアントの L2層入力次元数（sqr 31 + crelu 31 = 62）
pub const LAYER_STACK_32X96_L2_IN: usize = LAYER_STACK_32X96_MAIN_DIM * 2; // 62

/// LayerStacks 48x96 バリアントの L1層出力次元数（main 47 + skip 1 = 48）
/// P4 で 1536-32-96 を更に拡張した表現力向上版 (2026-05-17)。
pub const LAYER_STACK_48X96_L1_OUT: usize = 48;

/// LayerStacks 48x96 バリアントの main 次元数
pub const LAYER_STACK_48X96_MAIN_DIM: usize = LAYER_STACK_48X96_L1_OUT - 1; // 47

/// LayerStacks 48x96 バリアントの L2層入力次元数（sqr 47 + crelu 47 = 94）
pub const LAYER_STACK_48X96_L2_IN: usize = LAYER_STACK_48X96_MAIN_DIM * 2; // 94

/// LayerStacks 64x96 バリアントの L1層出力次元数（main 63 + skip 1 = 64）
/// P5 で 1536-32-96 を 2x 拡張した密度拡張版 (2026-05-17)。
pub const LAYER_STACK_64X96_L1_OUT: usize = 64;

/// LayerStacks 64x96 バリアントの main 次元数
pub const LAYER_STACK_64X96_MAIN_DIM: usize = LAYER_STACK_64X96_L1_OUT - 1; // 63

/// LayerStacks 64x96 バリアントの L2層入力次元数（sqr 63 + crelu 63 = 126）
pub const LAYER_STACK_64X96_L2_IN: usize = LAYER_STACK_64X96_MAIN_DIM * 2; // 126

// =============================================================================
// LayerStacks 16x48 バリアント (NPS 回復版、 2026-05-16)
// =============================================================================
//
// L1_OUT=16 / L2_IN=30 は 16x96 バリアントと同一寸法 (共通の Feature Transformer
// 出力 layout を持つ)。 異なるのは L3=48 (≒ 16x96 の半分) のみ。
//
// **重要**: 値は LAYER_STACK_16X96_* と数値同一だが、 用途上の意味付け (将来
// L1_OUT 系の独立調整余地) を保つため、 別名 const として明示する。

/// LayerStacks 16x48 バリアントの L1層出力次元数（main 15 + skip 1 = 16）。
pub const LAYER_STACK_16X48_L1_OUT: usize = 16;

/// LayerStacks 16x48 バリアントの main 次元数。
pub const LAYER_STACK_16X48_MAIN_DIM: usize = LAYER_STACK_16X48_L1_OUT - 1; // 15

/// LayerStacks 16x48 バリアントの L2層入力次元数（sqr 15 + crelu 15 = 30）。
pub const LAYER_STACK_16X48_L2_IN: usize = LAYER_STACK_16X48_MAIN_DIM * 2; // 30

/// LayerStacks 16x48 バリアントの L3層出力次元数 (hidden2)。
pub const LAYER_STACK_16X48_L3: usize = 48;

/// nnue-pytorch の隠れ層重みスケール
pub const NNUE_PYTORCH_WEIGHT_SCALE_HIDDEN: i32 = 64;

/// nnue-pytorch の出力層重みスケール
pub const NNUE_PYTORCH_WEIGHT_SCALE_OUT: i32 = 16;

/// nnue-pytorch の量子化単位
pub const NNUE_PYTORCH_QUANTIZED_ONE: i32 = 127;

// =============================================================================
// SCReLU (Squared Clipped ReLU) 用定数
// =============================================================================

/// SCReLU 量子化係数 (bullet-shogi 準拠)
///
/// SCReLU では clamp(x, 0, QA)² を計算する。
/// QA = 127 のとき、最大出力は 127² = 16,129。
///
/// スケーリング設計:
/// - 入力: i16 (FeatureTransformer出力、範囲 [-QA, QA])
/// - 出力: i32 (最大 QA² = 16,129)
/// - オーバーフロー検証: 16,129 × 127 × 512 < i32_MAX ✓
pub const SCRELU_QA: i16 = 127;

/// SCReLU L1層以降の量子化係数 (bullet-shogi 準拠)
///
/// L1層以降では QB = 64 を使用。
/// i32 として定義（中間層の i32 演算で使用するため）
pub const SCRELU_QB: i32 = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(TRANSFORMED_FEATURE_DIMENSIONS, 256);
        assert_eq!(HIDDEN1_DIMENSIONS, 32);
        assert_eq!(HIDDEN2_DIMENSIONS, 32);
        assert_eq!(OUTPUT_DIMENSIONS, 1);
    }

    /// LayerStacks 8x32 バリアント (FT768/L1_8/L2_32) の次元整合。
    /// L1_OUT=8 (= main 7 + skip 1)、 L2_IN=14 (= 7+7)、 L2_PADDED=32、 L3=32。
    /// 8X96 系と L1_OUT/MAIN_DIM/L2_IN は同値で、 L3 のみ異なる。
    #[test]
    fn test_layer_stack_8x32_dims() {
        assert_eq!(LAYER_STACK_8X32_L1_OUT, 8);
        assert_eq!(LAYER_STACK_8X32_MAIN_DIM, 7);
        assert_eq!(LAYER_STACK_8X32_L2_IN, 14);
        // skip connection: main + 1 = L1_OUT
        assert_eq!(LAYER_STACK_8X32_MAIN_DIM + 1, LAYER_STACK_8X32_L1_OUT);
        // L2_IN = main × 2 (sqr concat crelu)
        assert_eq!(LAYER_STACK_8X32_MAIN_DIM * 2, LAYER_STACK_8X32_L2_IN);
        // 8X96 系と L1_OUT/MAIN_DIM/L2_IN は数値一致 (L3 のみ 96→32)
        assert_eq!(LAYER_STACK_8X32_L1_OUT, LAYER_STACK_8X96_L1_OUT);
        assert_eq!(LAYER_STACK_8X32_MAIN_DIM, LAYER_STACK_8X96_MAIN_DIM);
        assert_eq!(LAYER_STACK_8X32_L2_IN, LAYER_STACK_8X96_L2_IN);
    }
}
