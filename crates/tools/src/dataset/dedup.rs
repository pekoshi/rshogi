//! Bloom filter ベースの SFEN 重複排除
//!
//! Pass 1 で使用し、SFEN 文字列の重複を検出する。

use crate::common::bloom::{BloomFilter, BloomParams};

/// データセット用の重複排除フィルタ
pub struct DedupFilter {
    bloom: BloomFilter,
}

impl DedupFilter {
    /// 推定要素数と偽陽性率からフィルタを構築する
    pub fn new(estimated_elements: u64, fpr: f64) -> Self {
        let params = BloomFilter::estimate(estimated_elements, fpr);
        log::info!(
            "Bloom filter: {:.1} GiB ({} blocks, k={}, FPR={:.4}%)",
            params.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            params.num_blocks,
            params.num_hashes,
            fpr * 100.0,
        );
        let bloom = BloomFilter::allocate(&params);
        Self { bloom }
    }

    /// フィルタサイズを事前計算する（メモリチェック用）
    pub fn estimate(estimated_elements: u64, fpr: f64) -> BloomParams {
        BloomFilter::estimate(estimated_elements, fpr)
    }

    /// SFEN 文字列を挿入し、重複かどうかを返す。
    ///
    /// - `true` = おそらく重複
    /// - `false` = 新規（挿入済み）
    #[inline]
    pub fn insert_or_check(&mut self, sfen: &str) -> bool {
        self.bloom.insert_or_check_sfen_str(sfen)
    }

    /// PackedSfen バイト列での重複チェック
    #[inline]
    pub fn insert_or_check_packed(&mut self, sfen: &[u8; 32]) -> bool {
        self.bloom.insert_or_check_packed_sfen(sfen)
    }
}
