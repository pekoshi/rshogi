//! Policy logits の TT-key 直結キャッシュ。
//!
//! ## 目的
//!
//! Transformer の forward は ~100μs-1ms と αβ の単 node より 10-100× 重い。
//! 同一局面の transposition を hit させて推論を **スキップ** することで、
//! 中盤 hook を有効化した時の NPS 退行を緩和する。
//!
//! ## 設計
//!
//! - 直接マップ (open addressing なし、 衝突時は replace)
//! - エントリ size: `POLICY_SIZE * 4 + 8` ≒ 8.76 KB / entry (大きいが個数は少ない)
//! - default 1024 entries ≒ 9 MB。 USI option `MidgamePolicyCacheEntries` で
//!   調整可能。
//! - lock-free **single-thread** 前提 (root 1 推論のみなので race なし)。 将来
//!   並列探索 hook 化する際に re-design 余地あり。
//! - `Box<[PolicyCacheEntry]>` を 1 度確保し、 ホットパスで再 alloc しない。

use crate::move_lookup::POLICY_SIZE;

/// 1 エントリ: zobrist key + raw logits。
///
/// `#[repr(C, align(64))]` で cache line 整列 (logits は 2187 × 4 = 8748 B、
/// 線形 walk なので prefetcher friendly)。
#[repr(C, align(64))]
pub struct PolicyCacheEntry {
    /// Zobrist key (0 = 空エントリ、 NaN/0 衝突対策で別途 occupied フラグは
    /// 持たない: 実 zobrist が 0 になる確率は 2^-64 で実用上無視可能)。
    key: u64,
    /// raw logits (softmax 前)。
    logits: [f32; POLICY_SIZE],
}

impl PolicyCacheEntry {
    /// 空エントリ。
    pub fn empty() -> Self {
        Self {
            key: 0,
            logits: [0.0f32; POLICY_SIZE],
        }
    }

    /// このエントリが指定 key と一致するか。
    #[inline]
    pub fn matches(&self, key: u64) -> bool {
        self.key == key && key != 0
    }

    /// 内部 logits slice (読み取り専用)。
    #[inline]
    pub fn logits(&self) -> &[f32; POLICY_SIZE] {
        &self.logits
    }
}

/// Policy cache の hit/miss 統計。
#[derive(Clone, Copy, Debug, Default)]
pub struct PolicyCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
}

impl PolicyCacheStats {
    /// hit 率 (0.0 - 1.0)。 一度も probe されていない場合は 0.0。
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// 直接マップ式 policy cache。
///
/// ## 使用パターン
///
/// ```ignore
/// let mut cache = PolicyCache::new(1024);
/// if let Some(logits) = cache.probe(pos.key()) {
///     // hit: 既存 logits を使う
/// } else {
///     // miss: 推論を実行し、 結果を store
///     evaluator.evaluate(pos, &mut buf)?;
///     cache.store(pos.key(), &buf);
/// }
/// ```
pub struct PolicyCache {
    entries: Box<[PolicyCacheEntry]>,
    mask: usize,
    stats: PolicyCacheStats,
}

impl PolicyCache {
    /// `num_entries` を power-of-2 に切り上げて確保 (最小 16)。
    ///
    /// メモリは即時 alloc される (entry あたり ~8.76 KB)。 大量に確保すると
    /// 物理メモリを食うため、 default は 1024 entries (≒ 9 MB) を推奨。
    pub fn new(num_entries: usize) -> Self {
        let size = num_entries.next_power_of_two().max(16);
        // `vec![]` で初期化すると Clone が要求されるため、 with_capacity + push で組み立てる。
        let mut entries: Vec<PolicyCacheEntry> = Vec::with_capacity(size);
        for _ in 0..size {
            entries.push(PolicyCacheEntry::empty());
        }
        Self {
            entries: entries.into_boxed_slice(),
            mask: size - 1,
            stats: PolicyCacheStats::default(),
        }
    }

    /// エントリ数 (power-of-2)。
    #[inline]
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// 統計取得。
    #[inline]
    pub fn stats(&self) -> PolicyCacheStats {
        self.stats
    }

    /// 全エントリクリア (統計は維持)。
    pub fn clear(&mut self) {
        for e in self.entries.iter_mut() {
            e.key = 0;
        }
    }

    /// 統計リセット。
    pub fn reset_stats(&mut self) {
        self.stats = PolicyCacheStats::default();
    }

    /// 指定 key で probe。 hit なら logits の参照を返す、 miss なら `None`。
    pub fn probe(&mut self, key: u64) -> Option<&[f32; POLICY_SIZE]> {
        let idx = (key as usize) & self.mask;
        let entry = &self.entries[idx];
        if entry.matches(key) {
            self.stats.hits += 1;
            Some(&entry.logits)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// 指定 key で logits を保存 (既存エントリは上書き)。
    pub fn store(&mut self, key: u64, logits: &[f32]) {
        debug_assert_eq!(logits.len(), POLICY_SIZE);
        if key == 0 {
            // sentinel と衝突するため store しない (極稀)。
            return;
        }
        let idx = (key as usize) & self.mask;
        let entry = &mut self.entries[idx];
        entry.key = key;
        entry.logits.copy_from_slice(logits);
        self.stats.inserts += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_rounds_up_to_power_of_two() {
        assert_eq!(PolicyCache::new(15).capacity(), 16);
        assert_eq!(PolicyCache::new(16).capacity(), 16);
        assert_eq!(PolicyCache::new(100).capacity(), 128);
        assert_eq!(PolicyCache::new(0).capacity(), 16);
    }

    #[test]
    fn store_then_probe_hits() {
        let mut cache = PolicyCache::new(16);
        let key = 0xDEAD_BEEF_CAFE_BABE;
        let mut logits = vec![0.0f32; POLICY_SIZE];
        logits[42] = 3.5;
        logits[2000] = -1.25;

        cache.store(key, &logits);
        let hit = cache.probe(key).expect("should hit");
        assert!((hit[42] - 3.5).abs() < 1e-6);
        assert!((hit[2000] - (-1.25)).abs() < 1e-6);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().inserts, 1);
    }

    #[test]
    fn probe_miss_then_store_then_hit() {
        let mut cache = PolicyCache::new(16);
        let key = 0x1234_5678;
        assert!(cache.probe(key).is_none());
        assert_eq!(cache.stats().misses, 1);

        let logits = vec![1.0f32; POLICY_SIZE];
        cache.store(key, &logits);
        assert!(cache.probe(key).is_some());
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn key_zero_is_not_stored() {
        // sentinel と衝突するので store はスキップされる。
        let mut cache = PolicyCache::new(16);
        let logits = vec![1.0f32; POLICY_SIZE];
        cache.store(0, &logits);
        // probe(0) は空エントリ判定で miss。
        assert!(cache.probe(0).is_none());
    }

    #[test]
    fn clear_drops_all_entries_but_keeps_stats() {
        let mut cache = PolicyCache::new(16);
        let logits = vec![0.5f32; POLICY_SIZE];
        cache.store(1, &logits);
        cache.store(2, &logits);
        let _ = cache.probe(1);
        assert_eq!(cache.stats().hits, 1);

        cache.clear();
        assert!(cache.probe(1).is_none());
        assert_eq!(cache.stats().hits, 1, "stats should be preserved by clear()");
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn collision_overwrites() {
        // 同じ low-bit を持つ 2 key で衝突 → 最後の store が勝つ。
        let mut cache = PolicyCache::new(16);
        let mask = (cache.capacity() - 1) as u64;
        let key1 = 0x1234_5670 & mask | 0x10_0000_0000;
        let key2 = 0xABCD_1230 & mask | 0x20_0000_0000;
        // 両 key の (key & mask) を一致させる
        let collision_idx = 0x5u64;
        let k1 = (1u64 << 32) | collision_idx;
        let k2 = (2u64 << 32) | collision_idx;
        assert_eq!(k1 as usize & cache.mask, k2 as usize & cache.mask);

        let _ = key1;
        let _ = key2;

        let logits1 = vec![0.1f32; POLICY_SIZE];
        let logits2 = vec![0.9f32; POLICY_SIZE];
        cache.store(k1, &logits1);
        cache.store(k2, &logits2);
        let probe = cache.probe(k1);
        assert!(probe.is_none(), "k1 should be evicted by k2 collision");
        let probe2 = cache.probe(k2).expect("k2 should hit");
        assert!((probe2[0] - 0.9).abs() < 1e-6);
    }
}
