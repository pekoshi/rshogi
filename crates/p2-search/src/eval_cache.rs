//! MCTS NN 評価キャッシュ (transposition で同一局面を再評価しない)
//!
//! 探索木では異なるパスから同一局面に到達することが頻繁にある (transposition)。
//! NN forward pass は重い (180M params で 2-5 ms / sample) ため、
//! 局面ハッシュをキーとして評価値をキャッシュすることで NPS を大幅に底上げする。
//!
//! 実測想定 hit 率:
//! - 序盤 (定跡圏): 5-20% (新局面ばかり)
//! - 中盤        : 30-50% (複数手順で同一局面)
//! - 終盤        : 50-70% (千日手回避、 詰み近傍の重複)
//! - 平均        : 30-40% → NPS +1.5x 程度
//!
//! 設計:
//! - 直接マップ式 (open addressing なし、 衝突時は上書き)
//! - エントリサイズ 8 byte (key32 + i32 value)
//! - サイズはコンフィグ指定 (power-of-2 に切り上げ)
//! - 同期 MCTS 想定 (単一スレッド)、 必要なら atomic 化

/// 単一エントリ: 8 byte (key32: u32 + value: i32)
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct CacheEntry {
    /// Zobrist key の上位 32 bit (validate 用)。 0 = 空エントリ
    key32: u32,
    /// 評価値 (i32 raw value、 評価器のスケールそのまま)
    value: i32,
}

/// MCTS NN 評価キャッシュ
pub struct MctsEvalCache {
    entries: Box<[CacheEntry]>,
    mask: usize,
    hits: u64,
    misses: u64,
    inserts: u64,
}

impl MctsEvalCache {
    /// 指定エントリ数で作成 (power-of-2 に切り上げ、 最小 16)
    pub fn new(num_entries: usize) -> Self {
        let size = num_entries.next_power_of_two().max(16);
        Self {
            entries: vec![CacheEntry::default(); size].into_boxed_slice(),
            mask: size - 1,
            hits: 0,
            misses: 0,
            inserts: 0,
        }
    }

    /// 全エントリをクリア (hits/misses はそのまま)
    pub fn clear(&mut self) {
        for e in self.entries.iter_mut() {
            *e = CacheEntry::default();
        }
    }

    /// 統計をリセット
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.inserts = 0;
    }

    /// 局面の評価値を引く
    ///
    /// `Some(value)` ならキャッシュヒット、 `None` ならミス。
    pub fn probe(&mut self, key: u64) -> Option<i32> {
        let key32 = (key >> 32) as u32;
        // key32 == 0 は空エントリ判定に使うため衝突回避: 0 のときは 1 に置換
        let key32 = if key32 == 0 { 1 } else { key32 };
        let idx = (key as usize) & self.mask;
        let entry = self.entries[idx];
        if entry.key32 == key32 {
            self.hits += 1;
            Some(entry.value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// 局面の評価値を格納 (衝突時は上書き)
    pub fn store(&mut self, key: u64, value: i32) {
        let key32 = (key >> 32) as u32;
        let key32 = if key32 == 0 { 1 } else { key32 };
        let idx = (key as usize) & self.mask;
        self.entries[idx] = CacheEntry { key32, value };
        self.inserts += 1;
    }

    /// hit / miss / insert 統計
    pub fn hits(&self) -> u64 {
        self.hits
    }
    pub fn misses(&self) -> u64 {
        self.misses
    }
    pub fn inserts(&self) -> u64 {
        self.inserts
    }
    /// hit rate (hits / (hits + misses))。 アクセスがなければ 0
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
    /// エントリ容量
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }
}

// ────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_miss_then_hit() {
        let mut cache = MctsEvalCache::new(64);
        assert!(cache.probe(0xdead_beef).is_none());
        assert_eq!(cache.misses(), 1);
        cache.store(0xdead_beef, 12345);
        assert_eq!(cache.probe(0xdead_beef), Some(12345));
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn collision_overwrites() {
        let mut cache = MctsEvalCache::new(16);
        // 同じ index に落ちる 2 つの異なるキー (mask=15、 同じ下位 4 bit)
        let key_a = 0x0000_0001_0000_0011u64;
        let key_b = 0x0000_0002_0000_0011u64;
        // 同じ index 、 異なる key32
        assert_eq!(key_a as usize & 15, key_b as usize & 15);
        assert_ne!(key_a >> 32, key_b >> 32);

        cache.store(key_a, 100);
        cache.store(key_b, 200); // 上書き
        assert!(cache.probe(key_a).is_none()); // a は失われた
        assert_eq!(cache.probe(key_b), Some(200));
    }

    #[test]
    fn high_zero_key32_handled() {
        // key32 == 0 ケース: 衝突回避のため 1 に置換
        let mut cache = MctsEvalCache::new(16);
        let key_zero_high = 0x0000_0000_dead_beefu64;
        cache.store(key_zero_high, 42);
        assert_eq!(cache.probe(key_zero_high), Some(42));
    }

    #[test]
    fn capacity_is_power_of_two() {
        let cache = MctsEvalCache::new(100);
        assert_eq!(cache.capacity(), 128); // 100 → 128
        let cache2 = MctsEvalCache::new(8);
        assert_eq!(cache2.capacity(), 16); // min 16
    }

    #[test]
    fn clear_resets_entries() {
        let mut cache = MctsEvalCache::new(16);
        cache.store(0xabcd_0001, 99);
        cache.clear();
        assert!(cache.probe(0xabcd_0001).is_none());
    }

    #[test]
    fn hit_rate_calculation() {
        let mut cache = MctsEvalCache::new(16);
        cache.store(1u64 << 33, 10);
        let _ = cache.probe(1u64 << 33); // hit
        let _ = cache.probe(2u64 << 33); // miss
        let _ = cache.probe(3u64 << 33); // miss
        assert!((cache.hit_rate() - 1.0 / 3.0).abs() < 1e-5);
    }
}
