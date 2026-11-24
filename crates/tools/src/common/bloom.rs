//! Cache-line blocked Bloom filter（共通モジュール）
//!
//! `psv_dedup_bloom` バイナリと `dataset` パイプラインの両方から利用する。

/// Cache-line blocked Bloom filter
///
/// 標準 Bloom filter は k 回のプローブが全域に散らばるため、
/// フィルタサイズが L3 キャッシュを超えるとメモリレイテンシが支配的になる。
///
/// Blocked Bloom Filter は全域を 512 bit (= 64 bytes = 1 cache line) のブロックに分割し、
/// h1 でブロックを選択、h2 で同一ブロック内に k 個のビットを配置する。
pub struct BloomFilter {
    blocks: Vec<u64>,
    num_blocks: u64,
    num_hashes: u32,
}

/// 1ブロック = 8 × u64 = 512 bits = 64 bytes = 1 cache line
const BLOCK_U64S: usize = 8;
const BLOCK_BITS: u32 = (BLOCK_U64S * 64) as u32; // 512

/// Bloom filter のサイズパラメータ（確保前に計算）
pub struct BloomParams {
    pub num_blocks: u64,
    pub num_hashes: u32,
    pub total_u64s: usize,
    pub size_bytes: u64,
}

impl BloomFilter {
    /// 必要なフィルタサイズを算出する（メモリ確保はしない）
    pub fn estimate(num_elements: u64, fpr: f64) -> BloomParams {
        let n = num_elements.max(1) as f64;
        // m = -n * ln(p) / (ln2)^2
        let m = (-n * fpr.ln() / (2.0_f64.ln().powi(2))).ceil() as u64;
        // k = (m/n) * ln2
        let k = ((m as f64 / n) * 2.0_f64.ln()).round().max(1.0) as u32;
        let num_blocks = m.div_ceil(BLOCK_BITS as u64);
        let total_u64s = num_blocks as usize * BLOCK_U64S;
        let size_bytes = total_u64s as u64 * 8;
        BloomParams {
            num_blocks,
            num_hashes: k,
            total_u64s,
            size_bytes,
        }
    }

    /// 算出済みパラメータでフィルタを確保する
    pub fn allocate(params: &BloomParams) -> Self {
        let blocks = vec![0u64; params.total_u64s];
        Self {
            blocks,
            num_blocks: params.num_blocks,
            num_hashes: params.num_hashes,
        }
    }

    /// フィルタに挿入し、挿入前に既に存在していた可能性があるかを返す。
    ///
    /// - `true` = おそらく重複（偽陽性あり）
    /// - `false` = 確実に新規
    #[inline]
    pub fn insert_or_check(&mut self, h1: u64, h2: u64) -> bool {
        let block_idx = (h1 % self.num_blocks) as usize;
        let block_offset = block_idx * BLOCK_U64S;

        // h2 からブロック内の k 個のプローブ位置を生成
        let h2a = h2 as u32;
        let h2b = (h2 >> 32) as u32 | 1;

        let mut all_set = true;
        for i in 0..self.num_hashes {
            let bit_pos = h2a.wrapping_add(i.wrapping_mul(h2b)) % BLOCK_BITS;
            let word_in_block = (bit_pos >> 6) as usize;
            let mask = 1u64 << (bit_pos & 63);
            // SAFETY: block_idx < num_blocks かつ word_in_block < BLOCK_U64S (8) なので
            // block_offset + word_in_block < blocks.len() が保証される。
            let w = unsafe { self.blocks.get_unchecked_mut(block_offset + word_in_block) };
            if *w & mask == 0 {
                all_set = false;
                *w |= mask;
            }
        }
        all_set
    }

    /// PackedSfen（32バイト）の重複チェック用ラッパー
    #[inline]
    pub fn insert_or_check_packed_sfen(&mut self, sfen: &[u8; 32]) -> bool {
        let (h1, h2) = hash_pair_bytes(sfen);
        self.insert_or_check(h1, h2)
    }

    /// SFEN 文字列の重複チェック用ラッパー
    #[inline]
    pub fn insert_or_check_sfen_str(&mut self, sfen: &str) -> bool {
        let (h1, h2) = hash_sfen_str(sfen);
        self.insert_or_check(h1, h2)
    }

    /// 存在チェックのみ（挿入しない）
    #[inline]
    pub fn contains(&self, h1: u64, h2: u64) -> bool {
        let block_idx = (h1 % self.num_blocks) as usize;
        let block_offset = block_idx * BLOCK_U64S;

        let h2a = h2 as u32;
        let h2b = (h2 >> 32) as u32 | 1;

        for i in 0..self.num_hashes {
            let bit_pos = h2a.wrapping_add(i.wrapping_mul(h2b)) % BLOCK_BITS;
            let word_in_block = (bit_pos >> 6) as usize;
            let mask = 1u64 << (bit_pos & 63);
            // SAFETY: 同上
            let w = unsafe { self.blocks.get_unchecked(block_offset + word_in_block) };
            if *w & mask == 0 {
                return false;
            }
        }
        true
    }
}

/// PackedSfen から 2 つの独立した 64bit FNV-1a ハッシュを生成する
#[inline]
pub fn hash_pair_bytes(data: &[u8]) -> (u64, u64) {
    let mut h1: u64 = 0xcbf29ce484222325; // FNV offset basis
    let mut h2: u64 = 0x6c62272e07bb0142; // 異なる初期値
    for &b in data.iter() {
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 ^= b as u64;
        h2 = h2.wrapping_mul(0x100000001b3);
    }
    (h1, h2 | 1)
}

/// SFEN 文字列からハッシュペアを生成する
#[inline]
pub fn hash_sfen_str(sfen: &str) -> (u64, u64) {
    hash_pair_bytes(sfen.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_basic_insert_and_check() {
        let params = BloomFilter::estimate(1000, 0.01);
        let mut bloom = BloomFilter::allocate(&params);

        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        assert!(!bloom.insert_or_check_sfen_str(sfen));
        assert!(bloom.insert_or_check_sfen_str(sfen));
    }

    #[test]
    fn bloom_packed_sfen() {
        let params = BloomFilter::estimate(1000, 0.01);
        let mut bloom = BloomFilter::allocate(&params);

        let sfen = [0u8; 32];
        assert!(!bloom.insert_or_check_packed_sfen(&sfen));
        assert!(bloom.insert_or_check_packed_sfen(&sfen));

        let sfen2 = [1u8; 32];
        assert!(!bloom.insert_or_check_packed_sfen(&sfen2));
    }

    #[test]
    fn bloom_contains_without_insert() {
        let params = BloomFilter::estimate(1000, 0.01);
        let mut bloom = BloomFilter::allocate(&params);

        let (h1, h2) = hash_sfen_str("test sfen");
        assert!(!bloom.contains(h1, h2));
        bloom.insert_or_check(h1, h2);
        assert!(bloom.contains(h1, h2));
    }
}
