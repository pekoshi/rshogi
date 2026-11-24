//! df-pn 専用置換表 (dlshogi 流 cluster + proof piece search 対応)
//!
//! ## 設計
//!
//! ### Cluster 構造
//! - 1 cluster = `CLUSTER_SIZE` entries (4)
//! - cluster index = `key & mask`
//! - cluster 内で hash_high + hand match を線形探索
//! - hit なしなら最古 generation の entry を上書き
//!
//! ### Proof piece 対応
//! - entry に `Hand` 情報を保持
//! - OR-node の同 hash_high で「より少ない手駒で詰む」 entry があれば、 親も同条件で詰む
//! - AND-node の同 hash_high で「より多い手駒でも不詰」 entry があれば、 親も同条件で不詰
//!
//! ### 制限
//! - 完全な proof piece dominance 判定は実装簡略化のため "exact match" のみ
//! - dlshogi 流の完全 dominance は将来拡張

use rshogi_core::types::{Hand, Move};

/// df-pn 置換表エントリ (一つの局面情報)
#[derive(Clone, Copy)]
pub struct DfpnTtEntry {
    /// キーの上位 32 bit (hash_high)
    pub hash_high: u32,
    /// その時の手番側の手駒
    pub hand: Hand,
    /// proof number
    pub pn: u32,
    /// disproof number
    pub dn: u32,
    /// 最善手
    pub best_move: Move,
    /// generation (世代、 上書き選択用)
    pub generation: u8,
    /// パディング (atomic でない事を明示するための align 微調整)
    _pad: [u8; 3],
}

impl DfpnTtEntry {
    pub const EMPTY: Self = Self {
        hash_high: 0,
        hand: Hand::EMPTY,
        pn: 1,
        dn: 1,
        best_move: Move::NONE,
        generation: 0,
        _pad: [0; 3],
    };

    /// 空の entry か
    #[inline]
    pub const fn is_empty(&self) -> bool {
        // EMPTY 判定は pn==1 && dn==1 で代用 (実 entry は pn==1 && dn==1 にはなりにくい)
        self.pn == 1 && self.dn == 1 && self.hash_high == 0
    }
}

/// 1 cluster あたりのエントリ数 (small で OK、 4 で十分な hit 率)
pub const CLUSTER_SIZE: usize = 4;

/// 1 cluster (連続したメモリ領域)
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct DfpnCluster {
    pub entries: [DfpnTtEntry; CLUSTER_SIZE],
}

impl DfpnCluster {
    pub const EMPTY: Self = Self {
        entries: [DfpnTtEntry::EMPTY; CLUSTER_SIZE],
    };
}

/// df-pn 置換表
pub struct DfpnTranspositionTable {
    clusters: Box<[DfpnCluster]>,
    mask: usize,
    generation: u8,
}

impl DfpnTranspositionTable {
    /// 指定サイズ (MB) で作成
    pub fn new(size_mb: usize) -> Self {
        let cluster_size = std::mem::size_of::<DfpnCluster>();
        let num_clusters = ((size_mb * 1024 * 1024) / cluster_size).next_power_of_two().max(16);
        let clusters = vec![DfpnCluster::EMPTY; num_clusters].into_boxed_slice();
        Self {
            clusters,
            mask: num_clusters - 1,
            generation: 0,
        }
    }

    /// 次の search 開始時に generation を進める
    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        // 0 は EMPTY の generation 値なので、 0 はスキップ
        if self.generation == 0 {
            self.generation = 1;
        }
    }

    /// エントリを検索 (hash_high + hand exact match)
    pub fn probe(&self, key: u64, hand: Hand) -> Option<&DfpnTtEntry> {
        let hash_high = (key >> 32) as u32;
        let cluster_idx = (key as usize) & self.mask;
        let cluster = &self.clusters[cluster_idx];
        cluster
            .entries
            .iter()
            .find(|&entry| entry.hash_high == hash_high && entry.hand == hand)
            .map(|v| v as _)
    }

    /// 詰み証明済 entry の dominance lookup (OR-node 用)
    ///
    /// 同 `hash_high` で `entry.pn == 0` (詰み証明済) かつ `entry.hand <= hand` の entry を探す。
    /// 見つかれば現在の局面も詰み証明 (より多くの手駒を持っていれば詰み方が増えるだけ)。
    pub fn probe_proof_dominance(&self, key: u64, hand: Hand) -> Option<&DfpnTtEntry> {
        let hash_high = (key >> 32) as u32;
        let cluster_idx = (key as usize) & self.mask;
        let cluster = &self.clusters[cluster_idx];
        cluster
            .entries
            .iter()
            .find(|&entry| {
                entry.hash_high == hash_high
                    && entry.pn == 0
                    && hand.is_superior_or_equal(entry.hand)
            })
            .map(|v| v as _)
    }

    /// 不詰証明済 entry の dominance lookup (AND-node 用)
    ///
    /// 同 `hash_high` で `entry.dn == 0` (不詰証明済) かつ `entry.hand >= hand` の entry を探す。
    /// 見つかれば現在の局面も不詰証明 (より少ない手駒では詰み手段が減るだけ)。
    pub fn probe_disproof_dominance(&self, key: u64, hand: Hand) -> Option<&DfpnTtEntry> {
        let hash_high = (key >> 32) as u32;
        let cluster_idx = (key as usize) & self.mask;
        let cluster = &self.clusters[cluster_idx];
        cluster
            .entries
            .iter()
            .find(|&entry| {
                entry.hash_high == hash_high
                    && entry.dn == 0
                    && entry.hand.is_superior_or_equal(hand)
            })
            .map(|v| v as _)
    }

    /// エントリを格納
    ///
    /// 既存 entry があれば更新、 なければ最古 generation の slot に上書き
    pub fn store(&mut self, key: u64, hand: Hand, pn: u32, dn: u32, best_move: Move) {
        let hash_high = (key >> 32) as u32;
        let cluster_idx = (key as usize) & self.mask;
        let generation = self.generation;
        let cluster = &mut self.clusters[cluster_idx];

        // 既存 entry があれば更新
        for entry in cluster.entries.iter_mut() {
            if entry.hash_high == hash_high && entry.hand == hand {
                entry.pn = pn;
                entry.dn = dn;
                entry.best_move = best_move;
                entry.generation = generation;
                return;
            }
        }

        // 空 slot を優先、 なければ最古 generation を上書き
        let mut victim_idx = 0;
        let mut victim_score = i32::MAX;
        for (i, entry) in cluster.entries.iter().enumerate() {
            if entry.is_empty() {
                victim_idx = i;
                break;
            }
            // score = 世代差 (大きいほど古い) — wrapping 込み
            let age = generation.wrapping_sub(entry.generation) as i32;
            // 解決済 (pn=0 or dn=0) entry は score を下げて温存
            let preserve = if entry.pn == 0 || entry.dn == 0 {
                -100
            } else {
                0
            };
            let score = -(age + preserve);
            if score < victim_score {
                victim_score = score;
                victim_idx = i;
            }
        }
        cluster.entries[victim_idx] = DfpnTtEntry {
            hash_high,
            hand,
            pn,
            dn,
            best_move,
            generation,
            _pad: [0; 3],
        };
    }

    /// テーブルをクリア
    pub fn clear(&mut self) {
        for cluster in self.clusters.iter_mut() {
            *cluster = DfpnCluster::EMPTY;
        }
        self.generation = 0;
    }
}
