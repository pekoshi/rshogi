//! 戦術的キャッシュ（詰み探索結果等の固定サイズハッシュテーブル）
//!
//! lockfree XOR trick で一貫性検証を行う。
//! キャッシュ対象: 1/3/5/7 手詰、df-pn 結果、check_moves count。

use rshogi_core::types::Move;

/// キャッシュエントリ（論理的な構造）
#[derive(Debug, Clone, Copy)]
pub struct TacticalCacheEntry {
    /// キーの下位 32 bit
    pub key32: u32,
    /// エントリ種別
    pub kind: u8,
    /// 探索深さ (ply)
    pub depth: u8,
    /// 最善手
    pub best_move: Move,
    /// 世代
    pub generation: u16,
}

/// 戦術的キャッシュ
///
/// 固定サイズハッシュテーブル。2-word XOR trick で lockfree 一貫性検証。
pub struct TacticalCache {
    /// テーブル本体: 各エントリは [data0, data1] の 2 ワード
    /// data0 = key32 XOR packed_data
    /// data1 = packed_data
    entries: Box<[[u64; 2]]>,
    mask: usize,
    generation: u16,
}

impl TacticalCache {
    /// 詰み発見
    pub const KIND_MATE: u8 = 1;
    /// 不詰確認
    pub const KIND_NO_MATE: u8 = 2;
    /// df-pn 証明済み
    pub const KIND_DFPN_PROVEN: u8 = 3;
    /// df-pn 不詰証明済み
    pub const KIND_DFPN_DISPROVEN: u8 = 4;

    /// 指定エントリ数で作成（2の冪に切り上げ）
    pub fn new(num_entries: usize) -> Self {
        let size = num_entries.next_power_of_two().max(16);
        let entries = vec![[0u64; 2]; size].into_boxed_slice();
        Self {
            entries,
            mask: size - 1,
            generation: 0,
        }
    }

    /// 世代を進める
    pub fn new_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// エントリを格納
    pub fn store(&mut self, key: u64, kind: u8, depth: u8, best_move: Move) {
        let key32 = (key >> 32) as u32;
        let idx = key as usize & self.mask;
        let packed = Self::pack(kind, depth, best_move, self.generation);
        self.entries[idx][0] = key32 as u64 ^ packed;
        self.entries[idx][1] = packed;
    }

    /// エントリを検索
    pub fn probe(&self, key: u64) -> Option<TacticalCacheEntry> {
        let key32 = (key >> 32) as u32;
        let idx = key as usize & self.mask;
        let data0 = self.entries[idx][0];
        let data1 = self.entries[idx][1];

        // XOR 一貫性検証
        let stored_key32 = (data0 ^ data1) as u32;
        if stored_key32 != key32 {
            return None;
        }

        Some(Self::unpack(key32, data1))
    }

    /// データをパック
    fn pack(kind: u8, depth: u8, best_move: Move, generation: u16) -> u64 {
        let move_raw = best_move.raw() as u64;
        (kind as u64) | ((depth as u64) << 8) | (move_raw << 16) | ((generation as u64) << 48)
    }

    /// データをアンパック
    fn unpack(key32: u32, packed: u64) -> TacticalCacheEntry {
        let kind = packed as u8;
        let depth = (packed >> 8) as u8;
        let move_raw = (packed >> 16) as u32;
        let generation = (packed >> 48) as u16;
        TacticalCacheEntry {
            key32,
            kind,
            depth,
            best_move: Move::from_u16(move_raw as u16),
            generation,
        }
    }
}
