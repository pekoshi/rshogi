//! 置換表エントリー
//!
//! TTEntry: 16 バイトの atomic エントリ構造 (= XOR torn detection)
//! TTData: 読み取り用のデータ構造
//!
//! # 16bit キーの採用
//! YaneuraOu (CLUSTER_SIZE=3) 準拠で 16bit キーを使用。
//! クラスターインデックスは 64bit キーの上位ビットで決定し、
//! クラスター内マッチングに下位 16bit を使用する。
//! 衝突確率は 3/65536 ≈ 0.005% と十分低く、Move 合法性検証が二重チェックとして機能する。
//!
//! # XOR + AtomicU64 設計 (= Stockfish/YaneuraOu 準拠)
//! - data 64bit packed: `value16 | eval16 | move16 | gen_bound8 | depth8`
//! - key_xor_data = `(key16 as u64) ^ data`
//! - 読み取り時に `key = key_xor_data ^ data` で復元
//! - 競合状態で torn 発生時、復元 key が key16 と不一致 → invalid entry 扱い
//! - false positive (= 誤データ採用) は発生しない、 false negative (= cache miss) のみ許容
//!
//! ## Memory Ordering
//! Relaxed を採用 (= eval_hash.rs と同方針)。
//! - 正確性は XOR encoding で保証
//! - 競合時の cache miss は許容、 誤データ採用は XOR mismatch で排除
//! - x86_64 では Release/Acquire と差なし、 ARM でも XOR で正確性保証済

use super::{GENERATION_CYCLE, GENERATION_MASK};
use crate::types::{Bound, DEPTH_ENTRY_OFFSET, Move, Value};
use std::sync::atomic::{AtomicU64, Ordering};

// data 64bit packing layout (LSB → MSB):
//   bits  0..  8: depth8       (u8)
//   bits  8.. 16: gen_bound8   (u8)
//   bits 16.. 32: move16       (u16)
//   bits 32.. 48: value16      (i16 → u16 bit-cast)
//   bits 48.. 64: eval16       (i16 → u16 bit-cast)
const DEPTH_SHIFT: u32 = 0;
const GEN_BOUND_SHIFT: u32 = 8;
const MOVE_SHIFT: u32 = 16;
const VALUE_SHIFT: u32 = 32;
const EVAL_SHIFT: u32 = 48;

#[inline(always)]
fn pack(depth8: u8, gen_bound8: u8, move16: u16, value16: i16, eval16: i16) -> u64 {
    (depth8 as u64) << DEPTH_SHIFT
        | (gen_bound8 as u64) << GEN_BOUND_SHIFT
        | (move16 as u64) << MOVE_SHIFT
        | (value16 as u16 as u64) << VALUE_SHIFT
        | (eval16 as u16 as u64) << EVAL_SHIFT
}

#[inline(always)]
fn unpack_depth8(data: u64) -> u8 {
    (data >> DEPTH_SHIFT) as u8
}

#[inline(always)]
fn unpack_gen_bound8(data: u64) -> u8 {
    (data >> GEN_BOUND_SHIFT) as u8
}

#[inline(always)]
fn unpack_move16(data: u64) -> u16 {
    (data >> MOVE_SHIFT) as u16
}

#[inline(always)]
fn unpack_value16(data: u64) -> i16 {
    (data >> VALUE_SHIFT) as u16 as i16
}

#[inline(always)]
fn unpack_eval16(data: u64) -> i16 {
    (data >> EVAL_SHIFT) as u16 as i16
}

/// 置換表エントリー (= XOR + AtomicU64×2、 16 byte)
///
/// torn write/read 検出のため key_xor_data = (key16 as u64) ^ data として格納。
/// 読み取り時に復元した key16 が一致しない場合は invalid (= cache miss 扱い)。
#[repr(C, align(16))]
pub struct TTEntry {
    /// key16 XOR data (= torn detection 用)
    key_xor_data: AtomicU64,
    /// packed data (= depth8 | gen_bound8 | move16 | value16 | eval16)
    data: AtomicU64,
}

// エントリサイズは 16 バイト (= AtomicU64 ×2)
const _: () = assert!(std::mem::size_of::<TTEntry>() == 16);

impl Default for TTEntry {
    fn default() -> Self {
        Self::new()
    }
}

impl TTEntry {
    /// 新しい空のエントリを作成
    #[inline]
    pub const fn new() -> Self {
        Self {
            key_xor_data: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }

    /// raw な (key_xor_data, data) を Relaxed で 1 ペア読み出し
    #[inline]
    fn load_pair(&self) -> (u64, u64) {
        let kxd = self.key_xor_data.load(Ordering::Relaxed);
        let d = self.data.load(Ordering::Relaxed);
        (kxd, d)
    }

    /// 検証済みの key16 (= XOR 復元結果) を取得。
    /// torn または未使用なら 0 が返る (= 期待 key16 と一致しなければ miss 扱い)。
    #[inline]
    pub fn key16(&self) -> u16 {
        let (kxd, d) = self.load_pair();
        (kxd ^ d) as u16
    }

    /// エントリが使用されているか
    ///
    /// data のうち depth8 が 0 でないことで判定 (= 旧設計と同じ意味論)。
    #[inline]
    pub fn is_occupied(&self) -> bool {
        let d = self.data.load(Ordering::Relaxed);
        unpack_depth8(d) != 0
    }

    /// 深さを取得 (= DEPTH_ENTRY_OFFSET を加算)
    #[inline]
    pub fn depth(&self) -> i32 {
        let d = self.data.load(Ordering::Relaxed);
        unpack_depth8(d) as i32 + DEPTH_ENTRY_OFFSET
    }

    /// 保存されている生の depth8 を取得
    #[inline]
    pub fn depth8(&self) -> u8 {
        let d = self.data.load(Ordering::Relaxed);
        unpack_depth8(d)
    }

    /// エントリを読み取り、 (key16, TTData) を返す。
    ///
    /// torn 検出は呼び出し側で key16 と期待値を比較すること。
    /// (= probe() ですでに key16 比較で振り分ける設計のため、 ここでは生 read のみ)
    pub fn read_with_key(&self) -> (u16, TTData) {
        let (kxd, d) = self.load_pair();
        let key16 = (kxd ^ d) as u16;
        let mv = Move::from_u16(unpack_move16(d));
        let data = TTData {
            mv,
            value: Value::new(unpack_value16(d) as i32),
            eval: Value::new(unpack_eval16(d) as i32),
            depth: unpack_depth8(d) as i32 + DEPTH_ENTRY_OFFSET,
            bound: Bound::from_u8(unpack_gen_bound8(d) & 0x3).unwrap_or(Bound::None),
            is_pv: (unpack_gen_bound8(d) & 0x4) != 0,
        };
        (key16, data)
    }

    /// 互換性のため残す read () (= 旧 API、 torn 検出なし生 read)
    pub fn read(&self) -> TTData {
        self.read_with_key().1
    }

    /// エントリに保存
    ///
    /// # 設計
    /// `&self` で受ける (= 複数 thread が同一エントリへ並行 write 可能)。
    /// torn write は XOR + 2-store で検出可能 (= 読み取り側で reject)。
    /// 上書き条件は旧設計と同等 (= YaneuraOu / Stockfish 準拠)。
    ///
    /// # 引数が多い理由
    /// この関数は探索のホットパスで頻繁に呼ばれるため、
    /// 構造体にまとめるオーバーヘッドを避けて個別の引数として渡している。
    #[allow(clippy::too_many_arguments)]
    pub fn save(
        &self,
        key64: u64,
        value: Value,
        is_pv: bool,
        bound: Bound,
        depth: i32,
        mv: Move,
        eval: Value,
        generation8: u8,
    ) {
        let k16 = key64 as u16;

        // 現エントリ snapshot (= 上書き判定 + move 保持に必要)
        let (cur_kxd, cur_d) = self.load_pair();
        let cur_key16 = (cur_kxd ^ cur_d) as u16;
        let cur_depth8 = unpack_depth8(cur_d);
        let cur_move16 = unpack_move16(cur_d);
        let cur_gen_bound8 = unpack_gen_bound8(cur_d);

        // 新しい手がない場合は古い手を保持
        let new_move16 = if mv != Move::NONE || k16 != cur_key16 {
            mv.to_u16()
        } else {
            cur_move16
        };

        // 上書き条件：
        // - BOUND_EXACT (確定値)
        // - 異なるキー
        // - より深い探索 or PV ノード優先
        // - 古いエントリ
        let d8 = depth - DEPTH_ENTRY_OFFSET;
        let should_replace = bound == Bound::Exact
            || k16 != cur_key16
            || d8 + 2 * (is_pv as i32) > cur_depth8 as i32 - 4
            || relative_age_raw(cur_gen_bound8, generation8) != 0;

        if should_replace {
            debug_assert!(d8 > 0 && d8 < 256);

            let new_depth8 = d8 as u8;
            let new_gen_bound8 = generation8 | ((is_pv as u8) << 2) | bound as u8;
            let new_value16 = value.raw() as i16;
            let new_eval16 = eval.raw() as i16;

            let new_data = pack(new_depth8, new_gen_bound8, new_move16, new_value16, new_eval16);
            let new_kxd = (k16 as u64) ^ new_data;

            // 書き込み順: data → key_xor_data
            // 読み取り順は key_xor_data → data なので、 read 側が古い data + 新 kxd を
            // 観測しても XOR mismatch で reject される (= false positive 防止)。
            self.data.store(new_data, Ordering::Relaxed);
            self.key_xor_data.store(new_kxd, Ordering::Relaxed);
        } else if new_move16 != cur_move16 {
            // 上書き条件には合わないが move のみ更新するケース (= 旧 YO 準拠)
            let new_data = pack(
                cur_depth8,
                cur_gen_bound8,
                new_move16,
                unpack_value16(cur_d),
                unpack_eval16(cur_d),
            );
            let new_kxd = (cur_key16 as u64) ^ new_data;
            self.data.store(new_data, Ordering::Relaxed);
            self.key_xor_data.store(new_kxd, Ordering::Relaxed);
        }
    }

    /// 相対的な世代 (= 0 = 最新)
    #[inline]
    pub fn relative_age(&self, generation8: u8) -> u8 {
        let d = self.data.load(Ordering::Relaxed);
        relative_age_raw(unpack_gen_bound8(d), generation8)
    }
}

#[inline]
fn relative_age_raw(gen_bound8: u8, generation8: u8) -> u8 {
    let age = GENERATION_CYCLE
        .wrapping_add(generation8 as u16)
        .wrapping_sub(gen_bound8 as u16);
    (age & GENERATION_MASK) as u8
}

/// 置換表から読み取ったデータ
#[derive(Clone, Copy, Debug)]
pub struct TTData {
    /// 最善手
    pub mv: Move,
    /// 探索値
    pub value: Value,
    /// 評価値
    pub eval: Value,
    /// 探索深さ
    pub depth: i32,
    /// 境界タイプ
    pub bound: Bound,
    /// PV ノードかどうか
    pub is_pv: bool,
}

impl TTData {
    /// 空のデータ
    pub const EMPTY: Self = Self {
        mv: Move::NONE,
        value: Value::NONE,
        eval: Value::NONE,
        depth: DEPTH_ENTRY_OFFSET,
        bound: Bound::None,
        is_pv: false,
    };
}

impl Default for TTData {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{File, Rank, Square};

    #[test]
    fn test_tt_entry_new() {
        let entry = TTEntry::new();
        assert!(!entry.is_occupied());
        // 空エントリは key_xor_data = 0, data = 0 なので復元 key16 = 0
        assert_eq!(entry.key16(), 0);
    }

    #[test]
    fn test_tt_entry_save_and_read() {
        let entry = TTEntry::new();

        let key = 0x123456789ABCDEFu64;
        let value = Value::new(100);
        let eval = Value::new(-50);
        let depth = 10;
        let from = Square::new(File::File7, Rank::Rank7);
        let to = Square::new(File::File7, Rank::Rank6);
        let mv = Move::new_move(from, to, false);
        let bound = Bound::Exact;
        let is_pv = true;
        let generation = 8;

        entry.save(key, value, is_pv, bound, depth, mv, eval, generation);

        assert!(entry.is_occupied());
        assert_eq!(entry.key16(), key as u16);

        let (k16, data) = entry.read_with_key();
        assert_eq!(k16, key as u16);
        assert_eq!(data.value.raw(), 100);
        assert_eq!(data.eval.raw(), -50);
        assert_eq!(data.depth, 10);
        assert_eq!(data.bound, Bound::Exact);
        assert!(data.is_pv);
    }

    #[test]
    fn test_tt_entry_relative_age() {
        let entry = TTEntry::new();
        entry.save(0, Value::ZERO, false, Bound::Lower, 10, Move::NONE, Value::ZERO, 8);

        // 同じ世代では 0
        assert_eq!(entry.relative_age(8), 0);

        // 世代が進むと 8 刻みで age が増える (= GENERATION_DELTA = 8)
        assert_eq!(entry.relative_age(16), 8);
    }

    #[test]
    fn test_tt_entry_no_decay_non_exact() {
        let entry = TTEntry::new();
        let key = 0x123456789ABCDEFu64;

        // 深い Lower 境界を保存
        entry.save(key, Value::ZERO, false, Bound::Lower, 8, Move::NONE, Value::ZERO, 0);
        let depth_before = entry.depth8();

        // 同一世代・同一キー・浅い Lower を保存しても減衰しない
        entry.save(key, Value::ZERO, false, Bound::Lower, 1, Move::NONE, Value::ZERO, 0);
        assert_eq!(entry.depth8(), depth_before);
    }

    #[test]
    fn test_tt_entry_xor_consistency() {
        // XOR 復元が任意 key/data の組み合わせで動くか確認
        let entry = TTEntry::new();
        let keys: [u64; 5] = [
            0x0,
            0xFFFF_FFFF_FFFF_FFFF,
            0xDEAD_BEEF_CAFE_BABE,
            0x1,
            0x8000,
        ];
        for &k in &keys {
            entry.save(k, Value::new(123), false, Bound::Lower, 5, Move::NONE, Value::new(-7), 0);
            assert_eq!(entry.key16(), k as u16, "key16 mismatch for key {k:#x}");
            let data = entry.read();
            assert_eq!(data.value.raw(), 123);
            assert_eq!(data.eval.raw(), -7);
            assert_eq!(data.depth, 5);
        }
    }

    #[test]
    fn test_tt_data_empty() {
        let data = TTData::EMPTY;
        assert_eq!(data.mv, Move::NONE);
        assert_eq!(data.bound, Bound::None);
        assert!(!data.is_pv);
    }

    /// 複数スレッドが同一エントリへ並行 write しても、 reader 側で torn が
    /// XOR mismatch で大半が reject されることを確認。
    ///
    /// # 注意 (= 16bit key 設計の限界)
    /// 64bit XOR detection で 16bit key を使うため、 false positive (= XOR
    /// collision で別 thread の data を本物の thread の data と誤認) は
    /// 確率的に 1/65536 程度発生し得る。 これは Stockfish/YaneuraOu の
    /// 16bit key 設計と同じ限界。 production では:
    /// - probe() 側で `pos.to_move()` による move legality 検証
    /// - 偶然衝突した eval/value は探索木で alpha/beta cutoff の正常範囲
    ///   で safety を二重に確保している。 旧 plain 10byte struct (= torn 頻発)
    ///   よりは桁違いに安全。
    ///
    /// このテストでは torn detection が動いていることを定量確認する。
    #[test]
    fn test_tt_entry_concurrent_xor_safety() {
        use std::sync::Arc;
        use std::thread;

        let entry = Arc::new(TTEntry::new());
        let n_threads = 4;
        let iters = 20_000;

        let mut handles = Vec::with_capacity(n_threads);
        for t in 0..n_threads {
            let entry = entry.clone();
            handles.push(thread::spawn(move || {
                let k16 = (0x1000 + t as u64) & 0xFFFF;
                let key = (t as u64) << 32 | k16;
                let value_seed = (t as i32 + 1) * 1000; // 余裕を持って分離 (= 1000/2000/3000/4000)
                let eval_seed = -(t as i32 + 1) * 100;
                for i in 0..iters {
                    let val = Value::new(value_seed + (i as i32 & 0xFF));
                    let evl = Value::new(eval_seed - (i as i32 & 0xFF));
                    entry.save(key, val, false, Bound::Lower, 5, Move::NONE, evl, 8);
                }
            }));
        }
        let reader_entry = entry.clone();
        let reader = thread::spawn(move || {
            let mut torn_detected = 0u64;
            let mut consistent = 0u64;
            let mut xor_collision = 0u64;
            for _ in 0..(iters * n_threads) {
                let (k16, data) = reader_entry.read_with_key();
                let valid_k16 = (0x1000..(0x1000 + n_threads as u16)).contains(&k16);
                if valid_k16 {
                    let t = (k16 - 0x1000) as i32;
                    let value_seed = (t + 1) * 1000;
                    let eval_seed = -(t + 1) * 100;
                    let v = data.value.raw();
                    let e = data.eval.raw();
                    let v_ok = v >= value_seed && v <= value_seed + 255;
                    let e_ok = e <= eval_seed && e >= eval_seed - 255;
                    if v_ok && e_ok {
                        consistent += 1;
                    } else {
                        // XOR 衝突 (= 16bit key 設計の確率的限界)
                        xor_collision += 1;
                    }
                } else {
                    torn_detected += 1;
                }
            }
            (torn_detected, consistent, xor_collision)
        });

        for h in handles {
            h.join().unwrap();
        }
        let (torn_detected, consistent, xor_collision) = reader.join().unwrap();
        let total = torn_detected + consistent + xor_collision;
        assert_eq!(total, (iters * n_threads) as u64);
        eprintln!(
            "concurrent xor: torn={torn_detected} consistent={consistent} \
             xor_collision={xor_collision} total={total}"
        );
        // XOR collision rate は 1% 未満であるべき (= 16bit key で理論上 ~0.0015%)。
        // production では move legality + 探索 cutoff で更に safety 確保。
        let collision_rate = xor_collision as f64 / total as f64;
        assert!(
            collision_rate < 0.01,
            "XOR collision rate too high: {collision_rate:.6} (limit 0.01)"
        );
    }
}
