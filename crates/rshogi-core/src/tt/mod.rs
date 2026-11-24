//! 置換表モジュール
//!
//! 探索結果をキャッシュする置換表 (= Transposition Table)。
//!
//! - `TTEntry`: エントリ (= 16 byte、 AtomicU64×2、 XOR torn detection)
//! - `Cluster`: エントリのグループ (= 48 byte、 64 byte align)
//! - `TranspositionTable`: テーブル本体
//! - 世代管理
//! - prefetch
//!
//! # XOR + AtomicU64 設計 (= Stockfish/YaneuraOu 準拠)
//!
//! クラスターインデックスは 64bit キーの上位ビットで決定し、
//! クラスター内マッチングに下位 16bit を使用する。
//! key_xor_data = (key16 as u64) ^ data として格納、 read 時に復元し torn 検出。
//! multi-thread search で同一 cluster へ並行 write しても false positive なし
//! (= XOR mismatch なら invalid 扱い = cache miss)。

mod alloc;
mod entry;
mod table;

pub use entry::{TTData, TTEntry};
pub use table::{ProbeResult, TranspositionTable};

/// クラスターサイズ（エントリ数）
/// YaneuraOu準拠: 10bytes × 3 + 2padding = 32bytes
pub const CLUSTER_SIZE: usize = 3;

/// Generation関連の定数
pub const GENERATION_BITS: u32 = 3;
pub const GENERATION_DELTA: u8 = 1 << GENERATION_BITS; // 8
pub const GENERATION_CYCLE: u16 = 255 + GENERATION_DELTA as u16;
pub const GENERATION_MASK: u16 = 0xF8; // (0xFF << GENERATION_BITS) as u8
