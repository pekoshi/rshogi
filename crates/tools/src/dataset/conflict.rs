//! SFEN 衝突検出
//!
//! 同一 SFEN に対して異なる best_move が存在するケースを検出する。

use std::collections::HashMap;

/// SFEN 衝突検出器
pub struct ConflictDetector {
    /// sfen → (best_move, count)
    seen: HashMap<String, (String, u32)>,
    conflicts: Vec<ConflictEntry>,
}

/// 衝突エントリ
#[derive(Debug, Clone)]
pub struct ConflictEntry {
    pub sfen: String,
    pub move_a: String,
    pub move_b: String,
}

impl Default for ConflictDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ConflictDetector {
    pub fn new() -> Self {
        Self {
            seen: HashMap::new(),
            conflicts: Vec::new(),
        }
    }

    /// SFEN と best_move を登録する。衝突があれば true を返す。
    pub fn check(&mut self, sfen: &str, best_move: &str) -> bool {
        if let Some((existing_move, _count)) = self.seen.get(sfen) {
            if existing_move != best_move {
                self.conflicts.push(ConflictEntry {
                    sfen: sfen.to_string(),
                    move_a: existing_move.clone(),
                    move_b: best_move.to_string(),
                });
                return true;
            }
        } else {
            self.seen.insert(sfen.to_string(), (best_move.to_string(), 1));
        }
        false
    }

    /// 衝突エントリを取得する
    pub fn into_conflicts(self) -> Vec<ConflictEntry> {
        self.conflicts
    }

    /// 衝突数
    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }
}
