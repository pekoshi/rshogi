//! シャッフル + train/valid/test 分割

use crate::dataset::config::SplitConfig;
use crate::dataset::types::UnifiedRecord;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

/// シャッフル結果
pub struct SplitResult {
    pub train: Vec<UnifiedRecord>,
    pub valid: Vec<UnifiedRecord>,
    pub test: Vec<UnifiedRecord>,
}

/// Fisher-Yates シャッフル + 分割
pub fn shuffle_and_split(
    mut records: Vec<UnifiedRecord>,
    config: &SplitConfig,
    seed: u64,
) -> SplitResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    records.shuffle(&mut rng);

    let n = records.len();
    let test_count = (n as f64 * config.test_ratio).round() as usize;
    let valid_count = (n as f64 * config.valid_ratio).round() as usize;

    // 末尾から分割
    let test_start = n.saturating_sub(test_count);
    let valid_start = test_start.saturating_sub(valid_count);

    let test = records.split_off(test_start);
    let valid = records.split_off(valid_start.min(records.len()));
    let train = records;

    SplitResult { train, valid, test }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(sfen: &str) -> UnifiedRecord {
        UnifiedRecord {
            sfen: sfen.to_string(),
            best_move: "7g7f".to_string(),
            legal_moves: Vec::new(),
            policy_target: None,
            value_cp: 100,
            result: 1,
            ply: 10,
            source: "test".to_string(),
            phase: "middle".to_string(),
            tags: Vec::new(),
        }
    }

    #[test]
    fn split_ratios_match() {
        let records: Vec<_> = (0..1000).map(|i| make_record(&format!("sfen{i}"))).collect();
        let config = SplitConfig {
            train_ratio: 0.98,
            valid_ratio: 0.01,
            test_ratio: 0.01,
        };
        let result = shuffle_and_split(records, &config, 42);

        assert_eq!(result.train.len() + result.valid.len() + result.test.len(), 1000);
        assert!(result.test.len() == 10);
        assert!(result.valid.len() == 10);
    }
}
