//! 層別レザバーサンプリング
//!
//! source × phase × value_bucket の層別でレザバーサンプリングを行う。
//! 各層に目標件数を割り当て、ストリーミングで均等にサンプリングする。

use crate::dataset::config::SamplingConfig;
use crate::dataset::types::Phase;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// レザバーサンプリングの層別キー
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StratumKey {
    pub source: String,
    pub phase: Phase,
    pub value_bucket: usize,
}

/// 単一層のレザバー
struct Reservoir<T> {
    capacity: usize,
    items: Vec<T>,
    seen: u64,
}

impl<T> Reservoir<T> {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: Vec::with_capacity(capacity.min(1024)),
            seen: 0,
        }
    }

    /// アイテムを追加する。レザバーがフルの場合はランダムに置換する。
    fn add(&mut self, item: T, rng: &mut ChaCha8Rng) -> bool {
        self.seen += 1;
        if self.items.len() < self.capacity {
            self.items.push(item);
            true
        } else {
            let j = rng.random_range(0..self.seen);
            if (j as usize) < self.capacity {
                self.items[j as usize] = item;
                true
            } else {
                false
            }
        }
    }

    fn into_items(self) -> Vec<T> {
        self.items
    }
}

/// 層別レザバーサンプラー
pub struct StratifiedSampler<T> {
    reservoirs: HashMap<StratumKey, Reservoir<T>>,
    rng: ChaCha8Rng,
    value_bucket_thresholds: Vec<i32>,
}

impl<T> StratifiedSampler<T> {
    /// サンプラーを構築する
    ///
    /// `source_names`: ソース名のリスト
    /// `config`: サンプリング設定
    /// `seed`: 乱数シード
    pub fn new(source_names: &[String], config: &SamplingConfig, seed: u64) -> Self {
        use rand::SeedableRng;

        let rng = ChaCha8Rng::seed_from_u64(seed);
        let total = if config.total_positions > 0 {
            config.total_positions
        } else {
            u64::MAX / 2 // 実質無制限
        };

        // value bucket の閾値を設定
        let value_bucket_thresholds: Vec<i32> =
            config.value_buckets.iter().map(|(t, _)| *t).collect();

        let phases = [Phase::Opening, Phase::Middle, Phase::Endgame];

        // フェーズ比率マップ
        let phase_ratio_map: HashMap<&str, f64> =
            config.phase_ratios.iter().map(|(k, v)| (k.as_str(), *v)).collect();

        // 各層の目標件数を計算
        let mut reservoirs = HashMap::new();
        let num_sources = source_names.len().max(1);

        for source in source_names {
            for &phase in &phases {
                let phase_ratio = phase_ratio_map.get(phase.as_str()).copied().unwrap_or(0.33);

                for (bucket_idx, (_, bucket_ratio)) in config.value_buckets.iter().enumerate() {
                    let capacity = (total as f64 * phase_ratio * bucket_ratio / num_sources as f64)
                        .ceil() as usize;
                    let key = StratumKey {
                        source: source.clone(),
                        phase,
                        value_bucket: bucket_idx,
                    };
                    reservoirs.insert(key, Reservoir::new(capacity));
                }
            }
        }

        Self {
            reservoirs,
            rng,
            value_bucket_thresholds,
        }
    }

    /// value_cp から bucket index を決定する
    pub fn value_bucket_index(&self, value_cp: i32) -> usize {
        let abs = value_cp.abs();
        for (i, &threshold) in self.value_bucket_thresholds.iter().enumerate() {
            if abs <= threshold {
                return i;
            }
        }
        self.value_bucket_thresholds.len().saturating_sub(1)
    }

    /// アイテムをサンプリングに追加する
    pub fn add(&mut self, key: StratumKey, item: T) -> bool {
        if let Some(reservoir) = self.reservoirs.get_mut(&key) {
            reservoir.add(item, &mut self.rng)
        } else {
            false
        }
    }

    /// 全てのサンプリング結果を取り出す
    pub fn into_all_items(self) -> Vec<T> {
        let mut all = Vec::new();
        for (_, reservoir) in self.reservoirs {
            all.extend(reservoir.into_items());
        }
        all
    }

    /// 層別の統計を取得
    pub fn stats(&self) -> HashMap<StratumKey, (usize, u64)> {
        self.reservoirs
            .iter()
            .map(|(k, r)| (k.clone(), (r.items.len(), r.seen)))
            .collect()
    }
}
