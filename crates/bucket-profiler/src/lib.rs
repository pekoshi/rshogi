//! bucket-profiler library
//!
//! PSV (PackedSfenValue, 40 byte/record、 YaneuraOu / Bullet 互換) から
//! `progress8kpabs` bucket idx (0..=7) を計算するための薄い関数群。
//!
//! ## 設計方針
//!
//! - `crates/tools/` の `packed_sfen.rs` から **PSV → SFEN 復号部分のみ**
//!   を独立 module として再実装した。 `tools` crate は L3 generic 不整合で
//!   現在 build 不可のため、 依存を切り離した形で同等機能を持たせる。
//! - 復号結果 SFEN は `rshogi_core::position::Position::set_sfen()` で
//!   `Position` に取り込み、 そのまま rshogi-core の
//!   `compute_progress8kpabs_sum()` + `progress_sum_to_bucket()` に渡す。
//! - bucket selector の閾値・進行度モデルは rshogi-core の API をそのまま
//!   使うため、 実機エンジンと完全に同じ bucket idx が出る。
//!
//! ## API
//!
//! - [`PackedSfenValue`]: 40 byte PSV のパース。
//! - [`unpack_packed_sfen_to_position`]: 32 byte packed sfen → `Position`。
//! - [`progress_bucket_for_record`]: PSV 1 件 → bucket idx 0..=7。
//! - [`BucketHistogram`]: bucket 集計コンテナ (8-bin、 JSON 出力対応)。
//! - [`load_progress_kpabs_weights`]: `progress.bin` (f64 KP 線形回帰重み) の loader。

pub mod packed_sfen;

use anyhow::{anyhow, Context, Result};
use rshogi_core::nnue::{
    compute_progress8kpabs_sum, progress_sum_to_bucket, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
};
use rshogi_core::position::Position;
use serde::Serialize;
use std::path::Path;

pub use packed_sfen::{unpack_sfen, PackedSfenValue, PSV_RECORD_SIZE};

/// bucket 数 (rshogi-core の `Progress8KPAbs` と整合)。
pub const NUM_BUCKETS: usize = 8;

/// PSV 1 レコードから bucket idx (0..=7) を計算。
///
/// `weights` は progress.bin から読み込んだ f32 配列 (`SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS` 要素)。
///
/// # Errors
///
/// PSV の復号失敗、 SFEN parse 失敗時は `Err` を返す。
pub fn progress_bucket_for_record(
    psv: &PackedSfenValue,
    weights: &[f32],
    scratch_pos: &mut Position,
) -> Result<usize> {
    let sfen = unpack_sfen(&psv.sfen).map_err(|e| anyhow!("unpack_sfen failed: {e}"))?;
    scratch_pos.set_sfen(&sfen).map_err(|e| anyhow!("set_sfen failed: {e}"))?;
    let sum = compute_progress8kpabs_sum(scratch_pos, weights);
    Ok(progress_sum_to_bucket(sum))
}

/// `Position` から bucket idx を計算する低レベル API。
pub fn progress_bucket_for_position(pos: &Position, weights: &[f32]) -> usize {
    let sum = compute_progress8kpabs_sum(pos, weights);
    progress_sum_to_bucket(sum)
}

/// progress.bin (f64 KP 線形回帰重み) を読み込み、 f32 へ変換して返す。
///
/// rshogi-usi の `load_progress_coeff_kpabs` と同等のロジック。
pub fn load_progress_kpabs_weights<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    let path_ref = path.as_ref();
    let bytes = std::fs::read(path_ref)
        .with_context(|| format!("progress.bin の読み込みに失敗: {}", path_ref.display()))?;
    let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f64>();
    if bytes.len() != expected {
        return Err(anyhow!(
            "progress.bin size mismatch: got {} bytes, expected {}",
            bytes.len(),
            expected
        ));
    }
    let weights: Vec<f32> = bytes
        .chunks_exact(std::mem::size_of::<f64>())
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("chunk size is checked")) as f32)
        .collect();
    debug_assert_eq!(weights.len(), SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS);
    Ok(weights)
}

/// bucket 集計結果 (8-bin、 stats 付き)。
#[derive(Debug, Clone, Serialize)]
pub struct BucketHistogram {
    /// 各 bucket の count。
    pub counts: [u64; NUM_BUCKETS],
    /// 復号成功した record の総数。
    pub total: u64,
    /// 復号失敗した record の数。
    pub failed: u64,
}

impl Default for BucketHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl BucketHistogram {
    pub fn new() -> Self {
        Self {
            counts: [0; NUM_BUCKETS],
            total: 0,
            failed: 0,
        }
    }

    /// bucket idx を 1 件加算。
    pub fn record(&mut self, bucket: usize) {
        debug_assert!(bucket < NUM_BUCKETS, "bucket out of range: {bucket}");
        let idx = bucket.min(NUM_BUCKETS - 1);
        self.counts[idx] += 1;
        self.total += 1;
    }

    /// 復号失敗を 1 件カウント。
    pub fn record_failure(&mut self) {
        self.failed += 1;
    }

    /// 各 bucket の比率 (0.0..=1.0)。 total=0 のときは 0.0。
    pub fn ratios(&self) -> [f64; NUM_BUCKETS] {
        let mut out = [0.0; NUM_BUCKETS];
        if self.total == 0 {
            return out;
        }
        let denom = self.total as f64;
        for (slot, &count) in out.iter_mut().zip(self.counts.iter()) {
            *slot = count as f64 / denom;
        }
        out
    }

    /// 比率の (min, min_idx, max, max_idx, stdev) を返す。
    pub fn stats(&self) -> BucketStats {
        let ratios = self.ratios();
        let mean = 1.0 / NUM_BUCKETS as f64;
        let mut min_r = f64::INFINITY;
        let mut max_r = f64::NEG_INFINITY;
        let mut min_idx = 0usize;
        let mut max_idx = 0usize;
        let mut var_sum = 0.0f64;
        for (i, &r) in ratios.iter().enumerate() {
            if r < min_r {
                min_r = r;
                min_idx = i;
            }
            if r > max_r {
                max_r = r;
                max_idx = i;
            }
            let d = r - mean;
            var_sum += d * d;
        }
        let stdev = (var_sum / NUM_BUCKETS as f64).sqrt();
        if !min_r.is_finite() {
            min_r = 0.0;
        }
        if !max_r.is_finite() {
            max_r = 0.0;
        }
        BucketStats {
            min_ratio: min_r,
            min_bucket: min_idx,
            max_ratio: max_r,
            max_bucket: max_idx,
            stdev,
            judgment: Self::judge(max_r),
            mean,
        }
    }

    fn judge(max_ratio: f64) -> &'static str {
        if max_ratio > 0.50 {
            "critical"
        } else if max_ratio > 0.25 {
            "warning"
        } else {
            "ok"
        }
    }
}

/// `BucketHistogram::stats()` の結果。
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BucketStats {
    pub min_ratio: f64,
    pub min_bucket: usize,
    pub max_ratio: f64,
    pub max_bucket: usize,
    pub stdev: f64,
    /// "ok" / "warning" / "critical"
    pub judgment: &'static str,
    /// 期待値 (= 1/NUM_BUCKETS)。
    pub mean: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_basic() {
        let mut h = BucketHistogram::new();
        for _ in 0..100 {
            h.record(4);
        }
        let stats = h.stats();
        assert_eq!(h.total, 100);
        assert_eq!(stats.max_bucket, 4);
        assert!((stats.max_ratio - 1.0).abs() < 1e-9);
        assert_eq!(stats.judgment, "critical");
    }

    #[test]
    fn histogram_uniform() {
        let mut h = BucketHistogram::new();
        for b in 0..8 {
            for _ in 0..10 {
                h.record(b);
            }
        }
        let stats = h.stats();
        assert!((stats.max_ratio - 0.125).abs() < 1e-9);
        assert!((stats.min_ratio - 0.125).abs() < 1e-9);
        assert!(stats.stdev < 1e-9);
        assert_eq!(stats.judgment, "ok");
    }

    #[test]
    fn judge_thresholds() {
        // max=30% (>25%、 ≤50%) → "warning"
        let mut h = BucketHistogram::new();
        for _ in 0..30 {
            h.record(0);
        }
        for _ in 0..30 {
            h.record(1);
        }
        for _ in 0..20 {
            h.record(2);
        }
        for _ in 0..20 {
            h.record(3);
        }
        assert_eq!(h.stats().max_ratio, 0.30);
        assert_eq!(h.stats().judgment, "warning");

        // max=60% (>50%) → "critical"
        let mut h2 = BucketHistogram::new();
        for _ in 0..60 {
            h2.record(0);
        }
        for _ in 0..40 {
            h2.record(1);
        }
        assert_eq!(h2.stats().judgment, "critical");

        // max=22% (≤25%) → "ok"
        let mut h3 = BucketHistogram::new();
        for _ in 0..22 {
            h3.record(0);
        }
        for _ in 0..14 {
            h3.record(1);
        }
        for _ in 0..14 {
            h3.record(2);
        }
        for _ in 0..14 {
            h3.record(3);
        }
        for _ in 0..14 {
            h3.record(4);
        }
        for _ in 0..14 {
            h3.record(5);
        }
        for _ in 0..4 {
            h3.record(6);
        }
        for _ in 0..4 {
            h3.record(7);
        }
        assert_eq!(h3.stats().judgment, "ok");
    }

    #[test]
    fn psv_unpack_hirate_bucket() {
        // 平手初期局面を pack → unpack → bucket 計算。 zero-weight の場合は bucket 4。
        use rshogi_core::nnue::SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;
        use rshogi_core::position::Position;

        let mut pos = Position::new();
        pos.set_hirate();

        let weights = vec![0.0f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
        let bucket = progress_bucket_for_position(&pos, &weights);
        // 全ゼロ重みで sum=0、 PROGRESS_BUCKET_THRESHOLDS[3]=0.0 と等しいので partition_point は 4。
        assert_eq!(bucket, 4);
    }
}
