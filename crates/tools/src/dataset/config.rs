//! データセットパイプラインの TOML 設定ファイル定義

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// データセットパイプライン設定
#[derive(Debug, Deserialize)]
pub struct DatasetConfig {
    /// 出力ディレクトリ
    pub output_dir: PathBuf,

    /// 乱数シード
    #[serde(default = "default_seed")]
    pub seed: u64,

    /// データソース定義
    pub sources: Vec<SourceConfig>,

    /// フィルタ設定
    #[serde(default)]
    pub filter: FilterConfig,

    /// 正規化設定
    #[serde(default)]
    pub normalize: NormalizeConfig,

    /// サンプリング設定
    #[serde(default)]
    pub sampling: SamplingConfig,

    /// train/valid/test 分割設定
    #[serde(default)]
    pub split: SplitConfig,

    /// Policy 生成設定
    #[serde(default)]
    pub policy: PolicyConfig,

    /// Bloom filter 設定
    #[serde(default)]
    pub bloom: BloomConfig,
}

fn default_seed() -> u64 {
    42
}

/// データソース定義
#[derive(Debug, Deserialize)]
pub struct SourceConfig {
    /// ソース名（レポート・タグに使用）
    pub name: String,

    /// 入力パス（ファイルまたはディレクトリ）
    pub path: PathBuf,

    /// ディレクトリ指定時の glob パターン
    #[serde(default = "default_pattern")]
    pub pattern: String,

    /// サンプリング優先度（大きいほど高い）
    #[serde(default = "default_priority")]
    pub priority: u32,

    /// ソースからのサンプル比率 (0.0–1.0)
    #[serde(default = "default_sample_ratio")]
    pub sample_ratio: f64,
}

fn default_pattern() -> String {
    "*.bin".to_string()
}

fn default_priority() -> u32 {
    1
}

fn default_sample_ratio() -> f64 {
    1.0
}

/// フィルタ設定
#[derive(Debug, Clone, Deserialize)]
pub struct FilterConfig {
    /// |score| の最大値（これを超えるレコードを除外）
    #[serde(default = "default_score_abs_max")]
    pub score_abs_max: i32,

    /// score のクリップ値
    #[serde(default = "default_score_clip")]
    pub score_clip: i32,

    /// 最小手数
    #[serde(default = "default_ply_min")]
    pub ply_min: u32,

    /// 最大手数
    #[serde(default = "default_ply_max")]
    pub ply_max: u32,

    /// mate 局面を除外するか
    #[serde(default = "default_true")]
    pub exclude_mate: bool,

    /// mate 判定の閾値
    #[serde(default = "default_mate_threshold")]
    pub mate_threshold: i32,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            score_abs_max: default_score_abs_max(),
            score_clip: default_score_clip(),
            ply_min: default_ply_min(),
            ply_max: default_ply_max(),
            exclude_mate: true,
            mate_threshold: default_mate_threshold(),
        }
    }
}

fn default_score_abs_max() -> i32 {
    32000
}
fn default_score_clip() -> i32 {
    2000
}
fn default_ply_min() -> u32 {
    1
}
fn default_ply_max() -> u32 {
    512
}
fn default_true() -> bool {
    true
}
fn default_mate_threshold() -> i32 {
    30000
}

/// 正規化設定
#[derive(Debug, Deserialize)]
pub struct NormalizeConfig {
    /// 目標スケール
    #[serde(default = "default_target_scale")]
    pub target_scale: String,

    /// cp スケールを自動検出するか
    #[serde(default = "default_true")]
    pub auto_detect_scale: bool,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            target_scale: default_target_scale(),
            auto_detect_scale: true,
        }
    }
}

fn default_target_scale() -> String {
    "cp".to_string()
}

/// サンプリング設定
#[derive(Debug, Deserialize)]
pub struct SamplingConfig {
    /// 目標総ポジション数（0 = 無制限）
    #[serde(default)]
    pub total_positions: u64,

    /// フェーズ別目標比率 [["opening", 0.25], ...]
    #[serde(default = "default_phase_ratios")]
    pub phase_ratios: Vec<(String, f64)>,

    /// 評価値バケット [[100, 0.25], ...]
    #[serde(default = "default_value_buckets")]
    pub value_buckets: Vec<(i32, f64)>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            total_positions: 0,
            phase_ratios: default_phase_ratios(),
            value_buckets: default_value_buckets(),
        }
    }
}

fn default_phase_ratios() -> Vec<(String, f64)> {
    vec![
        ("opening".to_string(), 0.25),
        ("middle".to_string(), 0.45),
        ("endgame".to_string(), 0.30),
    ]
}

fn default_value_buckets() -> Vec<(i32, f64)> {
    vec![(100, 0.25), (500, 0.35), (1200, 0.25), (2000, 0.15)]
}

/// train/valid/test 分割設定
#[derive(Debug, Deserialize)]
pub struct SplitConfig {
    #[serde(default = "default_train_ratio")]
    pub train_ratio: f64,

    #[serde(default = "default_valid_ratio")]
    pub valid_ratio: f64,

    #[serde(default = "default_test_ratio")]
    pub test_ratio: f64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: default_train_ratio(),
            valid_ratio: default_valid_ratio(),
            test_ratio: default_test_ratio(),
        }
    }
}

fn default_train_ratio() -> f64 {
    0.98
}
fn default_valid_ratio() -> f64 {
    0.01
}
fn default_test_ratio() -> f64 {
    0.01
}

/// Policy 生成設定
#[derive(Debug, Deserialize)]
pub struct PolicyConfig {
    /// 合法手リストを生成するか
    #[serde(default = "default_true")]
    pub generate_legal_moves: bool,

    /// ターゲット種別: "onehot" | "soft"
    #[serde(default = "default_target_type")]
    pub target_type: String,

    /// soft target の温度パラメータ
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            generate_legal_moves: true,
            target_type: default_target_type(),
            temperature: default_temperature(),
        }
    }
}

fn default_target_type() -> String {
    "onehot".to_string()
}
fn default_temperature() -> f64 {
    150.0
}

/// Bloom filter 設定
#[derive(Debug, Deserialize)]
pub struct BloomConfig {
    /// 偽陽性率 (0.0–1.0)
    #[serde(default = "default_fpr")]
    pub fpr: f64,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self { fpr: default_fpr() }
    }
}

fn default_fpr() -> f64 {
    0.001
}

impl DatasetConfig {
    /// TOML ファイルから設定を読み込む
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("設定ファイル読み込み失敗: {}", path.display()))?;
        let config: Self = toml::from_str(&content)
            .with_context(|| format!("設定ファイルのパース失敗: {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// 設定値の整合性チェック
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.sources.is_empty(), "sources が空です");
        anyhow::ensure!(
            self.bloom.fpr > 0.0 && self.bloom.fpr < 1.0,
            "bloom.fpr は 0.0〜1.0 の範囲で指定してください"
        );

        let split_sum = self.split.train_ratio + self.split.valid_ratio + self.split.test_ratio;
        anyhow::ensure!(
            (split_sum - 1.0).abs() < 0.001,
            "split ratios の合計が 1.0 になりません: {split_sum}"
        );

        for src in &self.sources {
            anyhow::ensure!(!src.name.is_empty(), "source の name が空です");
            anyhow::ensure!(
                src.sample_ratio > 0.0 && src.sample_ratio <= 1.0,
                "source '{}' の sample_ratio が範囲外です: {}",
                src.name,
                src.sample_ratio
            );
        }

        Ok(())
    }
}
