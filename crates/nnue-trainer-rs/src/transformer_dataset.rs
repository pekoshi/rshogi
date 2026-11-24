//! NNUE-B Transformer 用の token dataset。
//!
//! 設計 (memo `nnue_b_transformer_plan_20260512`):
//! - 入力: 事前 tokenize された `.tokens` バイナリ (1 record = MAX_TOKENS × i16 + value/policy 補助)
//! - tokenize は bullet fork の `psv_to_tokens.rs` (未実装) が PSV → tokens で生成する想定
//! - 1 record の layout (案、 実装は psv_to_tokens 側と整合させること):
//!   ```text
//!   bytes [0..MAX_TOKENS*2):              tokens [i16; MAX_TOKENS]   (0=PAD)
//!   bytes [MAX_TOKENS*2..+2):             score i16  (stm 視点 cp)
//!   bytes [+2..+1):                       result u8  (0=Loss, 1=Draw, 2=Win, stm 視点)
//!   bytes [+1..+1):                       bucket u8  (progress bucket、 任意)
//!   bytes [+1..+2):                       policy_idx u16 LE (0..2186、 0xFFFF=no policy)
//!   total = MAX_TOKENS*2 + 6 byte/record
//!   ```
//! - DirectSequentialDataLoader 互換 fixed-size record で memmap 風読込
//!
//! 実装状況: スケルトンのみ。 psv_to_tokens 実装後に enable する。

#![allow(dead_code)]

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::path::Path;

/// Transformer の最大 token 数 (CLS + 81 board + 14 hand + 2 king)。
pub const MAX_TOKENS: usize = 98;

/// 1 record のバイト数。
pub const TOKEN_RECORD_SIZE: usize = MAX_TOKENS * 2 + 6;

/// policy_idx sentinel
pub const POLICY_SENTINEL: u16 = 0xFFFF;

/// Transformer 用 dataset (PSV → tokens 事前変換済みファイル)。
pub struct TokenDataset {
    raw: Vec<u8>,
    pub total: usize,
}

impl TokenDataset {
    /// 全件 RAM ロード。
    pub fn load_in_memory(path: &Path, max_records: Option<usize>) -> Result<Self> {
        use std::fs::File;
        use std::io::Read;
        let mut f = File::open(path)?;
        let size = f.metadata()?.len() as usize;
        let mut total = size / TOKEN_RECORD_SIZE;
        if let Some(max) = max_records {
            total = total.min(max);
        }
        let nbytes = total * TOKEN_RECORD_SIZE;
        let mut raw = vec![0u8; nbytes];
        f.read_exact(&mut raw)?;
        eprintln!(
            "[token_dataset] loaded {} records ({} MB, record_size={})",
            total,
            nbytes / 1024 / 1024,
            TOKEN_RECORD_SIZE,
        );
        Ok(Self { raw, total })
    }

    /// indices で指定された record の batch を構築。
    pub fn make_batch(&self, indices: &[usize], device: &Device) -> Result<TokenBatch> {
        let b = indices.len();
        let mut tokens = vec![0i64; b * MAX_TOKENS];
        let mut score = vec![0f32; b];
        let mut result = vec![0f32; b];
        let mut policy = vec![-1i64; b];

        for (i, &idx) in indices.iter().enumerate() {
            let off = idx * TOKEN_RECORD_SIZE;
            for k in 0..MAX_TOKENS {
                let lo = self.raw[off + k * 2] as u16;
                let hi = self.raw[off + k * 2 + 1] as u16;
                tokens[i * MAX_TOKENS + k] = ((hi << 8) | lo) as i16 as i64;
            }
            let sc_off = off + MAX_TOKENS * 2;
            let lo = self.raw[sc_off] as u16;
            let hi = self.raw[sc_off + 1] as u16;
            score[i] = (((hi << 8) | lo) as i16) as f32;
            result[i] = self.raw[sc_off + 2] as f32 / 2.0;
            // bucket: sc_off + 3 (未使用)
            let p_lo = self.raw[sc_off + 4] as u16;
            let p_hi = self.raw[sc_off + 5] as u16;
            let p_u16 = (p_hi << 8) | p_lo;
            policy[i] = if p_u16 == POLICY_SENTINEL {
                -1
            } else {
                p_u16 as i64
            };
        }

        Ok(TokenBatch {
            tokens: Tensor::from_vec(tokens, (b, MAX_TOKENS), device)?,
            score: Tensor::from_vec(score, b, device)?,
            result: Tensor::from_vec(result, b, device)?,
            policy_idx: Tensor::from_vec(policy, b, device)?,
        })
    }
}

pub struct TokenBatch {
    /// `[B, MAX_TOKENS]` i64、 0=PAD
    pub tokens: Tensor,
    /// `[B]` f32、 stm 視点 cp
    pub score: Tensor,
    /// `[B]` f32、 stm 視点 result (0=loss, 0.5=draw, 1=win)
    pub result: Tensor,
    /// `[B]` i64、 -1=sentinel または 0..2186
    pub policy_idx: Tensor,
}
