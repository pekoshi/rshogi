//! P2 NNUE バイナリファイルフォーマット
//!
//! ヘッダ 32 バイト + Layer 1/2/3 の重みデータ。
//! 総サイズ ≈ 2.13 MB。

use std::io::{self, Read};

use crate::types::{P2_INPUT_DIM, P2_L1_DIM, P2_L2_DIM, P2_OUTPUT_DIM};

/// マジックナンバー "P2NN" (LE)
pub const P2_NNUE_MAGIC: u32 = 0x4E4E3250;

/// ファイルフォーマットバージョン
///
/// v1: value のみ
/// v2: value + policy head (256→2187, dlshogi 形式)
/// v3: + outlook head (256→1, 大局観 scalar)
pub const P2_NNUE_VERSION: u32 = 3;

/// head_flags ビット定義（reserved1 フィールドに格納）
pub const HEAD_FLAG_VALUE: u32 = 1 << 0;
/// bit 1: policy hint 有効
pub const HEAD_FLAG_POLICY: u32 = 1 << 1;
/// bit 2: uncertainty head 有効（v1 未使用）
pub const HEAD_FLAG_UNCERTAINTY: u32 = 1 << 2;

/// 既知の head_flags ビットマスク
const KNOWN_HEAD_FLAGS: u32 = HEAD_FLAG_VALUE | HEAD_FLAG_POLICY | HEAD_FLAG_UNCERTAINTY;

/// P2 NNUE ファイルヘッダ (32 バイト)
#[derive(Debug, Clone, Copy)]
pub struct P2NnueHeader {
    pub magic: u32,
    pub version: u32,
    pub input_dim: u32,
    pub l1_dim: u32,
    pub l2_dim: u32,
    pub output_dim: u32,
    /// policy metadata 用予約
    pub reserved1: u32,
    pub reserved2: u32,
}

impl P2NnueHeader {
    /// デフォルトの有効なヘッダを生成
    pub fn default_valid() -> Self {
        Self {
            magic: P2_NNUE_MAGIC,
            version: P2_NNUE_VERSION,
            input_dim: P2_INPUT_DIM as u32,
            l1_dim: P2_L1_DIM as u32,
            l2_dim: P2_L2_DIM as u32,
            output_dim: P2_OUTPUT_DIM as u32,
            reserved1: HEAD_FLAG_VALUE | HEAD_FLAG_POLICY,
            reserved2: 0,
        }
    }

    /// バイナリストリームからヘッダを読み込み
    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut buf = [0u8; 32];
        reader.read_exact(&mut buf)?;

        Ok(Self {
            magic: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            version: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            input_dim: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            l1_dim: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            l2_dim: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
            output_dim: u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]),
            reserved1: u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]),
            reserved2: u32::from_le_bytes([buf[28], buf[29], buf[30], buf[31]]),
        })
    }

    /// ヘッダの妥当性を検証
    pub fn validate(&self) -> io::Result<()> {
        if self.magic != P2_NNUE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "不正なマジックナンバー: expected 0x{:08X}, got 0x{:08X}",
                    P2_NNUE_MAGIC, self.magic
                ),
            ));
        }
        if self.version != P2_NNUE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("未対応バージョン: expected {}, got {}", P2_NNUE_VERSION, self.version),
            ));
        }
        if self.input_dim != P2_INPUT_DIM as u32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("入力次元不一致: expected {}, got {}", P2_INPUT_DIM, self.input_dim),
            ));
        }
        if self.l1_dim != P2_L1_DIM as u32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("L1 次元不一致: expected {}, got {}", P2_L1_DIM, self.l1_dim),
            ));
        }
        if self.l2_dim != P2_L2_DIM as u32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("L2 次元不一致: expected {}, got {}", P2_L2_DIM, self.l2_dim),
            ));
        }
        if self.output_dim != P2_OUTPUT_DIM as u32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("出力次元不一致: expected {}, got {}", P2_OUTPUT_DIM, self.output_dim),
            ));
        }
        // reserved1 の未知ビットチェック
        if self.reserved1 != 0 && (self.reserved1 & !KNOWN_HEAD_FLAGS) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("未知の head_flags ビット: 0x{:08X}", self.reserved1),
            ));
        }
        Ok(())
    }

    /// ヘッダを 32 バイト LE に変換
    pub fn to_bytes(self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.input_dim.to_le_bytes());
        buf[12..16].copy_from_slice(&self.l1_dim.to_le_bytes());
        buf[16..20].copy_from_slice(&self.l2_dim.to_le_bytes());
        buf[20..24].copy_from_slice(&self.output_dim.to_le_bytes());
        buf[24..28].copy_from_slice(&self.reserved1.to_le_bytes());
        buf[28..32].copy_from_slice(&self.reserved2.to_le_bytes());
        buf
    }

    /// head_flags を取得（reserved1 == 0 の場合は VALUE のみとみなす）
    pub fn head_flags(&self) -> u32 {
        if self.reserved1 == 0 {
            HEAD_FLAG_VALUE
        } else {
            self.reserved1
        }
    }

    /// policy hint が有効か
    pub fn has_policy_hint(&self) -> bool {
        self.head_flags() & HEAD_FLAG_POLICY != 0
    }
}
