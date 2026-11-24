//! V2 features: 48 次元 (既存 32 + nodchip soft 8 ch + 補助 8 ch)
//!
//! ## 構成 (48 dims)
//!
//! ```text
//!  0..32  : 既存 `compute_features` (features.rs) の出力を完全コピー
//! 32..40  : nodchip progress8kpabs bucket soft 8 channel
//!           progress_weights が Some なら bucket index 0..7 周辺に
//!           gauss-like soft 値 (該当 1.0 / ±1 = 0.5 / ±2 = 0.25)
//!           None の場合は全 0.0
//! 40..48  : 補助 8 ch (軽量、 王位置 + 駒数 + 王手 / ply)
//! ```
//!
//! ## ホットパス制約
//! - allocation 0、 stack のみ
//! - 分岐最小化 (progress_weights branch 1 つのみ)
//! - 既存 V1 (`MlProgressPredictor`) と互換: 先頭 32 dim は完全一致
//!
//! 旧 64 dim 版は board_effect / pos.key 等の重い API を呼んでいたため、
//! ホットパス利用を考慮して 48 dim の軽量設計に置換した。

use crate::features::{compute_features, FEATURE_DIM};
use rshogi_core::nnue::{compute_progress8kpabs_sum, progress_sum_to_bucket};
use rshogi_core::position::Position;
use rshogi_core::types::{Color, PieceType};

/// V2 feature ベクトルの次元数
pub const FEATURE_DIM_V2: usize = 48;

/// nodchip progress bucket の数 (PROGRESS_BUCKET_THRESHOLDS = 7 -> 8 buckets)
pub const NUM_PROGRESS_BUCKETS: usize = 8;

/// nodchip soft bucket の開始 offset
pub const NODCHIP_OFFSET: usize = FEATURE_DIM; // 32

/// 補助 8 ch の開始 offset
pub const AUX_OFFSET: usize = NODCHIP_OFFSET + NUM_PROGRESS_BUCKETS; // 40

const _: () = assert!(AUX_OFFSET + 8 == FEATURE_DIM_V2);

/// 48 次元 V2 feature をスタック上で保持。
#[derive(Debug, Clone, Copy)]
pub struct ProgressFeaturesV2 {
    pub data: [f32; FEATURE_DIM_V2],
}

impl Default for ProgressFeaturesV2 {
    fn default() -> Self {
        Self {
            data: [0.0; FEATURE_DIM_V2],
        }
    }
}

impl ProgressFeaturesV2 {
    /// 推論モデルに渡す slice 参照。
    #[inline]
    pub fn as_slice(&self) -> &[f32; FEATURE_DIM_V2] {
        &self.data
    }

    /// 先頭 32 dim (V1 互換) の参照を返す helper。
    ///
    /// V1 model (`MlProgressPredictor`) は `ProgressFeatures` を受け取るため、
    /// 利用側で copy を組む際に使う。
    #[inline]
    pub fn head32(&self) -> &[f32; FEATURE_DIM] {
        // SAFETY: data の先頭 32 個は [f32; 32] と layout 互換。
        // FEATURE_DIM_V2 >= FEATURE_DIM の制約は const _ assert で確保済。
        unsafe { &*(self.data.as_ptr() as *const [f32; FEATURE_DIM]) }
    }
}

/// nodchip 8-bucket index を soft 値に書き込む。
///
/// 該当 bucket = 1.0、 ±1 = 0.5、 ±2 = 0.25。 ±3 以遠は 0.0。
/// 端点は自然にクリッピング (range out は書かない)。
#[inline]
fn fill_nodchip_soft(data: &mut [f32; FEATURE_DIM_V2], bucket: usize) {
    let bucket = bucket.min(NUM_PROGRESS_BUCKETS - 1);
    // 中心
    data[NODCHIP_OFFSET + bucket] = 1.0;
    // ±1
    if bucket >= 1 {
        data[NODCHIP_OFFSET + bucket - 1] = 0.5;
    }
    if bucket + 1 < NUM_PROGRESS_BUCKETS {
        data[NODCHIP_OFFSET + bucket + 1] = 0.5;
    }
    // ±2
    if bucket >= 2 {
        data[NODCHIP_OFFSET + bucket - 2] = 0.25;
    }
    if bucket + 2 < NUM_PROGRESS_BUCKETS {
        data[NODCHIP_OFFSET + bucket + 2] = 0.25;
    }
}

const BOARD_PIECE_TYPES_ALL: [PieceType; 13] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Gold,
    PieceType::ProPawn,
    PieceType::ProLance,
    PieceType::ProKnight,
    PieceType::ProSilver,
    PieceType::Horse,
    PieceType::Dragon,
];

const HAND_PIECE_TYPES_AUX: [PieceType; 6] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Bishop,
    PieceType::Rook,
];

/// 補助 8 ch を埋める。
///
/// ```text
/// [40]: own king file (centred file - 4) / 4.0     // file index 0..=8、 center=4
/// [41]: own king rank (centred rank - 4) / 4.0
/// [42]: enemy king file 同上
/// [43]: enemy king rank 同上
/// [44]: total piece count / 40.0   (盤上 + 持ち駒、 玉除く)
/// [45]: hand value bucket / 30.0   (Pawn×1 + Lance/Knight/Silver/Gold×3 + Bishop/Rook×5)
/// [46]: in_check ? 1.0 : 0.0       (既存 [30] と重複だが V2 features 内に閉じる)
/// [47]: game_ply / 256.0           (既存 [31] と重複だが V2 features 内に閉じる)
/// ```
#[inline]
fn fill_aux(data: &mut [f32; FEATURE_DIM_V2], pos: &Position) {
    let stm = pos.side_to_move();
    let own = pos.king_square(stm);
    let enemy = pos.king_square(stm.opponent());

    // 王位置 (file/rank index は 0..=8、 中心 4)
    data[AUX_OFFSET] = (own.file().index() as f32 - 4.0) / 4.0;
    data[AUX_OFFSET + 1] = (own.rank().index() as f32 - 4.0) / 4.0;
    data[AUX_OFFSET + 2] = (enemy.file().index() as f32 - 4.0) / 4.0;
    data[AUX_OFFSET + 3] = (enemy.rank().index() as f32 - 4.0) / 4.0;

    // total piece count: 盤上 (玉除く) + 持ち駒 (歩〜飛)
    let mut total = 0u32;
    for pt in BOARD_PIECE_TYPES_ALL {
        total += pos.pieces(Color::Black, pt).count();
        total += pos.pieces(Color::White, pt).count();
    }
    for c in [Color::Black, Color::White] {
        let h = pos.hand(c);
        for &pt in &HAND_PIECE_TYPES_AUX {
            total += h.count(pt);
        }
    }
    data[AUX_OFFSET + 4] = (total as f32 / 40.0).clamp(0.0, 1.0);

    // hand value bucket: Pawn×1 + Lance/Knight/Silver/Gold×3 + Bishop/Rook×5
    let mut hv: i32 = 0;
    for c in [Color::Black, Color::White] {
        let h = pos.hand(c);
        hv += h.count(PieceType::Pawn) as i32;
        hv += 3 * h.count(PieceType::Lance) as i32;
        hv += 3 * h.count(PieceType::Knight) as i32;
        hv += 3 * h.count(PieceType::Silver) as i32;
        hv += 3 * h.count(PieceType::Gold) as i32;
        hv += 5 * h.count(PieceType::Bishop) as i32;
        hv += 5 * h.count(PieceType::Rook) as i32;
    }
    data[AUX_OFFSET + 5] = (hv as f32 / 30.0).clamp(0.0, 1.0);

    data[AUX_OFFSET + 6] = if pos.in_check() { 1.0 } else { 0.0 };
    data[AUX_OFFSET + 7] = (pos.game_ply() as f32 / 256.0).clamp(0.0, 1.0);
}

/// Position から 48 次元 V2 特徴量を抽出。 allocation 0。
///
/// - `eval_cp`: 既存 [0..32) の eval_magnitude 計算に使用 (0 でも可)
/// - `progress_weights`: nodchip progress8kpabs の重み。 None なら [32..40) は 0
pub fn compute_features_v2(
    pos: &Position,
    eval_cp: i32,
    progress_weights: Option<&[f32]>,
) -> ProgressFeaturesV2 {
    let mut out = ProgressFeaturesV2::default();

    // [0..32) 既存 V1 features
    let v1 = compute_features(pos, eval_cp);
    out.data[..FEATURE_DIM].copy_from_slice(&v1.data);

    // [32..40) nodchip progress bucket soft 8 channel
    if let Some(weights) = progress_weights {
        let sum = compute_progress8kpabs_sum(pos, weights);
        let bucket = progress_sum_to_bucket(sum);
        fill_nodchip_soft(&mut out.data, bucket);
    }
    // None の場合は default の 0.0 のまま

    // [40..48) 補助 8 ch
    fill_aux(&mut out.data, pos);

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::position::Position;

    fn startpos() -> Position {
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        pos
    }

    #[test]
    fn shape_is_48() {
        assert_eq!(FEATURE_DIM_V2, 48);
        let pos = startpos();
        let f = compute_features_v2(&pos, 0, None);
        assert_eq!(f.data.len(), 48);
    }

    #[test]
    fn head32_matches_v1() {
        let pos = startpos();
        let v1 = compute_features(&pos, 123);
        let v2 = compute_features_v2(&pos, 123, None);
        for i in 0..FEATURE_DIM {
            assert!(
                (v1.data[i] - v2.data[i]).abs() < 1e-6,
                "head[{}] mismatch: v1={} v2={}",
                i,
                v1.data[i],
                v2.data[i]
            );
        }
        // head32 helper も同値
        let head = v2.head32();
        assert_eq!(&head[..], &v1.data[..]);
    }

    #[test]
    fn nodchip_zero_when_weights_none() {
        let pos = startpos();
        let f = compute_features_v2(&pos, 0, None);
        for i in NODCHIP_OFFSET..AUX_OFFSET {
            assert_eq!(f.data[i], 0.0, "nodchip[{}] should be 0", i - NODCHIP_OFFSET);
        }
    }

    #[test]
    fn nodchip_soft_center() {
        // center=3 (中央)
        let mut data = [0.0f32; FEATURE_DIM_V2];
        fill_nodchip_soft(&mut data, 3);
        assert_eq!(data[NODCHIP_OFFSET + 1], 0.25);
        assert_eq!(data[NODCHIP_OFFSET + 2], 0.5);
        assert_eq!(data[NODCHIP_OFFSET + 3], 1.0);
        assert_eq!(data[NODCHIP_OFFSET + 4], 0.5);
        assert_eq!(data[NODCHIP_OFFSET + 5], 0.25);
        // 範囲外は 0
        assert_eq!(data[NODCHIP_OFFSET], 0.0);
        assert_eq!(data[NODCHIP_OFFSET + 6], 0.0);
        assert_eq!(data[NODCHIP_OFFSET + 7], 0.0);
    }

    #[test]
    fn nodchip_soft_endpoint_clipping() {
        // bucket=0 (端)
        let mut data = [0.0f32; FEATURE_DIM_V2];
        fill_nodchip_soft(&mut data, 0);
        assert_eq!(data[NODCHIP_OFFSET], 1.0);
        assert_eq!(data[NODCHIP_OFFSET + 1], 0.5);
        assert_eq!(data[NODCHIP_OFFSET + 2], 0.25);

        // bucket=7 (端)
        let mut data = [0.0f32; FEATURE_DIM_V2];
        fill_nodchip_soft(&mut data, 7);
        assert_eq!(data[NODCHIP_OFFSET + 7], 1.0);
        assert_eq!(data[NODCHIP_OFFSET + 6], 0.5);
        assert_eq!(data[NODCHIP_OFFSET + 5], 0.25);
    }

    #[test]
    fn aux_at_startpos_is_in_range() {
        let pos = startpos();
        let f = compute_features_v2(&pos, 0, None);

        // 王 file: 中央 (5筋 = file index 4) なので (4-4)/4=0
        assert!((f.data[AUX_OFFSET]).abs() < 0.01);
        assert!((f.data[AUX_OFFSET + 2]).abs() < 0.01);
        // 王 rank はそれぞれ端 (index 0 or 8) -> [-1.0, 1.0]
        assert!((-1.0..=1.0).contains(&f.data[AUX_OFFSET + 1]));
        assert!((-1.0..=1.0).contains(&f.data[AUX_OFFSET + 3]));

        // total piece count: 平手で玉除き 38 駒 -> 0.95
        let total = f.data[AUX_OFFSET + 4];
        assert!(total > 0.9 && total <= 1.0, "total={}", total);

        // 平手は手駒 0
        assert_eq!(f.data[AUX_OFFSET + 5], 0.0);
        // in_check 0
        assert_eq!(f.data[AUX_OFFSET + 6], 0.0);
        // ply 1/256
        assert!(f.data[AUX_OFFSET + 7] < 0.02);
    }
}
