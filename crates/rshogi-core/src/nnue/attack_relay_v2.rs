//! Attack Relay Head v2 (= 構造化 980 weight 仕様、 2026-06-10 改訂)。
//!
//! 旧 v2 (= 1024 cell 直積、 attackers × overlap × supported × persp × bucket)
//! を **完全置換** する新仕様。 6 特徴クラス + 4 段階 phase を 5 テーブルへ
//! 因子分解し、 先後で重みを共有する (= version 3、 2026-06-11 heavy 軸追加)。
//!
//! # 特徴クラス (= zone は version 2 で 7x7 / 5x5 へ拡張済)
//! - `attackers` 0..3: 敵玉中心 7x7 へ利きを持つ攻撃側ユニーク駒数
//! - `balance` 0..8: 敵玉中心 5x5 の (attack − defense) を [-4, 4] へ clamp + 4
//! - `overlap` 0..3: 敵玉中心 5x5 における一升への最大攻撃駒数 − 1
//! - `supported` 0..3: 7x7 attackers のうち味方支援駒ありの数
//! - `alpha` 0..4: 5x5 内の weak_squares + critical_squares
//! - `heavy` 0..3: 守備側玉 8 近傍の逃げ場数 (= escape、 version 5。 攻撃側が利かず守備自駒で
//!   塞がれない近傍数。 詰めろ近接度。 v3 重駒数 / v4 玉3x3近接数が redundant のため置換。 field 名据置)
//! - `phase` 0..3: 既存 bucket / 2 (= 4 段階進行度)
//!
//! # モデル
//! ```text
//! side_score
//!     = W_base[a][b][o][s]              // 576 i16
//!     + W_alpha[alpha][a][b]             // 180 i16
//!     + W_phase[phase][a][b]             // 144 i16
//!     + W_alpha_shape[alpha][o][s]       //  80 i16
//!     + W_heavy[heavy][b]                //  36 i16
//! eval = (side_score(stm) − side_score(opp)) × OUTPUT_SCALE
//! ```
//!
//! 合計 1016 i16。 先後共有 weight (= perspective 別テーブルなし)。
//!
//! # 基準スライス
//! `W_alpha[0][*][*]` / `W_phase[0][*][*]` / `W_alpha_shape[0][*][*]` は学習側で
//! 常にゼロへ固定する (= 重複表現抑制)。 推論側は単に 0 を加算するだけ。
//! `W_heavy` は基準スライス無し (= W_base と同様、 heavy==0 も有効な weight)。
//!
//! # Disk layout
//! 旧 P2AREL2 形式 (= 1024 cell) とは非互換。 magic / version で識別。
//! ```text
//! sub_magic: "P2ARS1\0\0"
//! version:   3
//! count:     1016
//! 連結順:    W_base → W_alpha → W_phase → W_alpha_shape → W_heavy
//! ```
//!
//! 詳細は `docs/attack_relay_head_v2_a5_design.md` (= 旧設計) と本 module。

use crate::bitboard::{
    Bitboard, RANK_BB, bishop_effect, dragon_effect, gold_effect, horse_effect, king_5x5_mask,
    king_7x7_mask, king_effect, knight_effect, lance_effect, pawn_effect, rook_effect,
    silver_effect,
};
use crate::position::Position;
use crate::types::{Color, PieceType, Square};
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicI32, Ordering};

// =============================================================================
// 軸サイズと weight 数
// =============================================================================

/// `attackers` クラス数 (= 0..3)。
pub const ARH_V2_NUM_ATTACKERS: usize = 4;
/// `balance` クラス数 (= -4..4 → 0..8)。
pub const ARH_V2_NUM_BALANCE: usize = 9;
/// `overlap` クラス数 (= 0..3)。
pub const ARH_V2_NUM_OVERLAP: usize = 4;
/// `supported` クラス数 (= 0..3)。
pub const ARH_V2_NUM_SUPPORTED: usize = 4;
/// `alpha` クラス数 (= 0..4)。
pub const ARH_V2_NUM_ALPHA: usize = 5;
/// `phase` クラス数 (= 0..3、 既存 bucket / 2)。
pub const ARH_V2_NUM_PHASE: usize = 4;
/// `heavy` クラス数 (= 敵玉 7x7 へ利く重駒ユニーク数 0..3)。
pub const ARH_V2_NUM_HEAVY: usize = 4;

// -----------------------------------------------------------------------------
// version 11 侵入 head (= 敵陣侵入 / 成りポテンシャル 3 軸 × 4 phase、 flat[0..80) のみ使用)
//
// 残差解析 (= dlsuisho240 本番 net の最大の穴 = 敵陣侵入 / 成りポテンシャルを +46cp/駒
// 過小評価、 t=14.9) を head でカバーするための特徴。 mp/race/idle (version 10) を全置換。
//
// - 軸1 intrusion (侵入駒数): persp の駒で敵陣 (= persp から見て最遠 3 段) に居る非玉駒数 0..5。
// - 軸2 promo_ready (成りポテンシャル): persp の未成の成れる駒 (歩香桂銀角飛) で敵陣に居るか
//   敵陣隣接 (= 次手で敵陣に入れる) 駒数 0..5。
// - 軸3 supported_intruder (支えられた侵入駒): 敵陣の自駒のうち自軍利きで支えられた駒数 0..7。
//
// 軸×4phase の flat layout / offset は intrusion/promo/support = 0/24/48。
// version 12 で第4軸 advanced_pawn (= 前進歩/垂れ歩) を offset 80 に末尾追加 (= 計 104)。
// version 13 で第5軸 hand_pieces (= 自軍持駒総枚数) を offset 104 に末尾追加 (= 計 128)。
// version 14 で第6軸 entering_king (= 入玉宣言残り点数) を offset 128 に末尾追加 (= 計 152)。
// flat[i] ↔ disk[i] ↔ graph weight[i] が恒等対応 (= W_base 先頭 152 要素を流用)。
// -----------------------------------------------------------------------------

/// 軸1 侵入駒数 (intrusion) の class 数 (= 0..5、 6 段階)。
pub const ARH_V2_NUM_INTRUSION: usize = 6;
/// 軸2 成りポテンシャル (promo_ready) の class 数 (= 0..5、 6 段階)。
pub const ARH_V2_NUM_PROMO: usize = 6;
/// 軸3 支えられた侵入駒 (supported_intruder) の class 数 (= 0..7、 8 段階)。
pub const ARH_V2_NUM_SUPPORT_INTR: usize = 8;
/// 軸4 前進歩/垂れ歩 (advanced_pawn) の class 数 (= 0..5、 6 段階、 version 12)。
///
/// 残差解析で確定した系統ギャップ (= 前進歩を約 -48cp 過大評価) を補正する軸。
/// 対象 = 自軍の未成歩 (Pawn)。 **最も進出した未成歩 1 枚** を persp 基準の
/// 「敵陣からの段数」で層化する (= count ではなく位置 class)。 バケット定義:
/// - 0: 進出した未成歩なし (= 敵陣隣接段より手前のみ)。
/// - 1: その他進出歩 (= 敵陣隣接段より手前だが進出済、 現状は到達しないため予約)。
/// - 2: 垂れ歩 (= 敵陣隣接段、 次手で成れる位置)。
/// - 3: 敵陣 浅段 (= 敵陣 3 段のうち最も自陣寄り)。
/// - 4: 敵陣 中段。
/// - 5: 敵陣 最深段 (= 最も敵玉寄り)。
pub const ARH_V2_NUM_ADV_PAWN: usize = 6;
/// 軸5 持駒枚数 (hand_pieces) の class 数 (= 0..5、 6 段階、 version 13)。
///
/// 残差解析で確定した系統ギャップ (= 自軍持駒を約 -11.5cp/枚 過大評価) を補正する軸。
/// 対象 = **persp 側 (= 自軍) の持駒総枚数** (= 歩〜飛、 玉以外)。 持駒は先後で別管理
/// されるため persp 側の hand のみを数える。 総枚数を以下の 6 bucket へ層化する:
/// - 0: 持駒 0 枚。
/// - 1: 1 枚。
/// - 2: 2 枚。
/// - 3: 3 枚。
/// - 4: 4 枚。
/// - 5: 5 枚以上。
pub const ARH_V2_NUM_HAND: usize = 6;
/// 軸6 入玉宣言 残り点数 (entering_king) の class 数 (= 0..5、 6 段階、 version 14)。
///
/// dlshogi 知見 (= 入玉宣言の残り点数を明示特徴化で value 正解率 0.76 → 0.97) に基づき、
/// 入玉宣言 (= 27 点法) の **残り点数** を明示 bucket 化する軸。 残り点数が小さいほど
/// 入玉宣言が近い (= 終盤入玉の勝勢) ことを表す。
///
/// # 点数計算 (= 簡便版 27 点法)
/// - 対象 = **persp 側 (= 自軍) の駒**。 持駒は先後別管理のため persp 側のみ。
/// - 現在の宣言点数 = 「敵陣 (= persp から見て最遠 3 段) 内の自駒 (= 玉除く) の点数合計」
///   + 「persp 側持駒の点数合計」。 飛角 (= Rook/Bishop + 成 Dragon/Horse) = 5 点、 他 = 1 点。
/// - **自玉が敵陣内に居ない場合は 0 点扱い** (= 入玉宣言は自玉入玉が前提条件のため)。
/// - 残り点数 = max(0, 27 − 現在の宣言点数)。 必要点は先手 28 / 後手 27 が一般だが、
///   簡便に一律 27 点とする (= 設計判断、 head の役割は「残り点数の大小」の表現で十分)。
///
/// # バケット定義 (= 残り点数を 6 段階へ層化)
/// - 0: 残り ≤ 0 (= 入玉宣言点 達成。 自玉敵陣 + 点数 27 以上)。
/// - 1: 残り 1..=5。
/// - 2: 残り 6..=10。
/// - 3: 残り 11..=15。
/// - 4: 残り 16..=20。
/// - 5: 残り 21..=27 (= 自玉非入玉を含む最遠ケース)。
///
/// eval = (自陣営 − 敵陣営) の差で評価されるため、 persp 側 1 軸で自他両側の入玉接近度の
/// 相対が自動的に取れる (= 相手側を別軸にせず YAGNI で weight を抑える)。
pub const ARH_V2_NUM_EK: usize = 6;
/// 侵入 head の intrusion テーブル flat offset (= 0)。
pub const ARH_V2_INTRUSION_OFFSET: usize = 0;
/// 侵入 head の promo_ready テーブル flat offset (= intrusion 6 × 4 phase = 24)。
pub const ARH_V2_PROMO_OFFSET: usize =
    ARH_V2_INTRUSION_OFFSET + ARH_V2_NUM_INTRUSION * ARH_V2_NUM_PHASE;
/// 侵入 head の supported_intruder テーブル flat offset (= promo の直後 = 48)。
pub const ARH_V2_SUPPORT_OFFSET: usize = ARH_V2_PROMO_OFFSET + ARH_V2_NUM_PROMO * ARH_V2_NUM_PHASE;
/// 侵入 head の advanced_pawn テーブル flat offset (= support の直後 = 80、 version 12 末尾追加)。
pub const ARH_V2_ADV_PAWN_OFFSET: usize =
    ARH_V2_SUPPORT_OFFSET + ARH_V2_NUM_SUPPORT_INTR * ARH_V2_NUM_PHASE;
/// 侵入 head の hand_pieces テーブル flat offset (= adv_pawn の直後 = 104、 version 13 末尾追加)。
pub const ARH_V2_HAND_OFFSET: usize =
    ARH_V2_ADV_PAWN_OFFSET + ARH_V2_NUM_ADV_PAWN * ARH_V2_NUM_PHASE;
/// 侵入 head の entering_king テーブル flat offset (= hand の直後 = 128、 version 14 末尾追加)。
pub const ARH_V2_EK_OFFSET: usize = ARH_V2_HAND_OFFSET + ARH_V2_NUM_HAND * ARH_V2_NUM_PHASE;
/// 侵入 head が使用する flat weight 総数 (= 152、 残り [152, 1016) は未使用ゼロ)。
pub const ARH_V2_NEW_HEAD_USED: usize = ARH_V2_EK_OFFSET + ARH_V2_NUM_EK * ARH_V2_NUM_PHASE;

const _: () = {
    assert!(ARH_V2_INTRUSION_OFFSET == 0);
    assert!(ARH_V2_PROMO_OFFSET == 24);
    assert!(ARH_V2_SUPPORT_OFFSET == 48);
    assert!(ARH_V2_ADV_PAWN_OFFSET == 80);
    assert!(ARH_V2_HAND_OFFSET == 104);
    assert!(ARH_V2_EK_OFFSET == 128);
    assert!(ARH_V2_NEW_HEAD_USED == 152);
    assert!(ARH_V2_NEW_HEAD_USED <= ARH_V2_WEIGHT_COUNT);
};

/// 軸1 侵入駒数の flat weight index (= intrusion*4 + phase、 0..24)。
#[inline]
pub const fn intrusion_idx(intrusion: usize, phase: usize) -> usize {
    ARH_V2_INTRUSION_OFFSET + intrusion * ARH_V2_NUM_PHASE + phase
}

/// 軸2 成りポテンシャルの flat weight index (= 24 + promo*4 + phase、 24..48)。
#[inline]
pub const fn promo_idx(promo: usize, phase: usize) -> usize {
    ARH_V2_PROMO_OFFSET + promo * ARH_V2_NUM_PHASE + phase
}

/// 軸3 支えられた侵入駒の flat weight index (= 48 + support*4 + phase、 48..80)。
#[inline]
pub const fn support_idx(support: usize, phase: usize) -> usize {
    ARH_V2_SUPPORT_OFFSET + support * ARH_V2_NUM_PHASE + phase
}

/// 軸4 前進歩/垂れ歩の flat weight index (= 80 + adv*4 + phase、 80..104)。
#[inline]
pub const fn adv_pawn_idx(adv: usize, phase: usize) -> usize {
    ARH_V2_ADV_PAWN_OFFSET + adv * ARH_V2_NUM_PHASE + phase
}
/// 軸5 持駒枚数の flat weight index (= 104 + hand*4 + phase、 104..128)。
#[inline]
pub const fn hand_idx(hand: usize, phase: usize) -> usize {
    ARH_V2_HAND_OFFSET + hand * ARH_V2_NUM_PHASE + phase
}
/// 軸6 入玉宣言 残り点数の flat weight index (= 128 + ek*4 + phase、 128..152)。
#[inline]
pub const fn ek_idx(ek: usize, phase: usize) -> usize {
    ARH_V2_EK_OFFSET + ek * ARH_V2_NUM_PHASE + phase
}
/// 既存 bucket 数 (= 8)。
pub const ARH_V2_NUM_BUCKETS: usize = 8;

/// `W_base[attackers][balance][overlap][supported]` の要素数 (= 576)。
pub const ARH_V2_W_BASE_COUNT: usize =
    ARH_V2_NUM_ATTACKERS * ARH_V2_NUM_BALANCE * ARH_V2_NUM_OVERLAP * ARH_V2_NUM_SUPPORTED;
/// `W_alpha[alpha][attackers][balance]` の要素数 (= 180)。
pub const ARH_V2_W_ALPHA_COUNT: usize =
    ARH_V2_NUM_ALPHA * ARH_V2_NUM_ATTACKERS * ARH_V2_NUM_BALANCE;
/// `W_phase[phase][attackers][balance]` の要素数 (= 144)。
pub const ARH_V2_W_PHASE_COUNT: usize =
    ARH_V2_NUM_PHASE * ARH_V2_NUM_ATTACKERS * ARH_V2_NUM_BALANCE;
/// `W_alpha_shape[alpha][overlap][supported]` の要素数 (= 80)。
pub const ARH_V2_W_ALPHA_SHAPE_COUNT: usize =
    ARH_V2_NUM_ALPHA * ARH_V2_NUM_OVERLAP * ARH_V2_NUM_SUPPORTED;
/// `W_heavy[heavy][balance]` の要素数 (= 36)。
pub const ARH_V2_W_HEAVY_COUNT: usize = ARH_V2_NUM_HEAVY * ARH_V2_NUM_BALANCE;

/// 全 weight 数 (= 1016)。
pub const ARH_V2_WEIGHT_COUNT: usize = ARH_V2_W_BASE_COUNT
    + ARH_V2_W_ALPHA_COUNT
    + ARH_V2_W_PHASE_COUNT
    + ARH_V2_W_ALPHA_SHAPE_COUNT
    + ARH_V2_W_HEAVY_COUNT;

/// 4 テーブルの連結 disk offset (要素単位)。
pub const ARH_V2_OFFSET_W_BASE: usize = 0;
/// `W_alpha` の disk offset (= W_base の直後、 576)。
pub const ARH_V2_OFFSET_W_ALPHA: usize = ARH_V2_W_BASE_COUNT;
/// `W_phase` の disk offset (= 756)。
pub const ARH_V2_OFFSET_W_PHASE: usize = ARH_V2_OFFSET_W_ALPHA + ARH_V2_W_ALPHA_COUNT;
/// `W_alpha_shape` の disk offset (= 900)。
pub const ARH_V2_OFFSET_W_ALPHA_SHAPE: usize = ARH_V2_OFFSET_W_PHASE + ARH_V2_W_PHASE_COUNT;
/// `W_heavy` の disk offset (= 980、 4 テーブルの直後)。
pub const ARH_V2_OFFSET_W_HEAVY: usize = ARH_V2_OFFSET_W_ALPHA_SHAPE + ARH_V2_W_ALPHA_SHAPE_COUNT;

/// ckpt sub-section magic (= "P2ARS1" + 2 byte NUL padding)。
pub const ARH_V2_MAGIC: &[u8; 8] = b"P2ARS1\0\0";
/// ckpt version (= 14、 version 13 の 5 軸 head に第6軸 entering_king を末尾追加。
/// intrusion (= 侵入駒数) offset 0 / promo_ready (= 成りポテンシャル) offset 24 /
/// supported_intruder (= 支えられた侵入駒) offset 48 / advanced_pawn (= 前進歩/垂れ歩)
/// offset 80 / hand_pieces (= 自軍持駒総枚数) offset 104 / entering_king (= 入玉宣言残り点数)
/// offset 128、 各 軸 × 4 phase。
/// flat[0..152) のみ使用、 残り [152, 1016) は未使用ゼロ。
/// flat[i] ↔ disk[i] ↔ graph weight[i] は恒等対応 (= W_base 先頭 152 要素を流用)。
/// weight 数 1016 据置。 version 13 ckpt は flat[128..152) が全ゼロのため、
/// version 番号のみ書き換えれば version 14 として読み込め (= 第6軸 zero init migration)。
/// 旧 P2AREL2 / version 9 / version 10 (mp/race/idle) とも非互換。
pub const ARH_V2_VERSION: u32 = 14;

/// `OUTPUT_SCALE` 初期値 (= 4)。
pub const ARH_V2_OUTPUT_SCALE_DEFAULT: i32 = 4;

/// USI option `ARH_V2_OUTPUT_SCALE` で runtime 上書き可能な eval scale。
pub static ARH_V2_OUTPUT_SCALE: AtomicI32 = AtomicI32::new(ARH_V2_OUTPUT_SCALE_DEFAULT);

/// USI option 経由で eval scale を更新する。
pub fn set_output_scale(scale: i32) {
    ARH_V2_OUTPUT_SCALE.store(scale, Ordering::Relaxed);
}

const _: () = {
    assert!(ARH_V2_W_BASE_COUNT == 576);
    assert!(ARH_V2_W_ALPHA_COUNT == 180);
    assert!(ARH_V2_W_PHASE_COUNT == 144);
    assert!(ARH_V2_W_ALPHA_SHAPE_COUNT == 80);
    assert!(ARH_V2_W_HEAVY_COUNT == 36);
    assert!(ARH_V2_WEIGHT_COUNT == 1016);
};

// =============================================================================
// 型
// =============================================================================

/// disk 上の i16 weight 4 テーブル。 先後共有。
#[derive(Clone)]
pub struct ArhV2Weights {
    /// `[attackers][balance][overlap][supported]`。
    pub w_base: [[[[i16; ARH_V2_NUM_SUPPORTED]; ARH_V2_NUM_OVERLAP]; ARH_V2_NUM_BALANCE];
        ARH_V2_NUM_ATTACKERS],
    /// `[alpha][attackers][balance]`。 `alpha == 0` は基準スライスで常にゼロ。
    pub w_alpha: [[[i16; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_ATTACKERS]; ARH_V2_NUM_ALPHA],
    /// `[phase][attackers][balance]`。 `phase == 0` は基準スライスで常にゼロ。
    pub w_phase: [[[i16; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_ATTACKERS]; ARH_V2_NUM_PHASE],
    /// `[alpha][overlap][supported]`。 `alpha == 0` は基準スライスで常にゼロ。
    pub w_alpha_shape: [[[i16; ARH_V2_NUM_SUPPORTED]; ARH_V2_NUM_OVERLAP]; ARH_V2_NUM_ALPHA],
    /// `[heavy][balance]`。 基準スライス無し (= W_base と同様常時 active)。
    pub w_heavy: [[i16; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_HEAVY],
}

impl ArhV2Weights {
    /// 全 weight ゼロの新規 instance を返す。
    pub fn zeroed() -> Box<Self> {
        Box::new(Self {
            w_base: [[[[0i16; ARH_V2_NUM_SUPPORTED]; ARH_V2_NUM_OVERLAP]; ARH_V2_NUM_BALANCE];
                ARH_V2_NUM_ATTACKERS],
            w_alpha: [[[0i16; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_ATTACKERS]; ARH_V2_NUM_ALPHA],
            w_phase: [[[0i16; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_ATTACKERS]; ARH_V2_NUM_PHASE],
            w_alpha_shape: [[[0i16; ARH_V2_NUM_SUPPORTED]; ARH_V2_NUM_OVERLAP]; ARH_V2_NUM_ALPHA],
            w_heavy: [[0i16; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_HEAVY],
        })
    }

    /// 基準スライス (= `alpha == 0` / `phase == 0`) が全てゼロか検証する。
    pub fn zero_baseline_satisfied(&self) -> bool {
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                if self.w_alpha[0][a][b] != 0 || self.w_phase[0][a][b] != 0 {
                    return false;
                }
            }
        }
        for o in 0..ARH_V2_NUM_OVERLAP {
            for s in 0..ARH_V2_NUM_SUPPORTED {
                if self.w_alpha_shape[0][o][s] != 0 {
                    return false;
                }
            }
        }
        true
    }

    /// 基準スライスを明示的にゼロへ書き戻す (= 学習後 / 量子化後の保険)。
    pub fn enforce_zero_baseline(&mut self) {
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                self.w_alpha[0][a][b] = 0;
                self.w_phase[0][a][b] = 0;
            }
        }
        for o in 0..ARH_V2_NUM_OVERLAP {
            for s in 0..ARH_V2_NUM_SUPPORTED {
                self.w_alpha_shape[0][o][s] = 0;
            }
        }
    }
}

/// 推論用 i32 weight (= 範囲拡張のため i16 → i32 へ展開済)。
#[derive(Clone)]
pub struct ArhV2RuntimeWeights {
    /// `[attackers][balance][overlap][supported]`。
    pub w_base: [[[[i32; ARH_V2_NUM_SUPPORTED]; ARH_V2_NUM_OVERLAP]; ARH_V2_NUM_BALANCE];
        ARH_V2_NUM_ATTACKERS],
    /// `[alpha][attackers][balance]`。
    pub w_alpha: [[[i32; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_ATTACKERS]; ARH_V2_NUM_ALPHA],
    /// `[phase][attackers][balance]`。
    pub w_phase: [[[i32; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_ATTACKERS]; ARH_V2_NUM_PHASE],
    /// `[alpha][overlap][supported]`。
    pub w_alpha_shape: [[[i32; ARH_V2_NUM_SUPPORTED]; ARH_V2_NUM_OVERLAP]; ARH_V2_NUM_ALPHA],
    /// `[heavy][balance]`。
    pub w_heavy: [[i32; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_HEAVY],
    /// 侵入 head 用 flat weight (= disk index 順、 intrude_disk_indices でアクセス)。
    pub flat: Box<[i32; ARH_V2_WEIGHT_COUNT]>,
}

/// 局面から導出した特徴クラスの組。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArhV2Classes {
    /// `attackers` (0..=3)。
    pub attackers: usize,
    /// `balance` (= raw + 4、 0..=8)。
    pub balance: usize,
    /// `overlap` (0..=3)。
    pub overlap: usize,
    /// `supported` (0..=3)。
    pub supported: usize,
    /// `alpha` (0..=4)。
    pub alpha: usize,
    /// `heavy` (= 敵玉 7x7 へ利く重駒ユニーク数、 0..=3)。
    pub heavy: usize,
}

// =============================================================================
// 索引計算 (= 推論側 / 学習側で共有)
// =============================================================================

/// `W_base` テーブル先頭からの要素 index。
#[inline]
pub const fn w_base_local_idx(
    attackers: usize,
    balance: usize,
    overlap: usize,
    supported: usize,
) -> usize {
    ((attackers * ARH_V2_NUM_BALANCE + balance) * ARH_V2_NUM_OVERLAP + overlap)
        * ARH_V2_NUM_SUPPORTED
        + supported
}

/// `W_alpha` テーブル先頭からの要素 index。
#[inline]
pub const fn w_alpha_local_idx(alpha: usize, attackers: usize, balance: usize) -> usize {
    (alpha * ARH_V2_NUM_ATTACKERS + attackers) * ARH_V2_NUM_BALANCE + balance
}

/// `W_phase` テーブル先頭からの要素 index。
#[inline]
pub const fn w_phase_local_idx(phase: usize, attackers: usize, balance: usize) -> usize {
    (phase * ARH_V2_NUM_ATTACKERS + attackers) * ARH_V2_NUM_BALANCE + balance
}

/// `W_alpha_shape` テーブル先頭からの要素 index。
#[inline]
pub const fn w_alpha_shape_local_idx(alpha: usize, overlap: usize, supported: usize) -> usize {
    (alpha * ARH_V2_NUM_OVERLAP + overlap) * ARH_V2_NUM_SUPPORTED + supported
}

/// disk 全体での `W_base` 要素 index。
#[inline]
pub const fn w_base_disk_idx(
    attackers: usize,
    balance: usize,
    overlap: usize,
    supported: usize,
) -> usize {
    ARH_V2_OFFSET_W_BASE + w_base_local_idx(attackers, balance, overlap, supported)
}

/// disk 全体での `W_alpha` 要素 index。
#[inline]
pub const fn w_alpha_disk_idx(alpha: usize, attackers: usize, balance: usize) -> usize {
    ARH_V2_OFFSET_W_ALPHA + w_alpha_local_idx(alpha, attackers, balance)
}

/// disk 全体での `W_phase` 要素 index。
#[inline]
pub const fn w_phase_disk_idx(phase: usize, attackers: usize, balance: usize) -> usize {
    ARH_V2_OFFSET_W_PHASE + w_phase_local_idx(phase, attackers, balance)
}

/// disk 全体での `W_alpha_shape` 要素 index。
#[inline]
pub const fn w_alpha_shape_disk_idx(alpha: usize, overlap: usize, supported: usize) -> usize {
    ARH_V2_OFFSET_W_ALPHA_SHAPE + w_alpha_shape_local_idx(alpha, overlap, supported)
}

/// `W_heavy` テーブル先頭からの要素 index。
#[inline]
pub const fn w_heavy_local_idx(heavy: usize, balance: usize) -> usize {
    heavy * ARH_V2_NUM_BALANCE + balance
}

/// disk 全体での `W_heavy` 要素 index。
#[inline]
pub const fn w_heavy_disk_idx(heavy: usize, balance: usize) -> usize {
    ARH_V2_OFFSET_W_HEAVY + w_heavy_local_idx(heavy, balance)
}

/// 既存 8 bucket から 4 段階 phase への単調写像。
#[inline]
pub const fn bucket_to_phase(bucket: usize) -> usize {
    debug_assert!(bucket < ARH_V2_NUM_BUCKETS);
    bucket / 2
}

// =============================================================================
// 特徴量計算
// =============================================================================

#[inline]
fn piece_attack_bb(pt: PieceType, color: Color, sq: Square, occupied: Bitboard) -> Bitboard {
    match pt {
        PieceType::Pawn => pawn_effect(color, sq),
        PieceType::Lance => lance_effect(color, sq, occupied),
        PieceType::Knight => knight_effect(color, sq),
        PieceType::Silver => silver_effect(color, sq),
        PieceType::Gold
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver => gold_effect(color, sq),
        PieceType::Bishop => bishop_effect(sq, occupied),
        PieceType::Rook => rook_effect(sq, occupied),
        PieceType::Horse => horse_effect(sq, occupied),
        PieceType::Dragon => dragon_effect(sq, occupied),
        PieceType::King => king_effect(sq),
    }
}

/// 局面 `pos` に対して `persp` 視点の 5 特徴クラスを計算する。
///
/// - `persp` が「攻撃側」、 `persp.opponent()` が「守備側」 とみなす。
/// - 攻撃 zone = 7x7 (= 玉中心 ±3、 最大 49 sq)、 内 zone = 5x5 (= 玉中心 ±2、
///   最大 25 sq)。 盤端では `king_7x7_mask` / `king_5x5_mask` が盤内 sq のみを
///   返すため盤外参照は発生しない。
///
/// 戻り値は範囲 clamp 済 ([`ArhV2Classes`])。
pub fn compute_arh_v2_classes(pos: &Position, persp: Color) -> ArhV2Classes {
    let opp = persp.opponent();
    let opp_king_sq = pos.king_square(opp);
    let own_king_sq = pos.king_square(persp);
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);
    let own_pieces = pos.pieces_c(persp);
    let opp_pieces = pos.pieces_c(opp);

    // version 2 (= 2026-06-11): attackers の zone = 7x7、 inner zone = 5x5。
    let inner_zone = king_5x5_mask(opp_king_sq);
    let zone_7x7 = king_7x7_mask(opp_king_sq);
    let own_king_bb = Bitboard::from_square(own_king_sq);
    let opp_king_bb = Bitboard::from_square(opp_king_sq);
    let non_own_king = !own_king_bb;
    let non_opp_king = !opp_king_bb;

    // 軸 1: attackers = 7x7 zone へ利く攻撃側ユニーク駒 (= 攻撃側玉除外)。
    // 同ループで own_attack_union (= escape 軸用、 攻撃側玉以外の全利き union) を蓄積。
    let candidates = own_pieces & non_own_king;
    let mut attackers_bb = Bitboard::EMPTY;
    let mut own_attack_union = Bitboard::EMPTY;
    let mut iter = candidates;
    while iter.is_not_empty() {
        let sq = iter.pop();
        let pt = pos.piece_on(sq).piece_type();
        let attack_bb = piece_attack_bb(pt, persp, sq, occupied);
        own_attack_union |= attack_bb;
        if !(attack_bb & zone_7x7).is_empty() {
            attackers_bb |= Bitboard::from_square(sq);
        }
    }
    let attackers_raw = attackers_bb.count() as usize;
    let attackers = attackers_raw.min(ARH_V2_NUM_ATTACKERS - 1);
    // 軸 6: escape = 敵玉 8 近傍のうち「我々が利いておらず・敵自駒で塞がれていない」逃げ場数
    // (= 詰めろ近接度、 version 5)。 玉の自由度を測る軸で 5 攻撃集約軸と直交。 0..3 へ clamp。
    let escape_count =
        (king_effect(opp_king_sq) & !own_attack_union & !opp_pieces).count() as usize;
    let heavy = escape_count.min(ARH_V2_NUM_HEAVY - 1);

    // 軸 2/3/5: 5x5 inner zone を 1 走査で balance / overlap / alpha まとめて計算。
    let mut attack_power: i32 = 0;
    let mut defense_power: i32 = 0;
    let mut max_stack: usize = 0;
    let mut weak_squares: usize = 0;
    let mut critical_squares: usize = 0;
    let mut iter_in = inner_zone;
    while iter_in.is_not_empty() {
        let sq = iter_in.pop();
        let attackers_to_sq = pos.attackers_to_occ(sq, occupied);
        let atk_count = ((attackers_to_sq & own_pieces) & non_own_king).count() as usize;
        let def_count = ((attackers_to_sq & opp_pieces) & non_opp_king).count() as usize;
        attack_power += atk_count.min(3) as i32;
        defense_power += def_count.min(3) as i32;
        if atk_count > max_stack {
            max_stack = atk_count;
        }
        if atk_count > def_count {
            weak_squares += 1;
            if atk_count >= 2 {
                critical_squares += 1;
            }
        }
    }
    let balance_signed = (attack_power - defense_power).clamp(-4, 4);
    let balance = (balance_signed + 4) as usize;
    let overlap = max_stack.saturating_sub(1).min(ARH_V2_NUM_OVERLAP - 1);
    let alpha = (weak_squares + critical_squares).min(ARH_V2_NUM_ALPHA - 1);

    // 軸 4: supported = 5x5 attackers のうち味方支援駒 (= 自分 / 攻撃側玉 除外) を持つ駒数。
    let mut supported_count: usize = 0;
    let mut iter_a = attackers_bb;
    while iter_a.is_not_empty() {
        let piece_sq = iter_a.pop();
        let supporters = pos.attackers_to_occ(piece_sq, occupied)
            & own_pieces
            & !Bitboard::from_square(piece_sq)
            & non_own_king;
        if supporters.is_not_empty() {
            supported_count += 1;
        }
    }
    let supported = supported_count.min(ARH_V2_NUM_SUPPORTED - 1);

    ArhV2Classes {
        attackers,
        balance,
        overlap,
        supported,
        alpha,
        heavy,
    }
}

/// 両 perspective (= Black / White) の 5 特徴クラスを 1 pass で計算する。
///
/// [`compute_arh_v2_classes`] を 2 回呼ぶのと **完全に同一の結果** を返す
/// hot path 用 fused 実装。 旧経路は perspective 毎に inner zone 25 マス ×
/// `attackers_to_occ` (= 逆引きで全駒種 effect を再計算) + attackers /
/// supported ループで同じ駒の利きを何度も再計算していた。 本実装は
/// 盤上の玉以外の全駒について `piece_attack_bb` を **1 回だけ** 計算し、
/// そこから両 perspective の全 5 軸を導出する。
///
/// 等価性の根拠 (= 旧実装との対応):
/// - `attackers_to_occ(sq) & pieces_c(c) & !king(c)` は 「色 c の玉以外の駒で
///   sq に利くもの」 = 前向き計算 `piece_attack_bb` が sq を含む駒の集合と一致。
/// - supported: 駒 P の supporters = 「P 以外の味方 (玉除く) で P のマスに
///   利く駒」。 駒は自マスへ利かないため、 「味方 (玉除く) の利き union が
///   P のマスを含む」 ことと等価。
///
/// 戻り値は `[classes_black, classes_white]` (= `Color::index()` で引く)。
pub fn compute_arh_v2_classes_both(pos: &Position) -> [ArhV2Classes; 2] {
    let bk_sq = pos.king_square(Color::Black);
    let wk_sq = pos.king_square(Color::White);
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);

    // perspective p の攻撃 zone は敵玉中心: Black → wk、 White → bk。
    let zone_7x7 = [king_7x7_mask(wk_sq), king_7x7_mask(bk_sq)];
    let inner_zone = [king_5x5_mask(wk_sq), king_5x5_mask(bk_sq)];
    let inner_union = inner_zone[0] | inner_zone[1];
    // escape 軸用: 各 color の玉マス / 自駒。 di (= 守備側 color index、 0=Black/1=White)。
    let king_sq = [bk_sq, wk_sq];
    let pieces = [pos.pieces_c(Color::Black), pos.pieces_c(Color::White)];

    let king_bb = [Bitboard::from_square(bk_sq), Bitboard::from_square(wk_sq)];

    // 色別: 玉以外の全駒の利き union / 7x7 zone へ利く駒集合 / inner union 内
    // per-square 利き数 (= 玉除外)。 [u8; 81] はスタック上 (ヒープ割り当てなし)。
    let mut attack_union = [Bitboard::EMPTY; 2];
    let mut attackers_bb = [Bitboard::EMPTY; 2];
    let mut counts = [[0u8; 81]; 2];

    for ci in 0..2 {
        let color = if ci == 0 { Color::Black } else { Color::White };
        let mut iter = pos.pieces_c(color) & !king_bb[ci];
        while iter.is_not_empty() {
            let sq = iter.pop();
            let pt = pos.piece_on(sq).piece_type();
            let attack_bb = piece_attack_bb(pt, color, sq, occupied);
            attack_union[ci] |= attack_bb;
            if (attack_bb & zone_7x7[ci]).is_not_empty() {
                attackers_bb[ci] |= Bitboard::from_square(sq);
            }
            let mut t = attack_bb & inner_union;
            while t.is_not_empty() {
                counts[ci][t.pop().index()] += 1;
            }
        }
    }

    let mut out = [ArhV2Classes {
        attackers: 0,
        balance: 0,
        overlap: 0,
        supported: 0,
        alpha: 0,
        heavy: 0,
    }; 2];
    for (pi, out_slot) in out.iter_mut().enumerate() {
        let di = pi ^ 1; // 守備側 (= 敵) の color index

        // 軸 1: attackers。 軸 6: escape (= 守備側玉の逃げ場数)。
        let attackers_raw = attackers_bb[pi].count() as usize;
        let attackers = attackers_raw.min(ARH_V2_NUM_ATTACKERS - 1);
        // escape: 守備側 (di) 玉 8 近傍のうち攻撃側 (pi) が利いておらず di 自駒で塞がれていない数。
        let escape_count =
            (king_effect(king_sq[di]) & !attack_union[pi] & !pieces[di]).count() as usize;
        let heavy = escape_count.min(ARH_V2_NUM_HEAVY - 1);

        // 軸 2/3/5: 5x5 inner zone を per-square count から導出。
        let mut attack_power: i32 = 0;
        let mut defense_power: i32 = 0;
        let mut max_stack: usize = 0;
        let mut weak_squares: usize = 0;
        let mut critical_squares: usize = 0;
        let mut iter_in = inner_zone[pi];
        while iter_in.is_not_empty() {
            let sq = iter_in.pop();
            let atk_count = counts[pi][sq.index()] as usize;
            let def_count = counts[di][sq.index()] as usize;
            attack_power += atk_count.min(3) as i32;
            defense_power += def_count.min(3) as i32;
            if atk_count > max_stack {
                max_stack = atk_count;
            }
            if atk_count > def_count {
                weak_squares += 1;
                if atk_count >= 2 {
                    critical_squares += 1;
                }
            }
        }
        let balance_signed = (attack_power - defense_power).clamp(-4, 4);
        let balance = (balance_signed + 4) as usize;
        let overlap = max_stack.saturating_sub(1).min(ARH_V2_NUM_OVERLAP - 1);
        let alpha = (weak_squares + critical_squares).min(ARH_V2_NUM_ALPHA - 1);

        // 軸 4: supported = attackers のうち味方 (玉除く) の利きが乗っている駒数。
        let supported_count = (attackers_bb[pi] & attack_union[pi]).count() as usize;
        let supported = supported_count.min(ARH_V2_NUM_SUPPORTED - 1);

        *out_slot = ArhV2Classes {
            attackers,
            balance,
            overlap,
            supported,
            alpha,
            heavy,
        };
    }
    out
}

// =============================================================================
// runtime weight 構築 + 評価
// =============================================================================

/// disk weight (i16) を runtime weight (i32) へ展開する。
pub fn build_runtime_weights(disk: &ArhV2Weights) -> Box<ArhV2RuntimeWeights> {
    let mut rt = Box::new(ArhV2RuntimeWeights {
        w_base: [[[[0i32; ARH_V2_NUM_SUPPORTED]; ARH_V2_NUM_OVERLAP]; ARH_V2_NUM_BALANCE];
            ARH_V2_NUM_ATTACKERS],
        w_alpha: [[[0i32; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_ATTACKERS]; ARH_V2_NUM_ALPHA],
        w_phase: [[[0i32; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_ATTACKERS]; ARH_V2_NUM_PHASE],
        w_alpha_shape: [[[0i32; ARH_V2_NUM_SUPPORTED]; ARH_V2_NUM_OVERLAP]; ARH_V2_NUM_ALPHA],
        w_heavy: [[0i32; ARH_V2_NUM_BALANCE]; ARH_V2_NUM_HEAVY],
        flat: Box::new([0i32; ARH_V2_WEIGHT_COUNT]),
    });
    // 侵入 head: disk index 順に flat へ展開 (= write_weights と同順)。
    {
        let mut idx = 0usize;
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                for o in 0..ARH_V2_NUM_OVERLAP {
                    for s in 0..ARH_V2_NUM_SUPPORTED {
                        rt.flat[idx] = disk.w_base[a][b][o][s] as i32;
                        idx += 1;
                    }
                }
            }
        }
        for al in 0..ARH_V2_NUM_ALPHA {
            for a in 0..ARH_V2_NUM_ATTACKERS {
                for b in 0..ARH_V2_NUM_BALANCE {
                    rt.flat[idx] = disk.w_alpha[al][a][b] as i32;
                    idx += 1;
                }
            }
        }
        for ph in 0..ARH_V2_NUM_PHASE {
            for a in 0..ARH_V2_NUM_ATTACKERS {
                for b in 0..ARH_V2_NUM_BALANCE {
                    rt.flat[idx] = disk.w_phase[ph][a][b] as i32;
                    idx += 1;
                }
            }
        }
        for al in 0..ARH_V2_NUM_ALPHA {
            for o in 0..ARH_V2_NUM_OVERLAP {
                for s in 0..ARH_V2_NUM_SUPPORTED {
                    rt.flat[idx] = disk.w_alpha_shape[al][o][s] as i32;
                    idx += 1;
                }
            }
        }
        for h in 0..ARH_V2_NUM_HEAVY {
            for b in 0..ARH_V2_NUM_BALANCE {
                rt.flat[idx] = disk.w_heavy[h][b] as i32;
                idx += 1;
            }
        }
        debug_assert_eq!(idx, ARH_V2_WEIGHT_COUNT);
    }
    for a in 0..ARH_V2_NUM_ATTACKERS {
        for b in 0..ARH_V2_NUM_BALANCE {
            for o in 0..ARH_V2_NUM_OVERLAP {
                for s in 0..ARH_V2_NUM_SUPPORTED {
                    rt.w_base[a][b][o][s] = disk.w_base[a][b][o][s] as i32;
                }
            }
        }
    }
    for al in 0..ARH_V2_NUM_ALPHA {
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                rt.w_alpha[al][a][b] = disk.w_alpha[al][a][b] as i32;
            }
        }
    }
    for ph in 0..ARH_V2_NUM_PHASE {
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                rt.w_phase[ph][a][b] = disk.w_phase[ph][a][b] as i32;
            }
        }
    }
    for al in 0..ARH_V2_NUM_ALPHA {
        for o in 0..ARH_V2_NUM_OVERLAP {
            for s in 0..ARH_V2_NUM_SUPPORTED {
                rt.w_alpha_shape[al][o][s] = disk.w_alpha_shape[al][o][s] as i32;
            }
        }
    }
    for h in 0..ARH_V2_NUM_HEAVY {
        for b in 0..ARH_V2_NUM_BALANCE {
            rt.w_heavy[h][b] = disk.w_heavy[h][b] as i32;
        }
    }
    rt
}

/// 1 perspective での最大 active 数 (= version 14 新 head は固定 6 index、 余裕含め 32)。
pub const ARH_V2_INTRUDE_MAX_ACTIVE: usize = 32;

/// `persp` 視点の敵陣ビットボード (= persp から見て最遠 3 段、 成り条件と同一)。
///
/// Black の敵陣 = rank 1-3 (= RANK_BB[0..=2])、 White の敵陣 = rank 7-9 (= RANK_BB[6..=8])。
#[inline]
fn enemy_camp(persp: Color) -> Bitboard {
    match persp {
        Color::Black => RANK_BB[0] | RANK_BB[1] | RANK_BB[2],
        Color::White => RANK_BB[6] | RANK_BB[7] | RANK_BB[8],
    }
}

/// `persp` 視点の敵陣隣接段ビットボード (= 次手で敵陣に入れる段)。
///
/// Black は rank 4 (= RANK_BB[3])、 White は rank 6 (= RANK_BB[5])。
#[inline]
fn enemy_camp_adjacent(persp: Color) -> Bitboard {
    match persp {
        Color::Black => RANK_BB[3],
        Color::White => RANK_BB[5],
    }
}

/// `persp` 視点の敵陣 3 段を [浅段, 中段, 最深段] の順 (= 自陣寄り → 敵玉寄り) で返す。
///
/// Black の敵陣 = rank 1-3。 自陣寄り (= 浅) から見て rank 3, 2, 1 の順。
/// White の敵陣 = rank 7-9。 自陣寄り (= 浅) から見て rank 7, 8, 9 の順。
/// `advanced_pawn` 軸の段位別 bucket (= 浅/中/深) 判定に用いる。
#[inline]
fn enemy_camp_ranks(persp: Color) -> [Bitboard; 3] {
    match persp {
        // [浅 (rank3), 中 (rank2), 深 (rank1)]
        Color::Black => [RANK_BB[2], RANK_BB[1], RANK_BB[0]],
        // [浅 (rank7), 中 (rank8), 深 (rank9)]
        Color::White => [RANK_BB[6], RANK_BB[7], RANK_BB[8]],
    }
}

/// 入玉宣言 (= 27 点法) における駒 1 枚の点数 (= 飛角および成りの竜馬 = 5 点、 他 = 1 点)。
///
/// 玉は宣言点数の対象外 (= 呼び出し側で除外する)。 成銀以下の小駒の成りも 1 点。
#[inline]
fn entering_king_point(pt: PieceType) -> u32 {
    match pt {
        PieceType::Rook | PieceType::Dragon | PieceType::Bishop | PieceType::Horse => 5,
        _ => 1,
    }
}

/// 入玉宣言の必要点数 (= 簡便版で先後とも一律 27 点。 設計判断)。
const ENTERING_KING_TARGET: u32 = 27;

/// `persp` 視点で 6 軸 (= 侵入駒数 / 成りポテンシャル / 支えられた侵入駒 / 前進歩 / 持駒枚数 /
/// 入玉宣言残り点数) の flat index を `out` へ push する (= 常に 6 件、 戻り値 6、 version 14)。
///
/// - 軸1 intrusion (侵入駒数): persp の駒で敵陣 (= persp から見て最遠 3 段) に居る非玉駒数
///   (= 成駒含む)、 0..=5 clamp。
/// - 軸2 promo_ready (成りポテンシャル): persp の未成の成れる駒 (= 歩香桂銀角飛) で敵陣に居るか
///   敵陣隣接段 (= 次手で敵陣に入れる) に居る駒数、 0..=5 clamp。
/// - 軸3 supported_intruder (支えられた侵入駒): 敵陣の自駒 (= 玉除く) のうち自軍の利き
///   (= 自分自身 / 自玉 除く) で支えられた駒数、 0..=7 clamp。
/// - 軸4 advanced_pawn (前進歩/垂れ歩): persp の **未成歩 (Pawn)** のうち最も進出した
///   1 枚を敵陣からの段数で層化した位置 class (= 0..=5、 [`ARH_V2_NUM_ADV_PAWN`] のバケット
///   定義参照)。 count ではなく最深 1 枚の位置を取る。
/// - 軸5 hand_pieces (持駒枚数): persp 側 (= 自軍) の持駒総枚数 (= 歩〜飛、 玉以外) を
///   0..=5 へ clamp した bucket (= 5 枚以上は 5、 [`ARH_V2_NUM_HAND`] のバケット定義参照)。
///   持駒は先後で別管理されるため persp 側の hand のみを数える。
/// - 軸6 entering_king (入玉宣言残り点数): persp の入玉宣言 (= 27 点法) の残り点数を 6 bucket へ
///   層化 (= 0..=5、 [`ARH_V2_NUM_EK`] のバケット定義参照)。 自玉が敵陣内に居る場合のみ
///   宣言点数を計上し (= 自玉非入玉時は 0 点 = bucket 5)、 残り = max(0, 27 − 点数)。
///   点数 = 敵陣内自駒 (= 玉除く) + persp 側持駒の点数和 (= 飛角 5 点 / 他 1 点)。
///
/// 「敵陣」は敵玉位置でなく段 (= rank) で判定する。 先後で段が反転する点に注意
/// ([`enemy_camp`] が persp を吸収)。
///
/// flat[0..152) のみ使用 (= intrusion 0..24 / promo 24..48 / support 48..80 / adv_pawn 80..104 /
/// hand 104..128 / ek 128..152)。
/// flat[i] ↔ disk[i] が恒等対応のため、 推論側 (= 重み和) と学習側 (= sparse fill) で
/// bit-exact を共有する。
#[inline]
pub fn intrude_disk_indices(pos: &Position, persp: Color, phase: usize, out: &mut [u32]) -> usize {
    let mut n = 0usize;
    let occ = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);
    let own = pos.pieces_c(persp);
    let own_king = pos.king_square(persp);
    let own_non_king = own & !Bitboard::from_square(own_king);

    let camp = enemy_camp(persp);
    let camp_adj = enemy_camp_adjacent(persp);

    // ---- 軸1 侵入駒数 + 軸2 成りポテンシャル + 軸3 支え + 軸6 敵陣点数 (自駒 1 ループで集計) ----
    let mut intrusion = 0usize; // 敵陣の非玉駒数
    let mut promo = 0usize; // 敵陣 or 敵陣隣接の未成の成れる駒数
    let mut support = 0usize; // 敵陣の自駒で味方利きに支えられた駒数
    let mut ek_camp_points = 0u32; // 軸6: 敵陣内自駒 (= 玉除く) の入玉宣言点数和
    let mut it = own_non_king;
    while it.is_not_empty() {
        let sq = it.pop();
        let pt = pos.piece_on(sq).piece_type();
        let sq_bb = Bitboard::from_square(sq);
        let in_camp = (sq_bb & camp).is_not_empty();

        if in_camp {
            intrusion += 1;
            ek_camp_points += entering_king_point(pt);
            // 軸3: この侵入駒を支える味方 (= 自分自身 / 自玉 除く) が居るか。
            // own_non_king (= own & !own_king、 ループ前に算出済) を再利用し、 camp 駒ごとの
            // Bitboard::from_square(own_king) 再計算を排除する (= bit-exact、 AND の交換律)。
            let supporters = pos.attackers_to_occ(sq, occ) & own_non_king & !sq_bb;
            if supporters.is_not_empty() {
                support += 1;
            }
        }

        // 軸2: 未成の成れる駒 (= 歩香桂銀角飛) で敵陣 or 敵陣隣接段に居る駒。
        if pt.can_promote() && (in_camp || (sq_bb & camp_adj).is_not_empty()) {
            promo += 1;
        }
    }
    let intrusion = intrusion.min(ARH_V2_NUM_INTRUSION - 1);
    let promo = promo.min(ARH_V2_NUM_PROMO - 1);
    let support = support.min(ARH_V2_NUM_SUPPORT_INTR - 1);

    // ---- 軸4 前進歩/垂れ歩 (= 未成歩のみ、 最深 1 枚の位置 class) ----
    // 自軍未成歩の bitboard を段別マスクと積を取り、 最も進出した段の bucket を採用する。
    // 浅 → 深の順に評価し、 より深い段に歩が居れば bucket を上書き (= 最深優先)。
    let own_pawns = pos.pieces(persp, PieceType::Pawn);
    let camp_ranks = enemy_camp_ranks(persp); // [浅, 中, 深]
    let mut adv_pawn = 0usize;
    // bucket 2: 垂れ歩 (= 敵陣隣接段)。
    if (own_pawns & camp_adj).is_not_empty() {
        adv_pawn = 2;
    }
    // bucket 3/4/5: 敵陣 浅/中/深段。 深いほど優先。
    if (own_pawns & camp_ranks[0]).is_not_empty() {
        adv_pawn = 3;
    }
    if (own_pawns & camp_ranks[1]).is_not_empty() {
        adv_pawn = 4;
    }
    if (own_pawns & camp_ranks[2]).is_not_empty() {
        adv_pawn = 5;
    }
    // bucket 0 = 進出歩なし、 bucket 1 = その他進出歩 (= 隣接段より手前) は現状予約。
    let adv_pawn = adv_pawn.min(ARH_V2_NUM_ADV_PAWN - 1);

    // ---- 軸5 持駒枚数 (= persp 側の持駒総枚数、 0..5 へ clamp) ----
    let hand = pos.hand(persp);
    let mut hand_total = 0u32;
    let mut ek_hand_points = 0u32; // 軸6: 持駒の入玉宣言点数和 (= 飛角 5 / 他 1)
    for &pt in &PieceType::HAND_PIECES {
        let c = hand.count(pt);
        hand_total += c;
        ek_hand_points += c * entering_king_point(pt);
    }
    let hand_bucket = (hand_total as usize).min(ARH_V2_NUM_HAND - 1);

    // ---- 軸6 入玉宣言 残り点数 (= 自玉が敵陣内のときのみ点数計上、 残り = max(0, 27 − 点数)) ----
    // 自玉が敵陣内に居ない場合は入玉宣言の前提を満たさないため点数 0 (= 残り 27 = bucket 5)。
    let own_king_in_camp = (Bitboard::from_square(own_king) & camp).is_not_empty();
    let ek_points = if own_king_in_camp {
        ek_camp_points + ek_hand_points
    } else {
        0
    };
    let ek_remaining = ENTERING_KING_TARGET.saturating_sub(ek_points);
    // bucket: 残り ≤0 → 0、 1..=5 → 1、 6..=10 → 2、 11..=15 → 3、 16..=20 → 4、 21..=27 → 5。
    let ek_bucket = if ek_remaining == 0 {
        0
    } else {
        (((ek_remaining - 1) / 5) as usize + 1).min(ARH_V2_NUM_EK - 1)
    };

    if n < out.len() {
        out[n] = intrusion_idx(intrusion, phase) as u32;
        n += 1;
    }
    if n < out.len() {
        out[n] = promo_idx(promo, phase) as u32;
        n += 1;
    }
    if n < out.len() {
        out[n] = support_idx(support, phase) as u32;
        n += 1;
    }
    if n < out.len() {
        out[n] = adv_pawn_idx(adv_pawn, phase) as u32;
        n += 1;
    }
    if n < out.len() {
        out[n] = hand_idx(hand_bucket, phase) as u32;
        n += 1;
    }
    if n < out.len() {
        out[n] = ek_idx(ek_bucket, phase) as u32;
        n += 1;
    }
    n
}

/// `persp` 視点の侵入 score (= [`intrude_disk_indices`] の weight 和)。
#[inline]
fn intrusion_score(rt: &ArhV2RuntimeWeights, pos: &Position, persp: Color, phase: usize) -> i32 {
    let mut out = [0u32; ARH_V2_INTRUDE_MAX_ACTIVE];
    let n = intrude_disk_indices(pos, persp, phase, &mut out);
    let mut score = 0i32;
    for &idx in &out[..n] {
        score = score.wrapping_add(rt.flat[idx as usize]);
    }
    score
}

/// 評価本体 (= version 11 侵入 head)。 `bucket` は LayerStacks output bucket、 `stm` は side-to-move。
/// eval = (自陣営の敵陣侵入駒 weight 和 − 敵陣営の自陣侵入駒 weight 和) × OUTPUT_SCALE。
#[inline]
pub fn evaluate_arh_v2(rt: &ArhV2RuntimeWeights, pos: &Position, bucket: usize, stm: Color) -> i32 {
    debug_assert!(bucket < ARH_V2_NUM_BUCKETS);
    // OUTPUT_SCALE == 0 のとき最終結果は必ず `× 0 == 0` になるため、
    // intrusion_score の bitboard scan を行わず即 0 を返す。
    // `× 0` と早期 return 0 は代数的に同値なので eval は全局面で bit-exact に不変であり、
    // head を無効化 (scale=0) した場合に per-node の侵入駒 scan を省ける純粋な NPS 改善となる。
    let scale = ARH_V2_OUTPUT_SCALE.load(Ordering::Relaxed);
    if scale == 0 {
        return 0;
    }
    let phase = bucket_to_phase(bucket);
    let us = intrusion_score(rt, pos, stm, phase);
    let them = intrusion_score(rt, pos, stm.opponent(), phase);
    us.saturating_sub(them).saturating_mul(scale)
}

// =============================================================================
// disk I/O (= magic / version / count 検証 + 980 i16 round-trip)
// =============================================================================

/// 980 weight を sub-section magic + version + count + 連結 i16 形式で write する。
pub fn write_weights<W: Write>(w: &mut W, disk: &ArhV2Weights) -> io::Result<()> {
    w.write_all(ARH_V2_MAGIC)?;
    w.write_all(&ARH_V2_VERSION.to_le_bytes())?;
    w.write_all(&(ARH_V2_WEIGHT_COUNT as u32).to_le_bytes())?;
    for a in 0..ARH_V2_NUM_ATTACKERS {
        for b in 0..ARH_V2_NUM_BALANCE {
            for o in 0..ARH_V2_NUM_OVERLAP {
                for s in 0..ARH_V2_NUM_SUPPORTED {
                    w.write_all(&disk.w_base[a][b][o][s].to_le_bytes())?;
                }
            }
        }
    }
    for al in 0..ARH_V2_NUM_ALPHA {
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                w.write_all(&disk.w_alpha[al][a][b].to_le_bytes())?;
            }
        }
    }
    for ph in 0..ARH_V2_NUM_PHASE {
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                w.write_all(&disk.w_phase[ph][a][b].to_le_bytes())?;
            }
        }
    }
    for al in 0..ARH_V2_NUM_ALPHA {
        for o in 0..ARH_V2_NUM_OVERLAP {
            for s in 0..ARH_V2_NUM_SUPPORTED {
                w.write_all(&disk.w_alpha_shape[al][o][s].to_le_bytes())?;
            }
        }
    }
    for h in 0..ARH_V2_NUM_HEAVY {
        for b in 0..ARH_V2_NUM_BALANCE {
            w.write_all(&disk.w_heavy[h][b].to_le_bytes())?;
        }
    }
    Ok(())
}

/// magic / version / count を検証した上で 980 weight を読み込む。
///
/// magic は呼び出し側で probe 済 (= 8 byte 一致確認済) と前提する。
/// version 不一致 / count 不一致は `io::Error` (`InvalidData`) で返す。
pub fn read_weights_after_magic<R: Read>(r: &mut R) -> io::Result<Box<ArhV2Weights>> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != ARH_V2_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "AttackRelay v2 (P2ARS1) version {version} unsupported (expected {ARH_V2_VERSION})"
            ),
        ));
    }
    r.read_exact(&mut buf4)?;
    let count = u32::from_le_bytes(buf4);
    if (count as usize) != ARH_V2_WEIGHT_COUNT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "AttackRelay v2 (P2ARS1) count mismatch: expected {ARH_V2_WEIGHT_COUNT}, got {count}"
            ),
        ));
    }
    let mut disk = ArhV2Weights::zeroed();
    let mut buf2 = [0u8; 2];
    for a in 0..ARH_V2_NUM_ATTACKERS {
        for b in 0..ARH_V2_NUM_BALANCE {
            for o in 0..ARH_V2_NUM_OVERLAP {
                for s in 0..ARH_V2_NUM_SUPPORTED {
                    r.read_exact(&mut buf2)?;
                    disk.w_base[a][b][o][s] = i16::from_le_bytes(buf2);
                }
            }
        }
    }
    for al in 0..ARH_V2_NUM_ALPHA {
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                r.read_exact(&mut buf2)?;
                disk.w_alpha[al][a][b] = i16::from_le_bytes(buf2);
            }
        }
    }
    for ph in 0..ARH_V2_NUM_PHASE {
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                r.read_exact(&mut buf2)?;
                disk.w_phase[ph][a][b] = i16::from_le_bytes(buf2);
            }
        }
    }
    for al in 0..ARH_V2_NUM_ALPHA {
        for o in 0..ARH_V2_NUM_OVERLAP {
            for s in 0..ARH_V2_NUM_SUPPORTED {
                r.read_exact(&mut buf2)?;
                disk.w_alpha_shape[al][o][s] = i16::from_le_bytes(buf2);
            }
        }
    }
    for h in 0..ARH_V2_NUM_HEAVY {
        for b in 0..ARH_V2_NUM_BALANCE {
            r.read_exact(&mut buf2)?;
            disk.w_heavy[h][b] = i16::from_le_bytes(buf2);
        }
    }
    Ok(disk)
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Position, SFEN_HIRATE};

    fn startpos() -> Position {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        pos
    }

    fn set_sfen(s: &str) -> Position {
        let mut pos = Position::new();
        pos.set_sfen(s).unwrap();
        pos
    }

    // -- 仕様 §必須テスト 1: 盤端 玉でも 5x5 / 3x3 盤外アクセスなし。
    #[test]
    fn edge_king_no_oob() {
        // 1九 (= rank 9, file 1)、 9九 (= rank 9, file 9)、 1一 (= rank 1, file 1) 等
        // で attackers / balance / overlap / supported / alpha を計算しても panic しない。
        for sfen in [
            "9/9/9/9/9/9/9/9/k1K6 b - 1",
            "9/9/9/9/9/9/9/9/6K1k b - 1",
            "k1K6/9/9/9/9/9/9/9/9 b - 1",
            "6K1k/9/9/9/9/9/9/9/9 b - 1",
        ] {
            let pos = set_sfen(sfen);
            let c_b = compute_arh_v2_classes(&pos, Color::Black);
            let c_w = compute_arh_v2_classes(&pos, Color::White);
            assert!(c_b.attackers < ARH_V2_NUM_ATTACKERS);
            assert!(c_b.balance < ARH_V2_NUM_BALANCE);
            assert!(c_b.overlap < ARH_V2_NUM_OVERLAP);
            assert!(c_b.supported < ARH_V2_NUM_SUPPORTED);
            assert!(c_b.alpha < ARH_V2_NUM_ALPHA);
            assert!(c_w.attackers < ARH_V2_NUM_ATTACKERS);
            assert!(c_w.balance < ARH_V2_NUM_BALANCE);
            assert!(c_w.overlap < ARH_V2_NUM_OVERLAP);
            assert!(c_w.supported < ARH_V2_NUM_SUPPORTED);
            assert!(c_w.alpha < ARH_V2_NUM_ALPHA);
        }
    }

    // -- 仕様 §必須テスト 2: attackers がユニーク駒数 (= 同駒で複数升へ利いても 1)。
    #[test]
    fn attackers_is_unique_piece_count() {
        // 5x5 zone 内 8升へ利く飛車 = 1 駒、 attackers = 1。
        // 局面: 黒 玉 9九、 白 玉 5五、 黒 飛車 5九 (= 5五 へ 飛車先利き)。
        let pos = set_sfen("9/9/9/9/4k4/9/9/9/4R3K b - 1");
        let c = compute_arh_v2_classes(&pos, Color::Black);
        // 飛車 1 駒のみ 5x5 (= 白玉 5五 中心) へ利く。
        assert_eq!(c.attackers, 1);
    }

    // -- 仕様 §必須テスト 3: 玉が attackers / supporters から除外。
    #[test]
    fn kings_excluded_from_attackers_and_supporters() {
        // 黒 玉 5七 が白玉 5五 の 5x5 zone 内に入る局面でも attackers = 0。
        let pos = set_sfen("9/9/9/9/4k4/9/4K4/9/9 b - 1");
        let c = compute_arh_v2_classes(&pos, Color::Black);
        assert_eq!(c.attackers, 0, "own king must not be counted as attacker");
        assert_eq!(c.supported, 0);
    }

    // -- 仕様 §必須テスト 4: balance が -4..4 へ clamp (= 0..8)。
    #[test]
    fn balance_in_range() {
        // 多数の攻撃 / 防御を集めても balance_class は 0..=8 内。
        let pos = startpos();
        let c = compute_arh_v2_classes(&pos, Color::Black);
        assert!(c.balance < ARH_V2_NUM_BALANCE);
    }

    // -- 仕様 §必須テスト 5: overlap が 0..3 へ clamp。
    #[test]
    fn overlap_in_range() {
        let pos = startpos();
        let c = compute_arh_v2_classes(&pos, Color::Black);
        assert!(c.overlap < ARH_V2_NUM_OVERLAP);
    }

    // -- 仕様 §必須テスト 6: supported が 0..3 へ clamp。
    #[test]
    fn supported_in_range() {
        let pos = startpos();
        let c = compute_arh_v2_classes(&pos, Color::Black);
        assert!(c.supported < ARH_V2_NUM_SUPPORTED);
    }

    // -- 仕様 §必須テスト 7: alpha が 0..4 へ clamp。
    #[test]
    fn alpha_in_range() {
        let pos = startpos();
        let c = compute_arh_v2_classes(&pos, Color::Black);
        assert!(c.alpha < ARH_V2_NUM_ALPHA);
    }

    // -- 仕様 §必須テスト 7b: heavy (= escape v5) が 0..3 へ clamp。
    // escape は攻撃カウントと独立なため attackers との大小関係は問わない。
    #[test]
    fn heavy_in_range() {
        let pos = startpos();
        let c = compute_arh_v2_classes(&pos, Color::Black);
        assert!(c.heavy < ARH_V2_NUM_HEAVY);
    }

    // -- 仕様 §必須テスト 8: bucket → phase 写像 0..3。
    #[test]
    fn bucket_to_phase_monotonic() {
        let expected = [0, 0, 1, 1, 2, 2, 3, 3];
        for (b, &e) in expected.iter().enumerate() {
            assert_eq!(bucket_to_phase(b), e);
        }
    }

    // -- 仕様 §必須テスト 9: 各テーブルの index が範囲内で一意。
    #[test]
    fn table_local_indices_unique() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                for o in 0..ARH_V2_NUM_OVERLAP {
                    for s in 0..ARH_V2_NUM_SUPPORTED {
                        let idx = w_base_local_idx(a, b, o, s);
                        assert!(idx < ARH_V2_W_BASE_COUNT);
                        assert!(set.insert(("base", idx)));
                    }
                }
            }
        }
        for al in 0..ARH_V2_NUM_ALPHA {
            for a in 0..ARH_V2_NUM_ATTACKERS {
                for b in 0..ARH_V2_NUM_BALANCE {
                    let idx = w_alpha_local_idx(al, a, b);
                    assert!(idx < ARH_V2_W_ALPHA_COUNT);
                    assert!(set.insert(("alpha", idx)));
                }
            }
        }
        for ph in 0..ARH_V2_NUM_PHASE {
            for a in 0..ARH_V2_NUM_ATTACKERS {
                for b in 0..ARH_V2_NUM_BALANCE {
                    let idx = w_phase_local_idx(ph, a, b);
                    assert!(idx < ARH_V2_W_PHASE_COUNT);
                    assert!(set.insert(("phase", idx)));
                }
            }
        }
        for al in 0..ARH_V2_NUM_ALPHA {
            for o in 0..ARH_V2_NUM_OVERLAP {
                for s in 0..ARH_V2_NUM_SUPPORTED {
                    let idx = w_alpha_shape_local_idx(al, o, s);
                    assert!(idx < ARH_V2_W_ALPHA_SHAPE_COUNT);
                    assert!(set.insert(("alpha_shape", idx)));
                }
            }
        }
        for h in 0..ARH_V2_NUM_HEAVY {
            for b in 0..ARH_V2_NUM_BALANCE {
                let idx = w_heavy_local_idx(h, b);
                assert!(idx < ARH_V2_W_HEAVY_COUNT);
                assert!(set.insert(("heavy", idx)));
            }
        }
        assert_eq!(set.len(), ARH_V2_WEIGHT_COUNT);
    }

    // -- 仕様 §必須テスト 10: 全 disk index 範囲が重ならず合計 980。
    #[test]
    fn disk_indices_partition_into_980() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                for o in 0..ARH_V2_NUM_OVERLAP {
                    for s in 0..ARH_V2_NUM_SUPPORTED {
                        assert!(set.insert(w_base_disk_idx(a, b, o, s)));
                    }
                }
            }
        }
        for al in 0..ARH_V2_NUM_ALPHA {
            for a in 0..ARH_V2_NUM_ATTACKERS {
                for b in 0..ARH_V2_NUM_BALANCE {
                    assert!(set.insert(w_alpha_disk_idx(al, a, b)));
                }
            }
        }
        for ph in 0..ARH_V2_NUM_PHASE {
            for a in 0..ARH_V2_NUM_ATTACKERS {
                for b in 0..ARH_V2_NUM_BALANCE {
                    assert!(set.insert(w_phase_disk_idx(ph, a, b)));
                }
            }
        }
        for al in 0..ARH_V2_NUM_ALPHA {
            for o in 0..ARH_V2_NUM_OVERLAP {
                for s in 0..ARH_V2_NUM_SUPPORTED {
                    assert!(set.insert(w_alpha_shape_disk_idx(al, o, s)));
                }
            }
        }
        for h in 0..ARH_V2_NUM_HEAVY {
            for b in 0..ARH_V2_NUM_BALANCE {
                assert!(set.insert(w_heavy_disk_idx(h, b)));
            }
        }
        assert_eq!(set.len(), ARH_V2_WEIGHT_COUNT);
        assert_eq!(*set.iter().max().unwrap() + 1, ARH_V2_WEIGHT_COUNT);
        assert_eq!(*set.iter().min().unwrap(), 0);
    }

    // -- 仕様 §必須テスト 11: 980 weight disk layout の round-trip。
    #[test]
    fn round_trip_disk_layout() {
        let mut disk = ArhV2Weights::zeroed();
        // 全 weight にユニークな値を埋める (= 基準スライスを含む)、 後で基準を 0 へ戻す。
        let mut counter: i16 = 1;
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                for o in 0..ARH_V2_NUM_OVERLAP {
                    for s in 0..ARH_V2_NUM_SUPPORTED {
                        disk.w_base[a][b][o][s] = counter;
                        counter = counter.wrapping_add(1);
                    }
                }
            }
        }
        for al in 1..ARH_V2_NUM_ALPHA {
            for a in 0..ARH_V2_NUM_ATTACKERS {
                for b in 0..ARH_V2_NUM_BALANCE {
                    disk.w_alpha[al][a][b] = counter;
                    counter = counter.wrapping_add(1);
                }
            }
        }
        for ph in 1..ARH_V2_NUM_PHASE {
            for a in 0..ARH_V2_NUM_ATTACKERS {
                for b in 0..ARH_V2_NUM_BALANCE {
                    disk.w_phase[ph][a][b] = counter;
                    counter = counter.wrapping_add(1);
                }
            }
        }
        for al in 1..ARH_V2_NUM_ALPHA {
            for o in 0..ARH_V2_NUM_OVERLAP {
                for s in 0..ARH_V2_NUM_SUPPORTED {
                    disk.w_alpha_shape[al][o][s] = counter;
                    counter = counter.wrapping_add(1);
                }
            }
        }
        // W_heavy は基準スライス無しのため全 (heavy, balance) を埋める。
        for h in 0..ARH_V2_NUM_HEAVY {
            for b in 0..ARH_V2_NUM_BALANCE {
                disk.w_heavy[h][b] = counter;
                counter = counter.wrapping_add(1);
            }
        }
        assert!(disk.zero_baseline_satisfied());
        let mut buf = Vec::new();
        buf.extend_from_slice(ARH_V2_MAGIC); // probe 用 magic
        write_weights(&mut buf, &disk).unwrap();
        // probe magic を skip して read_weights_after_magic 経由で復元。
        let mut cur = std::io::Cursor::new(&buf);
        let mut magic = [0u8; 8];
        cur.read_exact(&mut magic).unwrap();
        assert_eq!(&magic, ARH_V2_MAGIC);
        // 内部 write_weights が冒頭に再度 magic を書くため、 2 つ目の magic も skip。
        let mut magic2 = [0u8; 8];
        cur.read_exact(&mut magic2).unwrap();
        assert_eq!(&magic2, ARH_V2_MAGIC);
        let restored = read_weights_after_magic(&mut cur).unwrap();
        // section 内全 weight が一致。
        for a in 0..ARH_V2_NUM_ATTACKERS {
            for b in 0..ARH_V2_NUM_BALANCE {
                for o in 0..ARH_V2_NUM_OVERLAP {
                    for s in 0..ARH_V2_NUM_SUPPORTED {
                        assert_eq!(restored.w_base[a][b][o][s], disk.w_base[a][b][o][s]);
                    }
                }
            }
        }
        for al in 0..ARH_V2_NUM_ALPHA {
            for a in 0..ARH_V2_NUM_ATTACKERS {
                for b in 0..ARH_V2_NUM_BALANCE {
                    assert_eq!(restored.w_alpha[al][a][b], disk.w_alpha[al][a][b]);
                }
            }
        }
        for ph in 0..ARH_V2_NUM_PHASE {
            for a in 0..ARH_V2_NUM_ATTACKERS {
                for b in 0..ARH_V2_NUM_BALANCE {
                    assert_eq!(restored.w_phase[ph][a][b], disk.w_phase[ph][a][b]);
                }
            }
        }
        for al in 0..ARH_V2_NUM_ALPHA {
            for o in 0..ARH_V2_NUM_OVERLAP {
                for s in 0..ARH_V2_NUM_SUPPORTED {
                    assert_eq!(restored.w_alpha_shape[al][o][s], disk.w_alpha_shape[al][o][s]);
                }
            }
        }
        for h in 0..ARH_V2_NUM_HEAVY {
            for b in 0..ARH_V2_NUM_BALANCE {
                assert_eq!(restored.w_heavy[h][b], disk.w_heavy[h][b]);
            }
        }
    }

    // -- 仕様 §必須テスト 12: magic / version / count 不一致がエラー。
    #[test]
    fn rejects_version_or_count_mismatch() {
        // wrong version。
        let mut buf = Vec::new();
        buf.extend_from_slice(&999u32.to_le_bytes());
        buf.extend_from_slice(&(ARH_V2_WEIGHT_COUNT as u32).to_le_bytes());
        buf.extend(std::iter::repeat_n(0u8, ARH_V2_WEIGHT_COUNT * 2));
        let err = read_weights_after_magic(&mut std::io::Cursor::new(&buf))
            .err()
            .expect("must error on wrong version");
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        // wrong count。
        let mut buf = Vec::new();
        buf.extend_from_slice(&ARH_V2_VERSION.to_le_bytes());
        buf.extend_from_slice(&1234u32.to_le_bytes());
        buf.extend(std::iter::repeat_n(0u8, 2));
        let err = read_weights_after_magic(&mut std::io::Cursor::new(&buf))
            .err()
            .expect("must error on wrong count");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    // -- 仕様 §必須テスト 13: 基準スライスが学習後・量子化後もゼロ。
    #[test]
    fn zero_baseline_enforced() {
        let mut disk = ArhV2Weights::zeroed();
        // 学習側が誤って基準スライスを更新したと仮定して非ゼロを混入。
        disk.w_alpha[0][2][3] = 42;
        disk.w_phase[0][1][5] = -17;
        disk.w_alpha_shape[0][2][2] = 99;
        assert!(!disk.zero_baseline_satisfied());
        disk.enforce_zero_baseline();
        assert!(disk.zero_baseline_satisfied());
        assert_eq!(disk.w_alpha[0][2][3], 0);
        assert_eq!(disk.w_phase[0][1][5], 0);
        assert_eq!(disk.w_alpha_shape[0][2][2], 0);
    }

    // -- 仕様 §必須テスト 15: 全 weight ゼロでは ARH 評価値 0 (= main 評価へ影響なし)。
    #[test]
    fn zero_weights_eval_is_zero() {
        let disk = ArhV2Weights::zeroed();
        let rt = build_runtime_weights(&disk);
        let pos = startpos();
        for bucket in 0..ARH_V2_NUM_BUCKETS {
            assert_eq!(evaluate_arh_v2(&rt, &pos, bucket, Color::Black), 0);
            assert_eq!(evaluate_arh_v2(&rt, &pos, bucket, Color::White), 0);
        }
    }

    // -- 仕様 §必須テスト 16: stm を反転すると評価値の符号が反転する (= 反対称性)。
    #[test]
    fn antisymmetric_under_stm_swap() {
        let mut disk = ArhV2Weights::zeroed();
        // 適当な非対称 weight を埋める。
        disk.w_base[1][5][1][1] = 30;
        disk.w_base[2][4][0][2] = -25;
        // 基準スライス外の alpha / phase / alpha_shape を 1 つずつ埋める。
        disk.w_alpha[1][1][5] = 12;
        disk.w_phase[1][1][5] = 8;
        disk.w_alpha_shape[1][1][1] = 5;
        let rt = build_runtime_weights(&disk);
        let pos = startpos();
        let scale_before = ARH_V2_OUTPUT_SCALE.load(Ordering::Relaxed);
        set_output_scale(1);
        for bucket in 0..ARH_V2_NUM_BUCKETS {
            let v_black = evaluate_arh_v2(&rt, &pos, bucket, Color::Black);
            let v_white = evaluate_arh_v2(&rt, &pos, bucket, Color::White);
            assert_eq!(v_black, -v_white, "ARH v2 eval must be antisymmetric under stm");
        }
        set_output_scale(scale_before);
    }

    // 追加: OUTPUT_SCALE 倍率の確認。
    #[test]
    fn output_scale_multiplies_eval() {
        let mut disk = ArhV2Weights::zeroed();
        disk.w_base[1][5][1][1] = 10;
        let rt = build_runtime_weights(&disk);
        let pos = startpos();
        let scale_before = ARH_V2_OUTPUT_SCALE.load(Ordering::Relaxed);
        set_output_scale(1);
        let v1 = evaluate_arh_v2(&rt, &pos, 0, Color::Black);
        set_output_scale(7);
        let v7 = evaluate_arh_v2(&rt, &pos, 0, Color::Black);
        assert_eq!(v7, v1 * 7);
        set_output_scale(scale_before);
    }

    /// flat index k (= < W_base count) に値を書く test helper (= flat[k] は disk 順で w_base)。
    fn set_flat(disk: &mut ArhV2Weights, k: usize, val: i16) {
        let a = k / (ARH_V2_NUM_BALANCE * ARH_V2_NUM_OVERLAP * ARH_V2_NUM_SUPPORTED);
        let rem = k % (ARH_V2_NUM_BALANCE * ARH_V2_NUM_OVERLAP * ARH_V2_NUM_SUPPORTED);
        let b = rem / (ARH_V2_NUM_OVERLAP * ARH_V2_NUM_SUPPORTED);
        let rem2 = rem % (ARH_V2_NUM_OVERLAP * ARH_V2_NUM_SUPPORTED);
        let o = rem2 / ARH_V2_NUM_SUPPORTED;
        let s = rem2 % ARH_V2_NUM_SUPPORTED;
        disk.w_base[a][b][o][s] = val;
    }

    // -- version 11 侵入 head: 敵陣に侵入した自駒の intrusion weight が反対称に効く。
    #[test]
    fn new_head_eval_basic() {
        // Black の銀 (Silver) が敵陣 (= rank 1-3) に居る局面 → Black の intrusion >= 1。
        // 白玉 5一 (= e1)、 Black 銀 5三 (= e3、 rank 3 = 敵陣)、 黒玉 5九 (= e9)。
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/4S4/9/9/9/9/9/4K4 b - 1").unwrap();
        let phase = bucket_to_phase(0);
        let mut out_b = [0u32; ARH_V2_INTRUDE_MAX_ACTIVE];
        let nb = intrude_disk_indices(&pos, Color::Black, phase, &mut out_b);
        assert_eq!(nb, 6);
        // out_b[0] = intrusion index (= 銀が敵陣に居るので intrusion >= 1)。
        let intr_b = out_b[0] as usize;
        assert!(
            (ARH_V2_INTRUSION_OFFSET..ARH_V2_PROMO_OFFSET).contains(&intr_b),
            "intr_b={intr_b}"
        );
        assert!(intr_b > intrusion_idx(0, phase), "Black 銀が敵陣に居る → intrusion >= 1");
        let mut disk = ArhV2Weights::zeroed();
        set_flat(&mut disk, intr_b, 30);
        let rt = build_runtime_weights(&disk);
        let sb = ARH_V2_OUTPUT_SCALE.load(Ordering::Relaxed);
        set_output_scale(1);
        let v_black = evaluate_arh_v2(&rt, &pos, 0, Color::Black);
        let v_white = evaluate_arh_v2(&rt, &pos, 0, Color::White);
        // White 側は侵入駒が無く intrusion=0 → them が同 index に該当しないため eval=30。
        assert_eq!(v_black, 30, "Black intrusion weight が効く");
        assert_eq!(v_white, -30, "反対称");
        set_output_scale(sb);
    }

    // 追加: fused 1-pass 実装 (= compute_arh_v2_classes_both) が reference
    // (= compute_arh_v2_classes × 2) と全 5 軸で bit-exact 一致。
    // 固定 SFEN (= 盤端玉 / 中盤 / 終盤入玉系) + random playout で検証する。
    #[test]
    fn fused_classes_match_reference() {
        use crate::movegen::{MoveList, generate_legal};

        fn assert_match(pos: &Position, label: &str) {
            let both = compute_arh_v2_classes_both(pos);
            let ref_b = compute_arh_v2_classes(pos, Color::Black);
            let ref_w = compute_arh_v2_classes(pos, Color::White);
            assert_eq!(both[Color::Black.index()], ref_b, "black mismatch: {label}");
            assert_eq!(both[Color::White.index()], ref_w, "white mismatch: {label}");
        }

        // 固定 SFEN 群。
        for sfen in [
            SFEN_HIRATE,
            // 盤端玉。
            "9/9/9/9/9/9/9/9/k1K6 b - 1",
            "9/9/9/9/9/9/9/9/6K1k b - 1",
            "k1K6/9/9/9/9/9/9/9/9 b - 1",
            "6K1k/9/9/9/9/9/9/9/9 b - 1",
            // 中盤 / 乱戦 (= NPS bench と同系統の局面)。
            "lnsgkgsnl/1r7/p1ppp1bpp/1p3pp2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b - 9",
            "l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",
            "6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1",
            "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
        ] {
            let pos = set_sfen(sfen);
            assert_match(&pos, sfen);
        }

        // random playout (= 決定的 LCG、 startpos から 60 手 × 20 game)。
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        for game in 0..20 {
            let mut pos = startpos();
            for ply in 0..60 {
                let mut list = MoveList::new();
                generate_legal(&pos, &mut list);
                if list.is_empty() {
                    break;
                }
                let mv = list.at(next() % list.len());
                let gc = pos.gives_check(mv);
                pos.do_move(mv, gc);
                assert_match(&pos, &format!("game {game} ply {ply}"));
            }
        }
    }

    // 追加: dim / magic / version の sanity。
    #[test]
    fn dims_magic_version_sanity() {
        assert_eq!(ARH_V2_WEIGHT_COUNT, 1016);
        assert_eq!(ARH_V2_W_BASE_COUNT, 576);
        assert_eq!(ARH_V2_W_ALPHA_COUNT, 180);
        assert_eq!(ARH_V2_W_PHASE_COUNT, 144);
        assert_eq!(ARH_V2_W_ALPHA_SHAPE_COUNT, 80);
        assert_eq!(ARH_V2_W_HEAVY_COUNT, 36);
        assert_eq!(ARH_V2_OFFSET_W_BASE, 0);
        assert_eq!(ARH_V2_OFFSET_W_ALPHA, 576);
        assert_eq!(ARH_V2_OFFSET_W_PHASE, 756);
        assert_eq!(ARH_V2_OFFSET_W_ALPHA_SHAPE, 900);
        assert_eq!(ARH_V2_OFFSET_W_HEAVY, 980);
        assert_eq!(ARH_V2_MAGIC, b"P2ARS1\0\0");
        assert_eq!(ARH_V2_VERSION, 14);
        // version 14 侵入 head: 6 軸 × 4 phase、 intrusion/promo/support/adv_pawn/hand/ek
        // offset 0/24/48/80/104/128、 計使用 152。
        assert_eq!(ARH_V2_NUM_INTRUSION, 6);
        assert_eq!(ARH_V2_NUM_PROMO, 6);
        assert_eq!(ARH_V2_NUM_SUPPORT_INTR, 8);
        assert_eq!(ARH_V2_NUM_ADV_PAWN, 6);
        assert_eq!(ARH_V2_NUM_HAND, 6);
        assert_eq!(ARH_V2_NUM_EK, 6);
        assert_eq!(ARH_V2_INTRUSION_OFFSET, 0);
        assert_eq!(ARH_V2_PROMO_OFFSET, 24);
        assert_eq!(ARH_V2_SUPPORT_OFFSET, 48);
        assert_eq!(ARH_V2_ADV_PAWN_OFFSET, 80);
        assert_eq!(ARH_V2_HAND_OFFSET, 104);
        assert_eq!(ARH_V2_EK_OFFSET, 128);
        assert_eq!(ARH_V2_NEW_HEAD_USED, 152);
        // ARH_V2_NEW_HEAD_USED <= ARH_V2_WEIGHT_COUNT は冒頭 const _ ブロックで静的検証済。
    }

    // version 12 侵入 head: 既知 SFEN で 4 index を分解 + 手計算照合 (= bit-exact 検証用、
    // bullet `intrude_active` も同 `intrude_disk_indices` を呼ぶため一致が保証される)。
    #[test]
    fn new_head_bitexact_decode() {
        // SFEN: 白玉 5一 (e1)、 黒銀 5三 (e3、 rank3 = Black 敵陣)、
        //       黒歩 5四 (e4、 rank4 = Black 敵陣隣接段)、 黒玉 5九 (e9)。 phase = 0。
        // 期待 (Black persp):
        //   軸1 intrusion: 敵陣 (rank1-3) の黒非玉駒 = 銀 e3 のみ → intrusion=1 → intrusion_idx(1,0)=4。
        //   軸2 promo_ready: 未成の成れる駒で敵陣 or 隣接段 = 銀 e3 (敵陣) + 歩 e4 (隣接) → promo=2
        //     → promo_idx(2,0)=32。
        //   軸3 support: 敵陣 (= 銀 e3) を支える味方利り = 歩 e4 は前方 e3 へ利く → support=1
        //     → support_idx(1,0)=52。
        //   軸4 adv_pawn: 黒未成歩 = e4 (rank4 = Black 敵陣隣接段 = 垂れ歩) のみ。 敵陣段に歩なし
        //     → adv_pawn=2 → adv_pawn_idx(2,0)=88。
        //   軸5 hand: Black 持駒なし (= "b - 1") → hand=0 → hand_idx(0,0)=104。
        //   軸6 ek: 黒玉 e9 は敵陣 (rank1-3) 外 → 点数 0 → 残り 27 → ek=5 → ek_idx(5,0)=148。
        let pos = set_sfen("4k4/9/4S4/4P4/9/9/9/9/4K4 b - 1");
        let mut out = [0u32; ARH_V2_INTRUDE_MAX_ACTIVE];
        let n = intrude_disk_indices(&pos, Color::Black, 0, &mut out);
        assert_eq!(n, 6);
        assert_eq!(out[0] as usize, intrusion_idx(1, 0), "intrusion index");
        assert_eq!(out[0], 4);
        assert_eq!(out[1] as usize, promo_idx(2, 0), "promo index");
        assert_eq!(out[1], 32);
        assert_eq!(out[2] as usize, support_idx(1, 0), "support index");
        assert_eq!(out[2], 52);
        assert_eq!(out[3] as usize, adv_pawn_idx(2, 0), "adv_pawn index");
        assert_eq!(out[3], 88);
        assert_eq!(out[4] as usize, hand_idx(0, 0), "hand index");
        assert_eq!(out[4], 104);
        assert_eq!(out[5] as usize, ek_idx(5, 0), "ek index");
        assert_eq!(out[5], 148);
    }

    // version 14 侵入 head: 平手初期局面で両 persp とも必ず 6 index、 各 index が
    // 正しい range (= intrusion∈[0,24) / promo∈[24,48) / support∈[48,80) /
    // adv_pawn∈[80,104) / hand∈[104,128) / ek∈[128,152)) に入る。
    #[test]
    fn new_head_startpos_six_indices_in_range() {
        let pos = startpos();
        let mut out = [0u32; ARH_V2_INTRUDE_MAX_ACTIVE];
        for persp in [Color::Black, Color::White] {
            let n = intrude_disk_indices(&pos, persp, 0, &mut out);
            assert_eq!(n, 6, "persp={persp:?} は常に 6 index");
            let intrusion = out[0] as usize;
            let promo = out[1] as usize;
            let support = out[2] as usize;
            let adv_pawn = out[3] as usize;
            let hand = out[4] as usize;
            let ek = out[5] as usize;
            assert!(
                (ARH_V2_INTRUSION_OFFSET..ARH_V2_PROMO_OFFSET).contains(&intrusion),
                "intrusion={intrusion} out of range persp={persp:?}"
            );
            assert!(
                (ARH_V2_PROMO_OFFSET..ARH_V2_SUPPORT_OFFSET).contains(&promo),
                "promo={promo} out of range persp={persp:?}"
            );
            assert!(
                (ARH_V2_SUPPORT_OFFSET..ARH_V2_ADV_PAWN_OFFSET).contains(&support),
                "support={support} out of range persp={persp:?}"
            );
            assert!(
                (ARH_V2_ADV_PAWN_OFFSET..ARH_V2_HAND_OFFSET).contains(&adv_pawn),
                "adv_pawn={adv_pawn} out of range persp={persp:?}"
            );
            assert!(
                (ARH_V2_HAND_OFFSET..ARH_V2_EK_OFFSET).contains(&hand),
                "hand={hand} out of range persp={persp:?}"
            );
            assert!(
                (ARH_V2_EK_OFFSET..ARH_V2_NEW_HEAD_USED).contains(&ek),
                "ek={ek} out of range persp={persp:?}"
            );
        }
    }

    /// `intrude_disk_indices` の adv_pawn (= out[3]) を bucket 値へ復号する helper。
    fn adv_pawn_bucket(pos: &Position, persp: Color) -> usize {
        let mut out = [0u32; ARH_V2_INTRUDE_MAX_ACTIVE];
        let n = intrude_disk_indices(pos, persp, 0, &mut out);
        assert_eq!(n, 6);
        let idx = out[3] as usize;
        assert!((ARH_V2_ADV_PAWN_OFFSET..ARH_V2_HAND_OFFSET).contains(&idx));
        (idx - ARH_V2_ADV_PAWN_OFFSET) / ARH_V2_NUM_PHASE
    }

    // -- version 12 軸4 必須テスト A: 進出歩のない局面では adv_pawn bucket = 0。
    // 玉のみの盤面 (= 歩なし) では当然 0。
    #[test]
    fn adv_pawn_zero_when_no_advanced_pawn() {
        let pos = set_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        assert_eq!(adv_pawn_bucket(&pos, Color::Black), 0);
        assert_eq!(adv_pawn_bucket(&pos, Color::White), 0);
        // 平手初期局面: 歩は rank7 (Black) / rank3 (White)、 いずれも敵陣隣接段に未到達 → 0。
        let start = startpos();
        assert_eq!(adv_pawn_bucket(&start, Color::Black), 0);
        assert_eq!(adv_pawn_bucket(&start, Color::White), 0);
    }

    // -- version 12 軸4 必須テスト B: 敵陣からの段数で bucket が段位別に決まる。
    // Black persp、 黒未成歩を 1 枚ずつ置いて bucket を確認 (= 浅3/中4/深5/垂れ歩2)。
    #[test]
    fn adv_pawn_bucket_by_rank() {
        // 垂れ歩 (= rank4 = Black 敵陣隣接段) → bucket 2。
        let p_adj = set_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1");
        assert_eq!(adv_pawn_bucket(&p_adj, Color::Black), 2);
        // 敵陣 浅段 (= rank3) → bucket 3。
        let p_shallow = set_sfen("4k4/9/4P4/9/9/9/9/9/4K4 b - 1");
        assert_eq!(adv_pawn_bucket(&p_shallow, Color::Black), 3);
        // 敵陣 中段 (= rank2) → bucket 4。
        let p_mid = set_sfen("4k4/4P4/9/9/9/9/9/9/4K4 b - 1");
        assert_eq!(adv_pawn_bucket(&p_mid, Color::Black), 4);
        // 敵陣 最深段 (= rank1) は玉と同段。 玉を別 file へ寄せて歩を置く。
        let p_deep = set_sfen("3kP4/9/9/9/9/9/9/9/4K4 b - 1");
        assert_eq!(adv_pawn_bucket(&p_deep, Color::Black), 5);
    }

    // -- version 12 軸4 必須テスト C: 複数歩では最深 1 枚が採用される (= count ではない)。
    #[test]
    fn adv_pawn_takes_deepest() {
        // 黒歩 rank4 (垂れ歩) + rank2 (敵陣中段) → 最深 = rank2 → bucket 4。
        let pos = set_sfen("4k4/4P4/9/4P4/9/9/9/9/4K4 b - 1");
        assert_eq!(adv_pawn_bucket(&pos, Color::Black), 4);
    }

    // -- version 12 軸4 必須テスト D: 先後対称 (= 鏡像局面で同 bucket)。
    // White の歩を rank6 (= White 敵陣隣接段) に置くと Black の rank4 と対称 → 同 bucket 2。
    #[test]
    fn adv_pawn_symmetric_black_white() {
        let black = set_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1");
        let white = set_sfen("4k4/9/9/9/9/4p4/9/9/4K4 w - 1");
        assert_eq!(
            adv_pawn_bucket(&black, Color::Black),
            adv_pawn_bucket(&white, Color::White),
            "鏡像局面で同 adv_pawn bucket (= 先後対称)"
        );
        // 敵陣最深段の対称: Black rank1 ↔ White rank9。
        let black_deep = set_sfen("3kP4/9/9/9/9/9/9/9/4K4 b - 1");
        let white_deep = set_sfen("4k4/9/9/9/9/9/9/9/3Kp4 w - 1");
        assert_eq!(
            adv_pawn_bucket(&black_deep, Color::Black),
            adv_pawn_bucket(&white_deep, Color::White),
        );
        assert_eq!(adv_pawn_bucket(&white_deep, Color::White), 5);
    }

    // -- version 12 軸4 必須テスト E: 成歩 (= ProPawn) は対象外 (= 未成歩のみ)。
    #[test]
    fn adv_pawn_ignores_promoted_pawn() {
        // 黒と金 (= ProPawn `+P`) を rank2 (敵陣中段) に置いても adv_pawn は 0 (= 未成歩なし)。
        let pos = set_sfen("4k4/4+P4/9/9/9/9/9/9/4K4 b - 1");
        assert_eq!(adv_pawn_bucket(&pos, Color::Black), 0);
    }

    /// `intrude_disk_indices` の hand (= out[4]) を bucket 値へ復号する helper。
    fn hand_bucket_of(pos: &Position, persp: Color) -> usize {
        let mut out = [0u32; ARH_V2_INTRUDE_MAX_ACTIVE];
        let n = intrude_disk_indices(pos, persp, 0, &mut out);
        assert_eq!(n, 6);
        let idx = out[4] as usize;
        assert!((ARH_V2_HAND_OFFSET..ARH_V2_EK_OFFSET).contains(&idx));
        (idx - ARH_V2_HAND_OFFSET) / ARH_V2_NUM_PHASE
    }

    // -- version 13 軸5 必須テスト A: 持駒のない局面では hand bucket = 0。
    #[test]
    fn hand_zero_when_no_hand_pieces() {
        let pos = set_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        assert_eq!(hand_bucket_of(&pos, Color::Black), 0);
        assert_eq!(hand_bucket_of(&pos, Color::White), 0);
        // 平手初期局面も持駒なし → 両 persp とも 0。
        let start = startpos();
        assert_eq!(hand_bucket_of(&start, Color::Black), 0);
        assert_eq!(hand_bucket_of(&start, Color::White), 0);
    }

    // -- version 13 軸5 必須テスト B: 持駒総枚数で bucket が枚数別に決まる (= 5 枚以上は clamp)。
    #[test]
    fn hand_bucket_by_count() {
        // Black 持駒: 歩 1 枚 → bucket 1。
        let p1 = set_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1");
        assert_eq!(hand_bucket_of(&p1, Color::Black), 1);
        // Black 持駒: 歩 3 枚 → bucket 3。
        let p3 = set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 3P 1");
        assert_eq!(hand_bucket_of(&p3, Color::Black), 3);
        // Black 持駒: 歩 4 枚 + 銀 1 枚 = 5 枚 → bucket 5 (= 5 枚以上 clamp)。
        let p5 = set_sfen("4k4/9/9/9/9/9/9/9/4K4 b S4P 1");
        assert_eq!(hand_bucket_of(&p5, Color::Black), 5);
        // Black 持駒: 歩 10 枚 = 10 枚 → bucket 5 へ clamp。
        let p10 = set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 10P 1");
        assert_eq!(hand_bucket_of(&p10, Color::Black), 5);
    }

    // -- version 13 軸5 必須テスト C: 持駒は persp 側 (= 自軍) のみ数える (= 先後で別管理)。
    #[test]
    fn hand_counts_only_own_side() {
        // Black が歩 2 枚、 White が歩 4 枚を持つ局面。
        let pos = set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2P4p 1");
        assert_eq!(hand_bucket_of(&pos, Color::Black), 2, "Black 自軍持駒 2 枚");
        assert_eq!(hand_bucket_of(&pos, Color::White), 4, "White 自軍持駒 4 枚");
    }

    // -- version 13 軸5 必須テスト D: phase が hand index に反映される。
    #[test]
    fn hand_index_uses_phase() {
        let pos = set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2P 1");
        let mut out = [0u32; ARH_V2_INTRUDE_MAX_ACTIVE];
        for bucket in 0..ARH_V2_NUM_BUCKETS {
            let phase = bucket_to_phase(bucket);
            let n = intrude_disk_indices(&pos, Color::Black, phase, &mut out);
            assert_eq!(n, 6);
            assert_eq!(out[4] as usize, hand_idx(2, phase), "bucket={bucket} phase={phase}");
        }
    }

    // -- version 13 軸5 必須テスト E: hand weight が反対称に効く (= feature 配線確認)。
    #[test]
    fn hand_weight_eval_basic() {
        // Black のみ持駒 (= 歩 2 枚) を持つ局面。 hand bucket = Black 2 / White 0。
        let pos = set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2P 1");
        let phase = bucket_to_phase(0);
        let mut disk = ArhV2Weights::zeroed();
        // flat[hand_idx(2, phase)] へ重みを書く (= flat[k] は disk 順で w_base へ写像)。
        set_flat(&mut disk, hand_idx(2, phase), 30);
        let rt = build_runtime_weights(&disk);
        let sb = ARH_V2_OUTPUT_SCALE.load(Ordering::Relaxed);
        set_output_scale(1);
        let v_black = evaluate_arh_v2(&rt, &pos, 0, Color::Black);
        let v_white = evaluate_arh_v2(&rt, &pos, 0, Color::White);
        // Black hand=2 (= 該当 index)、 White hand=0 (= 別 index) → us=30, them=0 → eval=30。
        assert_eq!(v_black, 30, "Black hand weight が効く");
        assert_eq!(v_white, -30, "反対称");
        set_output_scale(sb);
    }

    /// `intrude_disk_indices` の ek (= out[5]) を bucket 値へ復号する helper。
    fn ek_bucket_of(pos: &Position, persp: Color) -> usize {
        let mut out = [0u32; ARH_V2_INTRUDE_MAX_ACTIVE];
        let n = intrude_disk_indices(pos, persp, 0, &mut out);
        assert_eq!(n, 6);
        let idx = out[5] as usize;
        assert!((ARH_V2_EK_OFFSET..ARH_V2_NEW_HEAD_USED).contains(&idx));
        (idx - ARH_V2_EK_OFFSET) / ARH_V2_NUM_PHASE
    }

    // -- version 14 軸6 必須テスト A: 空盤面 (= 玉のみ) は自玉非入玉 → 点数 0 → 残り 27 → bucket 5。
    #[test]
    fn ek_empty_board_is_max_remaining() {
        let pos = set_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        assert_eq!(ek_bucket_of(&pos, Color::Black), 5);
        assert_eq!(ek_bucket_of(&pos, Color::White), 5);
        // 平手初期局面も自玉非入玉 → 両 persp とも bucket 5。
        let start = startpos();
        assert_eq!(ek_bucket_of(&start, Color::Black), 5);
        assert_eq!(ek_bucket_of(&start, Color::White), 5);
    }

    // -- version 14 軸6 必須テスト B: 自玉が敵陣に居ても点数不足なら残りが大きい。
    // 黒玉 rank1 (= 敵陣) のみ。 非玉駒なし・持駒なし → 点数 0 → 残り 27 → bucket 5。
    #[test]
    fn ek_king_in_camp_but_no_points() {
        // 黒玉 1一 (a1、 rank1 = Black 敵陣)、 白玉 5九。
        let pos = set_sfen("K3k4/9/9/9/9/9/9/9/9 b - 1");
        assert_eq!(ek_bucket_of(&pos, Color::Black), 5);
    }

    // -- version 14 軸6 必須テスト C: 自玉が敵陣外なら盤上点数があっても 0 点扱い (= 入玉前提)。
    // 黒玉 rank9 (敵陣外) + 黒飛 rank1 (敵陣、 5 点) でも入玉前提を満たさず残り 27 → bucket 5。
    #[test]
    fn ek_zero_when_king_not_in_camp() {
        // 黒飛 1一 (a1、 rank1)、 白玉 5五、 黒玉 5九 (rank9 = 敵陣外)。
        let pos = set_sfen("R8/9/9/9/4k4/9/9/9/4K4 b - 1");
        assert_eq!(ek_bucket_of(&pos, Color::Black), 5, "自玉敵陣外は点数 0");
    }

    // -- version 14 軸6 必須テスト D: 残り点数で bucket が段階的に決まる。
    // 黒玉 rank1 (敵陣) + 敵陣の自駒 / 持駒で点数を加算し残りを変化させる。
    #[test]
    fn ek_bucket_by_remaining() {
        // 黒玉 1一 + 黒飛 2一 (= 5 点) のみ。 点数 5 → 残り 22 → bucket 5 (= 21..=27)。
        let p5 = set_sfen("KR7/9/9/9/9/9/9/9/9 b - 1");
        assert_eq!(ek_bucket_of(&p5, Color::Black), 5);
        // 黒玉 1一 + 黒飛 2一 + 黒角 3一 (= 5+5=10 点) → 残り 17 → bucket 4 (= 16..=20)。
        let p10 = set_sfen("KRB6/9/9/9/9/9/9/9/9 b - 1");
        assert_eq!(ek_bucket_of(&p10, Color::Black), 4);
        // 黒玉 1一 + 飛角竜馬 (= 5×4=20 点) → 残り 7 → bucket 2 (= 6..=10)。
        let p20 = set_sfen("KRB+R+B4/9/9/9/9/9/9/9/9 b - 1");
        assert_eq!(ek_bucket_of(&p20, Color::Black), 2);
    }

    // -- version 14 軸6 必須テスト E: 入玉宣言点 達成 (= 残り 0) で bucket 0。
    // 黒玉 1一 (敵陣) + 飛 +飛 角 +角 (= 大駒 4 枚 = 20 点) + 金 4 + 銀 3 (= 7 点) = 27 点 → 残り 0 → bucket 0。
    // (= 将棋の大駒は飛車 2 / 角 2、 金 4 が上限のため board に 20 点 + 小駒で 27 へ到達させる。)
    #[test]
    fn ek_achieved_is_bucket_zero() {
        // rank1: K R +R B +B G G G G (= 金 4)、 rank2: S S S (= 銀 3)。 全て Black 敵陣 (rank1-3) 内。
        let pos = set_sfen("KR+RB+BGGGG/SSS6/9/9/9/9/9/9/4k4 b - 1");
        assert_eq!(ek_bucket_of(&pos, Color::Black), 0, "27 点以上で達成 bucket 0");
    }

    // -- version 14 軸6 必須テスト F: 持駒も点数に算入される (= 自玉が敵陣に居る前提下)。
    #[test]
    fn ek_hand_points_count_when_in_camp() {
        // 黒玉 1一 (敵陣) のみ + 持駒 飛 5 枚相当はないため、 持駒 飛角 (= 10) と
        // 小駒 4 枚 (= 4) を持たせ点数 14 → 残り 13 → bucket 3 (= 11..=15)。
        let pos = set_sfen("K8/9/9/9/9/9/9/9/9 b RB2S2N 1");
        // 持駒点数 = 飛 5 + 角 5 + 銀 1×2 + 桂 1×2 = 14。 残り 27-14=13 → bucket 3。
        assert_eq!(ek_bucket_of(&pos, Color::Black), 3);
    }

    // -- version 14 軸6 必須テスト G: ek weight が反対称に効く (= feature 配線確認)。
    #[test]
    fn ek_weight_eval_basic() {
        // Black 自玉 1一 + 飛角竜馬 (= 20 点) → 残り 7 → bucket 2。 White は自玉非入玉 → bucket 5。
        let pos = set_sfen("KRB+R+B4/9/9/9/9/9/9/9/4k4 b - 1");
        assert_eq!(ek_bucket_of(&pos, Color::Black), 2);
        assert_eq!(ek_bucket_of(&pos, Color::White), 5);
        let phase = bucket_to_phase(0);
        let mut disk = ArhV2Weights::zeroed();
        // flat[ek_idx(2, phase)] へ重みを書く (= Black 該当 / White 非該当)。
        set_flat(&mut disk, ek_idx(2, phase), 30);
        let rt = build_runtime_weights(&disk);
        let sb = ARH_V2_OUTPUT_SCALE.load(Ordering::Relaxed);
        set_output_scale(1);
        let v_black = evaluate_arh_v2(&rt, &pos, 0, Color::Black);
        let v_white = evaluate_arh_v2(&rt, &pos, 0, Color::White);
        // Black ek=2 (= 該当 index)、 White ek=5 (= 別 index) → us=30, them=0 → eval=30。
        assert_eq!(v_black, 30, "Black ek weight が効く");
        assert_eq!(v_white, -30, "反対称");
        set_output_scale(sb);
    }

    // -- version 14 軸6 必須テスト H: phase が ek index に反映される。
    #[test]
    fn ek_index_uses_phase() {
        // Black 自玉 1一 + 飛 2一 (= 5 点) → 残り 22 → bucket 5。
        let pos = set_sfen("KR7/9/9/9/9/9/9/9/9 b - 1");
        let mut out = [0u32; ARH_V2_INTRUDE_MAX_ACTIVE];
        for bucket in 0..ARH_V2_NUM_BUCKETS {
            let phase = bucket_to_phase(bucket);
            let n = intrude_disk_indices(&pos, Color::Black, phase, &mut out);
            assert_eq!(n, 6);
            assert_eq!(out[5] as usize, ek_idx(5, phase), "bucket={bucket} phase={phase}");
        }
    }

    // -- version 14 軸6 必須テスト I: 全 weight ゼロ (= feature 実質 OFF) では ek が eval に影響しない。
    // zero_weights_eval_is_zero と同趣旨だが ek 配線後も保たれることを明示確認。
    #[test]
    fn ek_off_when_weights_zero() {
        let disk = ArhV2Weights::zeroed();
        let rt = build_runtime_weights(&disk);
        // 入玉局面でも全 weight ゼロなら eval 0。
        let pos = set_sfen("KRB+R+B4/9/9/9/9/9/9/9/4k4 b - 1");
        for bucket in 0..ARH_V2_NUM_BUCKETS {
            assert_eq!(evaluate_arh_v2(&rt, &pos, bucket, Color::Black), 0);
            assert_eq!(evaluate_arh_v2(&rt, &pos, bucket, Color::White), 0);
        }
    }
}
