//! ODH (Optimism Damp Head) v0 = **eval 絶対値の calibration / damp head** (2026-07-02)。
//!
//! main NNUE (= 他 head / psqt を全部足した合算 raw eval = `main_raw`) が大きく
//! **優勢方向 (= main_raw > 0)** に振れているが、 局面条件的にその優勢を信用しにくい
//! (= 攻めが無いのに高評価 / 自玉が危険なのに高評価) 時に、 eval の絶対値を
//! **縮める (= 楽観を減衰させる)** 補正 head。
//!
//! KZT / KDGR 等の「加点 head」とは別物で、 **符号反転は狙わず** 正側の eval を
//! `main_raw - odh` に縮めるだけ (= 負側 = 較正済とみなし触らない、 互角付近も不動)。
//!
//! # 減衰式
//!
//! ```text
//! abs_main = abs(main_raw)                       // main_raw = ODH 適用前の合算 raw eval (fv_scale 適用前)
//! if main_raw <= 0 || abs_main < threshold: odh = 0   // 正側のみ、 互角付近も不動
//! else: odh = damp[class] * (min(abs_main, cap) - threshold) / 256
//! final_raw = main_raw - odh                     // 優勢評価を縮める (符号反転しない)
//! ```
//!
//! - `threshold` = 100 cp、 `cap` = 800 cp (raw 単位 = cp × fv_scale で扱う。 cp 基準で
//!   設定 → raw 換算)。 いずれも USI option (`ODH_THRESHOLD` / `ODH_CAP`) で cp 単位設定。
//! - `ODH_OUTPUT_SCALE` (= 既定 0 = 無効 = bit-exact) で damp 全体を runtime scale。
//!   `ODH_SCALE_REF` (= 64) を基準にした線形倍率 (= scale=64 で damp table の設計値)。
//!
//! # class 軸 (= 解析注入版 v0)
//!
//! 「優勢側」 (= main_raw > 0 のとき stm) の視点で以下 3 軸を離散化する。
//!
//! - **force_class[4]** (= 攻めの強制力、 主軸): `att_in_ek5` (= 敵玉 5x5 内の自駒数) と
//!   `own_hand_ty` (= 自手駒の駒種数) の合成を 4 class 化。 高 force (= 実攻めあり) →
//!   damp 小、 低 force (= 攻め無し) → damp 大。 解析: 低 force 高 eval = 76% 敗、
//!   高 force = 39% 敗 (= sep 37pt)。
//! - **own_king_risk[4]**: 自玉中心 3x3 の敵駒重み付き数 (= king_danger と同定義) を
//!   4 class 化。 高危険 → damp 増 (= 独立第 2 軸)。
//! - **bucket[8]**: 既存 progress8kpabs bucket (= 呼び出し側から受け取る)。
//!
//! class index = `(bucket * 4 + own_king_risk) * 4 + force_class` (= 0..128)。
//!
//! # damp table (= 解析注入 hand-set、 学習前 A/B 用)
//!
//! [`BASE_DAMP`] は `[own_king_risk 4][force_class 4]` の 16 値 (= bucket 非依存 v0)。
//! [`DAMP_TABLE`] はこれを 8 bucket に複製した 128 値。 stage2 (= 学習版) は
//! この 128 class table を backprop で学習する想定 (= 効果確認後ユーザ判断)。
//!
//! 目安 magnitude (= mix15 が敗局 eval を 627→464cp ≒ −26% 下げて改善): 低 force・
//! 高 risk cell で実効 damp 上限 ~150-250cp (= `ODH_OUTPUT_SCALE=64` 基準)、 高 force
//! cell は ~0。

use crate::bitboard::{Bitboard, king_5x5_mask, king_effect};
use crate::position::Position;
use crate::types::{Color, PieceType};
use std::sync::atomic::{AtomicI32, Ordering};

/// force_class 数 (= 攻めの強制力軸)。
pub const ODH_NUM_FORCE: usize = 4;
/// own_king_risk class 数 (= 自玉危険軸)。
pub const ODH_NUM_OKR: usize = 4;
/// bucket 数 (= progress8kpabs、 既存 head と揃える)。
pub const ODH_NUM_BUCKET: usize = 8;
/// damp table 総エントリ数 (= 128)。
pub const ODH_CLASS_COUNT: usize = ODH_NUM_BUCKET * ODH_NUM_OKR * ODH_NUM_FORCE;

/// 減衰式の除数 (= `damp * excess / 256`)。
pub const ODH_DAMP_DIVISOR: i64 = 256;
/// `ODH_OUTPUT_SCALE` の基準値 (= scale=64 で damp table 設計値、 線形倍率)。
pub const ODH_SCALE_REF: i64 = 64;

/// `ODH_OUTPUT_SCALE` 初期値 (= 0 = 無効 = bit-exact)。
pub const ODH_OUTPUT_SCALE_DEFAULT: i32 = 0;
/// `ODH_THRESHOLD` 初期値 (= 100 cp、 互角付近は不動)。
pub const ODH_THRESHOLD_DEFAULT: i32 = 100;
/// `ODH_CAP` 初期値 (= 800 cp、 これ以上の優勢は excess 頭打ち)。
pub const ODH_CAP_DEFAULT: i32 = 800;

/// USI option `ODH_OUTPUT_SCALE` で runtime 上書き可能な damp 全体倍率。
pub static ODH_OUTPUT_SCALE: AtomicI32 = AtomicI32::new(ODH_OUTPUT_SCALE_DEFAULT);
/// USI option `ODH_THRESHOLD` (= cp 単位) で runtime 上書き可能な減衰開始閾値。
pub static ODH_THRESHOLD: AtomicI32 = AtomicI32::new(ODH_THRESHOLD_DEFAULT);
/// USI option `ODH_CAP` (= cp 単位) で runtime 上書き可能な excess 上限。
pub static ODH_CAP: AtomicI32 = AtomicI32::new(ODH_CAP_DEFAULT);

/// USI option 経由で damp 全体倍率を更新する。
pub fn set_output_scale(scale: i32) {
    ODH_OUTPUT_SCALE.store(scale, Ordering::Relaxed);
}

/// USI option 経由で減衰開始閾値 (= cp) を更新する。
pub fn set_threshold(threshold_cp: i32) {
    ODH_THRESHOLD.store(threshold_cp, Ordering::Relaxed);
}

/// USI option 経由で excess 上限 (= cp) を更新する。
pub fn set_cap(cap_cp: i32) {
    ODH_CAP.store(cap_cp, Ordering::Relaxed);
}

/// 解析注入 damp base table `[own_king_risk 4][force_class 4]` (= bucket 非依存 v0)。
///
/// - 行 = own_king_risk (= 0 安全 .. 3 危険)、 列 = force_class (= 0 攻め無し .. 3 実攻め)。
/// - force 主軸: 列が進む (= 攻めあり) ほど damp 減、 force 3 (= 実攻め) は ~0。
/// - okr 第 2 軸: 行が進む (= 自玉危険) ほど damp 増。
///
/// 値は `ODH_OUTPUT_SCALE=64` 基準の fixed-point (= `damp * excess / 256` で raw 減算量)。
/// 高 eval (~627cp) の最悪 cell (= force0/okr3=90) で ~185cp 減 (≒ −29%) を狙う較正。
pub const BASE_DAMP: [[i32; ODH_NUM_FORCE]; ODH_NUM_OKR] = [
    // force: 0    1    2    3
    [30, 18, 8, 0],   // okr 0 (= 自玉安全)
    [50, 32, 16, 0],  // okr 1
    [70, 48, 26, 6],  // okr 2
    [90, 64, 38, 14], // okr 3 (= 自玉危険)
];

/// [`BASE_DAMP`] を 8 bucket に複製した 128 class damp table。
///
/// class index = [`class_index`] `(bucket, okr, force)`。 v0 は bucket 非依存
/// (= 全 bucket 同値)。 stage2 学習版はこの table を per-class に学習する想定。
pub const DAMP_TABLE: [i32; ODH_CLASS_COUNT] = build_damp_table();

/// [`BASE_DAMP`] から 128 class の [`DAMP_TABLE`] を生成する const fn。
const fn build_damp_table() -> [i32; ODH_CLASS_COUNT] {
    let mut table = [0i32; ODH_CLASS_COUNT];
    let mut bucket = 0;
    while bucket < ODH_NUM_BUCKET {
        let mut okr = 0;
        while okr < ODH_NUM_OKR {
            let mut force = 0;
            while force < ODH_NUM_FORCE {
                let idx = (bucket * ODH_NUM_OKR + okr) * ODH_NUM_FORCE + force;
                table[idx] = BASE_DAMP[okr][force];
                force += 1;
            }
            okr += 1;
        }
        bucket += 1;
    }
    table
}

/// class index (= `(bucket * 4 + own_king_risk) * 4 + force_class`)。
#[inline]
pub const fn class_index(bucket: usize, okr: usize, force: usize) -> usize {
    (bucket * ODH_NUM_OKR + okr) * ODH_NUM_FORCE + force
}

/// 敵 **重駒** mask (= 飛角金銀竜馬 + と/成香/成桂/成銀)。
/// [`crate::nnue::king_danger::enemy_heavy_mask`] と同一集合 (= odh を feature 独立に
/// するため内製)。
#[inline]
fn enemy_heavy_mask(pos: &Position, enemy: Color) -> Bitboard {
    pos.pieces(enemy, PieceType::Rook)
        | pos.pieces(enemy, PieceType::Bishop)
        | pos.pieces(enemy, PieceType::Gold)
        | pos.pieces(enemy, PieceType::Silver)
        | pos.pieces(enemy, PieceType::Dragon)
        | pos.pieces(enemy, PieceType::Horse)
        | pos.pieces(enemy, PieceType::ProPawn)
        | pos.pieces(enemy, PieceType::ProLance)
        | pos.pieces(enemy, PieceType::ProKnight)
        | pos.pieces(enemy, PieceType::ProSilver)
}

/// 敵 **軽駒** mask (= 歩香桂)。
#[inline]
fn enemy_light_mask(pos: &Position, enemy: Color) -> Bitboard {
    pos.pieces(enemy, PieceType::Pawn)
        | pos.pieces(enemy, PieceType::Lance)
        | pos.pieces(enemy, PieceType::Knight)
}

/// 手駒の駒種数 (= 0..=7)。 打ちの多様性 proxy。
#[inline]
fn hand_type_count(pos: &Position, c: Color) -> i32 {
    const HAND_TYPES: [PieceType; 7] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
    ];
    let h = pos.hand(c);
    HAND_TYPES.iter().filter(|&&pt| h.count(pt) > 0).count() as i32
}

/// perspective 側の自玉中心 3x3 (= 玉自身含む) の敵駒重み付き合計 (= 自玉危険度)。
/// [`crate::nnue::king_danger::king_danger_weighted_sum`] と同定義 (= 重×3 + 軽×1)。
#[inline]
pub fn own_king_danger_sum(pos: &Position, persp: Color) -> i32 {
    let king_sq = pos.king_square(persp);
    let enemy = persp.opponent();
    let zone = king_effect(king_sq) | Bitboard::from_square(king_sq);
    let heavy = (enemy_heavy_mask(pos, enemy) & zone).count() as i32;
    let light = (enemy_light_mask(pos, enemy) & zone).count() as i32;
    heavy * 3 + light
}

/// 攻めの強制力 raw 値 (= `att_in_ek5 * 2 + own_hand_ty`)。
///
/// - `att_in_ek5` = 敵玉 5x5 内の自軍駒数 (= 直接的な攻め駒の量)。
/// - `own_hand_ty` = 自手駒の駒種数 (= 打ち込みの脅威、 weight 小)。
#[inline]
pub fn force_raw(pos: &Position, persp: Color) -> i32 {
    let enemy = persp.opponent();
    let enemy_king = pos.king_square(enemy);
    let att_in_ek5 = (pos.pieces_c(persp) & king_5x5_mask(enemy_king)).count() as i32;
    let own_hand_ty = hand_type_count(pos, persp);
    att_in_ek5 * 2 + own_hand_ty
}

/// 攻めの強制力 class (= 0 攻め無し .. 3 実攻め)。
///
/// [`force_raw`] を 4 分位に離散化 (= 解析注入の手設定境界)。
#[inline]
pub fn compute_force_class(pos: &Position, persp: Color) -> usize {
    match force_raw(pos, persp) {
        0 => 0,
        1..=2 => 1,
        3..=5 => 2,
        _ => 3,
    }
}

/// 自玉危険 class (= 0 安全 .. 3 危険)。
///
/// [`own_king_danger_sum`] を 4 分位に離散化 (= 解析注入の手設定境界)。
#[inline]
pub fn compute_own_king_risk_class(pos: &Position, persp: Color) -> usize {
    match own_king_danger_sum(pos, persp) {
        0 => 0,
        1..=3 => 1,
        4..=6 => 2,
        _ => 3,
    }
}

/// ODH 減衰量 (= `main_raw` から **引く** raw 単位の量、 常に `0..=main_raw`) を返す。
///
/// `ODH_OUTPUT_SCALE == 0` (= default) で即 0 → eval bit-exact (= 無効時無害)。
///
/// # Arguments
/// * `main_raw`  - ODH 適用前の合算 raw eval (= fv_scale 適用前、 stm 視点)
/// * `pos`       - 局面
/// * `stm`       - 手番 (= side to move、 main_raw > 0 のとき優勢側)
/// * `bucket`    - progress8kpabs bucket (= 0..=7)
/// * `fv_scale`  - 現在の fv_scale (= cp → raw 換算用)
///
/// # 仕様
/// - `main_raw <= 0` (= 負側 or 互角) → 0 (= 較正済とみなし触らない)。
/// - `abs(main_raw) < threshold` (= 互角付近) → 0。
/// - それ以外: `damp[class] * (min(abs_main, cap) - threshold) / 256 * scale / 64`。
/// - 返り値は `[0, main_raw]` に clamp (= 符号反転しない)。
#[inline]
pub fn apply_odh(main_raw: i32, pos: &Position, stm: Color, bucket: usize, fv_scale: i32) -> i32 {
    let scale = ODH_OUTPUT_SCALE.load(Ordering::Relaxed);
    // scale 0 → 減衰なし = bit-exact (= class 計算も skip して NPS も温存)。
    if scale == 0 {
        return 0;
    }
    // 負側 / 互角は不動 (= 正側の優勢のみ縮める)。
    if main_raw <= 0 {
        return 0;
    }
    let fv = fv_scale.max(1);
    let threshold_raw = (ODH_THRESHOLD.load(Ordering::Relaxed)).saturating_mul(fv);
    if main_raw < threshold_raw {
        return 0;
    }
    let cap_raw = (ODH_CAP.load(Ordering::Relaxed)).saturating_mul(fv);
    // excess = min(abs_main, cap) - threshold (>= 0)。 cap < threshold の異常設定は 0 clamp。
    let capped = main_raw.min(cap_raw);
    let excess = (capped - threshold_raw).max(0) as i64;

    let force = compute_force_class(pos, stm);
    let okr = compute_own_king_risk_class(pos, stm);
    let damp = DAMP_TABLE[class_index(bucket, okr, force)] as i64;

    // odh = damp * excess / 256 * scale / 64 (= i64 中間で overflow 回避)。
    let odh = damp * excess * (scale as i64) / (ODH_DAMP_DIVISOR * ODH_SCALE_REF);
    // 符号反転しない (= final_raw >= 0 を保証)。 damp は非負なので下限は 0。
    odh.clamp(0, main_raw as i64) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Position, SFEN_HIRATE};
    use std::sync::Mutex;

    // ODH の runtime option は **global static AtomicI32** のため、 これを書き換える
    // test を並列実行すると値が race する。 cargo test はデフォルトで並列なので、
    // option を触る test は本 lock で直列化する (= 他 test の set が読み値を汚さない)。
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// TEST_LOCK を取得 (= poison しても中身を取り出して継続)。
    fn lock_options() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set_pos(sfen: &str) -> Position {
        let mut pos = Position::new();
        pos.set_sfen(sfen).unwrap();
        pos
    }

    fn make_pos() -> Position {
        set_pos(SFEN_HIRATE)
    }

    /// option を default に戻すヘルパ (= 他 test への副作用防止)。
    fn reset_options() {
        set_output_scale(ODH_OUTPUT_SCALE_DEFAULT);
        set_threshold(ODH_THRESHOLD_DEFAULT);
        set_cap(ODH_CAP_DEFAULT);
    }

    /// apply_odh の整数式を test 側で再現 (= 期待値計算、 floor 切り捨て一致)。
    fn expected_odh(damp: i64, excess_raw: i64, scale: i64) -> i32 {
        (damp * excess_raw * scale / (ODH_DAMP_DIVISOR * ODH_SCALE_REF)) as i32
    }

    #[test]
    fn test_class_index_range() {
        assert_eq!(class_index(0, 0, 0), 0);
        assert_eq!(class_index(7, 3, 3), ODH_CLASS_COUNT - 1);
        assert_eq!(ODH_CLASS_COUNT, 128);
    }

    #[test]
    fn test_damp_table_matches_base() {
        // 全 bucket で BASE_DAMP と一致 (= bucket 非依存 v0)。
        for bucket in 0..ODH_NUM_BUCKET {
            for okr in 0..ODH_NUM_OKR {
                for force in 0..ODH_NUM_FORCE {
                    assert_eq!(DAMP_TABLE[class_index(bucket, okr, force)], BASE_DAMP[okr][force]);
                }
            }
        }
        // force 単調減少 / okr 単調増加 (= 設計方向) を確認。
        for okr in 0..ODH_NUM_OKR {
            for force in 0..ODH_NUM_FORCE - 1 {
                assert!(BASE_DAMP[okr][force] >= BASE_DAMP[okr][force + 1]);
            }
        }
        for force in 0..ODH_NUM_FORCE {
            for okr in 0..ODH_NUM_OKR - 1 {
                assert!(BASE_DAMP[okr][force] <= BASE_DAMP[okr + 1][force]);
            }
        }
    }

    #[test]
    fn test_scale_zero_is_bit_exact() {
        let _g = lock_options();
        reset_options();
        // scale=0 (= default) で常に 0 (= どんな main_raw / 局面でも)。
        let pos = make_pos();
        for main_raw in [-5000, -100, 0, 100, 5000, 100000] {
            assert_eq!(apply_odh(main_raw, &pos, Color::Black, 0, 14), 0);
        }
        reset_options();
    }

    #[test]
    fn test_negative_and_equal_untouched() {
        let _g = lock_options();
        reset_options();
        set_output_scale(64);
        let pos = make_pos();
        // 負側 (= main_raw <= 0) は不動。
        for main_raw in [-100000, -1400, -1, 0] {
            assert_eq!(apply_odh(main_raw, &pos, Color::Black, 0, 14), 0);
        }
        reset_options();
    }

    #[test]
    fn test_below_threshold_untouched() {
        let _g = lock_options();
        reset_options();
        set_output_scale(64);
        let pos = make_pos();
        let fv = 14;
        // threshold = 100cp → raw 1400 未満は不動。
        assert_eq!(apply_odh(1399, &pos, Color::Black, 0, fv), 0);
        // 閾値ちょうどでは excess=0 → 0。
        assert_eq!(apply_odh(1400, &pos, Color::Black, 0, fv), 0);
        reset_options();
    }

    #[test]
    fn test_low_force_damps_high_force_not() {
        let _g = lock_options();
        reset_options();
        set_output_scale(64);
        let fv = 14;
        // 高 eval を仮定 (= 627cp raw = 8778)。
        let main_raw = 627 * fv;

        // 低 force (= 攻め無し): 黒が優勢だが敵玉 5x5 に黒駒無し + 手駒無し + 自玉安全。
        // 黒玉 5九、 白玉 5一、 盤上駒最小 → force0/okr0。
        let low = set_pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        assert_eq!(compute_force_class(&low, Color::Black), 0);
        assert_eq!(compute_own_king_risk_class(&low, Color::Black), 0);
        let odh_low = apply_odh(main_raw, &low, Color::Black, 0, fv);
        assert!(odh_low > 0, "low force high eval should be damped, got {odh_low}");

        // 高 force (= 実攻め): 敵玉 5x5 に黒駒多数 + 黒に潤沢な手駒。
        // 白玉 5一 周辺 (= 5一中心 5x5 = 段1-3、 筋3-7) に黒の金銀桂を配置。
        let high = set_pos("4k4/3GSG3/3S1N3/9/9/9/9/9/4K4 b RB2P 1");
        assert_eq!(compute_force_class(&high, Color::Black), 3);
        let odh_high = apply_odh(main_raw, &high, Color::Black, 0, fv);
        assert_eq!(odh_high, 0, "real attack (force3/okr0) damp is 0, got {odh_high}");

        // 低 force の減衰量は force0/okr0 damp=30 の設計値と一致するはず。
        // excess = 627*14 - 100*14 = 527*14 = 7378。 odh(scale64) = 30 * 7378 / 256 = 864。
        assert_eq!(odh_low, expected_odh(30, (527 * fv) as i64, 64));
        reset_options();
    }

    #[test]
    fn test_own_king_risk_increases_damp() {
        let _g = lock_options();
        reset_options();
        set_output_scale(64);
        let fv = 14;
        let main_raw = 627 * fv;

        // 自玉危険 (= 黒玉 5九 周辺 3x3 に白重駒複数) かつ 低 force。
        // 6八 白飛 (重3)、 5八 白角 (重3) → okr sum=6 → okr class 2。 敵玉 5x5 に黒駒無し。
        let risky = set_pos("4k4/9/9/9/9/9/9/3rb4/4K4 b - 1");
        let okr = compute_own_king_risk_class(&risky, Color::Black);
        assert_eq!(okr, 2, "two heavy pieces in 3x3 (sum6) => okr class 2");
        let force = compute_force_class(&risky, Color::Black);
        // damp = BASE_DAMP[okr][force]。 okr>0 なので force0 damp より大。
        let odh = apply_odh(main_raw, &risky, Color::Black, 0, fv);
        assert_eq!(odh, expected_odh(BASE_DAMP[okr][force] as i64, (527 * fv) as i64, 64));
        assert!(
            odh > expected_odh(30, (527 * fv) as i64, 64),
            "risky king should damp more than safe"
        );
        reset_options();
    }

    #[test]
    fn test_cap_limits_excess() {
        let _g = lock_options();
        reset_options();
        set_output_scale(64);
        let fv = 14;
        // 低 force 局面。
        let low = set_pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        // main = 5000cp は cap=800cp で頭打ち → excess = (800-100)*14 = 700*14。
        let odh_huge = apply_odh(5000 * fv, &low, Color::Black, 0, fv);
        let odh_at_cap = apply_odh(800 * fv, &low, Color::Black, 0, fv);
        assert_eq!(odh_huge, odh_at_cap, "excess should saturate at cap");
        assert_eq!(odh_huge, expected_odh(30, (700 * fv) as i64, 64));
        reset_options();
    }

    #[test]
    fn test_scale_linear() {
        let _g = lock_options();
        reset_options();
        let fv = 14;
        let low = set_pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        let main_raw = 627 * fv;
        let excess = (527 * fv) as i64;
        set_output_scale(64);
        let v64 = apply_odh(main_raw, &low, Color::Black, 0, fv);
        set_output_scale(128);
        let v128 = apply_odh(main_raw, &low, Color::Black, 0, fv);
        set_output_scale(32);
        let v32 = apply_odh(main_raw, &low, Color::Black, 0, fv);
        // scale は基準 64 に対する線形倍率 (= floor 切り捨て込みで exact 一致を確認)。
        assert_eq!(v64, expected_odh(30, excess, 64));
        assert_eq!(v128, expected_odh(30, excess, 128));
        assert_eq!(v32, expected_odh(30, excess, 32));
        // 単調増加 (= scale 大きいほど減衰大)。
        assert!(v32 < v64 && v64 < v128);
        reset_options();
    }

    #[test]
    fn test_never_flips_sign() {
        let _g = lock_options();
        reset_options();
        // 極端な damp (= scale 最大) でも odh <= main_raw (= 符号反転しない)。
        set_output_scale(256);
        set_cap(30000);
        let low = set_pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        let main_raw = 200 * 14;
        let odh = apply_odh(main_raw, &low, Color::Black, 0, 14);
        assert!(odh <= main_raw, "odh {odh} must not exceed main_raw {main_raw}");
        assert!(odh >= 0);
        reset_options();
    }

    #[test]
    fn test_startpos_symmetry_and_force() {
        let _g = lock_options();
        reset_options();
        // 開幕は敵玉 5x5 に自駒無し + 手駒無し → force0、 自玉安全 → okr0。
        let pos = make_pos();
        assert_eq!(compute_force_class(&pos, Color::Black), 0);
        assert_eq!(compute_force_class(&pos, Color::White), 0);
        assert_eq!(compute_own_king_risk_class(&pos, Color::Black), 0);
        assert_eq!(compute_own_king_risk_class(&pos, Color::White), 0);
        reset_options();
    }
}
