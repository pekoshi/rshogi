//! MovePicker（指し手オーダリング）
//!
//! 探索中に指し手を効率的に順序付けして返すコンポーネント。
//! Alpha-Beta探索の効率を最大化するため、カットオフを起こしやすい手を先に返す。
//!
//! ## Lazy Generation
//!
//! MovePickerは指し手を段階的に生成する（lazy generation）。
//! LMP等の枝刈り条件が成立したら、`skip_quiets()`を呼び出すことで
//! 残りのquiet手の生成をスキップできる。
//!
//! ## History参照を保持しない設計
//!
//! 再帰呼び出し時の参照エイリアス問題を避けるため、MovePickerはHistory参照を
//! フィールドとして保持しない。代わりに、`next_move()`メソッドでHistoryTables
//! への参照を受け取る。
//!
//! ## Stage
//!
//! 指し手生成は複数の段階（Stage）に分けて行われる：
//!
//! ### 通常探索（王手なし）
//! 1. MainTT - 置換表の指し手
//! 2. CaptureInit - 捕獲手の生成
//! 3. GoodCapture - 良い捕獲手（SEE >= threshold）
//! 4. QuietInit - 静かな手の生成
//! 5. GoodQuiet - 良い静かな手
//! 6. BadCapture - 悪い捕獲手
//! 7. BadQuiet - 悪い静かな手
//!
//! ### 王手回避
//! 1. EvasionTT - 置換表の指し手
//! 2. EvasionInit - 回避手の生成
//! 3. Evasion - 回避手
//!
//! ### 静止探索
//! 1. QSearchTT - 置換表の指し手
//! 2. QCaptureInit - 捕獲手の生成
//! 3. QCapture - 捕獲手

use super::{HistoryTables, LOW_PLY_HISTORY_SIZE, PieceToHistory};
use crate::bitboard::{Bitboard, bishop_effect, king_5x5_mask, rook_effect};
use crate::movegen::{ExtMove, ExtMoveBuffer};
use crate::position::Position;
use crate::types::{Color, DEPTH_QS, Depth, Move, Piece, PieceType, Square, Value};

// =============================================================================
// Stage（指し手生成の段階）
// =============================================================================

/// 指し手生成の段階
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Stage {
    // 通常探索（王手なし）
    /// 置換表の指し手
    MainTT,
    /// ProbCut: 置換表の指し手
    ProbCutTT,
    /// 捕獲手の生成
    CaptureInit,
    /// ProbCut: 捕獲手生成
    ProbCutInit,
    /// ProbCut: SEEしきい値付き捕獲
    ProbCut,
    /// 良い捕獲手（SEE >= threshold）
    GoodCapture,
    /// 静かな手の生成
    QuietInit,
    /// 良い静かな手
    GoodQuiet,
    /// 悪い捕獲手
    BadCapture,
    /// 悪い静かな手
    BadQuiet,

    // 王手回避
    /// 置換表の指し手（回避）
    EvasionTT,
    /// 回避手の生成
    EvasionInit,
    /// 回避手
    Evasion,

    // 静止探索
    /// 置換表の指し手（静止探索）
    QSearchTT,
    /// 静止探索用捕獲手の生成
    QCaptureInit,
    /// 静止探索用捕獲手
    QCapture,
}

impl Stage {
    pub fn next(self) -> Self {
        match self {
            Stage::MainTT => Stage::CaptureInit,
            Stage::CaptureInit => Stage::GoodCapture,
            Stage::ProbCutTT => Stage::ProbCutInit,
            Stage::ProbCutInit => Stage::ProbCut,
            Stage::ProbCut => Stage::ProbCut,
            Stage::GoodCapture => Stage::QuietInit,
            Stage::QuietInit => Stage::GoodQuiet,
            Stage::GoodQuiet => Stage::BadCapture,
            Stage::BadCapture => Stage::BadQuiet,
            Stage::BadQuiet => Stage::BadQuiet,
            Stage::EvasionTT => Stage::EvasionInit,
            Stage::EvasionInit => Stage::Evasion,
            Stage::Evasion => Stage::Evasion,
            Stage::QSearchTT => Stage::QCaptureInit,
            Stage::QCaptureInit => Stage::QCapture,
            Stage::QCapture => Stage::QCapture,
        }
    }
}
pub struct MovePicker {
    // History参照は保持しない（next_move時に渡す）

    // ContinuationHistory参照（スコアリング用、ply毎に異なるため保持）
    continuation_history: [*const PieceToHistory; 6],

    // 状態
    stage: Stage,
    tt_move: Move,
    probcut_threshold: Option<Value>,
    /// 探索の深さ（部分ソートの閾値計算に使用）
    depth: Depth,
    ply: i32,
    skip_quiets: bool,
    generate_all_legal_moves: bool,

    // 初期化時にキャッシュする情報
    side_to_move: Color,
    pawn_history_index: usize,

    // 指し手バッファ（MaybeUninitにより初期化コストゼロ）
    moves: ExtMoveBuffer,
    cur: usize,
    end_cur: usize,
    end_bad_captures: usize,
    end_captures: usize,
    end_generated: usize,
    /// partial_insertion_sort の sorted 領域末尾位置（未使用だが YO 構造保持用）
    end_good_quiets: usize,

    /// Phase A (2026-06-05): 直近 next_move で return した move の出所 stage。
    /// alpha_beta cutoff 時に stage 別 first_cut_rate を集計するため。
    /// `Stage::MainTT` 等が「stage の遷移前」 を保持 (= TT return 後 stage 自体は CaptureInit に進むため)。
    last_returned_stage: Stage,
}

/// Phase A: Stage を 7 bucket に集約 (= cutoff_by_stage の index 算出)。
/// 0=MainTT(系), 1=GoodCapture, 2=GoodQuiet, 3=BadCapture, 4=BadQuiet, 5=Evasion(系), 6=Other
#[inline]
pub fn stage_to_idx(stage: Stage) -> usize {
    match stage {
        Stage::MainTT | Stage::EvasionTT | Stage::QSearchTT | Stage::ProbCutTT => 0,
        Stage::GoodCapture => 1,
        Stage::GoodQuiet => 2,
        Stage::BadCapture => 3,
        Stage::BadQuiet => 4,
        Stage::Evasion | Stage::EvasionInit | Stage::QCapture | Stage::QCaptureInit => 5,
        _ => 6,
    }
}

/// Phase B-1 (2026-06-05): score_quiets 内 history coefficient (= USI option 経由 SPSA tune)
/// default = 既存 baseline と同等 (= 全 1)、 increase で boost。
pub static HIST_CH_COEFF: core::sync::atomic::AtomicI32 = core::sync::atomic::AtomicI32::new(1);
pub static HIST_MAIN_COEFF: core::sync::atomic::AtomicI32 = core::sync::atomic::AtomicI32::new(2);
pub static HIST_PAWN_COEFF: core::sync::atomic::AtomicI32 = core::sync::atomic::AtomicI32::new(2);
pub static HIST_LP_COEFF: core::sync::atomic::AtomicI32 = core::sync::atomic::AtomicI32::new(8);

pub fn set_hist_ch_coeff(v: i32) {
    HIST_CH_COEFF.store(v.clamp(0, 16), core::sync::atomic::Ordering::Relaxed);
}
pub fn set_hist_main_coeff(v: i32) {
    HIST_MAIN_COEFF.store(v.clamp(0, 16), core::sync::atomic::Ordering::Relaxed);
}
pub fn set_hist_pawn_coeff(v: i32) {
    HIST_PAWN_COEFF.store(v.clamp(0, 16), core::sync::atomic::Ordering::Relaxed);
}
pub fn set_hist_lp_coeff(v: i32) {
    HIST_LP_COEFF.store(v.clamp(0, 32), core::sync::atomic::Ordering::Relaxed);
}

// =============================================================================
// 敵玉近傍 drop の move ordering bonus (2026-06-16)
// =============================================================================
//
// 診断 (Dewar 戦 ply88): 敵玉近傍への強力な drop (例 4九飛 R*4i) を浅い探索で
// 発見できず崩壊した。 これらの「敵玉 5x5 zone への駒打ち」を quiet move ordering で
// 高スコア付与し、 浅い段階でも上位探索 → 早期に深掘り・危険認識させる。
//
// ★ 害の診断 (2026-06-16): mode 3 (= 全局面一律 5x5 加点) は同ノード A/B で
//   vs AobaNNUE -17.6pp。 通常中盤局面 (median ply88) の 3-4% で bestmove を
//   drop に書き換え、 さらに毎ノード「玉近傍の大半が悪手の drop」を quiet 先頭に
//   押し上げて YO 調整済 ordering を乱し、 cutoff 効率を落とすため。
//   → mode で「稀な致命ケースのみ救う」狭い版に切替可能にした (SPSA / A/B 用)。
//
// score_quiets の history score (典型 ±8000 程度) と王手 bonus (+16384) の中間 tier に
// 置く。 駒種重み = 飛/竜/角/馬 最優先、 金銀 次、 歩香桂 低め。 飛角の遠隔利きが
// 玉ライン (= 玉升へ届く) に乗る drop は zone 内外問わず追加 bonus。

/// 重駒系 (飛/竜/角/馬) を敵玉 5x5 zone へ drop した時の bonus。
const KING_ZONE_DROP_BONUS_HEAVY: i32 = 14000;
/// 金/銀を敵玉 5x5 zone へ drop した時の bonus。
const KING_ZONE_DROP_BONUS_MID: i32 = 9000;
/// 歩/香/桂を敵玉 5x5 zone へ drop した時の bonus。
const KING_ZONE_DROP_BONUS_LIGHT: i32 = 4000;
/// 飛/角 drop の利きが敵玉升へ届く (= 玉ラインに乗る) 場合の追加 bonus。
const KING_LINE_DROP_BONUS: i32 = 6000;

/// 敵玉 drop bonus のモード (= USI option `KingDropBonusMode`)。
///
/// - 0 = off: 一切加点しない (= base 相当)。
/// - 1 = line-only: 飛/角の遠隔利きが玉升に乗る drop のみ加点 (= 最も狭く安全)。
/// - 2 = danger-gated (default): line bonus は常時、 5x5 zone bonus は
///   「敵玉が強い攻撃下にある (= 自軍駒が玉 5x5 内に `KING_DROP_DANGER_MIN` 枚以上)」
///   局面に限定。 通常局面では zone flood を抑え、 寄せ局面でのみ drop を浅く読ませる。
/// - 3 = full: 全局面で 5x5 zone + line を一律加点 (= 旧 dropmo、 害 -17.6pp、 A/B 参照用)。
///
/// default = 2 (danger-gated)。 害の主因 (= 通常局面の zone flood) を除去しつつ
/// 寄せ局面の致命 drop 救済を維持する設計。
///
/// ★ 害の診断補足 (2026-06-16): 単純な「自軍駒が玉 5x5 内に 1 枚以上」ゲートは
///   中盤 (median ply88) で頻繁に開き、 mode3 とほぼ同じ非drop→drop率 (6.2% vs 5.0%) で
///   不乱効果が薄かった。 → 閾値 `KING_DROP_DANGER_MIN` (default 3) で「複数駒が密集する
///   真の寄せ局面」のみゲートを開く。 単発接近の中盤局面ではゲートを閉じる。
pub static KING_DROP_BONUS_MODE: core::sync::atomic::AtomicI32 =
    core::sync::atomic::AtomicI32::new(2);

/// mode 2 (danger-gated) で zone bonus を有効化する「自軍駒が敵玉 5x5 内」の最小枚数
/// (= USI option `KingDropDangerMin`)。 default 2。 値を上げるほどゲートが厳しくなり
/// (通常局面を除外)、 下げるほど緩くなる (= 1 で旧挙動)。
///
/// ★ 閾値 sweep (2026-06-16, arh_e0 局面 + 客観最善救済4件): min2 が「救済全4件維持 ×
///   非drop→drop率 1.2% (= line-only 同等、 mode3 full 5.0% の 1/4)」の最適点。
///   min3/min4 は不乱度が上がる上に S*6g/G*6g (金銀の5x5救済) を取りこぼす (厳しすぎ)。
pub static KING_DROP_DANGER_MIN: core::sync::atomic::AtomicI32 =
    core::sync::atomic::AtomicI32::new(2);

/// `KingDropBonusMode` を設定する (0..=3 に clamp)。
pub fn set_king_drop_bonus_mode(v: i32) {
    KING_DROP_BONUS_MODE.store(v.clamp(0, 3), core::sync::atomic::Ordering::Relaxed);
}

/// `KingDropDangerMin` を設定する (1..=8 に clamp)。
pub fn set_king_drop_danger_min(v: i32) {
    KING_DROP_DANGER_MIN.store(v.clamp(1, 8), core::sync::atomic::Ordering::Relaxed);
}

/// 敵玉近傍 drop の move ordering bonus を計算する (= quiet move 専用)。
///
/// `m` が駒打ちでない場合は 0。 駒打ちのとき (mode に応じて):
/// - 着手升が敵玉 5x5 zone 内 → 駒種重みに応じた bonus (mode 2 では `king_in_danger` ゲート)。
/// - 飛/角 drop の利き (現盤面 occupancy) が敵玉升へ届く → 追加 bonus (全 mode 共通、 稀)。
///
/// 玉に直接利く歩香桂 / 詰めろ系は既存の check bonus (+16384) が別途加算されるため、
/// ここでは zone 内であれば light でも加点し「投下手を浅く読ませる」ことに専念する。
///
/// `mode` は呼出側で 1 度だけ atomic load した値を渡す (= per-node hoist)。
/// `king_in_danger` は mode 2 用の danger gate (= 呼出側で per-node に 1 度判定)。
#[inline]
fn king_zone_drop_bonus(
    m: Move,
    enemy_king_sq: Square,
    enemy_king_zone: Bitboard,
    occupied: Bitboard,
    mode: i32,
    king_in_danger: bool,
) -> i32 {
    if mode == 0 || !m.is_drop() {
        return 0;
    }
    let to = m.to();
    let pt = m.drop_piece_type();

    let mut bonus = 0;

    // 5x5 zone bonus。 mode 1 (line-only) では完全に無効。
    // mode 2 (danger-gated) では敵玉が危険な局面に限定。 mode 3 (full) は一律。
    let zone_enabled = match mode {
        1 => false,
        2 => king_in_danger,
        _ => true, // 3 = full
    };
    if zone_enabled && enemy_king_zone.contains(to) {
        bonus += match pt {
            // 飛角 (打てる持駒は不成の Rook / Bishop のみ。 竜馬は持駒に存在しない)
            PieceType::Rook | PieceType::Bishop => KING_ZONE_DROP_BONUS_HEAVY,
            PieceType::Gold | PieceType::Silver => KING_ZONE_DROP_BONUS_MID,
            PieceType::Pawn | PieceType::Lance | PieceType::Knight => KING_ZONE_DROP_BONUS_LIGHT,
            _ => 0,
        };
    }

    // 飛角の遠隔利きが敵玉升に乗る drop (= 開き筋 / 王手含み)。 zone 内外を問わない。
    // 稀かつ意味的 (= Dewar 4九飛 型) なため全 mode (1/2/3) で常時加点。
    let line_hits_king = match pt {
        PieceType::Rook => rook_effect(to, occupied).contains(enemy_king_sq),
        PieceType::Bishop => bishop_effect(to, occupied).contains(enemy_king_sq),
        _ => false,
    };
    if line_hits_king {
        bonus += KING_LINE_DROP_BONUS;
    }

    bonus
}

impl MovePicker {
    /// 通常探索・静止探索用コンストラクタ（History参照を保持しない）
    ///
    /// `pos`は初期化時のみ使用し、フィールドとして保持しない。
    /// `continuation_history`はスコアリング時に使用するため、ポインタとして保持する。
    ///
    /// # Safety
    ///
    /// `continuation_history`のポインタは、MovePickerのライフタイム中有効でなければならない。
    /// これは探索ノード内でのみMovePickerを使用することで保証される。
    pub fn new(
        pos: &Position,
        tt_move: Move,
        depth: Depth,
        ply: i32,
        continuation_history: [&PieceToHistory; 6],
        generate_all_legal_moves: bool,
    ) -> Self {
        let stage = if pos.in_check() {
            // 王手回避
            if tt_move.is_some() && pos.pseudo_legal_with_all(tt_move, generate_all_legal_moves) {
                Stage::EvasionTT
            } else {
                Stage::EvasionInit
            }
        } else if depth > DEPTH_QS {
            // 通常探索
            if tt_move.is_some() && pos.pseudo_legal_with_all(tt_move, generate_all_legal_moves) {
                Stage::MainTT
            } else {
                Stage::CaptureInit
            }
        } else {
            // 静止探索
            if tt_move.is_some() && pos.pseudo_legal_with_all(tt_move, generate_all_legal_moves) {
                Stage::QSearchTT
            } else {
                Stage::QCaptureInit
            }
        };

        Self {
            continuation_history: [
                continuation_history[0] as *const _,
                continuation_history[1] as *const _,
                continuation_history[2] as *const _,
                continuation_history[3] as *const _,
                continuation_history[4] as *const _,
                continuation_history[5] as *const _,
            ],
            stage,
            tt_move,
            probcut_threshold: None,
            depth,
            ply,
            skip_quiets: false,
            generate_all_legal_moves,
            side_to_move: pos.side_to_move(),
            pawn_history_index: pos.pawn_history_index(),
            moves: ExtMoveBuffer::new(),
            cur: 0,
            end_cur: 0,
            end_bad_captures: 0,
            end_captures: 0,
            end_generated: 0,
            end_good_quiets: 0,
            last_returned_stage: Stage::MainTT,
        }
    }

    /// 王手回避専用コンストラクタ（シンプル版）
    pub fn new_evasions(
        pos: &Position,
        tt_move: Move,
        ply: i32,
        continuation_history: [&PieceToHistory; 6],
        generate_all_legal_moves: bool,
    ) -> Self {
        debug_assert!(pos.in_check());

        let stage =
            if tt_move.is_some() && pos.pseudo_legal_with_all(tt_move, generate_all_legal_moves) {
                Stage::EvasionTT
            } else {
                Stage::EvasionInit
            };

        Self {
            continuation_history: [
                continuation_history[0] as *const _,
                continuation_history[1] as *const _,
                continuation_history[2] as *const _,
                continuation_history[3] as *const _,
                continuation_history[4] as *const _,
                continuation_history[5] as *const _,
            ],
            stage,
            tt_move,
            probcut_threshold: None,
            depth: DEPTH_QS,
            ply,
            skip_quiets: false,
            generate_all_legal_moves,
            side_to_move: pos.side_to_move(),
            pawn_history_index: pos.pawn_history_index(),
            moves: ExtMoveBuffer::new(),
            cur: 0,
            end_cur: 0,
            end_bad_captures: 0,
            end_captures: 0,
            end_generated: 0,
            end_good_quiets: 0,
            last_returned_stage: Stage::MainTT,
        }
    }

    /// ProbCut専用コンストラクタ
    pub fn new_probcut(
        pos: &Position,
        tt_move: Move,
        threshold: Value,
        ply: i32,
        continuation_history: [&PieceToHistory; 6],
        generate_all_legal_moves: bool,
    ) -> Self {
        debug_assert!(!pos.in_check());

        let stage = if tt_move.is_some()
            && pos.capture_stage(tt_move)
            && pos.pseudo_legal_with_all(tt_move, generate_all_legal_moves)
        {
            Stage::ProbCutTT
        } else {
            Stage::ProbCutInit
        };

        Self {
            continuation_history: [
                continuation_history[0] as *const _,
                continuation_history[1] as *const _,
                continuation_history[2] as *const _,
                continuation_history[3] as *const _,
                continuation_history[4] as *const _,
                continuation_history[5] as *const _,
            ],
            stage,
            tt_move,
            probcut_threshold: Some(threshold),
            depth: DEPTH_QS,
            ply,
            skip_quiets: false,
            generate_all_legal_moves,
            side_to_move: pos.side_to_move(),
            pawn_history_index: pos.pawn_history_index(),
            moves: ExtMoveBuffer::new(),
            cur: 0,
            end_cur: 0,
            end_bad_captures: 0,
            end_captures: 0,
            end_generated: 0,
            end_good_quiets: 0,
            last_returned_stage: Stage::MainTT,
        }
    }

    /// quiet手の生成をスキップ（LMP条件成立時に呼び出す）
    ///
    /// YaneuraOu準拠: フラグのみ設定する。実際のステージ遷移は next_move() 側で処理される。
    pub fn skip_quiets(&mut self) {
        self.skip_quiets = true;
    }

    /// 現在のステージがquiet段階かどうかを返す
    ///
    /// LMP発火条件の判定に使用する。quiet段階（QuietInit, GoodQuiet, BadQuiet）で
    /// のみLMPを発火させ、captureは残す。
    #[inline]
    pub fn is_quiet_stage(&self) -> bool {
        matches!(self.stage, Stage::QuietInit | Stage::GoodQuiet | Stage::BadQuiet)
    }

    /// 現在のステージを取得（デバッグ用）
    #[inline]
    pub fn stage(&self) -> Stage {
        self.stage
    }

    /// Phase A: 直近 next_move で return した move の出所 stage を取得
    #[inline]
    pub fn last_returned_stage(&self) -> Stage {
        self.last_returned_stage
    }

    /// 次の指し手を返す
    ///
    /// 指し手が尽きたら `Move::NONE` を返す。
    ///
    /// ## 引数
    /// - `pos`: 現在の局面への参照（各呼び出しで一時的に借用）
    /// - `history`: HistoryTablesへの参照（スコアリング時に使用）
    ///
    /// ## 使用パターン
    ///
    /// ```ignore
    /// let mv = ctx.history.with_read(|h| mp.next_move(pos, h));
    /// ```
    pub fn next_move(&mut self, pos: &Position, history: &HistoryTables) -> Move {
        loop {
            match self.stage {
                // ==============================
                // TT手を返す
                // ==============================
                Stage::MainTT | Stage::EvasionTT | Stage::QSearchTT | Stage::ProbCutTT => {
                    self.last_returned_stage = self.stage;
                    self.stage = self.stage.next();
                    return self.tt_move;
                }

                // ==============================
                // 捕獲手の生成
                // ==============================
                Stage::CaptureInit | Stage::QCaptureInit | Stage::ProbCutInit => {
                    self.init_captures(pos, history);
                }

                // ==============================
                // 良い捕獲手を返す
                // ==============================
                Stage::GoodCapture => {
                    if let Some(m) = self.select_good_capture(pos) {
                        self.last_returned_stage = Stage::GoodCapture;
                        return m;
                    }
                    // YaneuraOu準拠: 常に QuietInit へ遷移。skip_quiets は
                    // QuietInit/GoodQuiet/BadQuiet 側で判定する。
                    self.stage = Stage::QuietInit;
                }

                // ==============================
                // 静かな手の生成
                // ==============================
                Stage::QuietInit => {
                    self.init_quiets(pos, history);
                }

                // ==============================
                // 良い静かな手を返す（YaneuraOu準拠: value > GOOD_QUIET_THRESHOLD）
                // partial_insertion_sortで高スコア手が先頭に来るが、
                // YaneuraOu同様に全quiet手を走査してthreshold超の手を返す
                // ==============================
                Stage::GoodQuiet => {
                    if !self.skip_quiets {
                        self.end_cur = self.end_generated;
                        if let Some(m) = self.select_good_quiet() {
                            self.last_returned_stage = Stage::GoodQuiet;
                            return m;
                        }
                    }

                    // 悪い捕獲手の準備
                    self.cur = 0;
                    self.end_cur = self.end_bad_captures;
                    self.stage = Stage::BadCapture;
                }

                // ==============================
                // 悪い捕獲手を返す
                // ==============================
                Stage::BadCapture => {
                    if let Some(m) = self.select_simple() {
                        self.last_returned_stage = Stage::BadCapture;
                        return m;
                    }

                    // YaneuraOu準拠: endCapturesからquiet手全体を再走査
                    self.cur = self.end_captures;
                    self.end_cur = self.end_generated;
                    self.stage = Stage::BadQuiet;
                }

                // ==============================
                // 悪い静かな手を返す（YaneuraOu準拠: value <= GOOD_QUIET_THRESHOLD）
                // ==============================
                Stage::BadQuiet => {
                    if !self.skip_quiets {
                        // YaneuraOu準拠: endCaptures から全quiet手を再走査し、
                        // value <= GOOD_QUIET_THRESHOLD の手のみ返す
                        if let Some(m) = self.select_bad_quiet() {
                            self.last_returned_stage = Stage::BadQuiet;
                            return m;
                        }
                    }
                    return Move::NONE;
                }

                // ==============================
                // 回避手の生成
                // ==============================
                Stage::EvasionInit => {
                    self.init_evasions(pos, history);
                }

                // ==============================
                // 回避手を返す
                // ==============================
                Stage::Evasion => {
                    let m = self.select_simple().unwrap_or(Move::NONE);
                    if m != Move::NONE {
                        self.last_returned_stage = Stage::Evasion;
                    }
                    return m;
                }

                // ==============================
                // 静止探索用捕獲手を返す
                // ==============================
                Stage::QCapture => {
                    let m = self.select_simple().unwrap_or(Move::NONE);
                    if m != Move::NONE {
                        self.last_returned_stage = Stage::QCapture;
                    }
                    return m;
                }

                // ==============================
                // ProbCut: SEEが閾値以上の捕獲のみ
                // ==============================
                Stage::ProbCut => {
                    if let Some(th) = self.probcut_threshold {
                        let m = self.select_probcut(pos, th).unwrap_or(Move::NONE);
                        if m != Move::NONE {
                            self.last_returned_stage = Stage::ProbCut;
                        }
                        return m;
                    }
                    return Move::NONE;
                }
            }
        }
    }

    // =========================================================================
    // スコアリング
    // =========================================================================

    #[inline(never)]
    fn init_captures(&mut self, pos: &Position, history: &HistoryTables) {
        self.cur = 0;
        self.end_bad_captures = 0;
        self.moves.clear();

        // YaneuraOu準拠: CaptureInit/QCaptureInit/ProbCutInit は同じ capture 生成を使う
        let count = if self.generate_all_legal_moves {
            crate::movegen::generate_with_type(
                pos,
                crate::movegen::GenType::CapturesAll,
                &mut self.moves,
                None,
            );
            self.moves.len()
        } else {
            crate::movegen::generate_with_type(
                pos,
                crate::movegen::GenType::Captures,
                &mut self.moves,
                None,
            );
            self.moves.len()
        };
        self.end_cur = count;
        self.end_captures = count;

        self.score_captures(pos, history);
        partial_insertion_sort(self.moves.as_mut_slice(), self.end_cur, i32::MIN);

        self.stage = self.stage.next();
    }

    #[inline(never)]
    fn init_quiets(&mut self, pos: &Position, history: &HistoryTables) {
        if !self.skip_quiets {
            // 静かな手を生成
            let count = if self.generate_all_legal_moves {
                crate::movegen::generate_with_type(
                    pos,
                    crate::movegen::GenType::QuietsAll,
                    &mut self.moves,
                    None,
                );
                self.moves.len() - self.end_captures
            } else {
                pos.generate_quiets(&mut self.moves, self.end_captures)
            };
            self.end_cur = self.end_captures + count;
            self.end_generated = self.end_cur;
            self.moves.set_len(self.end_cur);

            self.cur = self.end_captures;

            self.score_quiets(pos, history);

            // YaneuraOu準拠: 深さベースの閾値で部分ソート
            let limit = -3560 * self.depth;
            let quiet_count = self.end_cur - self.end_captures;
            debug_assert!(self.end_captures <= self.moves.len());
            // SAFETY: end_captures <= moves.len() は MovePicker の不変条件。
            let sorted_end = partial_insertion_sort(
                unsafe { self.moves.as_mut_slice().get_unchecked_mut(self.end_captures..) },
                quiet_count,
                limit,
            );
            self.end_good_quiets = self.end_captures + sorted_end;
        } else {
            self.end_good_quiets = self.end_captures;
        }
        self.stage = Stage::GoodQuiet;
    }

    #[inline(never)]
    fn init_evasions(&mut self, pos: &Position, history: &HistoryTables) {
        // 回避手を生成
        let count = if self.generate_all_legal_moves {
            self.moves.clear();
            crate::movegen::generate_with_type(
                pos,
                crate::movegen::GenType::EvasionsAll,
                &mut self.moves,
                None,
            );
            self.moves.len()
        } else {
            pos.generate_evasions_ext(&mut self.moves)
        };
        self.cur = 0;
        self.end_cur = count;
        self.end_generated = count;

        self.score_evasions(pos, history);
        partial_insertion_sort(self.moves.as_mut_slice(), self.end_cur, i32::MIN);

        self.stage = Stage::Evasion;
    }
    /// 捕獲手のスコアを計算
    #[inline(never)]
    fn score_captures(&mut self, pos: &Position, history: &HistoryTables) {
        debug_assert!(self.cur <= self.end_cur && self.end_cur <= self.moves.len());
        // SAFETY: cur <= end_cur <= moves.len() は MovePicker の不変条件。
        for ext in unsafe { self.moves.as_mut_slice().get_unchecked_mut(self.cur..self.end_cur) } {
            let m = ext.mv;
            let to = m.to();
            let pc = m.moved_piece_after();
            let captured = pos.piece_on(to);
            let captured_pt = captured.piece_type();

            // MVV + CaptureHistory + 王手ボーナス
            let mut value = history.capture_history.get(pc, to, captured_pt) as i32;
            value += 7 * piece_value(captured);

            ext.value = value;
        }
    }

    /// 静かな手のスコアを計算
    #[inline(never)]
    fn score_quiets(&mut self, pos: &Position, history: &HistoryTables) {
        let us = self.side_to_move;
        let pawn_idx = self.pawn_history_index;
        debug_assert!(self.cur <= self.end_cur && self.end_cur <= self.moves.len());
        // SAFETY: cur <= end_cur <= moves.len() は MovePicker の不変条件。
        let moves = unsafe { self.moves.as_mut_slice().get_unchecked_mut(self.cur..self.end_cur) };
        // SAFETY:
        // - continuation_history のポインタは MovePicker の寿命中有効。
        // - ここでは読み取り専用でのみ参照する。
        // - moves とは無関係な領域なのでエイリアスは発生しない。
        let (ch0, ch1, ch2, ch3, ch5) = unsafe {
            (
                &*self.continuation_history[0],
                &*self.continuation_history[1],
                &*self.continuation_history[2],
                &*self.continuation_history[3],
                &*self.continuation_history[5],
            )
        };

        // C-6 hoist: check_squares を loop 外で pt-indexed テーブルに展開。
        // 各反復で `cur_state()` ポインタ追跡 + `check_sq_index` の match を消し、
        // `cs_by_pt[pt as usize]` の単一 indexed load に置換。
        let cs_by_pt = pos.check_squares_by_pt_table();

        // 敵玉近傍 drop bonus 用の loop 外 hoist (2026-06-16)。
        // enemy = 手番側の相手。 玉升 / 5x5 zone / 現盤面 occupancy を 1 度だけ取得。
        let enemy_king_sq = pos.king_square(us.opponent());
        let enemy_king_zone = king_5x5_mask(enemy_king_sq);
        let occupied = pos.occupied();
        // mode + danger gate も per-node に 1 度だけ評価 (per-iter コスト 0)。
        let drop_bonus_mode = KING_DROP_BONUS_MODE.load(Relaxed);
        // danger gate (mode 2 用): 自軍駒が敵玉 5x5 zone 内に閾値枚数以上集まる
        // (= 真の寄せ局面) なら danger とみなす。 通常中盤の単発接近ではゲート閉。
        // 1 回の bitboard-AND + popcount で O(1)。 mode 0/1/3 では未使用。
        let king_in_danger = drop_bonus_mode == 2 && {
            let near = (pos.pieces_c(us) & enemy_king_zone).count() as i32;
            near >= KING_DROP_DANGER_MIN.load(Relaxed)
        };

        // P1 hoist (2026-06-06): atomic coefficients を関数冒頭で 1 度 load。
        // USI option の動的変更は per-search で取り直しでよく、 score_quiets 1 回呼出
        // (per-node) の中では「整合した coeff set」 を維持。 per-iter 0 atomic load。
        use core::sync::atomic::Ordering::Relaxed;
        let ch_c = HIST_CH_COEFF.load(Relaxed);
        let main_c = HIST_MAIN_COEFF.load(Relaxed);
        let pawn_c = HIST_PAWN_COEFF.load(Relaxed);
        let lp_c = HIST_LP_COEFF.load(Relaxed);

        if self.ply < LOW_PLY_HISTORY_SIZE as i32 {
            debug_assert!(self.ply >= 0, "ply must be non-negative: {}", self.ply);
            let low_ply_idx = self.ply as usize;
            let low_ply_div = 1 + self.ply;

            for ext in moves {
                let m = ext.mv;
                let to = m.to();
                let pc = m.moved_piece_after();
                let pt = pc.piece_type();

                let mut value = main_c * history.main_history.get(us, m) as i32;
                value += pawn_c * history.pawn_history.get(pawn_idx, pc, to) as i32;
                value += ch_c * ch0.get(pc, to) as i32;
                value += ch_c * ch1.get(pc, to) as i32;
                value += ch_c * ch2.get(pc, to) as i32;
                value += ch_c * ch3.get(pc, to) as i32;
                value += ch_c * ch5.get(pc, to) as i32;

                if cs_by_pt[pt as usize].contains(to) && pos.see_ge(m, Value::new(-75)) {
                    value += 16384;
                }

                // 敵玉近傍 drop bonus (2026-06-16): 投下手を浅く読ませる。
                value += king_zone_drop_bonus(
                    m,
                    enemy_king_sq,
                    enemy_king_zone,
                    occupied,
                    drop_bonus_mode,
                    king_in_danger,
                );

                // ply >= 0 (debug_assert 済み) なので low_ply_div >= 1。
                value += lp_c * history.low_ply_history.get(low_ply_idx, m) as i32 / low_ply_div;

                ext.value = value;
            }
        } else {
            for ext in moves {
                let m = ext.mv;
                let to = m.to();
                let pc = m.moved_piece_after();
                let pt = pc.piece_type();

                let mut value = main_c * history.main_history.get(us, m) as i32;
                value += pawn_c * history.pawn_history.get(pawn_idx, pc, to) as i32;
                value += ch_c * ch0.get(pc, to) as i32;
                value += ch_c * ch1.get(pc, to) as i32;
                value += ch_c * ch2.get(pc, to) as i32;
                value += ch_c * ch3.get(pc, to) as i32;
                value += ch_c * ch5.get(pc, to) as i32;

                if cs_by_pt[pt as usize].contains(to) && pos.see_ge(m, Value::new(-75)) {
                    value += 16384;
                }

                // 敵玉近傍 drop bonus (2026-06-16): 投下手を浅く読ませる。
                value += king_zone_drop_bonus(
                    m,
                    enemy_king_sq,
                    enemy_king_zone,
                    occupied,
                    drop_bonus_mode,
                    king_in_danger,
                );

                ext.value = value;
            }
        }
    }

    /// 回避手のスコアを計算
    #[inline(never)]
    fn score_evasions(&mut self, pos: &Position, history: &HistoryTables) {
        let us = self.side_to_move;
        // SAFETY:
        // - continuation_history[0] は MovePicker の寿命中有効。
        // - 読み取り専用でのみ使用し、moves とは無関係な領域を指す。
        let ch = unsafe { &*self.continuation_history[0] };

        debug_assert!(self.cur <= self.end_cur && self.end_cur <= self.moves.len());
        // SAFETY: cur <= end_cur <= moves.len() は MovePicker の不変条件。
        for ext in unsafe { self.moves.as_mut_slice().get_unchecked_mut(self.cur..self.end_cur) } {
            let m = ext.mv;
            let to = m.to();
            let pc = m.moved_piece_after();

            if pos.capture_stage(m) {
                // 捕獲手は駒価値 + 大きなボーナス
                let captured = pos.piece_on(to);
                ext.value = piece_value(captured) + (1 << 28);
            } else {
                // 静かな手はHistory
                let mut value = history.main_history.get(us, m) as i32;
                value += ch.get(pc, to) as i32;
                ext.value = value;
            }
        }
    }

    // =========================================================================
    // ヘルパー
    // =========================================================================

    /// 良い捕獲手を選択（SEE >= threshold）
    fn select_good_capture(&mut self, pos: &Position) -> Option<Move> {
        while self.cur < self.end_cur {
            let ext = self.moves.get(self.cur);
            self.cur += 1;

            // TT手は既に返したのでスキップ
            if ext.mv == self.tt_move {
                continue;
            }

            // SEEで閾値以上の手のみ
            let threshold = Value::new(-ext.value / 18);
            if pos.see_ge(ext.mv, threshold) {
                return Some(ext.mv);
            } else {
                // 悪い捕獲手は後回し
                self.moves.swap(self.end_bad_captures, self.cur - 1);
                self.end_bad_captures += 1;
            }
        }
        None
    }

    /// GoodQuiet用: value > GOOD_QUIET_THRESHOLD の手のみ返す（YaneuraOu準拠）
    fn select_good_quiet(&mut self) -> Option<Move> {
        const GOOD_QUIET_THRESHOLD: i32 = -14000;
        while self.cur < self.end_cur {
            let ext = self.moves.get(self.cur);
            self.cur += 1;

            if ext.mv == self.tt_move {
                continue;
            }

            if ext.value > GOOD_QUIET_THRESHOLD {
                return Some(ext.mv);
            }
        }
        None
    }

    /// BadQuiet用: value <= GOOD_QUIET_THRESHOLD の手のみ返す（YaneuraOu準拠）
    fn select_bad_quiet(&mut self) -> Option<Move> {
        const GOOD_QUIET_THRESHOLD: i32 = -14000;
        while self.cur < self.end_cur {
            let ext = self.moves.get(self.cur);
            self.cur += 1;

            if ext.mv == self.tt_move {
                continue;
            }

            if ext.value <= GOOD_QUIET_THRESHOLD {
                return Some(ext.mv);
            }
        }
        None
    }

    /// シンプルな手の選択（TT手スキップのみ）
    fn select_simple(&mut self) -> Option<Move> {
        while self.cur < self.end_cur {
            let ext = self.moves.get(self.cur);
            self.cur += 1;

            // TT手は既に返したのでスキップ
            if ext.mv == self.tt_move {
                continue;
            }

            return Some(ext.mv);
        }
        None
    }

    /// ProbCut用の手の選択（SEE閾値チェック）
    fn select_probcut(&mut self, pos: &Position, threshold: Value) -> Option<Move> {
        while self.cur < self.end_cur {
            let ext = self.moves.get(self.cur);
            self.cur += 1;

            // TT手は既に返したのでスキップ
            if ext.mv == self.tt_move {
                continue;
            }

            if pos.see_ge(ext.mv, threshold) {
                return Some(ext.mv);
            }
        }
        None
    }
}

// MovePickerIter は History 参照を引数で渡す新しい設計では使用できないため削除
// 代わりに、探索ループ内で直接 mp.next_move(pos, history) を呼び出す

// =============================================================================
// ユーティリティ関数
// =============================================================================

/// 部分ソート（ハイブリッド方式）
///
/// YaneuraOu準拠の部分挿入ソート。
///
/// `limit` 以上のスコアの手を先頭に集め、降順でソートする。
/// 先頭要素（index 0）を初期 sorted 領域とみなし、index 1 から走査する。
///
/// ## 戻り値
/// sorted 領域の末尾インデックス（sorted_end）を返す。
/// sorted 領域は `[0, sorted_end]`（inclusive）。
/// - `end <= 1` の場合は 0 を返す（ソート不要）
///
/// ## 計算量
/// - 挿入ソート相当
#[inline]
fn partial_insertion_sort(moves: &mut [ExtMove], end: usize, limit: i32) -> usize {
    // YaneuraOu: movepick.cpp partial_insertion_sort()
    //   for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
    //       if (p->value >= limit) {
    //           ExtMove tmp = *p, *q;
    //           *p = *++sortedEnd;
    //           for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
    //               *q = *(q - 1);
    //           *q = tmp;
    //       }
    if end <= 1 {
        return 0;
    }

    let mut sorted_end: usize = 0;
    let base = moves.as_mut_ptr();
    for p in 1..end {
        // SAFETY:
        // - p は常に 1..end の範囲で、end は呼び出し元が `moves.len()` 以下で渡す。
        // - sorted_end は 0..p の範囲を保つため、add() でアクセスする要素は常に同一 slice 内。
        // - ExtMove は Copy なので、要素の読み書き・シフトで drop 問題は発生しない。
        unsafe {
            let tmp = *base.add(p);
            if tmp.value >= limit {
                sorted_end += 1;
                if p != sorted_end {
                    *base.add(p) = *base.add(sorted_end);
                }
                let mut q = sorted_end;
                while q > 0 && (*base.add(q - 1)).value < tmp.value {
                    *base.add(q) = *base.add(q - 1);
                    q -= 1;
                }
                *base.add(q) = tmp;
            }
        }
    }
    sorted_end
}

/// 駒の価値（MVV用）
#[inline]
pub(crate) fn piece_value(pc: Piece) -> i32 {
    if pc.is_none() {
        return 0;
    }
    use PieceType::*;
    match pc.piece_type() {
        Pawn => 90,
        Lance => 315,
        Knight => 405,
        Silver => 495,
        Gold | ProPawn | ProLance | ProKnight | ProSilver => 540,
        Bishop => 855,
        Rook => 990,
        // YaneuraOu Eval::PieceValue 準拠
        Horse => 945,
        Dragon => 1395,
        King => 15000,
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_next() {
        assert_eq!(Stage::MainTT.next(), Stage::CaptureInit);
        assert_eq!(Stage::CaptureInit.next(), Stage::GoodCapture);
        assert_eq!(Stage::GoodCapture.next(), Stage::QuietInit);
        assert_eq!(Stage::QuietInit.next(), Stage::GoodQuiet);
        assert_eq!(Stage::GoodQuiet.next(), Stage::BadCapture);
        assert_eq!(Stage::BadCapture.next(), Stage::BadQuiet);
        assert_eq!(Stage::BadQuiet.next(), Stage::BadQuiet);

        assert_eq!(Stage::EvasionTT.next(), Stage::EvasionInit);
        assert_eq!(Stage::EvasionInit.next(), Stage::Evasion);
        assert_eq!(Stage::Evasion.next(), Stage::Evasion);

        assert_eq!(Stage::QSearchTT.next(), Stage::QCaptureInit);
        assert_eq!(Stage::QCaptureInit.next(), Stage::QCapture);
        assert_eq!(Stage::QCapture.next(), Stage::QCapture);
    }

    #[test]
    fn test_partial_insertion_sort() {
        // YaneuraOu準拠: index 0 を初期 sorted 領域とし、index 1 から走査
        let mut moves = vec![
            ExtMove::new(Move::NONE, 100),
            ExtMove::new(Move::NONE, 50),
            ExtMove::new(Move::NONE, 200),
            ExtMove::new(Move::NONE, 10),
            ExtMove::new(Move::NONE, 150),
        ];

        let len = moves.len();
        let sorted_end = partial_insertion_sort(&mut moves, len, 100);

        // index 1 以降で >= 100 の手は 200, 150 の2つ → sorted_end=2
        assert_eq!(sorted_end, 2);
        // sorted 領域 [0..=2] が降順
        assert_eq!(moves[0].value, 200);
        assert_eq!(moves[1].value, 150);
        assert_eq!(moves[2].value, 100);
    }

    #[test]
    fn test_partial_insertion_sort_boundary_value() {
        // 境界値テスト: value == limit の手は閾値以上として扱われる
        let mut moves = vec![
            ExtMove::new(Move::NONE, 99),
            ExtMove::new(Move::NONE, 100), // ちょうど閾値
            ExtMove::new(Move::NONE, 101),
        ];

        let len = moves.len();
        let sorted_end = partial_insertion_sort(&mut moves, len, 100);

        // index 1 以降で >= 100: 100, 101 の2つ → sorted_end=2
        assert_eq!(sorted_end, 2);
        assert_eq!(moves[0].value, 101);
        assert_eq!(moves[1].value, 100);
        assert_eq!(moves[2].value, 99);
    }

    #[test]
    fn test_partial_insertion_sort_large_array() {
        // 20要素の配列
        let mut moves: Vec<ExtMove> = (0..20).map(|i| ExtMove::new(Move::NONE, i * 10)).collect();

        let len = moves.len();
        // limit=100: index 1 以降で >= 100 の手は 100, 110, ..., 190 の10個
        let sorted_end = partial_insertion_sort(&mut moves, len, 100);

        assert_eq!(sorted_end, 10);
        // sorted 領域先頭に降順配置
        assert_eq!(moves[0].value, 190);
        assert_eq!(moves[1].value, 180);
        assert_eq!(moves[9].value, 100);
    }

    #[test]
    fn test_partial_insertion_sort_no_good_moves() {
        // 閾値を満たす手が0個の場合
        let mut moves = vec![
            ExtMove::new(Move::NONE, 10),
            ExtMove::new(Move::NONE, 20),
            ExtMove::new(Move::NONE, 30),
        ];

        let len = moves.len();
        let sorted_end = partial_insertion_sort(&mut moves, len, 100);

        assert_eq!(sorted_end, 0);
        // 順序は変わらない（swapは発生しない）
    }

    #[test]
    fn test_partial_insertion_sort_all_good_moves() {
        // 全ての手が閾値以上の場合
        let mut moves = vec![
            ExtMove::new(Move::NONE, 100),
            ExtMove::new(Move::NONE, 200),
            ExtMove::new(Move::NONE, 150),
        ];

        let len = moves.len();
        let sorted_end = partial_insertion_sort(&mut moves, len, 50);

        // index 1 以降で >= 50: 200, 150 の2つ → sorted_end=2
        assert_eq!(sorted_end, 2);
        // 全てソートされる（降順）
        assert_eq!(moves[0].value, 200);
        assert_eq!(moves[1].value, 150);
        assert_eq!(moves[2].value, 100);
    }

    #[test]
    fn test_partial_insertion_sort_full_sort() {
        // limit = i32::MIN の場合は全要素ソート
        let mut moves = vec![
            ExtMove::new(Move::NONE, 50),
            ExtMove::new(Move::NONE, -100),
            ExtMove::new(Move::NONE, 200),
            ExtMove::new(Move::NONE, 0),
        ];

        let len = moves.len();
        let sorted_end = partial_insertion_sort(&mut moves, len, i32::MIN);

        // index 1 以降の全3要素が >= i32::MIN → sorted_end=3
        assert_eq!(sorted_end, 3);
        // 全体がソートされる（降順）
        assert_eq!(moves[0].value, 200);
        assert_eq!(moves[1].value, 50);
        assert_eq!(moves[2].value, 0);
        assert_eq!(moves[3].value, -100);
    }

    #[test]
    fn test_partial_insertion_sort_empty() {
        // 空配列
        let mut moves: Vec<ExtMove> = vec![];
        let sorted_end = partial_insertion_sort(&mut moves, 0, 100);
        assert_eq!(sorted_end, 0);
    }

    #[test]
    fn test_partial_insertion_sort_single_element() {
        // 1要素の配列: ループ 1..1 は実行されない → sorted_end=0
        let mut moves = vec![ExtMove::new(Move::NONE, 150)];
        let sorted_end = partial_insertion_sort(&mut moves, 1, 100);
        assert_eq!(sorted_end, 0);
        assert_eq!(moves[0].value, 150);

        let mut moves2 = vec![ExtMove::new(Move::NONE, 50)];
        let sorted_end2 = partial_insertion_sort(&mut moves2, 1, 100);
        assert_eq!(sorted_end2, 0);
    }

    #[test]
    fn test_piece_value() {
        assert_eq!(piece_value(Piece::B_PAWN), 90);
        assert_eq!(piece_value(Piece::W_GOLD), 540);
        assert_eq!(piece_value(Piece::B_ROOK), 990);
        assert_eq!(piece_value(Piece::W_HORSE), 945);
        assert_eq!(piece_value(Piece::W_DRAGON), 1395);
    }

    // =========================================================================
    // 敵玉近傍 drop bonus のテスト (2026-06-16)
    // =========================================================================

    use crate::types::{File, Rank};

    /// 玉升を (file, rank) で構築するヘルパー (= 0-index)。
    fn sq(file: usize, rank: usize) -> Square {
        Square::new(File::ALL[file], Rank::ALL[rank])
    }

    // テストは mode 3 (= full、 旧 dropmo 相当 = zone 無条件 + line) と danger=true で
    // 既存の zone tier / line bonus セマンティクスを検証する。
    const FULL: i32 = 3;

    #[test]
    fn test_drop_bonus_non_drop_is_zero() {
        // 駒打ちでない手 (= 移動手) は 0。
        let king = sq(4, 4); // 5五
        let zone = king_5x5_mask(king);
        let m = Move::new_move(sq(0, 0), sq(0, 1), false);
        assert_eq!(king_zone_drop_bonus(m, king, zone, Bitboard::EMPTY, FULL, true), 0);
    }

    #[test]
    fn test_drop_bonus_heavy_in_zone() {
        // 敵玉 5五 の 5x5 zone 内 (= 4六 等) への飛車打ちは heavy bonus 以上。
        let king = sq(4, 4); // 5五
        let zone = king_5x5_mask(king);
        let to = sq(3, 5); // 4六 (file diff 1, rank diff 1 → zone 内)
        assert!(zone.contains(to));
        let m = Move::new_drop(PieceType::Rook, to);
        let bonus = king_zone_drop_bonus(m, king, zone, Bitboard::EMPTY, FULL, true);
        assert!(
            bonus >= KING_ZONE_DROP_BONUS_HEAVY,
            "heavy drop in zone should get >= heavy bonus, got {bonus}"
        );
    }

    #[test]
    fn test_drop_bonus_light_lower_than_heavy() {
        // zone 内でも歩は light、 飛は heavy。 順序が保たれる。
        let king = sq(4, 4);
        let zone = king_5x5_mask(king);
        let to = sq(4, 6); // 5七 (同 file、 rank diff 2 → zone 内)
        assert!(zone.contains(to));
        // occupancy 空だと飛の利きが玉升 5五 に届き line bonus も付くため、
        // 純粋な zone tier 比較は別升で行う。 ここでは light < heavy のみ確認。
        let light = king_zone_drop_bonus(
            Move::new_drop(PieceType::Pawn, to),
            king,
            zone,
            Bitboard::EMPTY,
            FULL,
            true,
        );
        assert_eq!(light, KING_ZONE_DROP_BONUS_LIGHT);
        let mid = king_zone_drop_bonus(
            Move::new_drop(PieceType::Gold, to),
            king,
            zone,
            Bitboard::EMPTY,
            FULL,
            true,
        );
        assert_eq!(mid, KING_ZONE_DROP_BONUS_MID);
        assert!(light < mid);
    }

    #[test]
    fn test_drop_bonus_outside_zone_no_line_is_zero() {
        // zone 外かつ利きが玉に届かない歩打ちは 0。
        let king = sq(0, 0); // 1一
        let zone = king_5x5_mask(king);
        let to = sq(8, 8); // 9九 (完全に zone 外)
        assert!(!zone.contains(to));
        let m = Move::new_drop(PieceType::Pawn, to);
        // occupancy をフルに埋めて利きを遮断 (歩は元々遠隔利きなし)。
        assert_eq!(king_zone_drop_bonus(m, king, zone, Bitboard::EMPTY, FULL, true), 0);
    }

    #[test]
    fn test_drop_bonus_rook_line_to_king() {
        // 飛を敵玉と同 file の zone 外升に打ち、 利きが玉升に届く場合は line bonus。
        let king = sq(4, 0); // 5一
        let zone = king_5x5_mask(king);
        let to = sq(4, 6); // 5七 (同 file、 rank diff 6 → zone 外)
        assert!(!zone.contains(to));
        let m = Move::new_drop(PieceType::Rook, to);
        // 盤面空 → 飛の縦利きが 5一 に届く。
        let bonus = king_zone_drop_bonus(m, king, zone, Bitboard::EMPTY, FULL, true);
        assert_eq!(
            bonus, KING_LINE_DROP_BONUS,
            "rook line to king should get line bonus, got {bonus}"
        );
    }

    #[test]
    fn test_drop_bonus_in_zone_with_line_stacks() {
        // zone 内 かつ 利きが玉に届く飛打ちは heavy + line を加算。
        let king = sq(4, 4); // 5五
        let zone = king_5x5_mask(king);
        let to = sq(4, 6); // 5七 (同 file zone 内、 縦利きが 5五 に届く)
        assert!(zone.contains(to));
        let m = Move::new_drop(PieceType::Rook, to);
        let bonus = king_zone_drop_bonus(m, king, zone, Bitboard::EMPTY, FULL, true);
        assert_eq!(bonus, KING_ZONE_DROP_BONUS_HEAVY + KING_LINE_DROP_BONUS);
    }

    #[test]
    fn test_drop_bonus_mode0_off() {
        // mode 0 = 全て 0 (= base 相当)。
        let king = sq(4, 4);
        let zone = king_5x5_mask(king);
        let to = sq(3, 5); // zone 内
        let m = Move::new_drop(PieceType::Rook, to);
        assert_eq!(king_zone_drop_bonus(m, king, zone, Bitboard::EMPTY, 0, true), 0);
    }

    #[test]
    fn test_drop_bonus_mode1_line_only() {
        // mode 1 = line-only。 zone 内でも line に乗らなければ 0。
        let king = sq(4, 4); // 5五
        let zone = king_5x5_mask(king);
        // 金を zone 内 (line に乗らない升) に打つ → mode 1 では 0。
        let to_gold = sq(3, 3); // 4四 (zone 内、 玉と非同筋非同段非同斜め)
        assert!(zone.contains(to_gold));
        let m_gold = Move::new_drop(PieceType::Gold, to_gold);
        assert_eq!(
            king_zone_drop_bonus(m_gold, king, zone, Bitboard::EMPTY, 1, true),
            0,
            "mode1: zone-only gold should get 0"
        );
        // 飛を同 file zone 外に打ち line に乗る → line bonus のみ。
        let to_rook = sq(4, 7); // 5八 (同 file、 zone 外)
        assert!(!zone.contains(to_rook));
        let m_rook = Move::new_drop(PieceType::Rook, to_rook);
        assert_eq!(
            king_zone_drop_bonus(m_rook, king, zone, Bitboard::EMPTY, 1, true),
            KING_LINE_DROP_BONUS,
            "mode1: rook line should get line bonus"
        );
    }

    #[test]
    fn test_drop_bonus_mode2_danger_gate() {
        // mode 2 = danger-gated。 zone bonus は king_in_danger=true のときのみ。
        let king = sq(4, 4); // 5五
        let zone = king_5x5_mask(king);
        let to = sq(3, 3); // 4四 (zone 内、 line に乗らない)
        let m = Move::new_drop(PieceType::Gold, to);
        // danger=false → zone bonus 無効 (line も乗らないので 0)。
        assert_eq!(
            king_zone_drop_bonus(m, king, zone, Bitboard::EMPTY, 2, false),
            0,
            "mode2 + safe: zone gold should be gated to 0"
        );
        // danger=true → zone bonus 有効。
        assert_eq!(
            king_zone_drop_bonus(m, king, zone, Bitboard::EMPTY, 2, true),
            KING_ZONE_DROP_BONUS_MID,
            "mode2 + danger: zone gold should get mid bonus"
        );
    }

    #[test]
    fn test_drop_bonus_mode2_line_always_on() {
        // mode 2 でも line bonus は danger に関係なく常時。
        let king = sq(4, 0); // 5一
        let zone = king_5x5_mask(king);
        let to = sq(4, 6); // 5七 (同 file zone 外、 line に乗る)
        let m = Move::new_drop(PieceType::Rook, to);
        assert_eq!(
            king_zone_drop_bonus(m, king, zone, Bitboard::EMPTY, 2, false),
            KING_LINE_DROP_BONUS,
            "mode2: line bonus must fire even when not in danger"
        );
    }

    /// partial_insertion_sort のソート順が YO と同型であることを検証
    #[test]
    fn test_partial_insertion_sort_order() {
        // YaneuraOu準拠: index 0 は初期 sorted 領域、index 1 以降を走査
        let mut moves = vec![
            ExtMove::new(Move::NONE, 100),  // index 0: 初期 sorted 領域
            ExtMove::new(Move::NONE, -200), // index 1: < 0, skip
            ExtMove::new(Move::NONE, 50),   // index 2: >= 0, sorted_end=1
            ExtMove::new(Move::NONE, 200),  // index 3: >= 0, sorted_end=2
            ExtMove::new(Move::NONE, -100), // index 4: < 0, skip
        ];

        let len = moves.len();
        let sorted_end = partial_insertion_sort(&mut moves, len, 0);

        // index 1 以降で >= 0 の手は 50, 200 の2つ
        assert_eq!(sorted_end, 2);

        // sorted 領域 [0..=2] は降順
        assert_eq!(moves[0].value, 200);
        assert_eq!(moves[1].value, 100);
        assert_eq!(moves[2].value, 50);
        // 残りは閾値未満
        assert!(moves[3].value < 0);
        assert!(moves[4].value < 0);
    }
}
