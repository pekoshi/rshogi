//! 終盤判定 + SearchMode 決定
//!
//! 局面の特徴量を計算し、GamePhase / SearchMode を決定する。
//! ヒステリシス付きのフェーズ遷移を提供する。

use rshogi_core::bitboard::king_effect;
use rshogi_core::position::Position;
use rshogi_core::types::{Color, PieceType};

/// ゲームフェーズ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamePhase {
    Opening,
    Middlegame,
    Endgame,
}

/// 探索モード（Ord 付き: Normal < Endgame < MateThreat < ForcedMate）
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum SearchMode {
    #[default]
    Normal,
    Endgame,
    MateThreat,
    ForcedMate,
}

/// 終盤判定用特徴量 (進行度精度向上のため engagement / promotion 追加)
#[derive(Debug, Clone)]
pub struct EndgameFeatures {
    /// 盤上 material / MAX_MATERIAL（低いほど終盤）
    pub material_phase: f32,
    /// 玉間チェビシェフ距離の近さ（0.0=遠い, 1.0=隣接）
    pub king_distance: f32,
    /// 手番側玉の移動可能マス数の少なさ
    pub mobility: f32,
    /// 相手利き優勢度
    pub attack_pressure: f32,
    /// 評価値の絶対値の大きさ
    pub eval_magnitude: f32,
    /// TT cutoff rate（MCTS では 0.0 固定）
    pub tt_cutoff_rate: f32,
    /// 王手頻度（MCTS ルートで追跡、初期は 0.0）
    pub check_frequency: f32,
    /// ルート子のうち王手手の比率（初期は 0.0）
    pub mcts_check_bias: f32,
    /// 取り捌き量 (両手番の持ち駒合計 / 正規化)。 0=未交換 (序盤)、 1=多数交換 (中終盤)
    pub engagement: f32,
    /// 成駒進捗 (盤上の成駒数 / 正規化)。 0=成駒なし (序盤)、 1=多数成駒 (終盤近辺)
    pub promotion_progress: f32,
}

/// フェーズ追跡器（ヒステリシス用状態保持）
#[derive(Debug)]
pub struct PhaseTracker {
    current_phase: GamePhase,
    /// 直近 determine_phase で観測した連続スコア (0.0..=1.0)
    /// 進行度 (progression) の参照用に exposed
    pub last_score: f32,
}

/// ヒステリシス閾値 (Middlegame ↔ Endgame)
pub const ENDGAME_ENTER_THRESHOLD: f32 = 0.65;
pub const ENDGAME_EXIT_THRESHOLD: f32 = 0.45;

/// ヒステリシス閾値 (Opening ↔ Middlegame)
pub const OPENING_EXIT_THRESHOLD: f32 = 0.20;
pub const OPENING_ENTER_THRESHOLD: f32 = 0.15;

/// material 正規化用の最大値（平手初期局面の概算）
const MAX_MATERIAL: f32 = 14_175.0;

/// 駒種ごとの基本評価値（detector.rs と同一）
const fn piece_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => 90,
        PieceType::Lance => 315,
        PieceType::Knight => 405,
        PieceType::Silver => 495,
        PieceType::Bishop => 855,
        PieceType::Rook => 990,
        PieceType::Gold => 540,
        PieceType::ProPawn | PieceType::ProLance | PieceType::ProKnight | PieceType::ProSilver => {
            540
        }
        PieceType::Horse => 945,
        PieceType::Dragon => 1395,
        PieceType::King => 0,
    }
}

/// material 計算対象の駒種（King 除外）
const MATERIAL_PIECE_TYPES: [PieceType; 13] = [
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

/// 特徴量の重み（合計 ≈ 1.0、 score 範囲は [0,1] にクランプ）
/// 進行度精度向上のため engagement / promotion を追加、 既存重みを再分配
const W_MATERIAL: f32 = 0.20;
const W_KING_DIST: f32 = 0.10;
const W_MOBILITY: f32 = 0.10;
const W_ATTACK: f32 = 0.10;
const W_EVAL_MAG: f32 = 0.10;
const W_TT_CUTOFF: f32 = 0.00; // MCTS では未使用
const W_CHECK_FREQ: f32 = 0.05;
const W_CHECK_BIAS: f32 = 0.05;
const W_ENGAGEMENT: f32 = 0.20;
const W_PROMOTION: f32 = 0.10;

/// engagement 正規化用 (10 駒交換で 1.0 飽和)
const MAX_HAND_PIECES: f32 = 10.0;
/// promotion 正規化用 (6 成駒で 1.0 飽和)
const MAX_PROMOTED_PIECES: f32 = 6.0;

/// 盤上 material を計算（両手番合算、King 除外）
pub fn compute_board_material(pos: &Position) -> i32 {
    let mut total = 0i32;
    for &pt in &MATERIAL_PIECE_TYPES {
        let count = pos.pieces(Color::Black, pt).count() + pos.pieces(Color::White, pt).count();
        total += count as i32 * piece_value(pt);
    }
    total
}

/// 終盤特徴量を計算
pub fn compute_features(pos: &Position, eval_cp: i32) -> EndgameFeatures {
    let us = pos.side_to_move();
    let them = !us;
    let our_king = pos.king_square(us);
    let their_king = pos.king_square(them);

    // material_phase: 1.0 = 序盤（駒多い）→ 低いほど終盤
    // 終盤スコアとしては 1 - material_phase を使う
    let board_mat = compute_board_material(pos);
    let material_phase = 1.0 - (board_mat as f32 / MAX_MATERIAL).clamp(0.0, 1.0);

    // king_distance: チェビシェフ距離（0-8）→ 近いほど終盤的
    let file_diff = (our_king.file() as i32 - their_king.file() as i32).unsigned_abs() as f32;
    let rank_diff = (our_king.rank() as i32 - their_king.rank() as i32).unsigned_abs() as f32;
    let chebyshev = file_diff.max(rank_diff);
    let king_distance = 1.0 - chebyshev / 8.0;

    // mobility: 手番側玉の移動可能マス数の少なさ
    let king_moves = king_effect(our_king) & !pos.occupied();
    let free_count = king_moves.count() as f32;
    let mobility = 1.0 - (free_count / 8.0).min(1.0);

    // attack_pressure: 相手利き優勢度
    let neighbors = king_effect(our_king);
    let mut effect_diff: i32 = 0;
    for sq in neighbors.iter() {
        effect_diff += pos.board_effect(them, sq) as i32 - pos.board_effect(us, sq) as i32;
    }
    let attack_pressure = (effect_diff as f32 / 10.0).clamp(0.0, 1.0);

    // eval_magnitude: 評価値の絶対値の大きさ
    let eval_magnitude = (eval_cp.unsigned_abs() as f32 / 3000.0).min(1.0);

    // engagement: 取り捌き量 (両手番の手駒合計、 開戦/中盤の指標)
    let total_hand_pieces: u32 = [Color::Black, Color::White]
        .iter()
        .map(|&c| {
            let h = pos.hand(c);
            h.count(PieceType::Pawn)
                + h.count(PieceType::Lance)
                + h.count(PieceType::Knight)
                + h.count(PieceType::Silver)
                + h.count(PieceType::Gold)
                + h.count(PieceType::Bishop)
                + h.count(PieceType::Rook)
        })
        .sum();
    let engagement = (total_hand_pieces as f32 / MAX_HAND_PIECES).clamp(0.0, 1.0);

    // promotion_progress: 盤上成駒数 (中終盤の指標)
    let promoted_count: u32 = [
        PieceType::ProPawn,
        PieceType::ProLance,
        PieceType::ProKnight,
        PieceType::ProSilver,
        PieceType::Horse,
        PieceType::Dragon,
    ]
    .iter()
    .map(|&pt| pos.pieces(Color::Black, pt).count() + pos.pieces(Color::White, pt).count())
    .sum::<u32>();
    let promotion_progress = (promoted_count as f32 / MAX_PROMOTED_PIECES).clamp(0.0, 1.0);

    EndgameFeatures {
        material_phase,
        king_distance,
        mobility,
        attack_pressure,
        eval_magnitude,
        tt_cutoff_rate: 0.0,
        check_frequency: 0.0,
        mcts_check_bias: 0.0,
        engagement,
        promotion_progress,
    }
}

/// 終盤スコアを計算（重み付き合計、[0,1] にクランプ）
pub fn compute_endgame_score(f: &EndgameFeatures) -> f32 {
    let raw = W_MATERIAL * f.material_phase
        + W_KING_DIST * f.king_distance
        + W_MOBILITY * f.mobility
        + W_ATTACK * f.attack_pressure
        + W_EVAL_MAG * f.eval_magnitude
        + W_TT_CUTOFF * f.tt_cutoff_rate
        + W_CHECK_FREQ * f.check_frequency
        + W_CHECK_BIAS * f.mcts_check_bias
        + W_ENGAGEMENT * f.engagement
        + W_PROMOTION * f.promotion_progress;
    raw.clamp(0.0, 1.0)
}

/// ヒステリシス付きフェーズ決定 (3-phase: Opening / Middlegame / Endgame)
///
/// score < OPENING_ENTER_THRESHOLD : Opening
/// OPENING_EXIT_THRESHOLD <= score < ENDGAME_ENTER_THRESHOLD : Middlegame
/// score >= ENDGAME_ENTER_THRESHOLD : Endgame
///
/// 各境界でヒステリシス (enter/exit 閾値分離) でチャタリング防止。
pub fn determine_phase(score: f32, tracker: &mut PhaseTracker) -> GamePhase {
    tracker.last_score = score;
    match tracker.current_phase {
        GamePhase::Opening => {
            if score >= ENDGAME_ENTER_THRESHOLD {
                tracker.current_phase = GamePhase::Endgame;
            } else if score >= OPENING_EXIT_THRESHOLD {
                tracker.current_phase = GamePhase::Middlegame;
            }
        }
        GamePhase::Middlegame => {
            if score >= ENDGAME_ENTER_THRESHOLD {
                tracker.current_phase = GamePhase::Endgame;
            } else if score < OPENING_ENTER_THRESHOLD {
                tracker.current_phase = GamePhase::Opening;
            }
        }
        GamePhase::Endgame => {
            if score < ENDGAME_EXIT_THRESHOLD {
                tracker.current_phase = if score < OPENING_ENTER_THRESHOLD {
                    GamePhase::Opening
                } else {
                    GamePhase::Middlegame
                };
            }
        }
    }
    tracker.current_phase
}

/// 進行度スコア (0.0=完全な序盤、 1.0=詰みレベル終盤)
///
/// `compute_endgame_score` の連続値そのまま。 ヒステリシス影響なし。
/// Hybrid 探索の routing 判断や policy 補完強度の調整に使用。
pub fn progress_score(features: &EndgameFeatures) -> f32 {
    compute_endgame_score(features)
}

/// score → GamePhase の即決変換 (ヒステリシスなし)
pub fn score_to_phase_immediate(score: f32) -> GamePhase {
    if score >= ENDGAME_ENTER_THRESHOLD {
        GamePhase::Endgame
    } else if score >= OPENING_EXIT_THRESHOLD {
        GamePhase::Middlegame
    } else {
        GamePhase::Opening
    }
}

/// SearchMode を決定
///
/// - `score`: endgame_score [0,1]
/// - `in_check`: 手番側が王手されているか
/// - `eval_cp`: 評価値 (centipawn)
/// - `check_freq`: 王手頻度
/// - `legal_check_moves`: 合法な王手手の数
/// - `mcts_top_check_ratio`: ルート上位ノードの王手比率
pub fn determine_search_mode(
    score: f32,
    in_check: bool,
    eval_cp: i32,
    check_freq: f32,
    legal_check_moves: u32,
    mcts_top_check_ratio: f32,
) -> SearchMode {
    // ForcedMate: 王手されている + 評価値が詰みレベル + 終盤深い
    if in_check && eval_cp.abs() > 2000 && score >= 0.80 {
        return SearchMode::ForcedMate;
    }

    // MateThreat: 王手手が多い or チェック頻度が高い + 終盤
    if score >= 0.65 && (check_freq >= 0.3 || legal_check_moves >= 3 || mcts_top_check_ratio >= 0.4)
    {
        return SearchMode::MateThreat;
    }

    // Endgame: 終盤スコアが閾値以上
    if score >= ENDGAME_ENTER_THRESHOLD {
        return SearchMode::Endgame;
    }

    SearchMode::Normal
}

impl PhaseTracker {
    /// 新しいトラッカーを作成（初期フェーズ = Opening）
    pub fn new() -> Self {
        Self {
            current_phase: GamePhase::Opening,
            last_score: 0.0,
        }
    }

    /// 現フェーズ参照
    pub fn phase(&self) -> GamePhase {
        self.current_phase
    }

    /// 直近の連続進行度スコア
    pub fn progression(&self) -> f32 {
        self.last_score
    }

    /// 現在のフェーズを取得 (`phase` のエイリアス、 後方互換用)
    pub fn current_phase(&self) -> GamePhase {
        self.current_phase
    }
}

impl Default for PhaseTracker {
    fn default() -> Self {
        Self::new()
    }
}
