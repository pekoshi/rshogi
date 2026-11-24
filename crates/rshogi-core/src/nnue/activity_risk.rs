//! Activity-Risk Interaction Head (= 2026-06-06、 中盤楽観バイアス補正)。
//!
//! # 仮説と動機
//!
//! Floodgate 強豪戦 21 局、 30〜82 手目の 147 局面を p2 と Suisho5 で再評価した結果:
//! - 敗局中盤で p2 が平均 **+49cp 楽観**
//! - depth 14 でも最大 **+557cp** の評価差が残る
//! - 楽観局面では自玉周辺の敵利きが **約 1.5〜1.7 倍**
//! - 自玉の歩の盾が薄い
//! - 同時に自軍大駒の可動性は **約 1.18 倍**
//!
//! 仮説: p2 は「大駒が働いているが、 自玉への継続的圧力のため手番を渡せない」
//! 中盤局面を過大評価している。
//!
//! # 設計
//!
//! PSQT scalar auxiliary head として、 局面ごと・perspective ごとに **1 entry** を発火する。
//!
//! ## Feature index
//!
//! `[perspective 2][bucket 8][major_activity 8][king_pressure 8][shield 4]`
//! → 2 × 8 × 8 × 8 × 4 = **4,096 i16 weight** (= ~8 KB)
//!
//! ## Disk layout
//!
//! `[bucket 8][perspective 2][activity 8][pressure 8][shield 4]` row-major。
//! 既存 PairShape (`P2PAIR\0\0`) section 直後に `P2ACTR\0\0` magic で配置する。
//!
//! ## 評価式
//!
//! 各 perspective で該当 weight を 1 個 lookup し、
//!
//! ```text
//! black 寄与 - white 寄与
//! ```
//!
//! として PSQT scalar 値へ加算、 既存 PairShape と同様に `fv_scale` 適用前へ統合する。
//! output scale は独立定数 `ACTIVITY_RISK_OUTPUT_SCALE = 1` で初期化、 学習・量子化側で吸収する。
//!
//! ## bucket 仕様
//!
//! ### major_activity (= 8 bucket)
//!
//! perspective 側の **飛・角・馬・竜** について、 現在位置から到達可能な升数を合計して bucket 化。
//! 初版は pseudo mobility を使い、 以下を数える:
//! - 駒の利きのうち、 自軍駒が占有していない升
//! - 敵駒の升は含む
//! - pin や王手放置による legal 判定は行わない
//! - 合計を bucket 境界 `0, 1-3, 4-6, 7-9, 10-12, 13-15, 16-19, 20+` で分類
//!
//! ### king_pressure (= 8 bucket)
//!
//! perspective 側の玉を中心とする **3x3 領域** (= 玉自身を含む、 盤外除外) について、
//! 敵側の利き数を合計する。
//! - 各升について `attackers_to_occ()` の敵駒数を popcount
//! - 合計を `0, 1, 2, 3, 4, 5, 6, 7+` で分類
//! - 占有状態によらず 9 升すべて対象
//!
//! ### shield (= 4 bucket)
//!
//! perspective 側の玉を中心とする **5x5 領域** (= 玉自身を除く、 盤外除外) にある、
//! perspective 側の以下の駒数を数える:
//! - Pawn / Silver / Gold / ProPawn / ProLance / ProKnight / ProSilver
//!
//! bucket 境界 = `0-2 / 3-4 / 5-6 / 7+`
//!
//! 「歩の盾」 だけだと囲い種類への依存が強いため、 初版は守備小駒密度とする。
//!
//! # 注意
//!
//! - 初版は per-eval full compute (= 各 evaluate で 4 計算)、 NPS 影響は実測する
//! - PairShape 比 NPS 97% 未満なら、 実装完了扱いにせず差分更新案を検討する
//! - feature gate `nnue-activity-risk-psqt` 無効時は hot path への分岐・追加コストを残さない

use crate::bitboard::{
    Bitboard, bishop_effect, dragon_effect, gold_effect, horse_effect, king_effect, rook_effect,
    silver_effect,
};
use crate::position::Position;
use crate::types::{Color, PieceType, Square};

// =============================================================================
// 定数
// =============================================================================

/// perspective 数 (= black + white)
pub const ACTIVITY_RISK_NUM_PERSPECTIVES: usize = 2;
/// bucket 数 (= PSQT bucket と同期、 8 = progress8 系)
pub const ACTIVITY_RISK_NUM_BUCKETS: usize = 8;
/// major_activity bucket 数
pub const ACTIVITY_RISK_NUM_ACTIVITY: usize = 8;
/// king_pressure bucket 数
pub const ACTIVITY_RISK_NUM_PRESSURE: usize = 8;
/// shield bucket 数
pub const ACTIVITY_RISK_NUM_SHIELD: usize = 4;

/// 1 perspective × 1 bucket あたりの weight 数 (= activity × pressure × shield)
pub const ACTIVITY_RISK_PER_PB_DIM: usize =
    ACTIVITY_RISK_NUM_ACTIVITY * ACTIVITY_RISK_NUM_PRESSURE * ACTIVITY_RISK_NUM_SHIELD;

/// 全 weight 数 = 2 × 8 × 8 × 8 × 4 = 4096
pub const ACTIVITY_RISK_WEIGHT_DIM: usize =
    ACTIVITY_RISK_NUM_BUCKETS * ACTIVITY_RISK_NUM_PERSPECTIVES * ACTIVITY_RISK_PER_PB_DIM;

/// disk section magic (= 他 section と衝突しない 8 byte、 P2 prefix + ACTR + null padding)
pub const ACTIVITY_RISK_SUB_MAGIC: [u8; 8] = *b"P2ACTR\0\0";

/// section version (= 3、 2026-06-08 で v1 base 維持 + tactical 1024 weight 追加)。
///
/// v1 = 1 (= 玉 3x3 内 敵利き合計 0..=7+、 4096 weight)
/// v2 = 2 (= 8 排他クラス、 4096 weight、 廃止確定)
/// v3 = 3 (= v1 base 4096 + tactical 1024 = 5120 weight、 heavy_attack + escape + hand_check 追加)
/// v1 ckpt は v3 build で **load 不可**、 ただし v1 → v3 変換 tool で base copy + tactical 0 で v3 ckpt 作成可能。
pub const ACTIVITY_RISK_VERSION: u32 = 3;

/// v3 base 重み数 (= v1 と同 4096、 activity × pressure_v1 × shield × bucket × persp)
pub const ACTIVITY_RISK_V3_BASE_DIM: usize = ACTIVITY_RISK_WEIGHT_DIM;

/// v3 tactical 軸 class 数 (= heavy_attack / escape / hand_check 各 4 class)
pub const ACTIVITY_RISK_V3_NUM_HEAVY: usize = 4;
pub const ACTIVITY_RISK_V3_NUM_ESCAPE: usize = 4;
pub const ACTIVITY_RISK_V3_NUM_HAND_CHECK: usize = 4;

/// v3 tactical 重み数 (= bucket × persp × heavy × escape × hand_check = 8 × 2 × 4 × 4 × 4 = 1024)
pub const ACTIVITY_RISK_V3_TACTICAL_DIM: usize = ACTIVITY_RISK_NUM_BUCKETS
    * ACTIVITY_RISK_NUM_PERSPECTIVES
    * ACTIVITY_RISK_V3_NUM_HEAVY
    * ACTIVITY_RISK_V3_NUM_ESCAPE
    * ACTIVITY_RISK_V3_NUM_HAND_CHECK;

/// v3 全 重み数 (= base 4096 + tactical 1024 = 5120)
pub const ACTIVITY_RISK_V3_WEIGHT_DIM: usize =
    ACTIVITY_RISK_V3_BASE_DIM + ACTIVITY_RISK_V3_TACTICAL_DIM;

/// section の expected weight count (= dimensionality check 用、 v3 = 5120)
pub const ACTIVITY_RISK_COUNT: u32 = ACTIVITY_RISK_V3_WEIGHT_DIM as u32;

/// 評価合成時の output scale (= 初版 1、 学習・量子化側で吸収)
pub const ACTIVITY_RISK_OUTPUT_SCALE: i32 = 1;

// =============================================================================
// bucket 境界
// =============================================================================

/// major_activity (= 飛角馬竜 pseudo mobility 合計) を 0..=7 の bucket に変換する。
///
/// 境界 (= 半開区間 `[lower, upper)`):
/// - 0: `[0, 1)` (= activity = 0)
/// - 1: `[1, 4)` (= 1, 2, 3)
/// - 2: `[4, 7)` (= 4, 5, 6)
/// - 3: `[7, 10)` (= 7, 8, 9)
/// - 4: `[10, 13)` (= 10, 11, 12)
/// - 5: `[13, 16)` (= 13, 14, 15)
/// - 6: `[16, 20)` (= 16, 17, 18, 19)
/// - 7: `[20, ∞)` (= 20+)
#[inline]
pub fn activity_to_bucket(activity: usize) -> usize {
    if activity == 0 {
        0
    } else if activity < 4 {
        1
    } else if activity < 7 {
        2
    } else if activity < 10 {
        3
    } else if activity < 13 {
        4
    } else if activity < 16 {
        5
    } else if activity < 20 {
        6
    } else {
        7
    }
}

/// king_pressure (= 玉 3x3 内の敵利き合計) を 0..=7 の bucket に変換する。
///
/// 境界: 0, 1, 2, 3, 4, 5, 6, 7+ (= clamp(p, 0, 7))
#[inline]
pub fn pressure_to_bucket(pressure: usize) -> usize {
    pressure.min(7)
}

/// shield (= 玉 5x5 内の自軍小駒数) を 0..=3 の bucket に変換する。
///
/// 境界 (= 半開区間):
/// - 0: `[0, 3)` (= 0, 1, 2)
/// - 1: `[3, 5)` (= 3, 4)
/// - 2: `[5, 7)` (= 5, 6)
/// - 3: `[7, ∞)` (= 7+)
#[inline]
pub fn shield_to_bucket(shield: usize) -> usize {
    if shield < 3 {
        0
    } else if shield < 5 {
        1
    } else if shield < 7 {
        2
    } else {
        3
    }
}

// =============================================================================
// Index 契約
// =============================================================================

/// disk layout flatten index 計算 (= `[bucket][perspective][activity][pressure][shield]` row-major)。
///
/// disk format での読み書きはこの順で sequential、 hot path 推論時もこの index で lookup する。
///
/// # 範囲
/// - `bucket < ACTIVITY_RISK_NUM_BUCKETS` (= 8)
/// - `perspective < ACTIVITY_RISK_NUM_PERSPECTIVES` (= 2)
/// - `activity < ACTIVITY_RISK_NUM_ACTIVITY` (= 8)
/// - `pressure < ACTIVITY_RISK_NUM_PRESSURE` (= 8)
/// - `shield < ACTIVITY_RISK_NUM_SHIELD` (= 4)
///
/// 返り値 = `[0, ACTIVITY_RISK_WEIGHT_DIM)`
#[inline]
pub const fn flatten_idx(
    bucket: usize,
    perspective: usize,
    activity: usize,
    pressure: usize,
    shield: usize,
) -> usize {
    let inner =
        ((activity * ACTIVITY_RISK_NUM_PRESSURE) + pressure) * ACTIVITY_RISK_NUM_SHIELD + shield;
    let pb = perspective * ACTIVITY_RISK_PER_PB_DIM + inner;
    bucket * ACTIVITY_RISK_NUM_PERSPECTIVES * ACTIVITY_RISK_PER_PB_DIM + pb
}

/// flatten_idx の逆変換 (= unit test 用)。
#[inline]
pub const fn unflatten_idx(idx: usize) -> (usize, usize, usize, usize, usize) {
    let per_persp = ACTIVITY_RISK_PER_PB_DIM;
    let per_bucket = ACTIVITY_RISK_NUM_PERSPECTIVES * per_persp;
    let bucket = idx / per_bucket;
    let rest = idx % per_bucket;
    let perspective = rest / per_persp;
    let inner = rest % per_persp;
    let shield = inner % ACTIVITY_RISK_NUM_SHIELD;
    let pr_act = inner / ACTIVITY_RISK_NUM_SHIELD;
    let pressure = pr_act % ACTIVITY_RISK_NUM_PRESSURE;
    let activity = pr_act / ACTIVITY_RISK_NUM_PRESSURE;
    (bucket, perspective, activity, pressure, shield)
}

// =============================================================================
// Weight 型
// =============================================================================

/// disk 上の weight 配列 (= [bucket][persp][activity][pressure][shield] row-major)。
pub type ActivityRiskWeights = [i16; ACTIVITY_RISK_WEIGHT_DIM];

// =============================================================================
// Feature 計算
// =============================================================================

/// perspective 側の major piece pseudo mobility (= 飛角馬竜) を計算する。
///
/// 各駒について `*_effect(sq, occupied)` の利き bitboard を取り、 自軍駒占有 sq を除いた
/// 升数を popcount。 全 major 駒の合計を返す。
///
/// 計算量 = 飛 (max 2) + 角 (max 2) + 馬 (max 2) + 竜 (max 2) = 最大 8 駒 × `*_effect` lookup。
///
/// pseudo mobility のため pin / 王手放置は判定しない (= legal 判定なし、 仕様通り)。
pub fn compute_major_activity(pos: &Position, persp: Color) -> usize {
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);
    let own = pos.pieces_c(persp);
    let mut total: usize = 0;

    // 飛車
    let rooks = pos.pieces(persp, PieceType::Rook);
    for sq in rooks.iter() {
        let attacks = rook_effect(sq, occupied);
        total += (attacks & !own).count() as usize;
    }
    // 角
    let bishops = pos.pieces(persp, PieceType::Bishop);
    for sq in bishops.iter() {
        let attacks = bishop_effect(sq, occupied);
        total += (attacks & !own).count() as usize;
    }
    // 馬
    let horses = pos.pieces(persp, PieceType::Horse);
    for sq in horses.iter() {
        let attacks = horse_effect(sq, occupied);
        total += (attacks & !own).count() as usize;
    }
    // 竜
    let dragons = pos.pieces(persp, PieceType::Dragon);
    for sq in dragons.iter() {
        let attacks = dragon_effect(sq, occupied);
        total += (attacks & !own).count() as usize;
    }
    total
}

/// perspective 側の玉中心 3x3 (= 玉自身を含む、 盤外除外) で敵側 attacker 合計を計算する。
///
/// **v1 (= 2026-06-06 以前)**: 玉 3x3 内 敵利き 合計 (= 0..=7+)、 [[ACTIVITY_RISK_VERSION]] 1 用。
/// v2 build では `compute_king_pressure_v2` を使う、 こちらは legacy 経路 (= 既存呼出 互換のため残置)。
pub fn compute_king_pressure(pos: &Position, persp: Color) -> usize {
    let king_sq = pos.king_square(persp);
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);
    let enemy = persp.opponent();
    let enemy_pieces = pos.pieces_c(enemy);

    // 9 sq = king_effect(king_sq) | king_sq 自身
    let zone = king_effect(king_sq) | Bitboard::from_square(king_sq);
    let mut total: usize = 0;
    for sq in zone.iter() {
        let attackers = pos.attackers_to_occ(sq, occupied) & enemy_pieces;
        total += attackers.count() as usize;
    }
    total
}

/// 敵 heavy piece bitboard を構築する (= v2 計算用 helper)。
///
/// heavy = Silver / Gold / Bishop / Rook / Horse / Dragon (+ Pro 系 4 種 = Gold 相当)。
/// Pawn / Lance / Knight / King は heavy に含めない。
///
/// 1 局面につき 1 回構築 + 各 sq loop で再利用、 NPS overhead 最小化。
#[inline]
pub fn enemy_heavy_mask(pos: &Position, enemy: Color) -> Bitboard {
    pos.pieces(enemy, PieceType::Silver)
        | pos.pieces(enemy, PieceType::Gold)
        | pos.pieces(enemy, PieceType::Bishop)
        | pos.pieces(enemy, PieceType::Rook)
        | pos.pieces(enemy, PieceType::Horse)
        | pos.pieces(enemy, PieceType::Dragon)
        | pos.pieces(enemy, PieceType::ProPawn)
        | pos.pieces(enemy, PieceType::ProLance)
        | pos.pieces(enemy, PieceType::ProKnight)
        | pos.pieces(enemy, PieceType::ProSilver)
}

/// perspective 側の king_pressure を **v2 8 排他クラス**で計算する (= 2026-06-06)。
///
/// 戻り値 = `0..=7` の class index、 そのまま `[ACTIVITY_RISK_NUM_PRESSURE]` 軸の index として使う
/// (= v2 では `pressure_to_bucket` は identity)。
///
/// # 8 排他クラス (= 優先順位順、 上から判定)
///
/// 0. `attacked_squares == 0`
/// 1. `attacked_squares == 1 && heavy_attacks == 0`
/// 2. `attacked_squares >= 2 && heavy_attacks == 0`
/// 3. `attacked_squares <= 2 && heavy_attacks == 1`
/// 4. `attacked_squares >= 3 && heavy_attacks == 1`
/// 5. `heavy_attacks == 2`
/// 6. `heavy_attacks >= 3 && attacked_squares <= 3`
/// 7. `heavy_attacks >= 3 && attacked_squares >= 4`
///
/// # 計算最適化
///
/// 1. `enemy_heavy_mask` を 1 回構築
/// 2. 玉 3x3 zone (= max 9 sq) 固定 loop
/// 3. 各 sq で `attackers_to_occ` 1 呼び + bitboard 演算 + popcount のみ
///
/// = `attackers_to_occ` を heavy piece type 毎に呼ぶ form 禁止 (= NPS 退行 risk)。
pub fn compute_king_pressure_v2(pos: &Position, persp: Color) -> usize {
    let king_sq = pos.king_square(persp);
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);
    let enemy = persp.opponent();
    let enemy_pieces = pos.pieces_c(enemy);
    let enemy_heavy = enemy_heavy_mask(pos, enemy);

    // 玉 3x3 zone (= max 9 sq、 盤外 自動除外)
    let zone = king_effect(king_sq) | Bitboard::from_square(king_sq);
    let mut attacked_squares: usize = 0;
    let mut heavy_attacks: usize = 0;

    for sq in zone.iter() {
        let attackers = pos.attackers_to_occ(sq, occupied);
        let enemy_attackers = attackers & enemy_pieces;
        if !enemy_attackers.is_empty() {
            attacked_squares += 1;
            let heavy_attackers = enemy_attackers & enemy_heavy;
            heavy_attacks += heavy_attackers.count() as usize;
        }
    }

    classify_pressure_v2(attacked_squares, heavy_attacks)
}

/// v2 8 排他クラス分類 (= 優先順位順)。
///
/// 入力 = `attacked_squares` (= 玉 3x3 内 敵利り升数) + `heavy_attacks` (= heavy piece 利り総数)。
/// 戻り値 = `0..=7` class index。
#[inline]
pub fn classify_pressure_v2(attacked_squares: usize, heavy_attacks: usize) -> usize {
    // 優先順位順に判定 (= 上から最初に match した class を採用)
    if attacked_squares == 0 {
        0
    } else if attacked_squares == 1 && heavy_attacks == 0 {
        1
    } else if attacked_squares >= 2 && heavy_attacks == 0 {
        2
    } else if attacked_squares <= 2 && heavy_attacks == 1 {
        3
    } else if attacked_squares >= 3 && heavy_attacks == 1 {
        4
    } else if heavy_attacks == 2 {
        5
    } else if heavy_attacks >= 3 && attacked_squares <= 3 {
        6
    } else {
        // heavy_attacks >= 3 && attacked_squares >= 4
        7
    }
}

// =============================================================================
// v3 tactical 計算 (= 2026-06-08 設計、 4.2-4.4 節)
// =============================================================================

/// v3 heavy attack count (= 玉 3x3 内 heavy piece attack 総数を 0..=3+ bucket 化)。
///
/// v2 の `heavy_attacks` と同 piece 定義 (= Silver/Gold/Bishop/Rook/Horse/Dragon + Pro 系)、
/// ただし v3 では「v1 attacked_squares 圧縮」 をしない、 純 heavy 数のみ捕捉。
///
/// 戻り値: 0/1/2/3 (= 3 は 「3+」 飽和)
pub fn compute_heavy_attack_class_v3(pos: &Position, persp: Color) -> usize {
    let king_sq = pos.king_square(persp);
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);
    let enemy = persp.opponent();
    let enemy_heavy = enemy_heavy_mask(pos, enemy);

    let zone = king_effect(king_sq) | Bitboard::from_square(king_sq);
    let mut heavy_attacks: usize = 0;
    for sq in zone.iter() {
        let attackers = pos.attackers_to_occ(sq, occupied);
        heavy_attacks += (attackers & enemy_heavy).count() as usize;
    }
    heavy_attacks.min(3)
}

/// v3 escape count (= 守備側玉の 8 隣接で legal escape 升数を 0..=3+ bucket 化)。
///
/// 各 隣接升で:
/// - 自軍駒が存在しない (= occupancy filter)
/// - 移動後 occupancy で敵利きがない (= 王手放置防止)
/// - 敵駒 capture 時は capture 後 occupancy で確認
///
/// 詰み判定ではない、 純 「逃げ道の少なさ」 表現。
pub fn compute_escape_class_v3(pos: &Position, persp: Color) -> usize {
    let king_sq = pos.king_square(persp);
    let own_pieces = pos.pieces_c(persp);
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);
    let enemy = persp.opponent();

    let king_zone = king_effect(king_sq); // 8 隣接 (= 玉自身除く)
    let mut escape_count: usize = 0;
    for to_sq in king_zone.iter() {
        let to_bb = Bitboard::from_square(to_sq);
        // 自軍駒が存在 → 移動不可
        if !(to_bb & own_pieces).is_empty() {
            continue;
        }
        // 移動後 occupancy: 玉 from 除外 + to_sq 追加 (= capture 時は to_sq の敵駒も除外)
        let from_bb = Bitboard::from_square(king_sq);
        let new_occ = (occupied & !from_bb & !to_bb) | to_bb;
        // 敵 attackers が new_occ で to_sq に利く → 違法
        let new_enemy_pieces = pos.pieces_c(enemy) & !to_bb; // 自玉 capture 後の敵駒
        let attackers = pos.attackers_to_occ(to_sq, new_occ) & new_enemy_pieces;
        if attackers.is_empty() {
            escape_count += 1;
        }
    }
    escape_count.min(3)
}

/// v3 hand_check count (= 攻撃側持ち駒からの王手駒打ち種類数を 0..=3+ bucket 化)。
///
/// 対象駒種: Pawn / Lance / Knight / Silver / Gold / Bishop / Rook (= 7 種)。
///
/// 各駒種 1 個でも持ち、 任意空き升への drop で defender king に王手なる場合 +1 種類。
/// 二歩 + 行き所のない駒は除外 (= pseudo-legal、 打ち歩詰めは判定なし)。
/// 同 駒種で複数王手升があっても 1 種類カウント。
pub fn compute_hand_check_class_v3(pos: &Position, persp: Color) -> usize {
    let attacker = persp.opponent();
    let king_sq = pos.king_square(persp);
    let occupied = pos.pieces_c(Color::Black) | pos.pieces_c(Color::White);
    let empty = !occupied;

    let hand = pos.hand(attacker);

    let mut count: usize = 0;
    for &pt in &[
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
    ] {
        if hand.count(pt) == 0 {
            continue;
        }
        // defender king に王手なる drop 升を 1 つでも探す (= 存在判定のみ)
        // king へ drop で利く逆方向 = attacker 視点で sq から king に利く piece
        // つまり sq = piece_attacks_inverse(king_sq) で空き + drop 合法
        let check_squares = drop_check_squares_v3(pos, attacker, pt, king_sq, occupied, empty);
        if !check_squares.is_empty() {
            count += 1;
        }
    }
    count.min(3)
}

/// piece type の drop で defender king に王手 になる 空き升 bitboard を返す helper。
fn drop_check_squares_v3(
    pos: &Position,
    attacker: Color,
    pt: PieceType,
    king_sq: Square,
    occupied: Bitboard,
    empty: Bitboard,
) -> Bitboard {
    // 攻撃側 piece が defender king sq に利く逆 = 攻撃側 from sq の集合
    // つまり 「king_sq から見て attacker 視点 pt attack 範囲」 = drop すれば王手なる候補
    let candidates = match pt {
        PieceType::Pawn => {
            // attacker 視点 pawn = king の真前 (= attacker の自軍方向逆) 1 升
            pawn_drop_check_from_king(king_sq, attacker, pos)
        }
        PieceType::Lance => {
            // attacker 視点 lance = king の真前から空き升まで一直線
            lance_drop_check_from_king(king_sq, attacker, occupied)
        }
        PieceType::Knight => {
            // attacker 視点 knight = king の前方 2 升 (= 飛び駒)
            knight_drop_check_from_king(king_sq, attacker)
        }
        PieceType::Silver => silver_effect(attacker.opponent(), king_sq),
        PieceType::Gold => gold_effect(attacker.opponent(), king_sq),
        PieceType::Bishop => bishop_effect(king_sq, occupied),
        PieceType::Rook => rook_effect(king_sq, occupied),
        _ => Bitboard::EMPTY,
    };
    candidates & empty
}

/// Pawn drop で king 王手なる sq (= 二歩除外、 行き所のない駒除外)。
fn pawn_drop_check_from_king(king_sq: Square, attacker: Color, pos: &Position) -> Bitboard {
    use crate::bitboard::pawn_effect;
    // king の真前 1 升 (= attacker から見て pawn が前進する方向、 つまり defender 視点 pawn の back)
    // pawn_effect(defender, king_sq) で defender 視点 pawn 利き = king sq の back 方向、 attacker drop pawn ≠ defender pawn
    // attacker の pawn drop で king に王手なる sq = pawn_effect(attacker.opponent(), king_sq)
    let candidate = pawn_effect(attacker.opponent(), king_sq);
    // 二歩除外: 既 同 file に attacker pawn ある場合 候補から除外
    let attacker_pawns = pos.pieces(attacker, PieceType::Pawn);
    let mut result = Bitboard::EMPTY;
    for sq in candidate.iter() {
        let file_mask = file_bitboard(sq);
        if (attacker_pawns & file_mask).is_empty() {
            // 行き所のない 1 段目 (attacker 視点) は除外
            if !is_dead_drop_pawn_lance(sq, attacker) {
                result |= Bitboard::from_square(sq);
            }
        }
    }
    result
}

/// Lance drop で king 王手なる sq (= king 真前一直線、 行き所のない駒除外)。
fn lance_drop_check_from_king(king_sq: Square, attacker: Color, occupied: Bitboard) -> Bitboard {
    use crate::bitboard::lance_effect;
    let candidate = lance_effect(attacker.opponent(), king_sq, occupied);
    let mut result = Bitboard::EMPTY;
    for sq in candidate.iter() {
        if !is_dead_drop_pawn_lance(sq, attacker) {
            result |= Bitboard::from_square(sq);
        }
    }
    result
}

/// Knight drop で king 王手なる sq (= king 前方 2 升、 行き所のない 2 段目除外)。
fn knight_drop_check_from_king(king_sq: Square, attacker: Color) -> Bitboard {
    use crate::bitboard::knight_effect;
    let candidate = knight_effect(attacker.opponent(), king_sq);
    let mut result = Bitboard::EMPTY;
    for sq in candidate.iter() {
        if !is_dead_drop_knight(sq, attacker) {
            result |= Bitboard::from_square(sq);
        }
    }
    result
}

/// sq が attacker 視点で「行き所のない pawn/lance 1 段目」 か。
fn is_dead_drop_pawn_lance(sq: Square, attacker: Color) -> bool {
    let rank = sq.index() % 9;
    match attacker {
        Color::Black => rank == 0, // 1 段目に black pawn/lance 不可
        Color::White => rank == 8, // 9 段目に white pawn/lance 不可
    }
}

/// sq が attacker 視点で「行き所のない knight 1-2 段目」 か。
fn is_dead_drop_knight(sq: Square, attacker: Color) -> bool {
    let rank = sq.index() % 9;
    match attacker {
        Color::Black => rank <= 1, // 1-2 段目に black knight 不可
        Color::White => rank >= 7, // 8-9 段目に white knight 不可
    }
}

/// sq と同 file の bitboard (= 9 升)。
fn file_bitboard(sq: Square) -> Bitboard {
    let file = sq.index() / 9;
    let mut bb = Bitboard::EMPTY;
    for rank in 0..9u8 {
        let idx = (file as u8) * 9 + rank;
        if let Some(s) = Square::from_u8(idx) {
            bb |= Bitboard::from_square(s);
        }
    }
    bb
}

// =============================================================================
// v3 weight 型 + flatten
// =============================================================================

/// v3 全 weight 配列 (= base 4096 + tactical 1024 = 5120)。
pub type ActivityRiskWeightsV3 = [i16; ACTIVITY_RISK_V3_WEIGHT_DIM];

/// v3 tactical flatten index (= `[bucket][persp][heavy][escape][hand_check]` row-major、 範囲 `[0, 1024)`).
#[inline]
pub const fn flatten_idx_v3_tactical(
    bucket: usize,
    perspective: usize,
    heavy: usize,
    escape: usize,
    hand_check: usize,
) -> usize {
    let inner = ((heavy * ACTIVITY_RISK_V3_NUM_ESCAPE) + escape) * ACTIVITY_RISK_V3_NUM_HAND_CHECK
        + hand_check;
    let per_pb =
        ACTIVITY_RISK_V3_NUM_HEAVY * ACTIVITY_RISK_V3_NUM_ESCAPE * ACTIVITY_RISK_V3_NUM_HAND_CHECK;
    let pb = perspective * per_pb + inner;
    bucket * ACTIVITY_RISK_NUM_PERSPECTIVES * per_pb + pb
}

/// v3 evaluate (= base + tactical 加算合成、 black - white)。
pub fn evaluate_activity_risk_v3(
    pos: &Position,
    weights: &ActivityRiskWeightsV3,
    bucket: usize,
) -> i32 {
    // base 項 (= v1 と同計算: activity / pressure_v1 / shield)
    let (base_w, tactical_w) = weights.split_at(ACTIVITY_RISK_V3_BASE_DIM);
    let base_arr: &ActivityRiskWeights = base_w.try_into().expect("v3 base weights len 4096");

    let black_act = compute_major_activity(pos, Color::Black);
    let black_pre = compute_king_pressure(pos, Color::Black); // v1 = 敵利き合計
    let black_shi = compute_shield(pos, Color::Black);
    let white_act = compute_major_activity(pos, Color::White);
    let white_pre = compute_king_pressure(pos, Color::White);
    let white_shi = compute_shield(pos, Color::White);

    let base_black =
        lookup_perspective(base_arr, bucket, Color::Black, black_act, black_pre, black_shi);
    let base_white =
        lookup_perspective(base_arr, bucket, Color::White, white_act, white_pre, white_shi);

    // tactical 項
    let black_heavy = compute_heavy_attack_class_v3(pos, Color::Black);
    let black_escape = compute_escape_class_v3(pos, Color::Black);
    let black_check = compute_hand_check_class_v3(pos, Color::Black);
    let white_heavy = compute_heavy_attack_class_v3(pos, Color::White);
    let white_escape = compute_escape_class_v3(pos, Color::White);
    let white_check = compute_hand_check_class_v3(pos, Color::White);

    let tactical_black_idx =
        flatten_idx_v3_tactical(bucket, 0, black_heavy, black_escape, black_check);
    let tactical_white_idx =
        flatten_idx_v3_tactical(bucket, 1, white_heavy, white_escape, white_check);
    let tactical_black = tactical_w[tactical_black_idx] as i32;
    let tactical_white = tactical_w[tactical_white_idx] as i32;

    let black = base_black + tactical_black;
    let white = base_white + tactical_white;
    (black - white) * ACTIVITY_RISK_OUTPUT_SCALE
}

/// v1 weight (= 4096 i16) を v3 weight (= 5120 i16) に変換 (= base copy + tactical 全 0)。
///
/// 変換直後 `evaluate_activity_risk_v3(pos, v3, bucket)` は v1 の `evaluate_activity_risk` と
/// **bit-exact 一致** (= tactical 全 0 で 加算項が 0、 base が v1 と同形)。
pub fn convert_v1_to_v3(v1_weights: &ActivityRiskWeights) -> Box<ActivityRiskWeightsV3> {
    let mut v3 = Box::new([0i16; ACTIVITY_RISK_V3_WEIGHT_DIM]);
    v3[..ACTIVITY_RISK_V3_BASE_DIM].copy_from_slice(v1_weights);
    // tactical = 全 0 (= Box init で既 0)
    v3
}

/// perspective 側の玉中心 5x5 (= 玉自身除く、 盤外除外) で perspective 側の小駒数を計算する。
///
/// 5x5 マスク (= `king_5x5_mask(king_sq) ^ from_square(king_sq)`) と
/// 自軍 (Pawn / Silver / Gold / ProPawn / ProLance / ProKnight / ProSilver) の AND popcount。
pub fn compute_shield(pos: &Position, persp: Color) -> usize {
    let king_sq = pos.king_square(persp);
    let mask = king_5x5_mask(king_sq) & !Bitboard::from_square(king_sq);
    let shield_pieces = pos.pieces(persp, PieceType::Pawn)
        | pos.pieces(persp, PieceType::Silver)
        | pos.pieces(persp, PieceType::Gold)
        | pos.pieces(persp, PieceType::ProPawn)
        | pos.pieces(persp, PieceType::ProLance)
        | pos.pieces(persp, PieceType::ProKnight)
        | pos.pieces(persp, PieceType::ProSilver);
    (shield_pieces & mask).count() as usize
}

/// 玉中心 5x5 領域 bitboard (= 旧 activity_risk 定義、 2026-06-09 で
/// `crate::bitboard::king_5x5_mask` へ移管)。
///
/// ARH (attack_relay) で feature gate 独立に参照するため共通 module 側へ移管。
/// 内部使用者は変更不要 (= 同名 symbol を re-export)。
pub use crate::bitboard::king_5x5_mask;

// =============================================================================
// 評価合成 (= eval 末端で使用)
// =============================================================================

/// 当該局面 + bucket で Activity-Risk weight を 1 個 lookup する。
///
/// 各 perspective で計算した major_activity / king_pressure / shield を bucket 化し、
/// disk layout flatten index で weight 配列を引く。
#[inline]
pub fn lookup_perspective(
    weights: &ActivityRiskWeights,
    bucket: usize,
    perspective: Color,
    activity: usize,
    pressure: usize,
    shield: usize,
) -> i32 {
    let persp_idx = perspective as usize;
    let a_b = activity_to_bucket(activity);
    let p_b = pressure_to_bucket(pressure);
    let s_b = shield_to_bucket(shield);
    let idx = flatten_idx(bucket, persp_idx, a_b, p_b, s_b);
    weights[idx] as i32
}

/// 評価値合成 (= `black 寄与 - white 寄与`、 fv_scale 適用前)。
///
/// 各 perspective で `lookup_perspective` を 1 回呼び、 差を返す。
/// output scale は `ACTIVITY_RISK_OUTPUT_SCALE` (= 初版 1)。
pub fn evaluate_activity_risk(pos: &Position, weights: &ActivityRiskWeights, bucket: usize) -> i32 {
    let black_act = compute_major_activity(pos, Color::Black);
    let black_pre = compute_king_pressure_v2(pos, Color::Black);
    let black_shi = compute_shield(pos, Color::Black);
    let white_act = compute_major_activity(pos, Color::White);
    let white_pre = compute_king_pressure_v2(pos, Color::White);
    let white_shi = compute_shield(pos, Color::White);

    let black = lookup_perspective(weights, bucket, Color::Black, black_act, black_pre, black_shi);
    let white = lookup_perspective(weights, bucket, Color::White, white_act, white_pre, white_shi);

    (black - white) * ACTIVITY_RISK_OUTPUT_SCALE
}

/// `piece_type` が shield 対象 (= Pawn / Silver / Gold / ProPawn / ProLance / ProKnight / ProSilver) か。
#[inline]
fn is_shield_piece_type(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Pawn
            | PieceType::Silver
            | PieceType::Gold
            | PieceType::ProPawn
            | PieceType::ProLance
            | PieceType::ProKnight
            | PieceType::ProSilver
    )
}

// =============================================================================
// Accumulator (= 2026-06-06 skeleton、 NPS -17% 退行解消用、 次 session 完成予定)
// =============================================================================

/// Activity-Risk Accumulator (= per-position cache、 NPS overhead 削減目的)。
///
/// **状態 (2026-06-06)**: skeleton のみ、 未統合。 次 session で完成予定。
///
/// # 設計方針
///
/// per-eval full compute (= 現状) で NPS -17% 退行が実測 (= sb 240 vs sb 11 selfplay -120 Elo 真因)。
/// alpha-beta 探索中の incremental update で NPS overhead を **-3% 以内** にする。
///
/// # 必須機能
///
/// 1. `refresh(pos)`: full compute (= 既存 `compute_*` 関数呼び出し)、 root + king move + capture/promote 時
/// 2. `update_with_delta(delta)`: piece move 1 件の incremental update
///    - shield: 自軍小駒 add/remove → 玉 5x5 mask 内なら +1/-1
///    - king_pressure: piece add/remove → 玉 3x3 内 sq 影響あり、 attackers_to 部分再計算
///    - major_activity: 飛角馬竜 move OR 大駒 line 上 piece move → 大駒 mobility 再計算
/// 3. `evaluate(weights, bucket)`: cached features で `lookup_perspective` 2 回
///
/// # StateInfo 統合
///
/// `position::StateInfo` に `activity_risk_accum: ActivityRiskAccumulator` field 追加、
/// `do_move` / `undo_move` で前 state copy / restore。 dirty flag で eval 時 lazy refresh。
///
/// # 工程見積 (= 次 session 推奨)
///
/// - Step 1: struct field + refresh + golden test = 1 hr
/// - Step 2: update_with_delta (= shield) = 30 min
/// - Step 3: update_with_delta (= king_pressure) = 1 hr
/// - Step 4: update_with_delta (= major_activity) = 2 hr (= 最複雑、 大駒 line / 斜上 影響範囲)
/// - Step 5: StateInfo 統合 (= do/undo_move 完全対応) = 1 hr
/// - Step 6: NPS bench (= -3% 以内 要件確認) + golden test = 30 min
/// - **合計 ~6 hr**
#[derive(Debug, Clone, Default)]
pub struct ActivityRiskAccumulator {
    /// 黒視点 major piece pseudo mobility (= 飛角馬竜 合計)
    pub black_activity: usize,
    /// 黒視点 king pressure (= 玉 3x3 敵利き合計)
    pub black_pressure: usize,
    /// 黒視点 shield (= 玉 5x5 自軍小駒数)
    pub black_shield: usize,
    /// 白視点 major piece pseudo mobility
    pub white_activity: usize,
    /// 白視点 king pressure
    pub white_pressure: usize,
    /// 白視点 shield
    pub white_shield: usize,
    /// `true` なら refresh 必須 (= incremental update 失敗 or root)
    pub dirty: bool,
}

impl ActivityRiskAccumulator {
    /// 局面から full compute で全 feature を再計算 (= NPS cost: 既存 `evaluate_activity_risk` と同等)。
    ///
    /// 2026-06-06: `pressure` は v2 (= 8 排他クラス) で計算する。
    pub fn refresh(&mut self, pos: &Position) {
        self.black_activity = compute_major_activity(pos, Color::Black);
        self.black_pressure = compute_king_pressure_v2(pos, Color::Black);
        self.black_shield = compute_shield(pos, Color::Black);
        self.white_activity = compute_major_activity(pos, Color::White);
        self.white_pressure = compute_king_pressure_v2(pos, Color::White);
        self.white_shield = compute_shield(pos, Color::White);
        self.dirty = false;
    }

    /// shield のみ incremental update (= 玉 5x5 内 自軍小駒 add/remove で ±1)。
    ///
    /// `king_moved` true なら full refresh (= 玉移動で zone 全 変動、 incremental 不可)。
    /// それ以外は piece add/remove で shield ±1。 dirty フラグは false のまま (= caller 確認)。
    ///
    /// **状態 (= 2026-06-06)**: shield のみ、 king_pressure / major_activity は Step 3/4 で。
    /// caller は dirty=true でない限り、 すべての piece move を順次 update_shield_delta に渡す
    /// 想定。 king_moved or major piece move 時は dirty=true で refresh path に戻す。
    pub fn update_shield_delta(
        &mut self,
        from_sq: Option<Square>,
        from_piece: Option<crate::types::Piece>,
        to_sq: Option<Square>,
        to_piece: Option<crate::types::Piece>,
        king_sq_black: Square,
        king_sq_white: Square,
    ) {
        use crate::bitboard::Bitboard;
        let mask_black = king_5x5_mask(king_sq_black) & !Bitboard::from_square(king_sq_black);
        let mask_white = king_5x5_mask(king_sq_white) & !Bitboard::from_square(king_sq_white);

        // remove from from_sq
        if let (Some(sq), Some(piece)) = (from_sq, from_piece)
            && is_shield_piece_type(piece.piece_type())
        {
            let bb = Bitboard::from_square(sq);
            match piece.color() {
                Color::Black => {
                    if !(bb & mask_black).is_empty() {
                        self.black_shield = self.black_shield.saturating_sub(1);
                    }
                }
                Color::White => {
                    if !(bb & mask_white).is_empty() {
                        self.white_shield = self.white_shield.saturating_sub(1);
                    }
                }
            }
        }

        // add to to_sq
        if let (Some(sq), Some(piece)) = (to_sq, to_piece)
            && is_shield_piece_type(piece.piece_type())
        {
            let bb = Bitboard::from_square(sq);
            match piece.color() {
                Color::Black => {
                    if !(bb & mask_black).is_empty() {
                        self.black_shield += 1;
                    }
                }
                Color::White => {
                    if !(bb & mask_white).is_empty() {
                        self.white_shield += 1;
                    }
                }
            }
        }
    }

    /// king_pressure を partial refresh (= 玉 3x3 zone 全 recompute、 現状 NPS 改善 0%)。
    ///
    /// 真 incremental (= piece add/remove で 玉 3x3 内 sq の attackers delta 計算) は
    /// piece type 多様性 (= 飛車 line / 香車 file / 桂馬 jump / 大駒 line 上 occupied 変動) で
    /// 複雑。 当面 partial refresh で API base 完成、 真 incremental は別 step。
    pub fn update_king_pressure_partial_refresh(&mut self, pos: &Position) {
        self.black_pressure = compute_king_pressure(pos, Color::Black);
        self.white_pressure = compute_king_pressure(pos, Color::White);
    }

    /// major_activity を partial refresh (= 飛角馬竜 mobility 全 recompute、 現状 NPS 改善 0%)。
    ///
    /// 真 incremental (= 大駒移動 + 大駒 line 上 piece move で mobility delta 計算) は
    /// 最複雑、 別 step で実装予定。 当面 partial refresh で API base 完成。
    pub fn update_major_activity_partial_refresh(&mut self, pos: &Position) {
        self.black_activity = compute_major_activity(pos, Color::Black);
        self.white_activity = compute_major_activity(pos, Color::White);
    }

    /// cached features で評価値合成。 `dirty=true` なら panic (= caller が refresh 必須)。
    pub fn evaluate(&self, weights: &ActivityRiskWeights, bucket: usize) -> i32 {
        debug_assert!(!self.dirty, "evaluate() called on dirty accumulator; refresh first");
        let black = lookup_perspective(
            weights,
            bucket,
            Color::Black,
            self.black_activity,
            self.black_pressure,
            self.black_shield,
        );
        let white = lookup_perspective(
            weights,
            bucket,
            Color::White,
            self.white_activity,
            self.white_pressure,
            self.white_shield,
        );
        (black - white) * ACTIVITY_RISK_OUTPUT_SCALE
    }
}

// =============================================================================
// Unit Test
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_dimension() {
        assert_eq!(ACTIVITY_RISK_WEIGHT_DIM, 4096);
        assert_eq!(ACTIVITY_RISK_NUM_BUCKETS, 8);
        assert_eq!(ACTIVITY_RISK_NUM_PERSPECTIVES, 2);
        assert_eq!(ACTIVITY_RISK_NUM_ACTIVITY, 8);
        assert_eq!(ACTIVITY_RISK_NUM_PRESSURE, 8);
        assert_eq!(ACTIVITY_RISK_NUM_SHIELD, 4);
    }

    #[test]
    fn test_magic_and_version() {
        assert_eq!(ACTIVITY_RISK_SUB_MAGIC, *b"P2ACTR\0\0");
        assert_eq!(ACTIVITY_RISK_VERSION, 3); // v3: base 4096 + tactical 1024
        assert_eq!(ACTIVITY_RISK_COUNT, 5120);
        // 既 section magic と衝突しないこと
        let pairshape_magic: [u8; 8] = *b"P2PAIR\0\0";
        assert_ne!(ACTIVITY_RISK_SUB_MAGIC, pairshape_magic);
    }

    #[test]
    fn test_flatten_boundaries() {
        // 先頭: (bucket=0, persp=0, activity=0, pressure=0, shield=0)
        assert_eq!(flatten_idx(0, 0, 0, 0, 0), 0);
        // 末尾: 最大 index
        assert_eq!(flatten_idx(7, 1, 7, 7, 3), ACTIVITY_RISK_WEIGHT_DIM - 1);
        // bucket 1 の先頭
        assert_eq!(
            flatten_idx(1, 0, 0, 0, 0),
            ACTIVITY_RISK_NUM_PERSPECTIVES * ACTIVITY_RISK_PER_PB_DIM
        );
        // perspective 1 の先頭 (= bucket 0 内)
        assert_eq!(flatten_idx(0, 1, 0, 0, 0), ACTIVITY_RISK_PER_PB_DIM);
    }

    #[test]
    fn test_flatten_inverse() {
        // 全 index で 順写像確認
        for idx in 0..ACTIVITY_RISK_WEIGHT_DIM {
            let (b, p, a, pr, s) = unflatten_idx(idx);
            assert_eq!(flatten_idx(b, p, a, pr, s), idx);
        }
    }

    #[test]
    fn test_activity_bucket() {
        assert_eq!(activity_to_bucket(0), 0);
        assert_eq!(activity_to_bucket(1), 1);
        assert_eq!(activity_to_bucket(3), 1);
        assert_eq!(activity_to_bucket(4), 2);
        assert_eq!(activity_to_bucket(6), 2);
        assert_eq!(activity_to_bucket(7), 3);
        assert_eq!(activity_to_bucket(9), 3);
        assert_eq!(activity_to_bucket(10), 4);
        assert_eq!(activity_to_bucket(12), 4);
        assert_eq!(activity_to_bucket(13), 5);
        assert_eq!(activity_to_bucket(15), 5);
        assert_eq!(activity_to_bucket(16), 6);
        assert_eq!(activity_to_bucket(19), 6);
        assert_eq!(activity_to_bucket(20), 7);
        assert_eq!(activity_to_bucket(100), 7);
    }

    #[test]
    fn test_pressure_bucket() {
        for p in 0..=6 {
            assert_eq!(pressure_to_bucket(p), p);
        }
        assert_eq!(pressure_to_bucket(7), 7);
        assert_eq!(pressure_to_bucket(50), 7);
    }

    #[test]
    fn test_shield_bucket() {
        assert_eq!(shield_to_bucket(0), 0);
        assert_eq!(shield_to_bucket(2), 0);
        assert_eq!(shield_to_bucket(3), 1);
        assert_eq!(shield_to_bucket(4), 1);
        assert_eq!(shield_to_bucket(5), 2);
        assert_eq!(shield_to_bucket(6), 2);
        assert_eq!(shield_to_bucket(7), 3);
        assert_eq!(shield_to_bucket(100), 3);
    }

    #[test]
    fn test_king_5x5_mask_center() {
        // 中央 5e (= file=4, rank=4) で 5x5 = 25 sq
        let center = Square::from_u8(4 * 9 + 4).unwrap();
        let mask = king_5x5_mask(center);
        assert_eq!(mask.count(), 25);
    }

    #[test]
    fn test_king_5x5_mask_corner() {
        // 1a (= file=0, rank=0) で 5x5 = 3x3 = 9 sq (= 盤端で clip)
        let corner = Square::from_u8(0).unwrap();
        let mask = king_5x5_mask(corner);
        assert_eq!(mask.count(), 9);
    }

    #[test]
    fn test_startpos_features() {
        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);
        // startpos = 飛角各 1、 自軍駒で大駒可動 制限大、 玉周辺敵利き 0
        let b_act = compute_major_activity(&pos, Color::Black);
        let w_act = compute_major_activity(&pos, Color::White);
        // 飛 (2八) + 角 (8八) の pseudo mobility 合計
        // 飛: 2 段目 7 sq (= 1-9 段の 2 段) 他、 角: 自由斜め 等
        // startpos では mobility 限定的、 ただし > 0 確認
        assert!(b_act > 0, "black activity should be > 0 at startpos");
        assert!(w_act > 0, "white activity should be > 0 at startpos");
        // 対称局面のため black == white を期待
        assert_eq!(b_act, w_act, "startpos symmetric: black_act == white_act");

        // 玉周辺 敵利き = 0 (= 初手前は接敵なし)
        let b_pre = compute_king_pressure(&pos, Color::Black);
        let w_pre = compute_king_pressure(&pos, Color::White);
        assert_eq!(b_pre, 0, "startpos black king pressure = 0");
        assert_eq!(w_pre, 0, "startpos white king pressure = 0");

        // shield = 玉 5x5 内 自軍小駒 (= 玉自身除く)
        let b_shi = compute_shield(&pos, Color::Black);
        let w_shi = compute_shield(&pos, Color::White);
        assert!(b_shi > 0, "startpos black shield > 0");
        assert!(w_shi > 0, "startpos white shield > 0");
        assert_eq!(b_shi, w_shi, "startpos symmetric: black_shield == white_shield");
    }

    #[test]
    fn test_zero_weight_evaluation() {
        // weight が全 0 なら、 評価値も 0
        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);
        let weights: ActivityRiskWeights = [0i16; ACTIVITY_RISK_WEIGHT_DIM];
        for bucket in 0..ACTIVITY_RISK_NUM_BUCKETS {
            let val = evaluate_activity_risk(&pos, &weights, bucket);
            assert_eq!(val, 0, "zero weight => zero contribution (bucket {bucket})");
        }
    }

    /// Accumulator refresh + evaluate が既存 `evaluate_activity_risk` と完全一致する
    /// (= golden test、 next-step incremental update 実装の base 契約)。
    #[test]
    fn test_accumulator_refresh_matches_evaluate() {
        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);

        // ランダムっぽい weight (= sb 1 育成相当)
        let mut weights: ActivityRiskWeights = [0i16; ACTIVITY_RISK_WEIGHT_DIM];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = ((i as i32 * 31 % 257) - 128) as i16;
        }

        let mut accum = ActivityRiskAccumulator::default();
        accum.refresh(&pos);
        assert!(!accum.dirty, "after refresh, dirty must be false");

        for bucket in 0..ACTIVITY_RISK_NUM_BUCKETS {
            let expected = evaluate_activity_risk(&pos, &weights, bucket);
            let actual = accum.evaluate(&weights, bucket);
            assert_eq!(
                actual, expected,
                "accumulator evaluate must match evaluate_activity_risk (bucket {bucket})"
            );
        }
    }

    // =========================================================================
    // v2 king_pressure 8 排他クラス test (= 2026-06-06)
    // =========================================================================

    /// classify_pressure_v2: class 0 = 敵利り 0
    #[test]
    fn test_pressure_v2_class0_no_attack() {
        assert_eq!(classify_pressure_v2(0, 0), 0);
        assert_eq!(classify_pressure_v2(0, 1), 0); // attacked=0 が最優先
        assert_eq!(classify_pressure_v2(0, 5), 0);
    }

    /// class 1 = 1 sq 攻撃、 heavy 0
    #[test]
    fn test_pressure_v2_class1_single_light() {
        assert_eq!(classify_pressure_v2(1, 0), 1);
    }

    /// class 2 = 2 sq 以上攻撃、 heavy 0
    #[test]
    fn test_pressure_v2_class2_multi_light() {
        assert_eq!(classify_pressure_v2(2, 0), 2);
        assert_eq!(classify_pressure_v2(5, 0), 2);
        assert_eq!(classify_pressure_v2(9, 0), 2);
    }

    /// class 3 = heavy 1 利り、 攻撃升 <= 2
    #[test]
    fn test_pressure_v2_class3_heavy1_few_sq() {
        assert_eq!(classify_pressure_v2(1, 1), 3);
        assert_eq!(classify_pressure_v2(2, 1), 3);
    }

    /// class 4 = heavy 1 利り、 攻撃升 >= 3
    #[test]
    fn test_pressure_v2_class4_heavy1_many_sq() {
        assert_eq!(classify_pressure_v2(3, 1), 4);
        assert_eq!(classify_pressure_v2(5, 1), 4);
        assert_eq!(classify_pressure_v2(9, 1), 4);
    }

    /// class 5 = heavy 2
    #[test]
    fn test_pressure_v2_class5_heavy2() {
        assert_eq!(classify_pressure_v2(1, 2), 5);
        assert_eq!(classify_pressure_v2(2, 2), 5);
        assert_eq!(classify_pressure_v2(5, 2), 5);
    }

    /// class 6 = heavy >= 3、 攻撃升 <= 3
    #[test]
    fn test_pressure_v2_class6_heavy3plus_few_sq() {
        assert_eq!(classify_pressure_v2(1, 3), 6);
        assert_eq!(classify_pressure_v2(3, 3), 6);
        assert_eq!(classify_pressure_v2(3, 5), 6);
    }

    /// class 7 = heavy >= 3、 攻撃升 >= 4
    #[test]
    fn test_pressure_v2_class7_heavy3plus_many_sq() {
        assert_eq!(classify_pressure_v2(4, 3), 7);
        assert_eq!(classify_pressure_v2(9, 5), 7);
        assert_eq!(classify_pressure_v2(5, 10), 7);
    }

    /// startpos で compute_king_pressure_v2 = class 0 (= 平手では玉周辺敵利りなし)
    #[test]
    fn test_pressure_v2_startpos_class0() {
        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);
        assert_eq!(compute_king_pressure_v2(&pos, Color::Black), 0);
        assert_eq!(compute_king_pressure_v2(&pos, Color::White), 0);
    }

    /// 平手で色反転で同 class (= 左右対称 startpos)
    #[test]
    fn test_pressure_v2_color_invariance_startpos() {
        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);
        let b = compute_king_pressure_v2(&pos, Color::Black);
        let w = compute_king_pressure_v2(&pos, Color::White);
        assert_eq!(b, w, "startpos: 色反転で同 class (左右対称)");
    }

    /// ACTIVITY_RISK_VERSION = 3 確定 (= v1/v2 ckpt 拒否のため)
    #[test]
    fn test_v2_version_constant() {
        assert_eq!(ACTIVITY_RISK_VERSION, 3);
    }

    /// pressure_to_bucket は v2 でも identity (= 0-7 そのまま、 class 値が bucket index)
    #[test]
    fn test_pressure_to_bucket_identity_for_v2() {
        for class in 0..8 {
            assert_eq!(pressure_to_bucket(class), class);
        }
    }

    // =========================================================================
    // v3 unit test (= 2026-06-08)
    // =========================================================================

    /// v3 dimensions が設計通り (= base 4096 + tactical 1024 = 5120)。
    #[test]
    fn test_v3_dimensions() {
        assert_eq!(ACTIVITY_RISK_V3_BASE_DIM, 4096);
        assert_eq!(ACTIVITY_RISK_V3_TACTICAL_DIM, 1024);
        assert_eq!(ACTIVITY_RISK_V3_WEIGHT_DIM, 5120);
        assert_eq!(ACTIVITY_RISK_V3_NUM_HEAVY, 4);
        assert_eq!(ACTIVITY_RISK_V3_NUM_ESCAPE, 4);
        assert_eq!(ACTIVITY_RISK_V3_NUM_HAND_CHECK, 4);
        assert_eq!(ACTIVITY_RISK_VERSION, 3);
        assert_eq!(ACTIVITY_RISK_COUNT, 5120);
    }

    /// v3 tactical flatten 先頭 + 末尾 + 範囲。
    #[test]
    fn test_v3_tactical_flatten_boundaries() {
        assert_eq!(flatten_idx_v3_tactical(0, 0, 0, 0, 0), 0);
        assert_eq!(flatten_idx_v3_tactical(7, 1, 3, 3, 3), ACTIVITY_RISK_V3_TACTICAL_DIM - 1);
        // bucket 1 先頭
        assert_eq!(
            flatten_idx_v3_tactical(1, 0, 0, 0, 0),
            ACTIVITY_RISK_NUM_PERSPECTIVES
                * ACTIVITY_RISK_V3_NUM_HEAVY
                * ACTIVITY_RISK_V3_NUM_ESCAPE
                * ACTIVITY_RISK_V3_NUM_HAND_CHECK
        );
    }

    /// v1 → v3 変換 + zero tactical で v1 evaluate と bit-exact 一致 (= 設計 6 節 合格条件)。
    #[test]
    fn test_v1_to_v3_zero_tactical_bit_exact() {
        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);

        // ランダムっぽい v1 weight
        let mut v1: ActivityRiskWeights = [0i16; ACTIVITY_RISK_WEIGHT_DIM];
        for (i, w) in v1.iter_mut().enumerate() {
            *w = ((i as i32 * 31 % 257) - 128) as i16;
        }

        let v3 = convert_v1_to_v3(&v1);

        for bucket in 0..ACTIVITY_RISK_NUM_BUCKETS {
            let v1_score = evaluate_activity_risk(&pos, &v1, bucket);
            let v3_score = evaluate_activity_risk_v3(&pos, &v3, bucket);
            assert_eq!(
                v3_score, v1_score,
                "v3 with zero tactical must match v1 evaluate bit-exact (bucket {bucket})"
            );
        }
    }

    /// v3 heavy_attack class が compute_heavy_attack_class_v3 で 0..=3 範囲。
    #[test]
    fn test_v3_heavy_attack_class_range() {
        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);
        let b = compute_heavy_attack_class_v3(&pos, Color::Black);
        let w = compute_heavy_attack_class_v3(&pos, Color::White);
        assert!(b <= 3);
        assert!(w <= 3);
        // startpos = 玉周辺に敵 heavy 利き なし
        assert_eq!(b, 0);
        assert_eq!(w, 0);
    }

    /// v3 escape class が compute_escape_class_v3 で 0..=3 範囲、 startpos 互角。
    #[test]
    fn test_v3_escape_class_range_and_symmetry() {
        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);
        let b = compute_escape_class_v3(&pos, Color::Black);
        let w = compute_escape_class_v3(&pos, Color::White);
        assert!(b <= 3);
        assert!(w <= 3);
        assert_eq!(b, w, "startpos symmetric escape count");
    }

    /// v3 hand_check class が compute_hand_check_class_v3 で 0..=3 範囲、 startpos = 0。
    #[test]
    fn test_v3_hand_check_class_range_and_startpos() {
        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);
        // 平手 = 持ち駒 0、 hand_check = 0
        assert_eq!(compute_hand_check_class_v3(&pos, Color::Black), 0);
        assert_eq!(compute_hand_check_class_v3(&pos, Color::White), 0);
    }

    /// shield incremental update が compute_shield と一致 (= delta apply 後の値が full compute と等価)。
    #[test]
    fn test_shield_delta_matches_full_compute() {
        use crate::types::{Color, Piece, PieceType};

        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);

        let king_b = pos.king_square(Color::Black);
        let king_w = pos.king_square(Color::White);

        let mut accum = ActivityRiskAccumulator::default();
        accum.refresh(&pos);

        // 黒 Pawn を 玉 5x5 内の任意マスから remove (= shield -1)
        // startpos 黒玉 = 5i (= sq idx 76)、 black 7g 歩は 5x5 内
        // 7g = file=6, rank=6 → idx = 6*9+6 = 60
        let pawn_sq = Square::from_u8(60).unwrap();
        let pawn_piece = Piece::new(Color::Black, PieceType::Pawn);
        let initial_black_shield = accum.black_shield;
        accum.update_shield_delta(Some(pawn_sq), Some(pawn_piece), None, None, king_b, king_w);
        // 5x5 内 + 黒 Pawn なら shield -1、 そうでなければ変化なし
        let mask_b = king_5x5_mask(king_b) & !crate::bitboard::Bitboard::from_square(king_b);
        let in_zone = !(crate::bitboard::Bitboard::from_square(pawn_sq) & mask_b).is_empty();
        if in_zone {
            assert_eq!(accum.black_shield, initial_black_shield - 1);
        } else {
            assert_eq!(accum.black_shield, initial_black_shield);
        }
    }

    /// Accumulator features が compute_* 関数と一致 (= refresh が正しく feature を埋める)。
    #[test]
    fn test_accumulator_features_match_compute() {
        let mut pos = Position::new();
        pos.set_startpos_with_pass_rights(0, 0);

        let mut accum = ActivityRiskAccumulator::default();
        accum.refresh(&pos);

        assert_eq!(accum.black_activity, compute_major_activity(&pos, Color::Black));
        assert_eq!(accum.black_pressure, compute_king_pressure(&pos, Color::Black));
        assert_eq!(accum.black_shield, compute_shield(&pos, Color::Black));
        assert_eq!(accum.white_activity, compute_major_activity(&pos, Color::White));
        assert_eq!(accum.white_pressure, compute_king_pressure(&pos, Color::White));
        assert_eq!(accum.white_shield, compute_shield(&pos, Color::White));
    }
}
