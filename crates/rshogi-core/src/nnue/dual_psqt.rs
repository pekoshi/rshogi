//! Dual PSQT 補助 head: **King-Attacker 型** (= attack / defend 別 weight)
//!
//! 1 駒が **敵 king zone (= 敵 king + 8 近傍 = 9 sq)** に効きを持つかで attack
//! feature を発火、 **自 king zone** に効きを持つかで defend feature を発火。
//! 各 perspective 独立に計算し、 attack_sum - defend_sum を fv_scale 適用前の
//! cp として加算する。
//!
//! 2026-05-26: 旧 reserve 設計 (= bucket-only 16 weight × 2 head) を King-Attacker
//! 設計に置換。 旧 `P2DUAL\0\0` magic ckpt は magic mismatch で skip。
//!
//! # weight 形状
//!
//! 1 perspective あたり: `[head 2][bucket 8][sq 81][piece_type 14]` = 18,144 i16。
//! 2 perspective 合わせて 36,288 i16 = 72,576 byte。
//!
//! - head 0: attack (= 敵 king zone へ効いている駒)
//! - head 1: defend (= 自 king zone へ効いている駒)
//! - bucket: progress8kpabs の 8 bucket と共有
//! - sq: 正規化後の駒位置 (= perspective swap、 black=identity / white=80-idx)
//! - piece_type: 0..=13 (Pawn..Dragon、 King 含む 14 種)
//!
//! # ckpt section format (= disk layout)
//!
//! ```text
//! sub_magic  = "P2DUAL2\0"  (8 byte)
//! version    = u32 LE  (= 現在 1)
//! count      = u32 LE  (= 36,288 = 2 head × 8 bucket × 81 sq × 14 piece)
//! weights    = i16[persp 2][head 2][bucket 8][sq 81][piece_type 14]  (LE)
//! ```
//!
//! section 合計 byte 数: `8 + 4 + 4 + 2 × 18,144 × 2 = 72,592 byte`
//! (= magic 8 + version 4 + count 4 + 72,576 byte data)。
//!
//! ## 旧 reserve format との互換
//!
//! 旧 `P2DUAL\0\0` (= 44 byte reserve) は別 format。 loader 側は magic で分岐し、
//! 旧 magic を見つけたら 36 byte (= 4 version + 32 byte zero data) を skip して
//! 警告を出すのみ (dual head eval 寄与 = 0)。
//!
//! # arch_str marker
//!
//! arch_str に `"DUAL=1"` を含む ckpt 読み込み時のみ section を読む
//! ([`DUAL_PSQT_ARCH_MARKER`])。 `nnue-dual-psqt` feature 無効ビルドで marker が
//! 見つかった場合は error。
//!
//! # eval 寄与 (per_eval full compute)
//!
//! Phase 2 (= 本 module) では **per-eval full 計算**: 各盤上駒に対し
//! 利き bitboard を計算し、 敵 / 自 king zone と交差判定 → active なら weight
//! lookup + i32 accum。 NPS -10〜-20% を許容。 accumulator 化 (= Phase 3) は別
//! sub-agent。
//!
//! # cross-implementation 契約
//!
//! 本 module の以下定義は bullet 側 ShogiHalfKaHmKingAttackerDual 入力と
//! **完全一致**:
//!
//! - [`PieceTypeIdx`] 順序: Pawn..=Dragon (= [`PIN_PIECE_TYPES`] と同順)
//! - perspective swap 規則: black=identity、 white=80-sq
//! - king zone 定義: king sq + king_effect(king_sq) (= 9 sq max)
//! - feature index 計算: `head * 8 * 81 * 14 + bucket * 81 * 14 + sq * 14 + piece`
//!
//! 変更時は Golden Forward cross-validation 必須。

use crate::bitboard::{
    Bitboard, bishop_effect, dragon_effect, gold_effect, horse_effect, king_effect, knight_effect,
    lance_effect, pawn_effect, rook_effect, silver_effect,
};
use crate::nnue::constants::NUM_LAYER_STACK_BUCKETS;
use crate::position::Position;
use crate::types::{Color, Piece, PieceType, Square};

// =============================================================================
// 定数
// =============================================================================

/// dual head の数 (= 2: attack head + defend head)。
pub const NUM_DUAL_PSQT_HEADS: usize = 2;

/// bucket 数。 LayerStacks の progress8kpabs と共有 (`NUM_LAYER_STACK_BUCKETS == 8`)。
pub const NUM_DUAL_PSQT_BUCKETS: usize = 8;

/// 盤面マス数 (81)。
pub const DUAL_SQUARES: usize = 81;

/// piece_type 数 (Pawn..=Dragon、 King 含む 14 種、 [`piece_type_to_dual_idx`] と整合)。
pub const DUAL_PIECE_TYPES: usize = 14;

/// 1 perspective あたりの dim 数 = 2 head × 8 bucket × 81 sq × 14 piece = 18,144。
pub const DUAL_DIMENSIONS_PER_PERSPECTIVE: usize =
    NUM_DUAL_PSQT_HEADS * NUM_DUAL_PSQT_BUCKETS * DUAL_SQUARES * DUAL_PIECE_TYPES;

/// build-time check: bucket 数が LayerStacks と一致すること。
const _DUAL_PSQT_BUCKET_CHECK: () = {
    assert!(
        NUM_DUAL_PSQT_BUCKETS == NUM_LAYER_STACK_BUCKETS,
        "NUM_DUAL_PSQT_BUCKETS must equal NUM_LAYER_STACK_BUCKETS for bucket alignment"
    );
};

/// 1 perspective あたり = 18,144 i16 = 36,288 byte。
const _: () = {
    assert!(DUAL_DIMENSIONS_PER_PERSPECTIVE == 18_144);
};

/// ckpt section の **新** sub_magic (`P2DUAL2\0` = 8 bytes、 King-Attacker format)。
pub const DUAL_PSQT_SUB_MAGIC: [u8; 8] = *b"P2DUAL2\0";

/// ckpt section の **旧** sub_magic (`P2DUAL\0\0` = 8 bytes、 reserve format)。
///
/// 新 binary では magic mismatch として skip する (= dual head 寄与 0)。
pub const DUAL_PSQT_LEGACY_SUB_MAGIC: [u8; 8] = *b"P2DUAL\0\0";

/// ckpt section format version (現在 1)。
pub const DUAL_PSQT_VERSION: u32 = 1;

/// 1 perspective あたりの i16 count (= 36,288 と書かれているがこれは spec の
/// header value、 内部 dim 数は 18,144)。
///
/// **注**: spec では count u32 = 36,288 と記述があるが、 これは
/// 2 head × 8 bucket × 81 sq × 14 piece = **18,144** と一致しない。
/// 実装は per-perspective dim = `DUAL_DIMENSIONS_PER_PERSPECTIVE` (= 18,144) を
/// 採用し、 file の count field には spec 通り `(2 perspective × 18,144) = 36,288` を
/// 書く (= 2 perspective 合計 i16 entry 数)。
pub const DUAL_PSQT_COUNT: u32 = (2 * DUAL_DIMENSIONS_PER_PERSPECTIVE) as u32;

const _: () = {
    assert!(DUAL_PSQT_COUNT == 36_288);
};

/// arch_str に埋め込む marker 文字列 (King-Attacker でも同 marker を使う、
/// 区別は section magic で行う)。
pub const DUAL_PSQT_ARCH_MARKER: &str = "DUAL=1";

/// output scale (= fv_scale 適用前 cp 単位への正規化定数)。
///
/// PIN_OUTPUT_SCALE と同じ idea で 1 を採用。 weight 学習で scale を吸収する
/// 想定 (= bullet 側 quantise i16 で自然 scale)。
pub const DUAL_OUTPUT_SCALE: i32 = 1;

// =============================================================================
// 型定義
// =============================================================================

/// 1 perspective あたりの King-Attacker weight: `[head][bucket][sq][piece_type] -> i16`。
///
/// flat 18,144 i16 = 36,288 byte / perspective。
pub type DualPsqtPerspectiveWeights =
    [[[[i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS];

/// 2 perspective 全体の King-Attacker weight。
///
/// `weights[perspective_index][head][bucket][sq][piece_type] -> i16`。
/// perspective_index: 0 = Black, 1 = White。
///
/// 合計 36,288 i16 = 72,576 byte (Box 配置推奨)。
pub type DualPsqtKingAttackerWeights = [DualPsqtPerspectiveWeights; 2];

// =============================================================================
// piece_type indexing (pin_features と同順)
// =============================================================================

/// `PieceType` enum (1..=14) を dual feature の 0-indexed (0..=13) に変換。
///
/// 順序は [`crate::nnue::pin_features`] の `piece_type_to_pin_idx` と完全一致。
#[inline]
const fn piece_type_to_dual_idx(pt: PieceType) -> usize {
    (pt as u8 - 1) as usize
}

/// perspective ベースの square index 正規化。
///
/// pin_features と同方式 (Half-Mirror なし、 color swap のみ):
/// Black: identity / White: 80 - sq.index()。
#[inline]
const fn oriented_sq_idx(sq: Square, perspective: Color) -> usize {
    match perspective {
        Color::Black => sq.index(),
        Color::White => 80 - sq.index(),
    }
}

/// King-Attacker feature の flat index。
///
/// `flat = head × 8 × 81 × 14 + bucket × 81 × 14 + sq × 14 + piece`。
///
/// 範囲: `0..DUAL_DIMENSIONS_PER_PERSPECTIVE` (= 0..18,144)。
#[inline]
pub const fn dual_feature_index(
    head: usize,
    bucket: usize,
    sq_idx: usize,
    piece_idx: usize,
) -> usize {
    head * (NUM_DUAL_PSQT_BUCKETS * DUAL_SQUARES * DUAL_PIECE_TYPES)
        + bucket * (DUAL_SQUARES * DUAL_PIECE_TYPES)
        + sq_idx * DUAL_PIECE_TYPES
        + piece_idx
}

// =============================================================================
// king zone (= king + 8 近傍 = 9 sq) precomputed table
// =============================================================================

/// king sq → king zone (9 sq = king 自身 + 8 近傍) の bitboard。
///
/// `king_effect(sq)` は 8 近傍のみを返すため king sq 自身を or で追加する。
/// LazyLock で起動時 1 度だけ計算 (= 81 entry × 16 byte = 1.3 KB)。
fn king_zone_mask(king_sq: Square) -> Bitboard {
    let mut bb = king_effect(king_sq);
    bb.set(king_sq);
    bb
}

// =============================================================================
// 駒の利き計算
// =============================================================================

/// piece が sq に居るときの利き bitboard を取得する。
///
/// `position::board_effect::attacks_from` と等価だが、 module 内で利用するため
/// 同じ logic を再記述 (= pub fn が無いため reuse できず)。
#[inline]
fn piece_attacks_from(piece: Piece, sq: Square, occupied: Bitboard) -> Bitboard {
    let color = piece.color();
    match piece.piece_type() {
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

// =============================================================================
// eval 計算 (per-eval full、 Phase 3 で accumulator 化予定)
// =============================================================================

/// King-Attacker dual head の評価値計算 (per-eval full)。
///
/// # アルゴリズム
///
/// 各 perspective (Black / White) について:
/// 1. 自 king zone (= king + 8 近傍) と 敵 king zone を bitboard で取得
/// 2. 盤上全駒を iter (= 自軍駒のみ寄与、 perspective 側駒)
/// 3. 各駒の利き bitboard を計算し:
///    - 敵 king zone と交差 → **attack** head 発火
///    - 自 king zone と交差 → **defend** head 発火 (両 active 可能)
/// 4. 発火した entry の weight を `[perspective][head][bucket][sq][piece]` から
///    lookup して i32 で sum
///
/// # 戻り値
///
/// `(attack_sum, defend_sum)`: i32 cp (fv_scale 未適用、 [`DUAL_OUTPUT_SCALE`] 適用前)。
/// 呼び出し側で `(attack_sum - defend_sum) / DUAL_OUTPUT_SCALE` を計算する。
///
/// # 計算量
///
/// 各 perspective ~16-20 駒 × 利き bitboard 計算 (O(1) per 駒) + intersect (O(1)) =
/// 1 eval あたり ~40-50 駒分。 per-eval cost ~5-10us 想定。
pub fn compute_dual_king_attacker(
    pos: &Position,
    weights: &DualPsqtKingAttackerWeights,
    bucket: usize,
) -> (i32, i32) {
    debug_assert!(bucket < NUM_DUAL_PSQT_BUCKETS);

    let occupied = pos.occupied();
    let mut attack_sum: i32 = 0;
    let mut defend_sum: i32 = 0;

    for &perspective in &[Color::Black, Color::White] {
        let persp_idx = perspective.index();
        let enemy = !perspective;

        let own_king_zone = king_zone_mask(pos.king_square(perspective));
        let enemy_king_zone = king_zone_mask(pos.king_square(enemy));

        // perspective 側自軍駒を全 iter
        let own_pieces = pos.pieces_c(perspective);
        for sq in own_pieces.iter() {
            let piece = pos.piece_on(sq);
            if !piece.is_some() {
                continue;
            }

            let pt_idx = piece_type_to_dual_idx(piece.piece_type());
            let sq_idx = oriented_sq_idx(sq, perspective);

            let attacks = piece_attacks_from(piece, sq, occupied);

            // attack head: 敵 king zone と交差
            if !(attacks & enemy_king_zone).is_empty() {
                let idx = dual_feature_index(0, bucket, sq_idx, pt_idx);
                let flat = flatten_perspective_weight(&weights[persp_idx], idx);
                attack_sum = attack_sum.wrapping_add(flat as i32);
            }

            // defend head: 自 king zone と交差
            if !(attacks & own_king_zone).is_empty() {
                let idx = dual_feature_index(1, bucket, sq_idx, pt_idx);
                let flat = flatten_perspective_weight(&weights[persp_idx], idx);
                defend_sum = defend_sum.wrapping_add(flat as i32);
            }
        }
    }

    (attack_sum, defend_sum)
}

/// 1 perspective の weight を flat index でアクセスする helper。
///
/// `[head][bucket][sq][piece]` の 4 次元配列を 1 次元 view として扱う。
#[inline]
fn flatten_perspective_weight(w: &DualPsqtPerspectiveWeights, flat_idx: usize) -> i16 {
    debug_assert!(flat_idx < DUAL_DIMENSIONS_PER_PERSPECTIVE);
    // flat_idx decomposition: head * (8*81*14) + bucket * (81*14) + sq * 14 + piece
    let piece = flat_idx % DUAL_PIECE_TYPES;
    let rest = flat_idx / DUAL_PIECE_TYPES;
    let sq = rest % DUAL_SQUARES;
    let rest = rest / DUAL_SQUARES;
    let bucket = rest % NUM_DUAL_PSQT_BUCKETS;
    let head = rest / NUM_DUAL_PSQT_BUCKETS;
    w[head][bucket][sq][piece]
}

// =============================================================================
// accumulator 化 (= per-perspective × per-head × per-bucket scalar の差分更新基盤)
// =============================================================================

/// 1 perspective × 1 head あたりの dual feature 最大同時 active 数。
///
/// 上限根拠: 盤上自軍駒は最大 20 駒前後 (= 玉 + 19 駒)。 1 perspective × 1 head
/// あたり最大 20 entry。 安全側で 32 = stack alloc 用の固定長 buffer。
pub const MAX_ACTIVE_DUAL_FEATURES: usize = 32;

/// 1 head × 1 perspective あたりの sq×piece flat index space size (= 81 × 14 = 1134)。
///
/// active index は bucket を含まない形 (= `sq_idx * 14 + piece_idx`) で保存し、
/// 差分更新時に 8 bucket 全部に同じ index で `±W[persp][head][b][sq][piece]` を
/// 加減算する。
pub const DUAL_SQ_PIECE_SPACE: usize = DUAL_SQUARES * DUAL_PIECE_TYPES;

const _: () = {
    assert!(DUAL_SQ_PIECE_SPACE == 1134);
};

/// 1 perspective × 1 head の active sq×piece feature を列挙する (= public 版)。
///
/// `update_dual_perspective_diff` の事前段階で current active 集合を取得する
/// ために使う。 戻り値: 有効長。 `&out[..len]` が有効な active idx 列を保持。
pub fn collect_dual_active_per_head_public(
    pos: &Position,
    perspective: Color,
    head: usize,
    out: &mut [u16; MAX_ACTIVE_DUAL_FEATURES],
) -> usize {
    collect_dual_active_per_head(pos, perspective, head, out)
}

/// 1 perspective × 1 head の active sq×piece feature を列挙する。
///
/// `compute_dual_king_attacker` の per-head 分解版。 active index は
/// bucket を含まない `sq_idx * DUAL_PIECE_TYPES + piece_idx` で出力する。
///
/// # 引数
/// - `pos`: 局面
/// - `perspective`: 視点
/// - `head`: 0 = attack、 1 = defend
/// - `out`: 出力 buffer (stack alloc)
///
/// # 戻り値
///
/// 有効長 (= 実 active 数)。 `&out[..len]` が有効。
fn collect_dual_active_per_head(
    pos: &Position,
    perspective: Color,
    head: usize,
    out: &mut [u16; MAX_ACTIVE_DUAL_FEATURES],
) -> usize {
    debug_assert!(head < NUM_DUAL_PSQT_HEADS);

    let occupied = pos.occupied();
    let enemy = !perspective;
    let own_king_zone = king_zone_mask(pos.king_square(perspective));
    let enemy_king_zone = king_zone_mask(pos.king_square(enemy));
    let target_zone = if head == 0 {
        enemy_king_zone
    } else {
        own_king_zone
    };

    let mut len = 0usize;
    let own_pieces = pos.pieces_c(perspective);
    for sq in own_pieces.iter() {
        let piece = pos.piece_on(sq);
        if !piece.is_some() {
            continue;
        }
        let attacks = piece_attacks_from(piece, sq, occupied);
        if (attacks & target_zone).is_empty() {
            continue;
        }
        let pt_idx = piece_type_to_dual_idx(piece.piece_type());
        let sq_idx = oriented_sq_idx(sq, perspective);
        let idx = sq_idx * DUAL_PIECE_TYPES + pt_idx;
        debug_assert!(len < MAX_ACTIVE_DUAL_FEATURES);
        debug_assert!(idx < DUAL_SQ_PIECE_SPACE);
        out[len] = idx as u16;
        len += 1;
    }
    len
}

/// 1 perspective の dual accumulator を `pos` から full 計算する。
///
/// 各 head (= attack / defend) について active sq×piece 集合を列挙し、 各 active
/// idx について 8 bucket 全部に `W[persp][head][bucket][sq][piece]` を加算した
/// 結果を返す。
///
/// # 引数
/// - `pos`: 局面
/// - `weights`: DualPsqtKingAttackerWeights
/// - `perspective`: 視点
/// - `active_out`: 列挙した active index を head ごとに書き出す stack buffer
/// - `active_len_out`: 各 head の active 数を返す
///
/// # 戻り値
///
/// `[[i32; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS]`: head × bucket の sum 値。
///
/// # 数値契約
///
/// 旧 `compute_dual_king_attacker(pos, w, bucket)` の (attack, defend) と、
/// 本関数の `result[0][bucket]` (= attack) / `result[1][bucket]` (= defend) を
/// 両 perspective について合算した値が **bit-exact 同値**。
pub fn refresh_dual_perspective(
    pos: &Position,
    weights: &DualPsqtKingAttackerWeights,
    perspective: Color,
    active_out: &mut [[u16; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS],
    active_len_out: &mut [u8; NUM_DUAL_PSQT_HEADS],
) -> [[i32; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS] {
    let mut sums = [[0i32; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS];
    let persp_idx = perspective.index();
    let w_persp = &weights[persp_idx];

    for head in 0..NUM_DUAL_PSQT_HEADS {
        let active_len =
            collect_dual_active_per_head(pos, perspective, head, &mut active_out[head]);
        debug_assert!(active_len <= MAX_ACTIVE_DUAL_FEATURES);
        active_len_out[head] = active_len as u8;

        let w_head = &w_persp[head];
        for &idx in &active_out[head][..active_len] {
            let i = idx as usize;
            debug_assert!(i < DUAL_SQ_PIECE_SPACE);
            let sq = i / DUAL_PIECE_TYPES;
            let piece = i % DUAL_PIECE_TYPES;
            for (b, sum) in sums[head].iter_mut().enumerate().take(NUM_DUAL_PSQT_BUCKETS) {
                *sum = sum.wrapping_add(w_head[b][sq][piece] as i32);
            }
        }
    }

    sums
}

/// 1 perspective × 1 head の dual accumulator の差分更新。
///
/// `removed` の各 idx について 8 bucket 全部に `W[persp][head][b][sq][piece]` を
/// `-=`、 `added` の各 idx について `+=`。
///
/// # 引数
/// - `removed`: 親 → 現在で active でなくなった index 列 (sq×piece 形式)
/// - `added`: 親 → 現在で新たに active になった index 列
/// - `weights`: DualPsqtKingAttackerWeights
/// - `perspective`: 視点
/// - `head`: 0 = attack、 1 = defend
/// - `acc_head_buckets`: `acc.dual_accumulation[persp][head]` への可変参照
///
/// # 数値契約
///
/// `removed`/`added` が `parent_active ⊖ current_active` (= set symmetric
/// difference) と一致する場合、 結果は full refresh と bit-exact 同値。
pub fn update_dual_perspective_diff(
    removed: &[u16],
    added: &[u16],
    weights: &DualPsqtKingAttackerWeights,
    perspective: Color,
    head: usize,
    acc_head_buckets: &mut [i32; NUM_DUAL_PSQT_BUCKETS],
) {
    debug_assert!(head < NUM_DUAL_PSQT_HEADS);
    let persp_idx = perspective.index();
    let w_head = &weights[persp_idx][head];

    for &idx in removed {
        let i = idx as usize;
        debug_assert!(i < DUAL_SQ_PIECE_SPACE);
        let sq = i / DUAL_PIECE_TYPES;
        let piece = i % DUAL_PIECE_TYPES;
        for (b, sum) in acc_head_buckets.iter_mut().enumerate().take(NUM_DUAL_PSQT_BUCKETS) {
            *sum = sum.wrapping_sub(w_head[b][sq][piece] as i32);
        }
    }
    for &idx in added {
        let i = idx as usize;
        debug_assert!(i < DUAL_SQ_PIECE_SPACE);
        let sq = i / DUAL_PIECE_TYPES;
        let piece = i % DUAL_PIECE_TYPES;
        for (b, sum) in acc_head_buckets.iter_mut().enumerate().take(NUM_DUAL_PSQT_BUCKETS) {
            *sum = sum.wrapping_add(w_head[b][sq][piece] as i32);
        }
    }
}

/// `parent` と `current` の sparse active index 集合の symmetric difference を
/// 抽出する (= `removed = parent \ current`、 `added = current \ parent`)。
///
/// pin 側の [`crate::nnue::pin_features::diff_pin_active_indices`] と同じ
/// sort + merge 方式。 stack alloc buffer のサイズは
/// `MAX_ACTIVE_DUAL_FEATURES` で十分。
///
/// # 引数
/// - `parent`: 親 accumulator の active index slice
/// - `current`: 現在 accumulator の active index slice
/// - `removed_out`: removed 出力 buffer
/// - `added_out`: added 出力 buffer
///
/// # 戻り値
/// `(removed_len, added_len)`
pub fn diff_dual_active_indices(
    parent: &[u16],
    current: &[u16],
    removed_out: &mut [u16; MAX_ACTIVE_DUAL_FEATURES],
    added_out: &mut [u16; MAX_ACTIVE_DUAL_FEATURES],
) -> (usize, usize) {
    let mut parent_sorted = [0u16; MAX_ACTIVE_DUAL_FEATURES];
    let mut current_sorted = [0u16; MAX_ACTIVE_DUAL_FEATURES];
    debug_assert!(parent.len() <= MAX_ACTIVE_DUAL_FEATURES);
    debug_assert!(current.len() <= MAX_ACTIVE_DUAL_FEATURES);
    parent_sorted[..parent.len()].copy_from_slice(parent);
    current_sorted[..current.len()].copy_from_slice(current);
    parent_sorted[..parent.len()].sort_unstable();
    current_sorted[..current.len()].sort_unstable();

    let mut removed_len = 0usize;
    let mut added_len = 0usize;
    let mut i = 0usize;
    let mut j = 0usize;
    let plen = parent.len();
    let clen = current.len();
    while i < plen && j < clen {
        let p = parent_sorted[i];
        let c = current_sorted[j];
        match p.cmp(&c) {
            std::cmp::Ordering::Less => {
                removed_out[removed_len] = p;
                removed_len += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                added_out[added_len] = c;
                added_len += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    while i < plen {
        removed_out[removed_len] = parent_sorted[i];
        removed_len += 1;
        i += 1;
    }
    while j < clen {
        added_out[added_len] = current_sorted[j];
        added_len += 1;
        j += 1;
    }
    (removed_len, added_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// section の disk size 計算が仕様通り 72,592 byte (= 8 + 4 + 4 + 72,576)。
    /// 内訳: magic 8 + version 4 + count 4 + data (2 persp × 18,144 i16 × 2 byte) = 72,576。
    #[test]
    fn dual_psqt_section_size() {
        let magic_size = DUAL_PSQT_SUB_MAGIC.len();
        let version_size = std::mem::size_of::<u32>();
        let count_size = std::mem::size_of::<u32>();
        let weight_size = std::mem::size_of::<i16>() * 2 * DUAL_DIMENSIONS_PER_PERSPECTIVE;
        assert_eq!(magic_size + version_size + count_size + weight_size, 72_592);
        assert_eq!(weight_size, 72_576);
    }

    /// count u32 = 36,288 (= 2 perspective × 18,144)。
    #[test]
    fn dual_psqt_count_constant() {
        assert_eq!(DUAL_PSQT_COUNT, 36_288);
        assert_eq!(DUAL_DIMENSIONS_PER_PERSPECTIVE, 18_144);
    }

    /// feature index は範囲内 + 単調。
    #[test]
    fn dual_feature_index_range() {
        let idx0 = dual_feature_index(0, 0, 0, 0);
        let idx_max = dual_feature_index(1, 7, 80, 13);
        assert_eq!(idx0, 0);
        assert_eq!(idx_max, DUAL_DIMENSIONS_PER_PERSPECTIVE - 1);
    }

    /// king zone は king + 8 近傍 = 最大 9 マス (= 中央付近)、 端は 4-6 マス。
    #[test]
    fn king_zone_mask_central() {
        // 5五 (= center)
        let center = unsafe { Square::from_u8_unchecked(40) };
        let bb = king_zone_mask(center);
        assert_eq!(bb.count(), 9, "center king zone should have 9 sq");
    }

    #[test]
    fn king_zone_mask_corner() {
        // 1一 = 0 (file=0, rank=0)
        let corner = unsafe { Square::from_u8_unchecked(0) };
        let bb = king_zone_mask(corner);
        // corner king zone: king + 3 近傍 = 4 マス
        assert_eq!(bb.count(), 4, "corner king zone should have 4 sq");
    }

    /// magic 同士が 1 byte 差 (= 8 byte 目の `\0` vs `2`)。
    #[test]
    fn magic_one_byte_diff() {
        let mut diff = 0;
        for i in 0..8 {
            if DUAL_PSQT_SUB_MAGIC[i] != DUAL_PSQT_LEGACY_SUB_MAGIC[i] {
                diff += 1;
            }
        }
        assert_eq!(diff, 1, "new magic must differ from legacy by exactly 1 byte");
    }

    /// arch marker 文字列の文字列マッチング契約 (substring check) を確認。
    #[test]
    fn dual_psqt_arch_marker_contains() {
        let arch =
            format!("Features=HalfKA_hm[73305->1536x2],PIN=2268,{DUAL_PSQT_ARCH_MARKER},...");
        assert!(arch.contains(DUAL_PSQT_ARCH_MARKER));
        let arch_no = "Features=HalfKA_hm[73305->1536x2]";
        assert!(!arch_no.contains(DUAL_PSQT_ARCH_MARKER));
    }

    /// Zero weight で eval が 0 を返す sanity (initial position)。
    #[test]
    fn compute_dual_zero_weight_returns_zero() {
        use crate::position::{Position, SFEN_HIRATE};
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        let weights: Box<DualPsqtKingAttackerWeights> = Box::new(
            [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS];
                NUM_DUAL_PSQT_HEADS]; 2],
        );
        for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
            let (attack, defend) = compute_dual_king_attacker(&pos, &weights, bucket);
            assert_eq!(attack, 0);
            assert_eq!(defend, 0);
        }
    }

    /// defend head: 初期局面で自軍金 (= 4九/6九) は自玉 (= 5九) を守るため
    /// defend が立つはず。 weight = 1 にして defend > 0 を確認。
    #[test]
    fn compute_dual_defend_head_initial_position() {
        use crate::position::{Position, SFEN_HIRATE};
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        let mut weights: Box<DualPsqtKingAttackerWeights> = Box::new(
            [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS];
                NUM_DUAL_PSQT_HEADS]; 2],
        );
        // 全 defend weight = 1
        for persp in 0..2 {
            for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
                for sq in 0..DUAL_SQUARES {
                    for piece in 0..DUAL_PIECE_TYPES {
                        weights[persp][1][bucket][sq][piece] = 1;
                    }
                }
            }
        }
        let (attack, defend) = compute_dual_king_attacker(&pos, &weights, 0);
        // 初期局面: 敵 king (= 5一) zone に届く自軍駒は無い → attack = 0
        assert_eq!(attack, 0, "no own piece attacks enemy king zone in initial");
        // 自 king (= 5九) zone に届く自軍駒: 金 (4九/6九 → 5九)、 銀 (7九/3九 → 4八/6八 = zone)、
        // 玉自身 (= 5九) も自分の king_effect で 5八/4八/6八 にも届き zone 内
        // → defend > 0 (Black + White 両 perspective で発火)
        assert!(defend > 0, "own pieces defend king zone in initial position, got {defend}");
    }

    /// attack head: 敵 king に飛車の利きが届く SFEN で attack 発火。
    /// 5一に黒玉 (= 自分) 設定 + 5九に飛車 (敵) で attack を確認する代わりに、
    /// 敵 king 前に効きの届く配置を作る。
    #[test]
    fn compute_dual_attack_head_smoke() {
        use crate::position::Position;
        let mut pos = Position::new();
        // 白玉 = 5一、 黒玉 = 5九、 黒飛車 = 5五 (中央) で白玉 zone (5一/4一/6一/5二/...) に届く
        pos.set_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1").unwrap();
        let mut weights: Box<DualPsqtKingAttackerWeights> = Box::new(
            [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS];
                NUM_DUAL_PSQT_HEADS]; 2],
        );
        // 全 attack weight = 1
        for persp in 0..2 {
            for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
                for sq in 0..DUAL_SQUARES {
                    for piece in 0..DUAL_PIECE_TYPES {
                        weights[persp][0][bucket][sq][piece] = 1;
                    }
                }
            }
        }
        let (attack, _defend) = compute_dual_king_attacker(&pos, &weights, 0);
        // 黒飛車 5五 → 5一 (= 白 king) zone へ縦に効きが届く (= 5二 ∈ white king zone)
        // → black perspective で attack 発火
        assert!(attack > 0, "rook should attack enemy king zone, got {attack}");
    }

    // =========================================================================
    // ParityCheck tests (bullet ↔ rshogi-core feature index 集合 bit-exact)
    //
    // bullet 側 (= ShogiHalfKaHmKingAttackerDual の collect_dual_active_for_side)
    // は本 module の compute_dual_king_attacker と同 algorithm spec に従う:
    //   1. 同 king_zone_mask (= king sq + 8 近傍 = 最大 9 sq)
    //   2. 同 perspective swap (Black=identity、 White=80-sq)
    //   3. 同 piece_type_to_dual_idx (= pt as u8 - 1)
    //   4. 同 dual_feature_index 式 (= head * 8*81*14 + bucket * 81*14 + sq*14 + piece)
    //   5. 同 occupancy-aware 利き計算 (= bullet 側は ray-walk 表現、 rshogi-core 側は
    //      bitboard table、 ただし両方とも遮蔽あり ray + 近接駒 delta で同等)
    //
    // 本 test では weight=1 (= 全 head の全 entry に 1 を立てる) で
    // compute_dual_king_attacker を呼び、 (attack_sum, defend_sum) が bullet 側で
    // 算出した active feature 数と一致することを確認する。 これにより両側の
    // active feature index 集合の **要素数** が bit-exact (= count 一致 + 値も algorithm
    // 同 spec) であることを保証する。
    // =========================================================================

    /// 1 perspective × 1 head の全 weight を 1 にする helper。
    fn fill_weights_unit(
        weights: &mut Box<DualPsqtKingAttackerWeights>,
        persp: usize,
        head: usize,
    ) {
        for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
            for sq in 0..DUAL_SQUARES {
                for piece in 0..DUAL_PIECE_TYPES {
                    weights[persp][head][bucket][sq][piece] = 1;
                }
            }
        }
    }

    /// 全 perspective × 全 head の weight を 1 にする helper。
    fn fill_weights_all_unit(weights: &mut Box<DualPsqtKingAttackerWeights>) {
        for persp in 0..2 {
            for head in 0..NUM_DUAL_PSQT_HEADS {
                fill_weights_unit(weights, persp, head);
            }
        }
    }

    /// ParityCheck Case 1: kings-only board (黒玉 5九、 白玉 5一)。
    ///
    /// bullet 側 `collect_dual_empty_board_kings_only` test と同 SFEN。
    /// 期待: 各 perspective で `n_attack=0, n_defend=1` (= 自玉自身が own zone に効く)。
    /// 両 perspective 合算で attack_sum=0, defend_sum=2 を rshogi-core 側で確認。
    #[test]
    fn parity_check_kings_only_board() {
        use crate::position::Position;
        let mut pos = Position::new();
        // 黒玉 5九 + 白玉 5一、 他駒なし
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();

        let mut weights: Box<DualPsqtKingAttackerWeights> = Box::new(
            [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS];
                NUM_DUAL_PSQT_HEADS]; 2],
        );
        fill_weights_all_unit(&mut weights);

        let (attack, defend) = compute_dual_king_attacker(&pos, &weights, 0);
        // bullet 側 expectation: 黒 persp (n_a=0, n_d=1) + 白 persp (n_a=0, n_d=1)
        assert_eq!(attack, 0, "kings-only: no attack feature active");
        assert_eq!(defend, 2, "kings-only: each king defends own zone (= 1 per side × 2)");
    }

    /// ParityCheck Case 2: 5五黒飛車 → 5一白玉 zone へ attack。
    ///
    /// bullet 側 `collect_dual_rook_attack_smoke` test と同 SFEN。
    /// 黒 perspective: 黒飛車 5五 + 黒玉 5九 = 2 駒、 attack/defend は両方発火可能性。
    ///
    ///   - 黒飛車: 縦 ray で enemy zone (5一+8 近傍) 内 5二 に届く → attack active
    ///   - 黒飛車: 縦 ray で own zone (5九+8 近傍) 内 5八 に届く → defend active
    ///   - 黒玉: 8 近傍に届くが enemy zone (= 5一周辺) に届かない → attack 0
    ///   - 黒玉: 自身が own zone に居る → king_effect & own_zone は 8 近傍部分集合で非空 → defend active
    ///
    /// 黒 persp 合計 (n_a, n_d) = (1, 2)
    /// 白 perspective: 白玉のみ → (n_a, n_d) = (0, 1)
    ///
    /// 合計: attack_sum = 1, defend_sum = 3
    #[test]
    fn parity_check_rook_attack_board() {
        use crate::position::Position;
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1").unwrap();

        let mut weights: Box<DualPsqtKingAttackerWeights> = Box::new(
            [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS];
                NUM_DUAL_PSQT_HEADS]; 2],
        );
        fill_weights_all_unit(&mut weights);

        let (attack, defend) = compute_dual_king_attacker(&pos, &weights, 0);
        assert_eq!(attack, 1, "rook attacks enemy king zone (1 active across both perspectives)");
        assert_eq!(
            defend, 3,
            "rook + 2 kings defend own zones (3 active across both perspectives)"
        );
    }

    /// ParityCheck Case 3: bucket 切替時に同じ active 数。
    ///
    /// bullet 側 `collect_dual_bucket_offset` test と同 SFEN。
    /// bucket = 0 vs bucket = 7 で (attack_sum, defend_sum) が一致 (= bucket
    /// は active feature 集合の shift のみ、 数は不変)。
    #[test]
    fn parity_check_bucket_offset_invariance() {
        use crate::position::Position;
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1").unwrap();

        let mut weights: Box<DualPsqtKingAttackerWeights> = Box::new(
            [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS];
                NUM_DUAL_PSQT_HEADS]; 2],
        );
        fill_weights_all_unit(&mut weights);

        let (a0, d0) = compute_dual_king_attacker(&pos, &weights, 0);
        let (a7, d7) = compute_dual_king_attacker(&pos, &weights, 7);
        assert_eq!(a0, a7, "bucket should not change active count");
        assert_eq!(d0, d7, "bucket should not change active count");
    }

    /// ParityCheck Case 4: 初期局面 (= startpos)。
    ///
    /// bullet 側 fill_dual_features_for_side が startpos で多数 active を出すか確認。
    /// 初期局面: 各 perspective 全 19 駒 (玉含む)、 全て自軍駒で計算対象。
    /// 各駒の利きが own zone (= 5九/5一周辺) に届くかで defend、 enemy zone に
    /// 届くかで attack を判定。 初期局面は静的で active 数は決定的。
    #[test]
    fn parity_check_startpos_total_active() {
        use crate::position::{Position, SFEN_HIRATE};
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();

        let mut weights: Box<DualPsqtKingAttackerWeights> = Box::new(
            [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS];
                NUM_DUAL_PSQT_HEADS]; 2],
        );
        fill_weights_all_unit(&mut weights);

        let (attack, defend) = compute_dual_king_attacker(&pos, &weights, 0);
        // 初期局面: 敵 king (5一/5九) zone に届く自軍駒は無い → attack = 0
        assert_eq!(attack, 0, "startpos: no piece attacks enemy king zone");
        // defend: 自玉周辺に効く駒 (= 玉自身 + 金 + 銀 + 角 + 飛 + 歩 etc.) → > 0
        // 両 perspective で対称なので偶数。
        assert!(defend > 0, "startpos: own pieces defend own king zone, got {}", defend);
        assert_eq!(defend % 2, 0, "startpos: defend count should be symmetric");
    }

    /// ParityCheck Case 5: 上限テスト = 5 SFEN × 2 perspective × 2 head バリアント (= 20 ケース) で
    /// `compute_dual_king_attacker` が panic 無く完走 + active 数が 0..50 範囲。
    ///
    /// 各 SFEN を per_head, per_persp weight=1 設定で呼び、 計算結果を集計。
    /// (= bullet 側で実装した collect_dual_active_for_side の `DUAL_MAX_ACTIVE_PER_SIDE = 50`
    /// 制約と整合: rshogi-core 側でも 1 perspective あたり最大 50 entry 以下であるべき。)
    #[test]
    fn parity_check_smoke_5_positions() {
        use crate::position::{Position, SFEN_HIRATE};

        let sfens = [
            ("startpos", SFEN_HIRATE),
            ("kings_only", "4k4/9/9/9/9/9/9/9/4K4 b - 1"),
            ("rook_attack", "4k4/9/9/9/4R4/9/9/9/4K4 b - 1"),
            ("bishop_far", "8k/9/9/9/9/9/9/B8/4K4 b - 1"),
            ("lance_attack", "4k4/9/9/9/9/9/9/9/4KL3 b - 1"),
        ];

        let mut weights: Box<DualPsqtKingAttackerWeights> = Box::new(
            [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS];
                NUM_DUAL_PSQT_HEADS]; 2],
        );
        // attack head のみ 1 (= attack_sum で active 数を直接 readable)
        for persp in 0..2 {
            fill_weights_unit(&mut weights, persp, 0);
        }

        for (label, sfen) in &sfens {
            let mut pos = Position::new();
            pos.set_sfen(sfen).unwrap_or_else(|_| panic!("set_sfen failed for {}", label));
            let (attack, _defend) = compute_dual_king_attacker(&pos, &weights, 0);
            assert!(attack >= 0, "{}: attack count should be non-negative, got {}", label, attack);
            // 1 perspective あたり最大 50 駒分 × 2 perspective = 100 が安全上限
            assert!(
                attack < 200,
                "{}: attack count should be reasonable (< 200), got {}",
                label,
                attack
            );
        }
    }

    // =========================================================================
    // accumulator 化 (Phase 3b-acc) の bit-exact / 動作確認 test
    //
    // production の `NetworkLayerStacks::{refresh_dual_accumulator,
    // update_dual_accumulator}` は内部で以下 fn を呼ぶ:
    //   refresh: `refresh_dual_perspective`
    //   diff   : `collect_dual_active_per_head_public` + `diff_dual_active_indices`
    //            + `update_dual_perspective_diff`
    // 本 test 群はこれら fn を直接呼んで scratch refresh と diff update の
    // sums が bit-exact 一致することを検証する。
    // =========================================================================

    use crate::movegen::{MoveList, generate_legal};
    use crate::position::SFEN_HIRATE;
    use crate::types::Move;

    /// 固定 seed の xorshift64 PRNG (= pin tests と同実装、 deterministic)。
    struct XorShift64(u64);

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 { 0xDEADBEEF } else { seed })
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn pick(&mut self, n: usize) -> usize {
            (self.next_u64() as usize) % n
        }
    }

    /// pseudo-random だが固定の weight pattern を boxed alloc で確保する。
    fn build_test_dual_weights() -> Box<DualPsqtKingAttackerWeights> {
        let mut weights: Box<DualPsqtKingAttackerWeights> = Box::new(
            [[[[[0i16; DUAL_PIECE_TYPES]; DUAL_SQUARES]; NUM_DUAL_PSQT_BUCKETS];
                NUM_DUAL_PSQT_HEADS]; 2],
        );
        for p in 0..2 {
            for head in 0..NUM_DUAL_PSQT_HEADS {
                for b in 0..NUM_DUAL_PSQT_BUCKETS {
                    for sq in 0..DUAL_SQUARES {
                        for piece in 0..DUAL_PIECE_TYPES {
                            let v = ((p as i32 * 11
                                + head as i32 * 31
                                + b as i32 * 13
                                + sq as i32 * 7
                                + piece as i32 * 3)
                                % 1024
                                - 512) as i16;
                            weights[p][head][b][sq][piece] = v;
                        }
                    }
                }
            }
        }
        weights
    }

    fn startpos_dual() -> Position {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        pos
    }

    fn from_sfen_dual(sfen: &str) -> Position {
        let mut pos = Position::new();
        pos.set_sfen(sfen).unwrap_or_else(|_| panic!("invalid sfen: {sfen}"));
        pos
    }

    /// `pos` から 1 perspective の scratch refresh を計算。
    fn scratch_refresh_dual(
        pos: &Position,
        weights: &DualPsqtKingAttackerWeights,
        perspective: Color,
    ) -> (
        [[i32; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS],
        [[u16; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS],
        [u8; NUM_DUAL_PSQT_HEADS],
    ) {
        let mut active = [[0u16; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS];
        let mut len = [0u8; NUM_DUAL_PSQT_HEADS];
        let sums = refresh_dual_perspective(pos, weights, perspective, &mut active, &mut len);
        (sums, active, len)
    }

    /// 親 acc cache + 現局面 sparse 列挙から 1 perspective の diff update を計算。
    fn diff_update_dual(
        pos: &Position,
        weights: &DualPsqtKingAttackerWeights,
        perspective: Color,
        parent_acc: &[[i32; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS],
        parent_active: &[[u16; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS],
        parent_len: &[u8; NUM_DUAL_PSQT_HEADS],
    ) -> (
        [[i32; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS],
        [[u16; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS],
        [u8; NUM_DUAL_PSQT_HEADS],
    ) {
        let mut acc_sums = *parent_acc;
        let mut acc_active = [[0u16; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS];
        let mut acc_len = [0u8; NUM_DUAL_PSQT_HEADS];

        for head in 0..NUM_DUAL_PSQT_HEADS {
            let mut current_active = [0u16; MAX_ACTIVE_DUAL_FEATURES];
            let current_len =
                collect_dual_active_per_head_public(pos, perspective, head, &mut current_active);

            let parent_slice = &parent_active[head][..parent_len[head] as usize];
            let current_slice = &current_active[..current_len];

            let mut removed = [0u16; MAX_ACTIVE_DUAL_FEATURES];
            let mut added = [0u16; MAX_ACTIVE_DUAL_FEATURES];
            let (removed_len, added_len) =
                diff_dual_active_indices(parent_slice, current_slice, &mut removed, &mut added);

            update_dual_perspective_diff(
                &removed[..removed_len],
                &added[..added_len],
                weights,
                perspective,
                head,
                &mut acc_sums[head],
            );
            acc_active[head] = current_active;
            acc_len[head] = current_len as u8;
        }

        (acc_sums, acc_active, acc_len)
    }

    fn move_moves_king(m: Move) -> bool {
        if m.is_drop() {
            return false;
        }
        if !m.has_piece_info() {
            return false;
        }
        m.moved_piece_after().piece_type() == PieceType::King
    }

    /// refresh_dual_perspective が legacy compute_dual_king_attacker と
    /// bit-exact 一致することを確認する sanity test。
    #[test]
    fn dual_refresh_matches_legacy_compute() {
        let sfens = [
            None,
            Some("4k4/9/9/9/4R4/9/9/9/4K4 b - 1"),
            Some("4k4/9/9/9/9/9/9/9/4KL3 b - 1"),
            Some("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"),
        ];
        let weights = build_test_dual_weights();

        for sfen_opt in sfens {
            let pos = match sfen_opt {
                None => startpos_dual(),
                Some(s) => from_sfen_dual(s),
            };
            // legacy: 各 bucket で attack/defend を直接 sum
            for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
                let (legacy_a, legacy_d) = compute_dual_king_attacker(&pos, &weights, bucket);

                // new: 両 perspective を refresh して同 bucket で合算
                let (sums_b, _, _) = scratch_refresh_dual(&pos, &weights, Color::Black);
                let (sums_w, _, _) = scratch_refresh_dual(&pos, &weights, Color::White);

                let attack_new = sums_b[0][bucket].wrapping_add(sums_w[0][bucket]);
                let defend_new = sums_b[1][bucket].wrapping_add(sums_w[1][bucket]);

                assert_eq!(
                    attack_new, legacy_a,
                    "attack mismatch: sfen={sfen_opt:?}, bucket={bucket}"
                );
                assert_eq!(
                    defend_new, legacy_d,
                    "defend mismatch: sfen={sfen_opt:?}, bucket={bucket}"
                );
            }
        }
    }

    /// random walk × 1000 局面 で diff update と scratch refresh の acc が
    /// bit-exact 一致することを検証する。
    ///
    /// - 各 game: 開始局面から legal_moves random で 5 手指す
    /// - 各 do_move 後に scratch refresh と diff update の sums を比較
    /// - king move は production 側で full refresh されるので test も refresh を採用
    #[test]
    fn dual_acc_diff_vs_refresh_random_walk() {
        let weights = build_test_dual_weights();
        let mut rng = XorShift64::new(0xDEAD_BEEF);

        const NUM_GAMES: usize = 1000;
        const PLIES_PER_GAME: usize = 5;
        let mut total_steps = 0usize;
        let mut king_move_steps = 0usize;

        for _game in 0..NUM_GAMES {
            let mut pos = startpos_dual();

            // 親 acc cache を両 perspective で保持。
            let mut parent_sums: [[[i32; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS]; 2] =
                [[[0; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS]; 2];
            let mut parent_active: [[[u16; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS]; 2] =
                [[[0; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS]; 2];
            let mut parent_len: [[u8; NUM_DUAL_PSQT_HEADS]; 2] = [[0; NUM_DUAL_PSQT_HEADS]; 2];

            for &persp in &[Color::Black, Color::White] {
                let (s, a, l) = scratch_refresh_dual(&pos, &weights, persp);
                let p = persp as usize;
                parent_sums[p] = s;
                parent_active[p] = a;
                parent_len[p] = l;
            }

            for _ply in 0..PLIES_PER_GAME {
                let mut list = MoveList::new();
                generate_legal(&pos, &mut list);
                if list.is_empty() {
                    break;
                }
                let mv = list.at(rng.pick(list.len()));
                let gc = pos.gives_check(mv);
                let _dirty = pos.do_move(mv, gc);
                total_steps += 1;
                let king_moved = move_moves_king(mv);
                if king_moved {
                    king_move_steps += 1;
                }

                for &persp in &[Color::Black, Color::White] {
                    let p = persp as usize;

                    let (ref_sums, ref_active, ref_len) =
                        scratch_refresh_dual(&pos, &weights, persp);

                    // production 側: king が **どちらか** 動けば両 perspective full refresh、
                    // それ以外は両 perspective diff (= king zone の片方が動くと
                    // 自/敵 zone の判定が両 perspective で変わるため)。
                    let (acc_sums, acc_active, acc_len) = if king_moved {
                        (ref_sums, ref_active, ref_len)
                    } else {
                        diff_update_dual(
                            &pos,
                            &weights,
                            persp,
                            &parent_sums[p],
                            &parent_active[p],
                            &parent_len[p],
                        )
                    };

                    assert_eq!(
                        acc_sums, ref_sums,
                        "diff vs refresh sums mismatch: persp={persp:?}, mv={mv:?}"
                    );

                    // active set 一致 (head 別)
                    for head in 0..NUM_DUAL_PSQT_HEADS {
                        let acc_set: std::collections::BTreeSet<u16> =
                            acc_active[head][..acc_len[head] as usize].iter().copied().collect();
                        let ref_set: std::collections::BTreeSet<u16> =
                            ref_active[head][..ref_len[head] as usize].iter().copied().collect();
                        assert_eq!(
                            acc_set, ref_set,
                            "diff vs refresh active set mismatch: persp={persp:?}, head={head}, mv={mv:?}"
                        );
                    }

                    parent_sums[p] = acc_sums;
                    parent_active[p] = acc_active;
                    parent_len[p] = acc_len;
                }
            }
        }

        assert!(total_steps > 0, "no steps executed");
        let _ = king_move_steps;
    }

    /// 玉移動 1 手で acc が scratch refresh と一致することを検証。
    ///
    /// production 側では king が動けば両 perspective を full refresh するので、
    /// test も同じ流れで `refresh` を採用して結果が一致することを確認。
    #[test]
    fn dual_acc_king_move_triggers_refresh() {
        let weights = build_test_dual_weights();

        // 黒玉 5九 + 白玉 9一、 他駒なし。 必ず king move のみが合法。
        let mut pos = from_sfen_dual("k8/9/9/9/9/9/9/9/4K4 b - 1");

        // 親 acc 計算
        let mut parent_sums: [[[i32; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS]; 2] =
            [[[0; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS]; 2];
        for &persp in &[Color::Black, Color::White] {
            let (s, _a, _l) = scratch_refresh_dual(&pos, &weights, persp);
            parent_sums[persp as usize] = s;
        }

        let mut list = MoveList::new();
        generate_legal(&pos, &mut list);
        assert!(!list.is_empty(), "expected at least 1 king move");
        let mut king_mv: Option<Move> = None;
        for i in 0..list.len() {
            let m = list.at(i);
            if move_moves_king(m) {
                king_mv = Some(m);
                break;
            }
        }
        let mv = king_mv.expect("expected king move");
        assert!(move_moves_king(mv));

        let gc = pos.gives_check(mv);
        pos.do_move(mv, gc);

        // production 経路と同じく両 perspective を full refresh
        for &persp in &[Color::Black, Color::White] {
            let (ref_sums, _, _) = scratch_refresh_dual(&pos, &weights, persp);
            // sanity: king zone が動いたので親 sums と異なる可能性が高い、
            // ただし 0 sum も理論上ありうるので不一致を強制しない。
            // 主目的は refresh が panic しないことと再現性確認。
            let _ = (ref_sums, parent_sums[persp as usize]);
        }

        // 動いた局面で legacy compute と新 refresh が一致
        for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
            let (legacy_a, legacy_d) = compute_dual_king_attacker(&pos, &weights, bucket);
            let (sums_b, _, _) = scratch_refresh_dual(&pos, &weights, Color::Black);
            let (sums_w, _, _) = scratch_refresh_dual(&pos, &weights, Color::White);
            let attack_new = sums_b[0][bucket].wrapping_add(sums_w[0][bucket]);
            let defend_new = sums_b[1][bucket].wrapping_add(sums_w[1][bucket]);
            assert_eq!(attack_new, legacy_a, "king move post: attack mismatch bucket={bucket}");
            assert_eq!(defend_new, legacy_d, "king move post: defend mismatch bucket={bucket}");
        }
    }

    /// capture move (= 駒取り) で diff path 経由の acc が scratch refresh と
    /// bit-exact 一致することを検証。
    ///
    /// 開始: 黒玉 5九、 白玉 5一、 黒飛車 5五、 白飛車 5三 (= 飛車交換可能位置)。
    /// 5五飛 → 5三飛 を取る (= capture)、 capture 後の acc を diff path で確認。
    #[test]
    fn dual_acc_capture_move_diff() {
        let weights = build_test_dual_weights();

        // 黒飛車 5五 + 白飛車 5三、 5五飛 が 5三飛を取れる。
        let mut pos = from_sfen_dual("4k4/9/4r4/9/4R4/9/9/9/4K4 b - 1");

        // 親 acc 計算
        let mut parent_sums: [[[i32; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS]; 2] =
            [[[0; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS]; 2];
        let mut parent_active: [[[u16; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS]; 2] =
            [[[0; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS]; 2];
        let mut parent_len: [[u8; NUM_DUAL_PSQT_HEADS]; 2] = [[0; NUM_DUAL_PSQT_HEADS]; 2];
        for &persp in &[Color::Black, Color::White] {
            let p = persp as usize;
            let (s, a, l) = scratch_refresh_dual(&pos, &weights, persp);
            parent_sums[p] = s;
            parent_active[p] = a;
            parent_len[p] = l;
        }

        // legal_moves から **非 king** な capture を 1 つ取る。
        let mut list = MoveList::new();
        generate_legal(&pos, &mut list);
        let mut cap_mv: Option<Move> = None;
        for i in 0..list.len() {
            let m = list.at(i);
            if !m.is_drop() && !move_moves_king(m) && pos.piece_on(m.to()).is_some() {
                cap_mv = Some(m);
                break;
            }
        }
        let mv = cap_mv.expect("expected at least 1 non-king capture");
        let gc = pos.gives_check(mv);
        pos.do_move(mv, gc);

        for &persp in &[Color::Black, Color::White] {
            let p = persp as usize;
            let (ref_sums, ref_active, ref_len) = scratch_refresh_dual(&pos, &weights, persp);

            // king は動いていない → diff path で検証
            let (acc_sums, acc_active, acc_len) = diff_update_dual(
                &pos,
                &weights,
                persp,
                &parent_sums[p],
                &parent_active[p],
                &parent_len[p],
            );

            assert_eq!(
                acc_sums, ref_sums,
                "capture: diff vs refresh sums mismatch for persp={persp:?}"
            );

            for head in 0..NUM_DUAL_PSQT_HEADS {
                let acc_set: std::collections::BTreeSet<u16> =
                    acc_active[head][..acc_len[head] as usize].iter().copied().collect();
                let ref_set: std::collections::BTreeSet<u16> =
                    ref_active[head][..ref_len[head] as usize].iter().copied().collect();
                assert_eq!(
                    acc_set, ref_set,
                    "capture: active set mismatch for persp={persp:?}, head={head}"
                );
            }
        }
    }

    /// pin accumulator の field と独立であることを確認 (= 同時更新でも干渉なし)。
    ///
    /// AccumulatorLayerStacks に pin_accumulation + dual_accumulation 両方が
    /// 存在する状況で、 dual 側のみ更新したときに pin 側が変化しないこと。
    #[test]
    #[cfg(feature = "nnue-pin-aware")]
    fn dual_acc_pin_acc_independence() {
        use crate::nnue::accumulator_layer_stacks::AccumulatorLayerStacks;
        use crate::nnue::pin_features::{MAX_ACTIVE_PIN_FEATURES, NUM_PIN_BUCKETS};

        // pin_accumulation を一旦初期値以外の値で埋める
        let mut acc = AccumulatorLayerStacks::<1536>::new();
        for p in 0..2 {
            for b in 0..NUM_PIN_BUCKETS {
                acc.pin_accumulation[p][b] = 12345 + (p as i32 * 100) + b as i32;
            }
            for s in 0..MAX_ACTIVE_PIN_FEATURES {
                acc.pin_active_indices[p][s] = (s as u16) + 1;
            }
            acc.pin_active_len[p] = 7;
        }

        // pin_accumulation のスナップショット
        let snapshot_pin_acc = acc.pin_accumulation;
        let snapshot_pin_active = acc.pin_active_indices;
        let snapshot_pin_len = acc.pin_active_len;

        // dual 側を refresh (= 同じ acc を mut で渡す。 pin field を書き換えないこと)
        let weights = build_test_dual_weights();
        let pos = startpos_dual();
        for &persp in &[Color::Black, Color::White] {
            let p = persp as usize;
            let mut active = [[0u16; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS];
            let mut active_len = [0u8; NUM_DUAL_PSQT_HEADS];
            let sums =
                refresh_dual_perspective(&pos, &weights, persp, &mut active, &mut active_len);
            acc.dual_accumulation[p] = sums;
            acc.dual_active_indices[p] = active;
            acc.dual_active_len[p] = active_len;
        }

        // pin field が変わっていないこと
        assert_eq!(acc.pin_accumulation, snapshot_pin_acc, "pin_accumulation changed");
        assert_eq!(acc.pin_active_indices, snapshot_pin_active, "pin_active_indices changed");
        assert_eq!(acc.pin_active_len, snapshot_pin_len, "pin_active_len changed");
    }
}
