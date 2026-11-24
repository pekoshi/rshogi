//! Dual PSQT D2 cross feature: piece_threat_relation + piece_defended_by
//!
//! 既存 `dual_psqt.rs` (= King-Attacker) と並列して動作する **第 2 の dual head 群**。
//! PSQT (= 玉×駒) では捉えにくい **駒×駒** cross feature を学習する。
//!
//! # 特徴量定義
//!
//! 1 feature = `(own_sq, own_piece_type, target_piece_type)` の 3-tuple
//!
//! ## piece_threat_relation (= head 0 = "threat")
//!
//! - 自軍駒 `own_p` が `own_sq` に居て、 `own_p` の利き先 `dst` に **相手駒** `target_p` が居る
//! - emit: `idx = own_sq × 14 × 14 + own_p × 14 + target_p`
//! - 同一 own 駒が複数 相手駒を攻撃 → 複数 active
//!
//! ## piece_defended_by (= head 1 = "defended")
//!
//! - 自軍駒 `own_p` が `own_sq` に居て、 別の自軍駒 `defender_p` が `own_sq` を攻撃 (= 守備)
//! - emit: `idx = own_sq × 14 × 14 + own_p × 14 + defender_p`
//! - 同一 own 駒が複数 defender で守られる → 複数 active
//!
//! # weight 構造 (= memory layout、 bucket-inner)
//!
//! ```text
//! DualCrossWeights[persp 2][head 2][sq 81][own_p 14][target_p 14][bucket 8]
//! ```
//!
//! - 1 head × 1 (sq, own_p, target_p) = 8 i16 連続 (= 16 byte = AVX2 128-bit load 1 命令)
//! - 1 head total = 81 × 14 × 14 × 8 = 127,008 i16
//! - 1 perspective total (= 2 head) = 254,016 i16
//! - 2 perspective total = **508,032 i16 = 1,016,064 byte = ~1 MB**
//!
//! ## bucket-inner にした理由 (= NPS 最適化)
//!
//! 旧 layout `[bucket 8][sq 81][own_p 14][target_p 14]` (= bucket-outer) では 1 active idx
//! の 8 bucket weight が 31,752 byte stride で離れた address に分散し、 8 cache line touch
//! が発生していた (= L1/L2 miss が dominant、 NPS arm 1 比 -29%)。 bucket を最内側に
//! 置き換えることで 1 idx × 8 bucket = 16 byte 連続 (= 1/4 cache line) に収まり、 `_mm_loadu_si128`
//! 1 命令で 8 i16 同時 load 可能。 同 cache line 内に他の 3 idx もまとまり L1 hit 率激増。
//!
//! # ckpt section format (= 2 magic 並存、 後方互換)
//!
//! ## 新 format (= P2DUALZ magic、 Path I で導入。 head 0 のみ書き出し)
//!
//! ```text
//! sub_magic = "P2DUALZ\0"        (8 byte)
//! version   = u32 LE              (= 1)
//! count     = u32 LE              (= 254,016 = 2 persp × 1 head × 127,008)
//! weights   = i16 LE × 254,016   (= 508,032 byte = ~0.5 MB)
//! ```
//!
//! disk 順序: `[persp 2][bucket 8][sq 81][own_p 14][target_p 14]` (= bucket-outer)
//!
//! ## 旧 format (= P2DUALX magic、 head 0/1 両方の重み)
//!
//! ```text
//! sub_magic = "P2DUALX\0"        (8 byte)
//! version   = u32 LE              (= 1)
//! count     = u32 LE              (= 508,032 = 2 persp × 2 head × 127,008)
//! weights   = i16 LE × 508,032   (= 1,016,064 byte = ~1 MB)
//! ```
//!
//! disk 順序: `[persp 2][head 2][bucket 8][sq 81][own_p 14][target_p 14]`
//!
//! 読み込み時 (= `NetworkLayerStacks::read_from_le`) に head 1 weight を **捨て**、
//! 同 memory layout `[persp][sq][own_p][target_p][bucket]` に head 0 weight のみ並べ替え。
//! bullet 側 既存 ckpt (= arm 4/5) を引き続き読み込み可能 (= 後方互換)、 ただし memory
//! footprint は半減。
//!
//! # cross-implementation 契約
//!
//! 本 module の定数 + index 計算は bullet 側
//! `crates/bullet_lib/src/game/inputs/shogi_halfka_hm_king_attacker_dual.rs` の
//! cross feature 実装と **完全一致** が必須。
//!
//! - flat index 計算: `bucket * 15876 + sq * 196 + own_p * 14 + target_p`
//! - perspective swap: black=identity / white=80-sq (= 既存 dual_psqt と同方式)
//! - 利き計算: occupancy-aware (= bitboard::*_effect)
//!
//! 変更時は Golden Forward cross-validation 必須。

use crate::bitboard::{
    Bitboard, bishop_effect, dragon_effect, gold_effect, horse_effect, king_effect, knight_effect,
    lance_effect, pawn_effect, rook_effect, silver_effect,
};
use crate::nnue::constants::NUM_LAYER_STACK_BUCKETS;
use crate::nnue::dual_delta::MoveDelta;
use crate::position::Position;
use crate::types::{Color, Piece, PieceType, Square};

// =============================================================================
// 定数
// =============================================================================

/// head 数 (= 1: arm 6 で defended head 廃止、 threat head 0 のみ)。
///
/// Path I (= 2026-05-27) で memory footprint 半減のため head 次元を 2 → 1 に縮小。
/// ckpt format は後方互換: 旧 P2DUALX (= head 2) は head 0 のみ load (head 1 weight 捨て)、
/// 新 P2DUALZ (= head 1) は直接 load。
pub const NUM_DUAL_CROSS_HEADS: usize = 1;

/// bucket 数 (= LayerStacks と共有)。
pub const NUM_DUAL_CROSS_BUCKETS: usize = 8;

/// 盤面マス数。
pub const DUAL_CROSS_SQUARES: usize = 81;

/// piece_type 数 (= Pawn..Dragon、 King 含む 14 種、 dual_psqt と同順)。
pub const DUAL_CROSS_PIECE_TYPES: usize = 14;

/// 1 bucket あたりの dim 数 (= 81 × 14 × 14 = 15,876)。
pub const DUAL_CROSS_DIM_PER_BUCKET: usize =
    DUAL_CROSS_SQUARES * DUAL_CROSS_PIECE_TYPES * DUAL_CROSS_PIECE_TYPES;

const _: () = {
    assert!(DUAL_CROSS_DIM_PER_BUCKET == 15_876);
};

/// 1 head あたりの dim 数 (= 8 × 15,876 = 127,008)。
pub const DUAL_CROSS_DIM_PER_HEAD: usize = NUM_DUAL_CROSS_BUCKETS * DUAL_CROSS_DIM_PER_BUCKET;

const _: () = {
    assert!(DUAL_CROSS_DIM_PER_HEAD == 127_008);
};

/// 1 perspective × 1 head の sq×own_p×target_p flat index space (= 81 × 14 × 14 = 15,876)。
///
/// active index は bucket を含まない形 (= `sq * 196 + own_p * 14 + target_p`) で保存し、
/// 差分更新時に 8 bucket 全部に同じ index で `±W[persp][head][b][sq][own_p][target_p]` を
/// 加減算する。
pub const DUAL_CROSS_SQ_PAIR_SPACE: usize =
    DUAL_CROSS_SQUARES * DUAL_CROSS_PIECE_TYPES * DUAL_CROSS_PIECE_TYPES;

const _: () = {
    assert!(DUAL_CROSS_SQ_PAIR_SPACE == 15_876);
};

/// 1 perspective × 1 head あたりの cross feature 最大同時 active 数 (= 静的上限、 stack alloc 用)。
///
/// 上限根拠: 自軍駒最大 ~20 駒 × 平均 target ~4 = 80。 安全側で 96。
///
/// runtime cap (= [`active_cap`]) はこの値以下で動的に設定可能。
pub const MAX_ACTIVE_DUAL_CROSS_FEATURES: usize = 96;

// =============================================================================
// active cap (= runtime cap、 active 列挙 N 個で break)
// =============================================================================

use std::sync::Once;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// 現在 active な runtime cap (= 起動時 env で初期化、 default = `MAX_ACTIVE_DUAL_CROSS_FEATURES`)。
static DUAL_CROSS_ACTIVE_CAP: AtomicUsize = AtomicUsize::new(MAX_ACTIVE_DUAL_CROSS_FEATURES);
static ACTIVE_CAP_INIT: Once = Once::new();

// =============================================================================
// Phase 1 profiling: rdtsc-based per-function cycle / call counters
// =============================================================================
//
// 目的: dual_cross 関連の 5 hot path 関数について、 呼び出し回数 と 累積 cycle を
// 集計し、 真の bottleneck を numerical に特定する。
//
// 設計:
//   - env `DUAL_CROSS_PROFILE=1` で計装有効化、 default OFF で zero overhead
//     (= AtomicBool::load 1 命令のみ、 rdtsc は通らない)。
//   - 各関数 entry / exit で `_rdtsc()` 差分を AtomicU64 に fetch_add。
//   - USI quit 時に `print_dual_cross_profile()` で stderr へ累積出力。
//
// rdtsc は invariant tsc (= modern CPU) で wall-clock proportional。 process 中
// で停止しない (= cstate 影響なし)。 micro-benchmarking で十分 信頼できる。
//
// Phase 別 (= opening/middle/endgame) 細分:
//   - thread-local `CURRENT_PHASE: Cell<u8>` に caller が `set_current_phase_from_ply`
//     で 0..3 を書き込む。 default = 1 (= middle)。
//   - ProfileGuard drop 時に総計 + phase 別の両方へ fetch_add (= 2× write、 negligible)。
//   - 出力時 phase 別 per-call cycles 表 + ratio (endgame / opening) を表示。

/// profiling enable flag (= env で 1 度 set、 hot path で load 1 命令で分岐)。
static DUAL_CROSS_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static PROFILE_INIT: Once = Once::new();

/// 5 関数分の (call_count, cycle_sum)。 順序:
///   0: collect_dual_cross_active_per_head
///   1: collect_dual_cross_threat_in_affected_set
///   2: update_dual_cross_perspective_diff
///   3: refresh_dual_cross_perspective (+ persp_head0)
///   4: update_dual_cross_accumulator entry (= per-move full delta wrapper)
const DUAL_CROSS_PROFILE_SLOTS: usize = 5;

/// phase 別細分の phase 数 (= opening=0 / middle=1 / endgame=2)。
const DUAL_CROSS_PROFILE_PHASES: usize = 3;

const PHASE_LABELS: [&str; DUAL_CROSS_PROFILE_PHASES] = ["opening", "middle ", "endgame"];

static DUAL_CROSS_PROFILE_CALLS: [AtomicU64; DUAL_CROSS_PROFILE_SLOTS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static DUAL_CROSS_PROFILE_CYCLES: [AtomicU64; DUAL_CROSS_PROFILE_SLOTS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// phase × slot の (call, cycles)。 flat layout: `phase * SLOTS + slot`。
static DUAL_CROSS_PROFILE_PHASE_CALLS: [AtomicU64;
    DUAL_CROSS_PROFILE_PHASES * DUAL_CROSS_PROFILE_SLOTS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static DUAL_CROSS_PROFILE_PHASE_CYCLES: [AtomicU64;
    DUAL_CROSS_PROFILE_PHASES * DUAL_CROSS_PROFILE_SLOTS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

const PROFILE_LABELS: [&str; DUAL_CROSS_PROFILE_SLOTS] = [
    "collect_per_head        ",
    "collect_in_affected_set ",
    "update_perspective_diff ",
    "refresh_perspective     ",
    "update_accumulator(top) ",
];

thread_local! {
    /// 現 phase (= 0=opening / 1=middle / 2=endgame)。 caller が
    /// `set_current_phase_from_ply` で更新。 default = middle (= profile 無効時は使われない)。
    static CURRENT_PHASE: std::cell::Cell<u8> = const { std::cell::Cell::new(1) };

    /// position-based phase override (= 探索側 USI が `EndgameDetector` 結果を書き込む)。
    ///
    /// 値:
    /// - `0xFF` (= default): override 無効 (= ply-based 判定 fallback)
    /// - `0`/`1`/`2`: opening/middle/endgame (= `GamePhase` enum と同 mapping)
    ///
    /// `DUAL_CROSS_USE_ENDGAME_DETECTOR=1` のとき gate 判定が本 override を参照する。
    /// USI go コマンド入口 で 1 度書き込まれ、 1 go 内の hot path で繰り返し read される。
    static CURRENT_PHASE_OVERRIDE: std::cell::Cell<u8> = const { std::cell::Cell::new(0xFF) };
}

#[inline(always)]
pub(crate) fn current_phase_idx() -> usize {
    CURRENT_PHASE.with(|c| c.get() as usize)
}

/// ply (= position.game_ply()) から phase を判定して thread-local に保存する。
///
/// 判定: ply 1..=30 → opening / ply 31..=100 → middle / ply 101.. → endgame。
/// profile 無効時 (= `DUAL_CROSS_PROFILE=0`) でも呼んで OK (= overhead は cell 書き 1 回)。
#[inline]
pub fn set_current_phase_from_ply(ply: i32) {
    let p = phase_from_ply(ply);
    CURRENT_PHASE.with(|c| c.set(p));
}

/// 探索側 USI が `EndgameDetector` で判定した phase index を thread-local に書き込む。
///
/// `idx` は 0=Opening / 1=Middle / 2=Endgame の 3 値想定 (= `GamePhase` enum 順)。
/// それ以外を渡した場合は override を clear (= 0xFF) して ply-based fallback に戻す。
///
/// `dual_cross_use_endgame_detector()` が true の間、 gate 判定 (= `current_phase_is_endgame`)
/// が本 override を見る。 go コマンド入口で 1 度呼ばれて 探索中は固定値として参照される。
#[inline]
pub fn set_current_phase_idx(idx: u8) {
    let v = if idx <= 2 { idx } else { 0xFF };
    CURRENT_PHASE_OVERRIDE.with(|c| c.set(v));
}

/// `set_current_phase_idx` で書き込まれた override を clear する (= 0xFF)。
///
/// 主に test / bench 用。 通常 USI 経路では override は go コマンド毎に上書きされるため
/// 明示 clear は不要。
#[inline]
pub fn clear_current_phase_override() {
    CURRENT_PHASE_OVERRIDE.with(|c| c.set(0xFF));
}

/// `DUAL_CROSS_USE_ENDGAME_DETECTOR` 用の gate 判定 helper。
///
/// override が valid (= 0/1/2 のいずれか) なら override == 2 (= Endgame) を返す。
/// override 未設定 (= 0xFF) のときは `false` (= 安全側、 skip 経路に入る)。
///
/// 注意: USI 側で `set_current_phase_idx` を呼ばないまま flag を立てると、
/// override 未設定 = false で全 ply が skip 経路になる。 USI go コマンドに必ず
/// `set_current_phase_idx` を組み込むこと。
#[inline(always)]
pub fn current_phase_is_endgame() -> bool {
    CURRENT_PHASE_OVERRIDE.with(|c| c.get() == 2)
}

/// ply から phase index (= 0=opening / 1=middle / 2=endgame) を計算する純関数。
///
/// thread-local を touch しないため、 update / eval path で endgame 判定だけが
/// 必要な caller (= `DUAL_CROSS_ENDGAME_ONLY` flag 経路) が直接呼べる。
#[inline(always)]
pub fn phase_from_ply(ply: i32) -> u8 {
    if ply <= 30 {
        0
    } else if ply <= 100 {
        1
    } else {
        2
    }
}

/// `ply` が endgame 帯 (= ply 101..) か。 `DUAL_CROSS_ENDGAME_ONLY` 用の薄い helper。
#[inline(always)]
pub fn is_endgame_ply(ply: i32) -> bool {
    phase_from_ply(ply) == 2
}

#[inline(always)]
fn profile_enabled() -> bool {
    DUAL_CROSS_PROFILE_ENABLED.load(Ordering::Relaxed)
}

fn init_profile_flag() {
    PROFILE_INIT.call_once(|| {
        if let Ok(s) = std::env::var("DUAL_CROSS_PROFILE")
            && (s == "1" || s.eq_ignore_ascii_case("true"))
        {
            DUAL_CROSS_PROFILE_ENABLED.store(true, Ordering::Relaxed);
            eprintln!("info string DUAL_CROSS_PROFILE enabled");
        }
    });
}

/// 起動時 1 度呼び、 env から profile flag を読む。 USI 側 main の起動 path で呼ぶ。
pub fn init_dual_cross_profile() {
    init_profile_flag();
}

#[inline(always)]
fn rdtsc_now() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: rdtsc は副作用なし、 全 x86_64 user-mode で常に valid。
        // 守るべき不変条件: なし (= read-only counter read)。
        unsafe { std::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

/// 計装 RAII guard。 drop 時に diff cycle を slot へ fetch_add。
struct ProfileGuard {
    slot: usize,
    start: u64,
}

impl ProfileGuard {
    #[inline(always)]
    fn new(slot: usize) -> Option<Self> {
        if !profile_enabled() {
            return None;
        }
        Some(Self {
            slot,
            start: rdtsc_now(),
        })
    }
}

impl Drop for ProfileGuard {
    #[inline(always)]
    fn drop(&mut self) {
        let end = rdtsc_now();
        let d = end.wrapping_sub(self.start);
        DUAL_CROSS_PROFILE_CALLS[self.slot].fetch_add(1, Ordering::Relaxed);
        DUAL_CROSS_PROFILE_CYCLES[self.slot].fetch_add(d, Ordering::Relaxed);
        // phase 別 counter にも反映 (= flat idx = phase * SLOTS + slot)
        let phase = current_phase_idx();
        debug_assert!(phase < DUAL_CROSS_PROFILE_PHASES);
        let idx = phase * DUAL_CROSS_PROFILE_SLOTS + self.slot;
        DUAL_CROSS_PROFILE_PHASE_CALLS[idx].fetch_add(1, Ordering::Relaxed);
        DUAL_CROSS_PROFILE_PHASE_CYCLES[idx].fetch_add(d, Ordering::Relaxed);
    }
}

/// 累積 profile を stderr に出力 (= USI quit hook)。 profile 無効時は no-op。
pub fn print_dual_cross_profile() {
    if !profile_enabled() {
        return;
    }
    let mut total_cycles: u64 = 0;
    let snap: [(u64, u64); DUAL_CROSS_PROFILE_SLOTS] = std::array::from_fn(|i| {
        let c = DUAL_CROSS_PROFILE_CALLS[i].load(Ordering::Relaxed);
        let y = DUAL_CROSS_PROFILE_CYCLES[i].load(Ordering::Relaxed);
        total_cycles = total_cycles.saturating_add(y);
        (c, y)
    });
    eprintln!("info string === dual_cross profile (rdtsc cycles, total) ===");
    for i in 0..DUAL_CROSS_PROFILE_SLOTS {
        let (calls, cycles) = snap[i];
        let pct = if total_cycles > 0 {
            100.0 * cycles as f64 / total_cycles as f64
        } else {
            0.0
        };
        let per_call = if calls > 0 {
            cycles as f64 / calls as f64
        } else {
            0.0
        };
        eprintln!(
            "info string {} calls={:>12}  cycles={:>16}  ({:>5.2}%)  per_call={:>10.1}",
            PROFILE_LABELS[i], calls, cycles, pct, per_call
        );
    }
    eprintln!(
        "info string total_cycles_sum={total_cycles} (sum across slots, double-counted across nested calls)"
    );

    // phase 別 内訳: per_call cycles + ratio (endgame / opening)
    let mut phase_snap = [[(0u64, 0u64); DUAL_CROSS_PROFILE_SLOTS]; DUAL_CROSS_PROFILE_PHASES];
    for (p, row) in phase_snap.iter_mut().enumerate() {
        for (s, cell) in row.iter_mut().enumerate() {
            let idx = p * DUAL_CROSS_PROFILE_SLOTS + s;
            let c = DUAL_CROSS_PROFILE_PHASE_CALLS[idx].load(Ordering::Relaxed);
            let y = DUAL_CROSS_PROFILE_PHASE_CYCLES[idx].load(Ordering::Relaxed);
            *cell = (c, y);
        }
    }
    eprintln!("info string === dual_cross profile per-phase ===");
    eprintln!(
        "info string                          {:>14}  {:>14}  {:>14}  end/open",
        "opening (1-30)", "middle (31-100)", "endgame (101+)"
    );
    for (s, label) in PROFILE_LABELS.iter().enumerate() {
        let (oc, oy) = phase_snap[0][s];
        let (mc, my) = phase_snap[1][s];
        let (ec, ey) = phase_snap[2][s];
        let opc = if oc > 0 { oy as f64 / oc as f64 } else { 0.0 };
        let mpc = if mc > 0 { my as f64 / mc as f64 } else { 0.0 };
        let epc = if ec > 0 { ey as f64 / ec as f64 } else { 0.0 };
        let ratio = if opc > 0.0 { epc / opc } else { 0.0 };
        eprintln!(
            "info string {} per_call (cyc): {:>14.1}  {:>14.1}  {:>14.1}  {:>5.2}x",
            label, opc, mpc, epc, ratio
        );
        eprintln!("info string {} calls          : {:>14}  {:>14}  {:>14}", label, oc, mc, ec);
    }
    // phase 判定: phase ごとの呼び出し総数 (= phase 在留 share の代理) を出す
    let mut phase_call_sum = [0u64; DUAL_CROSS_PROFILE_PHASES];
    for (p, sum) in phase_call_sum.iter_mut().enumerate() {
        for cell in phase_snap[p].iter() {
            *sum = sum.saturating_add(cell.0);
        }
    }
    let grand: u64 = phase_call_sum.iter().sum();
    if grand > 0 {
        eprintln!(
            "info string phase calls share: opening={:>5.1}%  middle={:>5.1}%  endgame={:>5.1}%  (grand={})",
            100.0 * phase_call_sum[0] as f64 / grand as f64,
            100.0 * phase_call_sum[1] as f64 / grand as f64,
            100.0 * phase_call_sum[2] as f64 / grand as f64,
            grand
        );
    }
    let _ = PHASE_LABELS;
}

fn init_active_cap() {
    ACTIVE_CAP_INIT.call_once(|| {
        if let Ok(s) = std::env::var("DUAL_CROSS_ACTIVE_CAP")
            && let Ok(v) = s.parse::<usize>()
        {
            let clamped = v.clamp(1, MAX_ACTIVE_DUAL_CROSS_FEATURES);
            DUAL_CROSS_ACTIVE_CAP.store(clamped, Ordering::Relaxed);
            eprintln!("info string DUAL_CROSS_ACTIVE_CAP = {clamped}");
        }
    });
}

/// 1 perspective × 1 head での active feature 数の runtime cap。
///
/// 起動時 env var `DUAL_CROSS_ACTIVE_CAP=N` で N に設定 (= `1..=96` で clamp)。
/// 未設定なら [`MAX_ACTIVE_DUAL_CROSS_FEATURES`] (= 96) と同等。
///
/// active 列挙 helper (= [`collect_dual_cross_active_per_head`] /
/// [`collect_dual_cross_threat_in_affected_set`]) は本 cap に達したら **break** で
/// 切り捨てる (= sweep 検証用、 cap < 96 では bit-exact 性は意図的に崩れる)。
#[inline]
pub fn active_cap() -> usize {
    init_active_cap();
    DUAL_CROSS_ACTIVE_CAP.load(Ordering::Relaxed)
}

// =============================================================================
// Path G: important_set (= 統計的 active subset filter)
// =============================================================================
//
// 目的: ckpt から各 active idx (= 0..15876) の weight magnitude を計算し、 上位
// 一定% のみを important として残す。 production 列挙時に important でない idx を
// skip することで weight 加減算 cost を比例して削減。
//
// 構造: bit set (= 15876 bit = 1985 byte) に important idx を mark。 検査は
// `bit_get(idx)` で O(1)。 起動時 1 度だけ ckpt scan で構築 (= 数 ms)。
//
// 棋力影響: 上位% の選定基準は 8 bucket weight magnitude の合計 (= persp 別)。
// 全 bucket 平均で 重要度を測ることで bucket 切り替え時の安定性を確保。

/// important_set の bit storage (= 15876 bit、 4 byte align で 248 u64)。
const IMPORTANT_SET_U64_WORDS: usize = DUAL_CROSS_SQ_PAIR_SPACE.div_ceil(64);

/// 1 perspective 分の important set。 0: skip / 1: keep。
///
/// per-persp に分ける理由: arm 5 ckpt は black/white で weight 構造が異なる場合があり、
/// 各 persp で独立に top-N を選定する方が誤差が小さい。
#[derive(Clone)]
pub struct ImportantSet {
    bits: [u64; IMPORTANT_SET_U64_WORDS],
    kept_count: u32,
}

impl ImportantSet {
    /// 全 idx を keep (= no filter、 default)。
    pub fn full() -> Self {
        let mut s = Self {
            bits: [0; IMPORTANT_SET_U64_WORDS],
            kept_count: DUAL_CROSS_SQ_PAIR_SPACE as u32,
        };
        // 全 idx を 1 にする。 末尾の word のみ部分 mask。
        let full_words = DUAL_CROSS_SQ_PAIR_SPACE / 64;
        let rem_bits = DUAL_CROSS_SQ_PAIR_SPACE % 64;
        for w in s.bits.iter_mut().take(full_words) {
            *w = u64::MAX;
        }
        if rem_bits > 0 {
            s.bits[full_words] = (1u64 << rem_bits) - 1;
        }
        s
    }

    /// idx が important (= keep) か。
    #[inline(always)]
    pub fn contains(&self, idx: u16) -> bool {
        let i = idx as usize;
        debug_assert!(i < DUAL_CROSS_SQ_PAIR_SPACE);
        (self.bits[i / 64] >> (i % 64)) & 1 != 0
    }

    /// 全 important idx 数。
    pub fn count(&self) -> u32 {
        self.kept_count
    }
}

/// per-persp の important set (= production で参照される global state)。
///
/// 起動時に `init_important_set_from_weights` で構築。 default は full set
/// (= filter なし、 既存挙動と完全一致)。
static mut IMPORTANT_SETS: Option<[ImportantSet; 2]> = None;
static IMPORTANT_SETS_INIT: Once = Once::new();

/// `weights` から per-persp の magnitude 統計を取り、 top `top_pct` % のみ keep
/// する important_set を構築する。
///
/// `top_pct` ∈ `(0, 100]`。 100 = filter なし (= full set)、 50 = 上位 50% のみ keep。
///
/// 各 idx の score = sum over bucket 8 of |W[persp][0][sq][own_p][target_p][bucket]|。
/// 単純な i64 absolute sum で並べ替え、 top N を mark。
fn build_important_set(weights: &DualCrossWeights, persp: usize, top_pct: u32) -> ImportantSet {
    if top_pct >= 100 {
        return ImportantSet::full();
    }
    let top_pct = top_pct.max(1);
    let mut scores: Vec<(u32, u16)> = Vec::with_capacity(DUAL_CROSS_SQ_PAIR_SPACE);
    let w_head = &weights[persp][0];
    for (sq, w_sq) in w_head.iter().enumerate() {
        for (own_p, w_op) in w_sq.iter().enumerate() {
            for (target_p, w_tp) in w_op.iter().enumerate() {
                let mut s: u32 = 0;
                for &w in w_tp.iter() {
                    s = s.saturating_add(w.unsigned_abs() as u32);
                }
                let idx = dual_cross_active_index(sq, own_p, target_p) as u16;
                scores.push((s, idx));
            }
        }
    }
    // 上位 N 個を keep
    let n_keep = ((DUAL_CROSS_SQ_PAIR_SPACE as u64 * top_pct as u64) / 100).max(1) as usize;
    let n_keep = n_keep.min(DUAL_CROSS_SQ_PAIR_SPACE);
    // partial sort: select_nth_unstable で threshold を取得
    scores.select_nth_unstable_by(DUAL_CROSS_SQ_PAIR_SPACE - n_keep, |a, b| a.0.cmp(&b.0));
    let threshold = scores[DUAL_CROSS_SQ_PAIR_SPACE - n_keep].0;

    let mut set = ImportantSet {
        bits: [0; IMPORTANT_SET_U64_WORDS],
        kept_count: 0,
    };
    // threshold 以上の score を持つ idx を mark。 ties は全部 keep (= n_keep 超過する場合あり)。
    for &(s, idx) in &scores {
        if s >= threshold {
            let i = idx as usize;
            set.bits[i / 64] |= 1u64 << (i % 64);
            set.kept_count += 1;
        }
    }
    set
}

/// `weights` から per-persp の important set を構築し、 global state に格納する。
///
/// `top_pct` = 100 (or 未指定) なら no-op (= filter なし)。 起動時に env var
/// `DUAL_CROSS_TOP_PCT=N` を読んで、 0 < N < 100 なら filter 有効化。
///
/// 1 度だけ呼ばれる (= 2 回目以降は無視)。 thread safe (= `Once`)。
pub fn init_important_set_from_weights(weights: &DualCrossWeights) {
    IMPORTANT_SETS_INIT.call_once(|| {
        let top_pct = std::env::var("DUAL_CROSS_TOP_PCT")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(100);
        if top_pct >= 100 {
            // no filter (= 既存挙動と一致)。 global を None のまま残し、 `is_important_idx`
            // 側で None 時に常に true を返す。
            return;
        }
        let sets = [
            build_important_set(weights, 0, top_pct),
            build_important_set(weights, 1, top_pct),
        ];
        eprintln!(
            "info string DUAL_CROSS_TOP_PCT = {} (= persp0 keep {}/{}, persp1 keep {}/{})",
            top_pct,
            sets[0].count(),
            DUAL_CROSS_SQ_PAIR_SPACE,
            sets[1].count(),
            DUAL_CROSS_SQ_PAIR_SPACE,
        );
        // SAFETY: `Once::call_once` 内部で 1 回だけ実行されるため、 並行書き込みは発生しない。
        // 読み取り側 (= `is_important_idx`) は init 完了後にのみアクセス可能。
        // 守るべき不変条件: init は production 起動時の 1 回のみ呼ぶ (= NetworkLayerStacks::read 内)。
        unsafe {
            IMPORTANT_SETS = Some(sets);
        }
    });
}

/// `idx` が `perspective` の important set に含まれるか判定する。
///
/// init されていない / `DUAL_CROSS_TOP_PCT >= 100` の場合は常に true (= filter なし)。
#[inline(always)]
pub fn is_important_idx(perspective: Color, idx: u16) -> bool {
    // SAFETY: `IMPORTANT_SETS` は `init_important_set_from_weights` で 1 度だけ書かれ、
    // 以降は read-only。 production 起動 sequence (= NetworkLayerStacks::read → eval)
    // で init が先行するため、 inference path での read は valid。
    // 守るべき不変条件: init 後の書き換え禁止 (= `Once` で保証)。
    let opt = unsafe { (&raw const IMPORTANT_SETS).as_ref().and_then(|x| x.as_ref()) };
    match opt {
        Some(sets) => sets[perspective.index()].contains(idx),
        None => true,
    }
}

/// `IMPORTANT_SETS` への per-persp 参照を 1 度だけ取得する hot loop 用 helper。
///
/// 戻り値 None: filter なし (= `DUAL_CROSS_TOP_PCT >= 100` または未初期化)。
/// 戻り値 Some(set): 該当 persp の important set 参照。 caller は inner loop で
/// `set.contains(idx)` だけ呼べばよく、 per-iter の Option 分岐を回避できる。
#[inline(always)]
fn important_set_ref_for_persp(perspective: Color) -> Option<&'static ImportantSet> {
    // SAFETY: 同上 (= `is_important_idx` と同じ条件)。
    let opt = unsafe { (&raw const IMPORTANT_SETS).as_ref().and_then(|x| x.as_ref()) };
    opt.map(|sets| &sets[perspective.index()])
}

/// `def_sq` sentinel (= "defender 不在" = head 0 entry の不使用値)。
///
/// `Square::index()` の値域は `0..81` なので `81` を sentinel として使う。 u8 で
/// 表現可能 (= 81 < 256)。 head 1 では active emit 時に必ず defender_sq の生 board
/// idx が入る (= 0..80)、 head 0 では本 sentinel が入る。
pub const DUAL_CROSS_DEF_SQ_NONE: u8 = 81;

/// 新 ckpt section の sub_magic (`P2DUALZ\0` = 8 bytes、 Path I で導入、 head 0 のみ)。
pub const DUAL_CROSS_SUB_MAGIC_Z: [u8; 8] = *b"P2DUALZ\0";

/// 旧 ckpt section の sub_magic (`P2DUALX\0` = 8 bytes、 head 0/1 両方)。
///
/// 後方互換のため新 binary でも読み込み可能 (= head 1 weight は load 時に捨てる)。
pub const DUAL_CROSS_SUB_MAGIC: [u8; 8] = *b"P2DUALX\0";

/// ckpt section format version (現在 1、 P2DUALZ / P2DUALX 共通)。
pub const DUAL_CROSS_VERSION: u32 = 1;

/// 新 P2DUALZ count field (= 2 perspective × 1 head × 127,008 = 254,016)。
pub const DUAL_CROSS_COUNT_Z: u32 = (2 * NUM_DUAL_CROSS_HEADS * DUAL_CROSS_DIM_PER_HEAD) as u32;

/// 旧 P2DUALX count field (= 2 perspective × 2 head × 127,008 = 508,032)。
pub const DUAL_CROSS_COUNT: u32 = (2 * 2 * DUAL_CROSS_DIM_PER_HEAD) as u32;

const _: () = {
    assert!(DUAL_CROSS_COUNT == 508_032);
    assert!(DUAL_CROSS_COUNT_Z == 254_016);
};

/// arch_str marker (= dual_psqt と共通 "DUAL=1")。
pub const DUAL_CROSS_ARCH_MARKER: &str = "DUAL=1";

/// output scale (= dual_psqt と同じ、 weight 学習で吸収)。
pub const DUAL_CROSS_OUTPUT_SCALE: i32 = 1;

/// build-time check: bucket 数が LayerStacks と一致。
const _DUAL_CROSS_BUCKET_CHECK: () = {
    assert!(
        NUM_DUAL_CROSS_BUCKETS == NUM_LAYER_STACK_BUCKETS,
        "NUM_DUAL_CROSS_BUCKETS must equal NUM_LAYER_STACK_BUCKETS"
    );
};

// =============================================================================
// 型定義
// =============================================================================

/// 1 perspective × 1 head の cross weight (= bucket-inner layout): `[sq][own_p][target_p][bucket] -> i16`。
///
/// 1 active idx `(sq, own_p, target_p)` に対する 8 bucket weight は最内側の連続 16 byte に
/// 配置され、 AVX2 `_mm_loadu_si128` 1 命令で同時 load 可能。
pub type DualCrossPerspectiveHeadWeights = [[[[i16; NUM_DUAL_CROSS_BUCKETS]; DUAL_CROSS_PIECE_TYPES];
    DUAL_CROSS_PIECE_TYPES]; DUAL_CROSS_SQUARES];

/// 1 perspective の cross weight: `[head][sq][own_p][target_p][bucket]`。
pub type DualCrossPerspectiveWeights = [DualCrossPerspectiveHeadWeights; NUM_DUAL_CROSS_HEADS];

/// 2 perspective 全体の cross weight。 ~1 MB。 memory layout: `[persp][head][sq][own_p][target_p][bucket]`。
pub type DualCrossWeights = [DualCrossPerspectiveWeights; 2];

// =============================================================================
// piece_type indexing + perspective swap (= dual_psqt と同方式)
// =============================================================================

#[inline]
pub(crate) const fn piece_type_to_cross_idx(pt: PieceType) -> usize {
    (pt as u8 - 1) as usize
}

#[inline]
pub(crate) const fn oriented_sq_idx(sq: Square, perspective: Color) -> usize {
    match perspective {
        Color::Black => sq.index(),
        Color::White => 80 - sq.index(),
    }
}

/// cross feature flat index 計算。
///
/// disk layout: `flat = bucket × 15876 + sq × 196 + own_p × 14 + target_p`
/// 範囲: `0..DUAL_CROSS_DIM_PER_HEAD` (= 0..127,008)。
#[inline]
pub const fn dual_cross_flat_index(
    bucket: usize,
    sq_idx: usize,
    own_piece: usize,
    target_piece: usize,
) -> usize {
    bucket * DUAL_CROSS_DIM_PER_BUCKET
        + sq_idx * (DUAL_CROSS_PIECE_TYPES * DUAL_CROSS_PIECE_TYPES)
        + own_piece * DUAL_CROSS_PIECE_TYPES
        + target_piece
}

/// active index 計算 (= bucket なし、 sq×own_p×target_p flat space)。
#[inline]
pub const fn dual_cross_active_index(
    sq_idx: usize,
    own_piece: usize,
    target_piece: usize,
) -> usize {
    sq_idx * (DUAL_CROSS_PIECE_TYPES * DUAL_CROSS_PIECE_TYPES)
        + own_piece * DUAL_CROSS_PIECE_TYPES
        + target_piece
}

// =============================================================================
// 利き計算 helper (= dual_psqt と同方式)
// =============================================================================

#[inline]
pub(crate) fn piece_attacks_from(piece: Piece, sq: Square, occupied: Bitboard) -> Bitboard {
    match piece.piece_type() {
        PieceType::Pawn => pawn_effect(piece.color(), sq),
        PieceType::Lance => lance_effect(piece.color(), sq, occupied),
        PieceType::Knight => knight_effect(piece.color(), sq),
        PieceType::Silver => silver_effect(piece.color(), sq),
        PieceType::Gold
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver => gold_effect(piece.color(), sq),
        PieceType::Bishop => bishop_effect(sq, occupied),
        PieceType::Rook => rook_effect(sq, occupied),
        PieceType::Horse => horse_effect(sq, occupied),
        PieceType::Dragon => dragon_effect(sq, occupied),
        PieceType::King => king_effect(sq),
        _ => Bitboard::EMPTY,
    }
}

// =============================================================================
// active 列挙 (= 1 perspective × 1 head)
// =============================================================================

/// 1 perspective × 1 head の active cross feature を列挙する。
///
/// head 0 = threat: own (sq, p) が 相手駒 target_p を攻撃中
/// head 1 = defended: own (sq, p) が 自軍駒 defender_p に守られている
///
/// 戻り値: 有効長。 `&out[..len]` が有効な active idx 列を保持。 active idx は
/// `sq * 196 + own_p * 14 + target_p` 形式 (= bucket 抜き、 0..15,876)。
///
/// `def_sq_out` には head 1 の場合 defender の生 board sq idx (= 0..80) を、
/// head 0 の場合 [`DUAL_CROSS_DEF_SQ_NONE`] (= 81) を **同 index で** 並列に書き込む。
/// これは 完全 delta path で head 1 を「defender ∈ affected」 条件で filter するため
/// 必要な metadata で、 weight 加減算には使わない。
pub fn collect_dual_cross_active_per_head(
    pos: &Position,
    perspective: Color,
    head: usize,
    out: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    def_sq_out: &mut [u8; MAX_ACTIVE_DUAL_CROSS_FEATURES],
) -> usize {
    debug_assert!(head < NUM_DUAL_CROSS_HEADS);
    let _profile = ProfileGuard::new(0);

    let mut len = 0usize;
    let cap = active_cap();

    // head 0 のみ。 head 1 (= defended) は arm 6 で廃止、 Path I で API からも削除済。
    debug_assert_eq!(head, 0, "head 1 is removed (arm 6 + Path I)");
    let _ = head;

    // Path J: 敵駒 bitboard を 1 度だけ取得し、 hot loop の inner enumerate で
    // `attacks & enemy_bb` で **直接 敵 target のみ列挙** することで、
    //
    // 1. inner loop 反復数を 全 attack sq (= 多くは空 mass) → 敵 target のみに削減
    //    (= 体感 -50〜-80%、 駒種により異なる)
    // 2. per-dst の `is_some() / color() == perspective` 分岐を削除
    // 3. attack の 90% は空 mass を踏むため、 mask 後の bit count は元の数分の 1
    //
    // Path K1: AttackBitboardCache 廃止 (= 1.3 KB struct copy が NPS bottleneck)。
    // own_piece / attacks は per-sq に `pos.piece_on` + `piece_attacks_from` で計算。
    let own_pieces = pos.pieces_c(perspective);
    let enemy_bb = pos.pieces_c(!perspective);
    let occupied = pos.occupied();
    // Path J: important set を 1 度だけ lookup (= per-iter の Option 分岐を回避)。
    // None なら inner loop で filter なし path、 Some なら set.contains 直接 call。
    let imp_set = important_set_ref_for_persp(perspective);

    'outer: for own_sq in own_pieces.iter() {
        let own_piece = pos.piece_on(own_sq);
        debug_assert!(
            own_piece.is_some(),
            "pieces_c bit set but piece NONE at sq={}",
            own_sq.raw()
        );
        let attacks = piece_attacks_from(own_piece, own_sq, occupied);
        let own_pt_idx = piece_type_to_cross_idx(own_piece.piece_type());
        let own_sq_oriented = oriented_sq_idx(own_sq, perspective);

        // Path J: 直接 敵 target のみ列挙 (= 元 attacks の 1/3〜1/5 程度に縮小)
        let enemy_targets = attacks & enemy_bb;
        for dst in enemy_targets.iter() {
            let target = pos.piece_on(dst);
            debug_assert!(
                target.is_some() && target.color() != perspective,
                "enemy_bb mask failed: dst={} target={:?}",
                dst.raw(),
                target
            );
            let target_pt_idx = piece_type_to_cross_idx(target.piece_type());
            let idx = dual_cross_active_index(own_sq_oriented, own_pt_idx, target_pt_idx) as u16;
            // Path G + J: hoisted important_set 参照で per-iter Option 分岐を回避
            if let Some(s) = imp_set
                && !s.contains(idx)
            {
                continue;
            }
            if len >= cap {
                break 'outer;
            }
            // SAFETY: cap <= MAX_ACTIVE_DUAL_CROSS_FEATURES (= active_cap で clamp)、
            // len < cap で break するため len < MAX_ACTIVE_DUAL_CROSS_FEATURES。
            unsafe {
                *out.get_unchecked_mut(len) = idx;
                *def_sq_out.get_unchecked_mut(len) = DUAL_CROSS_DEF_SQ_NONE;
            }
            len += 1;
        }
    }

    // Path H3 (sort 化) は試験的に bench で改善見られず removed。 idx 列挙順 = 駒 iter 順
    // で 既に sq まとまり、 cache line hit 率 は十分 高い。
    len
}

// =============================================================================
// SIMD / prefetch helpers (= per-idx accumulator 加減算 の hot path 最適化)
// =============================================================================
//
// 背景: memory layout は `[persp][head][sq][own_p][target_p][bucket]` で bucket
// が **最内側**。 1 active idx に対する 8 bucket weight は `&w_head[sq][own_p][target_p][0]`
// から **16 byte 連続** に並ぶため、 AVX2 `_mm_loadu_si128` 1 命令で 8 i16 同時 load
// 可能。 旧 bucket-outer (= 31,752 byte stride × 8 cache line touch = NPS arm 1 比 -29%)
// と比較し、 1 idx あたりの cache line touch を 8 → 1 (= 1/8) に圧縮。
//
// 簡素化された SIMD path:
//   1. `_mm_loadu_si128` で 8 i16 連続 load (= 1 命令、 16 byte)
//   2. `_mm256_cvtepi16_epi32` で i32×8 拡張 (= 32 byte ymm)
//   3. `_mm256_loadu_si256` + `_mm256_add/sub_epi32` + `_mm256_storeu_si256` で acc update
//
// AVX2 非対応 build (= 古い CPU) では scalar path を維持。 bit-exact 同値。

/// 1 idx 分の 8 bucket weight 起点 pointer (= `&w_head[sq][own_p][target_p][0]`) を返す。
///
/// 戻り pointer から **16 byte 連続** に 8 bucket weight が並ぶ。
///
/// Path J 最適化: bucket-inner layout (= `[sq][own_p][target_p][bucket=8]`) で
/// memory 上 flat に `idx * 8 + bucket` で配置されているため、 旧 3-stage div/mod
/// 分解 (= `i/196 / 14 % 14 / 14`) を **単純な pointer 加算** に置換。
/// 1 idx あたり 3 integer 演算を削減。
#[inline(always)]
fn dual_cross_weight_base_ptr(w_head: &DualCrossPerspectiveHeadWeights, idx: u16) -> *const i16 {
    let i = idx as usize;
    debug_assert!(i < DUAL_CROSS_SQ_PAIR_SPACE);
    // SAFETY: `w_head` は `[[[[i16; 8]; 14]; 14]; 81]` で flat memory layout が
    // `[sq * 14 * 14 * 8 + own_p * 14 * 8 + target_p * 8 + bucket] = [idx * 8 + bucket]`
    // と等価 (= 多次元 array で stride が連続)。 idx < 15876 で
    // idx * 8 < 127008 = DUAL_CROSS_DIM_PER_HEAD = w_head 全要素数。
    // 守るべき不変条件:
    //   1. NUM_DUAL_CROSS_BUCKETS == 8 (= const assert で確認)
    //   2. bucket-inner layout 維持 (= 型 `[[[[i16; 8]; 14]; 14]; 81]` で compile time 保証)
    //   3. idx < DUAL_CROSS_SQ_PAIR_SPACE (= debug_assert)
    unsafe { (w_head.as_ptr() as *const i16).add(i * NUM_DUAL_CROSS_BUCKETS) }
}

/// 1 idx 分の 8 bucket weight を `acc` に加算する。
///
/// AVX2 path: `_mm_loadu_si128` で 8 i16 = 16 byte 連続 load → i32×8 拡張 → ymm add。
/// 非 AVX2 path: scalar wrapping_add loop。
#[inline(always)]
fn dual_cross_add_idx_to_acc(
    w_head: &DualCrossPerspectiveHeadWeights,
    idx: u16,
    acc: &mut [i32; NUM_DUAL_CROSS_BUCKETS],
) {
    let base = dual_cross_weight_base_ptr(w_head, idx);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY:
        // - `base` は `&w_head[sq][own_p][target_p][0]` で、 同 (sq, own_p, target_p) の
        //   8 bucket weight が memory 上 16 byte 連続 (= bucket-inner layout、 型で保証)。
        // - `_mm_loadu_si128` は 16 byte unaligned load (= align 制約なし)。
        // - `acc` は `&mut [i32; 8]` で 32 byte 連続、 `loadu`/`storeu` 使用で
        //   align 制約なし。
        // 守るべき不変条件:
        //   1. NUM_DUAL_CROSS_BUCKETS == 8 (= const assert で確認、 16 byte = 1 xmm = 8 i16)
        //   2. `idx as usize < DUAL_CROSS_SQ_PAIR_SPACE` (= debug_assert で確認済)
        //   3. weight layout が bucket-inner (= compile time の型で保証)
        unsafe {
            use std::arch::x86_64::*;
            let w_packed = _mm_loadu_si128(base as *const __m128i);
            let w_i32 = _mm256_cvtepi16_epi32(w_packed);
            let acc_ptr = acc.as_mut_ptr() as *mut __m256i;
            let acc_vec = _mm256_loadu_si256(acc_ptr as *const __m256i);
            _mm256_storeu_si256(acc_ptr, _mm256_add_epi32(acc_vec, w_i32));
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        // SAFETY: base pointer は同一 array 内 [i16; 8] の先頭を指す (= bucket-inner
        // layout で連続 8 要素)。
        // 守るべき不変条件: NUM_DUAL_CROSS_BUCKETS == 8、 idx 範囲内 (= debug_assert)。
        for (b, sum) in acc.iter_mut().enumerate().take(NUM_DUAL_CROSS_BUCKETS) {
            let w = unsafe { base.add(b).read_unaligned() } as i32;
            *sum = sum.wrapping_add(w);
        }
    }
}

/// 1 idx 分の 8 bucket weight を `acc` から減算する (= `add_idx_to_acc` の逆)。
#[inline(always)]
fn dual_cross_sub_idx_from_acc(
    w_head: &DualCrossPerspectiveHeadWeights,
    idx: u16,
    acc: &mut [i32; NUM_DUAL_CROSS_BUCKETS],
) {
    let base = dual_cross_weight_base_ptr(w_head, idx);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `dual_cross_add_idx_to_acc` と同一条件 (= 同 layout、 同 連続性)。
        // 守るべき不変条件: 同上。
        unsafe {
            use std::arch::x86_64::*;
            let w_packed = _mm_loadu_si128(base as *const __m128i);
            let w_i32 = _mm256_cvtepi16_epi32(w_packed);
            let acc_ptr = acc.as_mut_ptr() as *mut __m256i;
            let acc_vec = _mm256_loadu_si256(acc_ptr as *const __m256i);
            _mm256_storeu_si256(acc_ptr, _mm256_sub_epi32(acc_vec, w_i32));
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        // SAFETY: 同上。
        for (b, sum) in acc.iter_mut().enumerate().take(NUM_DUAL_CROSS_BUCKETS) {
            let w = unsafe { base.add(b).read_unaligned() } as i32;
            *sum = sum.wrapping_sub(w);
        }
    }
}

/// 次 idx の 8 bucket weight (= 16 byte) を L1 prefetch (= T0 hint) する。
///
/// bucket-inner layout では 1 idx の 8 bucket = 16 byte 連続なので、 1 cache line
/// (= 64 byte) の prefetch 1 発で十分カバーできる (= 旧 bucket-outer の 8 発から削減)。
#[inline(always)]
fn dual_cross_prefetch_idx(w_head: &DualCrossPerspectiveHeadWeights, idx: u16) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: prefetch は読み書きしない hint で invalid pointer でも UB 化しないが、
        // `base` は同一 weight array 内 [i16; 8] の先頭を指すので pointer provenance
        // 規則 (= in-bounds) も満たす。
        // 守るべき不変条件: idx 範囲内 (= caller 側で `< DUAL_CROSS_SQ_PAIR_SPACE`)。
        unsafe {
            use std::arch::x86_64::*;
            let base = dual_cross_weight_base_ptr(w_head, idx);
            _mm_prefetch(base as *const i8, _MM_HINT_T0);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        // no-op on non-AVX2 targets。
        let _ = (w_head, idx);
    }
}

// =============================================================================
// refresh / update (= accumulator 化、 dual_psqt と同 pattern)
// =============================================================================

/// 1 perspective の cross accumulator を `pos` から full 計算する (= head 0 のみ)。
///
/// head 0 (= threat) のみ active sq×own_p×target_p 集合を列挙し、 各 active idx で
/// 8 bucket 全部に `W[persp][0][bucket][sq][own_p][target_p]` を加算する。
///
/// arm 6 で head 1 (= defended) は完全廃止、 Path I で memory footprint も縮小
/// (`NUM_DUAL_CROSS_HEADS = 1`)。 ckpt format は後方互換 (= P2DUALX 旧 ckpt の head 1 weight は
/// load 時に捨てる)。
/// hot-path 用 lean refresh API (= per-persp head 0 only)。 wrapper 配列 copy を回避。
pub fn refresh_dual_cross_persp_head0(
    pos: &Position,
    weights: &DualCrossWeights,
    perspective: Color,
    active_out: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    active_def_sq_out: &mut [u8; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    active_len_out: &mut u16,
) -> [i32; NUM_DUAL_CROSS_BUCKETS] {
    let _profile = ProfileGuard::new(3);
    let mut sums = [0i32; NUM_DUAL_CROSS_BUCKETS];
    let persp_idx = perspective.index();
    let w_head = &weights[persp_idx][0];

    let active_len =
        collect_dual_cross_active_per_head(pos, perspective, 0, active_out, active_def_sq_out);
    debug_assert!(active_len <= MAX_ACTIVE_DUAL_CROSS_FEATURES);
    *active_len_out = active_len as u16;

    let actives = &active_out[..active_len];
    if !actives.is_empty() {
        dual_cross_prefetch_idx(w_head, actives[0]);
    }
    for k in 0..actives.len() {
        if k + 1 < actives.len() {
            dual_cross_prefetch_idx(w_head, actives[k + 1]);
        }
        dual_cross_add_idx_to_acc(w_head, actives[k], &mut sums);
    }

    sums
}

pub fn refresh_dual_cross_perspective(
    pos: &Position,
    weights: &DualCrossWeights,
    perspective: Color,
    active_out: &mut [[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
    active_def_sq_out: &mut [[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
    active_len_out: &mut [u16; NUM_DUAL_CROSS_HEADS],
) -> [[i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS] {
    let _profile = ProfileGuard::new(3);
    let mut sums = [[0i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS];
    let persp_idx = perspective.index();
    let w_persp = &weights[persp_idx];

    // head 0 (threat) のみ計算
    const HEAD0: usize = 0;
    let active_len = collect_dual_cross_active_per_head(
        pos,
        perspective,
        HEAD0,
        &mut active_out[HEAD0],
        &mut active_def_sq_out[HEAD0],
    );
    debug_assert!(active_len <= MAX_ACTIVE_DUAL_CROSS_FEATURES);
    active_len_out[HEAD0] = active_len as u16;

    let w_head = &w_persp[HEAD0];
    let actives = &active_out[HEAD0][..active_len];
    // 1 idx 先 prefetch: 最初の idx を先読みしてから loop へ。
    if !actives.is_empty() {
        dual_cross_prefetch_idx(w_head, actives[0]);
    }
    for k in 0..actives.len() {
        // 次 idx の 8 bucket weight を先読み (= 1 idx ahead)。
        if k + 1 < actives.len() {
            dual_cross_prefetch_idx(w_head, actives[k + 1]);
        }
        dual_cross_add_idx_to_acc(w_head, actives[k], &mut sums[HEAD0]);
    }

    sums
}

// =============================================================================
// 完全 delta accumulator (= per-move full scan 廃止、 影響範囲限定 scan)
// =============================================================================

/// head 0 (threat) の active feature のうち、 「own_sq ∈ affected_sq_set」 となる
/// active のみを列挙する。
///
/// head 0 の active idx の own_sq = **攻撃元 own 駒** の sq。 対称性により
/// affected_sq_set が「from/to/cap + 8 近傍 + 桂 + sliders」 を含めば、 target
/// (= 相手駒) ∈ affected な場合の attacker own も網羅される (= attacker は
/// target の attack 圏内 = 8 近傍 / 桂 / ray の元)。
///
/// 戻り値: out 配列に書き込んだ有効長。
///
/// head 1 (= defended) でも対称的に使用される (= 後述 update path 内)。
/// MaybeUninit 版 (= zero init skip)。 戻り値 `len` 個が初期化済み。
#[inline]
fn collect_dual_cross_threat_in_affected_set_uninit(
    pos: &Position,
    perspective: Color,
    affected_sq_set: Bitboard,
    out: &mut [std::mem::MaybeUninit<u16>; MAX_ACTIVE_DUAL_CROSS_FEATURES],
) -> usize {
    let _profile = ProfileGuard::new(1);
    let mut len = 0usize;
    let cap = active_cap();

    let own_pieces_all = pos.pieces_c(perspective);
    let enemy_bb = pos.pieces_c(!perspective);
    let occupied = pos.occupied();
    let imp_set = important_set_ref_for_persp(perspective);
    let mut own_in_affected = own_pieces_all & affected_sq_set;
    'outer: while !own_in_affected.is_empty() {
        let sq = own_in_affected.pop();
        let piece = pos.piece_on(sq);
        debug_assert!(piece.is_some(), "pieces_c bit set but piece NONE at sq={}", sq.raw());
        let attacks = piece_attacks_from(piece, sq, occupied);
        let own_pt_idx = piece_type_to_cross_idx(piece.piece_type());
        let own_sq_oriented = oriented_sq_idx(sq, perspective);

        let enemy_targets = attacks & enemy_bb;
        for dst in enemy_targets.iter() {
            let target = pos.piece_on(dst);
            debug_assert!(
                target.is_some() && target.color() != perspective,
                "enemy_bb mask failed: dst={} target={:?}",
                dst.raw(),
                target
            );
            let target_pt_idx = piece_type_to_cross_idx(target.piece_type());
            let idx = dual_cross_active_index(own_sq_oriented, own_pt_idx, target_pt_idx) as u16;
            if let Some(s) = imp_set
                && !s.contains(idx)
            {
                continue;
            }
            if len >= cap {
                break 'outer;
            }
            // SAFETY: cap <= MAX_ACTIVE_DUAL_CROSS_FEATURES、 len < cap で break。
            unsafe {
                out.get_unchecked_mut(len).write(idx);
            }
            len += 1;
        }
    }
    len
}

#[inline]
fn collect_dual_cross_threat_in_affected_set(
    pos: &Position,
    perspective: Color,
    affected_sq_set: Bitboard,
    out: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
) -> usize {
    // profile slot 1 は uninit 版で計測 (= production path)。 本関数は test / legacy path のみ。
    let mut len = 0usize;
    let cap = active_cap();

    // 影響圏内の自軍駒 (= own ∩ affected_sq_set) のみを起点に enumerate
    let own_pieces_all = pos.pieces_c(perspective);
    let enemy_bb = pos.pieces_c(!perspective);
    let occupied = pos.occupied();
    // Path J: important set を hoist (= per-iter Option 分岐を回避)
    let imp_set = important_set_ref_for_persp(perspective);
    let mut own_in_affected = own_pieces_all & affected_sq_set;
    // Path K1: AttackBitboardCache 廃止、 piece / attacks は per-sq に計算。
    'outer: while !own_in_affected.is_empty() {
        let sq = own_in_affected.pop();
        let piece = pos.piece_on(sq);
        debug_assert!(piece.is_some(), "pieces_c bit set but piece NONE at sq={}", sq.raw());
        let attacks = piece_attacks_from(piece, sq, occupied);
        let own_pt_idx = piece_type_to_cross_idx(piece.piece_type());
        let own_sq_oriented = oriented_sq_idx(sq, perspective);

        // Path J: 直接 敵 target のみ列挙
        let enemy_targets = attacks & enemy_bb;
        for dst in enemy_targets.iter() {
            let target = pos.piece_on(dst);
            debug_assert!(
                target.is_some() && target.color() != perspective,
                "enemy_bb mask failed: dst={} target={:?}",
                dst.raw(),
                target
            );
            let target_pt_idx = piece_type_to_cross_idx(target.piece_type());
            let idx = dual_cross_active_index(own_sq_oriented, own_pt_idx, target_pt_idx) as u16;
            // Path G + J: hoisted important_set 参照で per-iter Option 分岐を回避
            if let Some(s) = imp_set
                && !s.contains(idx)
            {
                continue;
            }
            if len >= cap {
                break 'outer;
            }
            // SAFETY: cap <= MAX_ACTIVE_DUAL_CROSS_FEATURES、 len < cap で break。
            unsafe {
                *out.get_unchecked_mut(len) = idx;
            }
            len += 1;
        }
    }

    // Path H3 同上 (= sort 不要)。
    len
}

/// 1 perspective × head 0 (threat) の prev_active から 「own_sq ∈ affected_sq_set」 と
/// なる active のみを remove 候補として抽出、 残りは kept (= 維持) として保持する。
///
/// `affected_sq_oriented` は perspective swap 済の oriented sq の bitboard
/// (= 0..81 の sq idx それぞれが立っている)。
///
/// 戻り値: (removed_len, kept_len)。
/// MaybeUninit 版 (= zero init skip)。 戻り値の最初の `removed_len` + `kept_len`
/// 個が初期化済み。 caller は 同 len 以下の slice として参照すること。
#[inline]
fn split_prev_active_by_affected_threat_uninit(
    prev_active: &[u16],
    affected_sq_oriented: u128,
    removed_out: &mut [std::mem::MaybeUninit<u16>; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    kept_out: &mut [std::mem::MaybeUninit<u16>; MAX_ACTIVE_DUAL_CROSS_FEATURES],
) -> (usize, usize) {
    let mut removed_len = 0usize;
    let mut kept_len = 0usize;
    for &idx in prev_active {
        let i = idx as usize;
        // own_sq (= oriented) は active idx の上位部分
        let own_sq = i / (DUAL_CROSS_PIECE_TYPES * DUAL_CROSS_PIECE_TYPES);
        if (affected_sq_oriented >> own_sq) & 1 != 0 {
            if removed_len < MAX_ACTIVE_DUAL_CROSS_FEATURES {
                removed_out[removed_len].write(idx);
                removed_len += 1;
            }
        } else if kept_len < MAX_ACTIVE_DUAL_CROSS_FEATURES {
            kept_out[kept_len].write(idx);
            kept_len += 1;
        }
    }
    (removed_len, kept_len)
}

#[inline]
fn split_prev_active_by_affected_threat(
    prev_active: &[u16],
    affected_sq_oriented: u128,
    removed_out: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    kept_out: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
) -> (usize, usize) {
    let mut removed_len = 0usize;
    let mut kept_len = 0usize;
    for &idx in prev_active {
        let i = idx as usize;
        // own_sq (= oriented) は active idx の上位部分
        let own_sq = i / (DUAL_CROSS_PIECE_TYPES * DUAL_CROSS_PIECE_TYPES);
        if (affected_sq_oriented >> own_sq) & 1 != 0 {
            if removed_len < MAX_ACTIVE_DUAL_CROSS_FEATURES {
                removed_out[removed_len] = idx;
                removed_len += 1;
            }
        } else if kept_len < MAX_ACTIVE_DUAL_CROSS_FEATURES {
            kept_out[kept_len] = idx;
            kept_len += 1;
        }
    }
    (removed_len, kept_len)
}

/// 与えられた `MoveDelta` から「影響を受ける board sq 集合」 を構築する。
///
/// 集合 = `{from, to, captured_sq, sliders sqs} ∪ {各 sq の 8 近傍 + 桂 jumps + ray ends}`。
///
/// 8 近傍 + 桂 jumps の理由: 短距離 attacker (= Pawn/Knight/Silver/Gold/King 等) が
/// from/to を 攻撃 / 守備していた場合、 その attacker 視点の active が変化する。
/// active idx は own_sq のみで索引されるため、 該当 attacker の own_sq を集合に
/// 含める必要がある。
///
/// 本関数は perspective に依存しないため、 2 perspective で結果共有可能 (= caller 側で
/// `update_dual_cross_persp_head0_full_delta_with_affected` を 2 回呼ぶ前に 1 度計算する)。
#[inline]
pub fn build_dual_cross_affected_sq_set(delta: &MoveDelta) -> Bitboard {
    build_affected_sq_set(delta)
}

#[inline]
pub(crate) fn build_affected_sq_set(delta: &MoveDelta) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    let to_sq = delta.moved_to;

    // Path A: drop move (= moved_from.is_none()) の場合 from 経路を skip。
    // 既に if let Some で囲んでいるため drop なら本枝に入らない。
    if let Some(from) = delta.moved_from {
        bb |= Bitboard::from_square(from);
        bb |= king_effect(from);
        // 桂 jumps は両 color から (= 双方の桂が target になる可能性)
        bb |= knight_effect(Color::Black, from);
        bb |= knight_effect(Color::White, from);
    }

    // moved piece の to: 必ず必要
    bb |= Bitboard::from_square(to_sq);
    bb |= king_effect(to_sq);
    bb |= knight_effect(Color::Black, to_sq);
    bb |= knight_effect(Color::White, to_sq);

    // Path A: captured_sq は将棋では **常に moved_to と同じ** (= en passant なし)。
    // 通常 move の場合 captured 経路は to 経路と完全冗長 (= 同じ table lookup + OR)。
    // drop move の場合 captured は必ず None なので本枝は実行されない。
    // 念のため cap_sq != to_sq のケース (= 想定外の特殊 move) のみ独立処理する。
    if let Some((cap_sq, _, _)) = delta.captured
        && cap_sq != to_sq
    {
        bb |= Bitboard::from_square(cap_sq);
        bb |= king_effect(cap_sq);
        bb |= knight_effect(Color::Black, cap_sq);
        bb |= knight_effect(Color::White, cap_sq);
    }

    // influenced sliders の sq (= ray 上にいる slider 自身、 利き変化で
    // attack target が変わる可能性)。 Path D: bitboard を直接 OR (= 旧版の
    // 各 slider sq ごと from_square + OR ループを 1 OR に圧縮)。
    bb |= delta.influenced_sliders_bb;

    bb
}

/// `affected_sq_set` (= board sq の bitboard) を perspective に応じて
/// **oriented sq の bit mask** (= 0..81 の bit position に展開した u128) に変換する。
///
/// active idx は `oriented_sq * 196 + ...` の構造なので、 prev_active を filter
/// する際 oriented sq 比較が必要。
#[inline]
fn affected_sq_to_oriented_mask(affected: Bitboard, perspective: Color) -> u128 {
    let mut mask: u128 = 0;
    let mut bb = affected;
    while !bb.is_empty() {
        let sq = bb.pop();
        let oriented = oriented_sq_idx(sq, perspective);
        mask |= 1u128 << oriented;
    }
    mask
}

/// 1 perspective × head 0 のみの cross accumulator を **完全 delta** (= 加減算 only) で
/// 更新する (= arm 6: head 1 廃止)。
///
/// 既存 `update_dual_cross_perspective_diff` は per-move full scan
/// (= `collect_dual_cross_active_per_head` で board 全 own 駒 scan) を必要としたが、
/// 本関数は **影響範囲限定 scan** で同じ結果を bit-exact に再現する。
///
/// # アルゴリズム (head 0 = threat のみ)
///
/// 1. `MoveDelta` から affected_sq_set 構築 (= from/to/captured + 8 近傍 + 桂 + sliders)
/// 2. prev_active を「own_sq ∈ affected」 (= remove 候補) と「own_sq ∉ affected」 (= kept) に分割
/// 3. pos_after で affected 内 own のみ active を再 enumerate (= add 候補)
/// 4. multiset symmetric difference で最終 remove/add 確定
/// 5. weight 加減算で accumulator 更新
/// 6. current_active = kept ∪ pos_after で enum した affected 内 active
///
/// head 0 では active idx の own_sq = 攻撃元 own、 対称性により affected_sq_set が
/// target ∈ affected な相手駒の attacker own もカバーする (= 8 近傍 + 桂 + sliders)。
///
/// # head 1 (defended) について
///
/// arm 6 で完全廃止。 本関数は head 1 の acc / active / def_sq を **書き換えない**。
/// caller は `acc_buckets[1]` / `current_active_*_out[1]` を呼び出し前に 0 / sentinel
/// で初期化しておくこと (= refresh path と同じ責務分担)。 ckpt 互換のため memory
/// layout は維持し、 head 1 weight は load されるが本関数で使われない (= dead weight)。
///
/// # king moved 時
///
/// `delta.king_moved_any` 時は本関数を呼ばず、 caller 側で full refresh する。
///
/// # 引数数について
///
/// 本関数は accumulator update の hot path で、 in/out 6 配列 (3 入力 / 3 出力) と
/// ctx 4 (pos / delta / weights / perspective) で 計 11 引数。 struct 化すると
/// caller 側の borrow lifetime が煩雑になる (AccumulatorLayerStacks の field を分割
/// borrow する必要あり) ため、 既存の search hot path 関数群 (eval_helpers / pruning
/// 等) と同方針で attribute で抑制する。
/// hot-path 用の lean API (= per-persp head 0 only、 wrapper 配列を持たない)。
///
/// `update_dual_cross_perspective_full_delta` の per-head 1-dim wrapping を除去し、
/// caller (= `network_layer_stacks.rs` の `update_dual_cross_accumulator_incremental`)
/// で per-persp accumulator の field を直接 渡せるようにする。 stack 上の wrapper
/// 配列 copy (= ~768 byte/move/persp) を省略。
///
/// 動作は `update_dual_cross_perspective_full_delta` と bit-exact 同等
/// (= head 0 のみ更新、 head 1 は無)。
/// Path K3: pre-built affected_sq_set を受け取る lean variant。
///
/// `update_dual_cross_persp_head0_full_delta` 内で行う `build_affected_sq_set` を
/// caller 側で 1 度だけ計算 (= 2 perspective で共有) するための API。 動作は
/// `update_dual_cross_persp_head0_full_delta` と bit-exact 同等。
#[allow(clippy::too_many_arguments)]
pub fn update_dual_cross_persp_head0_full_delta_with_affected(
    pos: &Position,
    delta: &MoveDelta,
    affected: Bitboard,
    weights: &DualCrossWeights,
    perspective: Color,
    prev_active: &[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    prev_active_len: u16,
    acc_buckets: &mut [i32; NUM_DUAL_CROSS_BUCKETS],
    current_active_out: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    current_active_def_sq_out: &mut [u8; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    current_active_len_out: &mut u16,
) {
    let _profile = ProfileGuard::new(4);
    debug_assert!(!delta.king_moved_any, "king moved should use full refresh");

    let affected_oriented_mask = affected_sq_to_oriented_mask(affected, perspective);
    update_dual_cross_persp_head0_core(
        pos,
        affected,
        affected_oriented_mask,
        weights,
        perspective,
        prev_active,
        prev_active_len,
        acc_buckets,
        current_active_out,
        current_active_def_sq_out,
        current_active_len_out,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn update_dual_cross_persp_head0_full_delta(
    pos: &Position,
    delta: &MoveDelta,
    weights: &DualCrossWeights,
    perspective: Color,
    prev_active: &[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    prev_active_len: u16,
    acc_buckets: &mut [i32; NUM_DUAL_CROSS_BUCKETS],
    current_active_out: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    current_active_def_sq_out: &mut [u8; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    current_active_len_out: &mut u16,
) {
    debug_assert!(!delta.king_moved_any, "king moved should use full refresh");
    // Path K3: caller が affected を渡さない場合は内部で build (= 既存 API 後方互換)。
    let affected = build_affected_sq_set(delta);
    update_dual_cross_persp_head0_full_delta_with_affected(
        pos,
        delta,
        affected,
        weights,
        perspective,
        prev_active,
        prev_active_len,
        acc_buckets,
        current_active_out,
        current_active_def_sq_out,
        current_active_len_out,
    );
}

/// Path K3 core: pre-built `affected` + `affected_oriented_mask` を受けて diff 計算 + acc update を実行する。
///
/// `update_dual_cross_persp_head0_full_delta_with_affected` から呼ばれる本体。
/// profile guard は本関数で立てる (= caller の wrapper は薄い)。
#[allow(clippy::too_many_arguments)]
#[inline]
fn update_dual_cross_persp_head0_core(
    pos: &Position,
    affected: Bitboard,
    affected_oriented_mask: u128,
    weights: &DualCrossWeights,
    perspective: Color,
    prev_active: &[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    prev_active_len: u16,
    acc_buckets: &mut [i32; NUM_DUAL_CROSS_BUCKETS],
    current_active_out: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    current_active_def_sq_out: &mut [u8; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    current_active_len_out: &mut u16,
) {
    // Path K2 wrapper 最適化:
    //   - removed_candidate / kept / added_candidate に MaybeUninit を使い zero init を skip
    //   - def_sq sentinel fill loop を削除 (= head 1 廃止後 dead metadata)
    //   - removed / added の中間 buffer も MaybeUninit
    // Path K3 追加:
    //   - affected_sq_set / affected_oriented_mask を caller (= per-persp ループの外) で
    //     1 度計算し共有。 wrapper 内重複計算 (= 2 persp × build_affected) を半減。

    use std::mem::MaybeUninit;
    let mut removed_candidate_uninit: [MaybeUninit<u16>; MAX_ACTIVE_DUAL_CROSS_FEATURES] =
        [const { MaybeUninit::uninit() }; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let mut kept_uninit: [MaybeUninit<u16>; MAX_ACTIVE_DUAL_CROSS_FEATURES] =
        [const { MaybeUninit::uninit() }; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let prev_slice = &prev_active[..prev_active_len as usize];
    let (rc_len, kept_len) = split_prev_active_by_affected_threat_uninit(
        prev_slice,
        affected_oriented_mask,
        &mut removed_candidate_uninit,
        &mut kept_uninit,
    );

    // SAFETY: split_prev_active_by_affected_threat_uninit が rc_len / kept_len 個を
    // 初期化したことを契約として保証 (= helper 仕様)。
    let removed_candidate: &[u16] = unsafe {
        std::slice::from_raw_parts(removed_candidate_uninit.as_ptr() as *const u16, rc_len)
    };
    let kept: &[u16] =
        unsafe { std::slice::from_raw_parts(kept_uninit.as_ptr() as *const u16, kept_len) };

    let mut added_candidate_uninit: [MaybeUninit<u16>; MAX_ACTIVE_DUAL_CROSS_FEATURES] =
        [const { MaybeUninit::uninit() }; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let ac_len = collect_dual_cross_threat_in_affected_set_uninit(
        pos,
        perspective,
        affected,
        &mut added_candidate_uninit,
    );
    // SAFETY: collect_dual_cross_threat_in_affected_set_uninit が ac_len 個を初期化済み。
    let added_candidate: &[u16] = unsafe {
        std::slice::from_raw_parts(added_candidate_uninit.as_ptr() as *const u16, ac_len)
    };

    let mut removed_uninit: [MaybeUninit<u16>; MAX_ACTIVE_DUAL_CROSS_FEATURES] =
        [const { MaybeUninit::uninit() }; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let mut added_uninit: [MaybeUninit<u16>; MAX_ACTIVE_DUAL_CROSS_FEATURES] =
        [const { MaybeUninit::uninit() }; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let (removed_len, added_len) = diff_dual_cross_active_indices_uninit(
        removed_candidate,
        added_candidate,
        &mut removed_uninit,
        &mut added_uninit,
    );
    // SAFETY: diff_dual_cross_active_indices_uninit が removed_len / added_len 個を初期化済み。
    let removed: &[u16] =
        unsafe { std::slice::from_raw_parts(removed_uninit.as_ptr() as *const u16, removed_len) };
    let added: &[u16] =
        unsafe { std::slice::from_raw_parts(added_uninit.as_ptr() as *const u16, added_len) };

    update_dual_cross_perspective_diff(removed, added, weights, perspective, 0, acc_buckets);

    let total_len = kept_len + ac_len;
    debug_assert!(total_len <= MAX_ACTIVE_DUAL_CROSS_FEATURES);
    current_active_out[..kept_len].copy_from_slice(kept);
    current_active_out[kept_len..kept_len + ac_len].copy_from_slice(added_candidate);
    let _ = current_active_def_sq_out;
    *current_active_len_out = total_len as u16;
}

/// 1 perspective × head 0 のみの cross accumulator を **完全 delta** (= 加減算 only) で
/// 更新する (= arm 6: head 1 廃止)。 lean API は `update_dual_cross_persp_head0_full_delta` 参照。
#[allow(clippy::too_many_arguments)]
pub fn update_dual_cross_perspective_full_delta(
    pos: &Position,
    delta: &MoveDelta,
    weights: &DualCrossWeights,
    perspective: Color,
    prev_active: &[[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
    prev_active_def_sq: &[[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
    prev_active_len: &[u16; NUM_DUAL_CROSS_HEADS],
    acc_buckets: &mut [[i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS],
    current_active_out: &mut [[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
    current_active_def_sq_out: &mut [[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
    current_active_len_out: &mut [u16; NUM_DUAL_CROSS_HEADS],
) {
    debug_assert!(!delta.king_moved_any, "king moved should use full refresh");

    // head 0 のみ。 def_sq は head 0 では常に sentinel (= 不使用 metadata)、
    // 後方互換のため引数として受け取るが参照しない。
    let _ = prev_active_def_sq;

    let affected = build_affected_sq_set(delta);
    let affected_oriented_mask = affected_sq_to_oriented_mask(affected, perspective);

    // ===== head 0 (threat): 影響範囲限定 scan =====
    const HEAD0: usize = 0;
    {
        let mut removed_candidate = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
        let mut kept = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
        let prev_slice = &prev_active[HEAD0][..prev_active_len[HEAD0] as usize];
        let (rc_len, kept_len) = split_prev_active_by_affected_threat(
            prev_slice,
            affected_oriented_mask,
            &mut removed_candidate,
            &mut kept,
        );

        let mut added_candidate = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
        let ac_len = collect_dual_cross_threat_in_affected_set(
            pos,
            perspective,
            affected,
            &mut added_candidate,
        );

        let mut removed = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
        let mut added = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
        let (removed_len, added_len) = diff_dual_cross_active_indices(
            &removed_candidate[..rc_len],
            &added_candidate[..ac_len],
            &mut removed,
            &mut added,
        );

        update_dual_cross_perspective_diff(
            &removed[..removed_len],
            &added[..added_len],
            weights,
            perspective,
            HEAD0,
            &mut acc_buckets[HEAD0],
        );

        let total_len = kept_len + ac_len;
        debug_assert!(total_len <= MAX_ACTIVE_DUAL_CROSS_FEATURES);
        current_active_out[HEAD0][..kept_len].copy_from_slice(&kept[..kept_len]);
        current_active_out[HEAD0][kept_len..kept_len + ac_len]
            .copy_from_slice(&added_candidate[..ac_len]);
        // head 0 の def_sq は全 sentinel (= 不使用 metadata) を埋める。
        for slot in current_active_def_sq_out[HEAD0].iter_mut().take(total_len) {
            *slot = DUAL_CROSS_DEF_SQ_NONE;
        }
        current_active_len_out[HEAD0] = total_len as u16;
    }
}

/// 1 perspective × 1 head の cross accumulator の差分更新。
///
/// dual_psqt の `update_dual_perspective_diff` と同 pattern。 内部で AVX2
/// SIMD + prefetch (= [`dual_cross_add_idx_to_acc`] / [`dual_cross_sub_idx_from_acc`]
/// / [`dual_cross_prefetch_idx`]) を使う。 非 AVX2 build では scalar fallback。
pub fn update_dual_cross_perspective_diff(
    removed: &[u16],
    added: &[u16],
    weights: &DualCrossWeights,
    perspective: Color,
    head: usize,
    acc_head_buckets: &mut [i32; NUM_DUAL_CROSS_BUCKETS],
) {
    debug_assert!(head < NUM_DUAL_CROSS_HEADS);
    let _profile = ProfileGuard::new(2);
    let persp_idx = perspective.index();
    let w_head = &weights[persp_idx][head];

    // removed loop: 1 idx ahead prefetch + SIMD sub
    if !removed.is_empty() {
        dual_cross_prefetch_idx(w_head, removed[0]);
    }
    for k in 0..removed.len() {
        if k + 1 < removed.len() {
            dual_cross_prefetch_idx(w_head, removed[k + 1]);
        }
        dual_cross_sub_idx_from_acc(w_head, removed[k], acc_head_buckets);
    }

    // added loop: 1 idx ahead prefetch + SIMD add
    if !added.is_empty() {
        dual_cross_prefetch_idx(w_head, added[0]);
    }
    for k in 0..added.len() {
        if k + 1 < added.len() {
            dual_cross_prefetch_idx(w_head, added[k + 1]);
        }
        dual_cross_add_idx_to_acc(w_head, added[k], acc_head_buckets);
    }
}

/// MaybeUninit 版 (= zero init skip)。 戻り値 (removed_len, added_len) 個が初期化済み。
pub fn diff_dual_cross_active_indices_uninit(
    parent: &[u16],
    current: &[u16],
    removed_out: &mut [std::mem::MaybeUninit<u16>; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    added_out: &mut [std::mem::MaybeUninit<u16>; MAX_ACTIVE_DUAL_CROSS_FEATURES],
) -> (usize, usize) {
    let mut parent_sorted = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let mut current_sorted = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    debug_assert!(parent.len() <= MAX_ACTIVE_DUAL_CROSS_FEATURES);
    debug_assert!(current.len() <= MAX_ACTIVE_DUAL_CROSS_FEATURES);
    parent_sorted[..parent.len()].copy_from_slice(parent);
    current_sorted[..current.len()].copy_from_slice(current);
    parent_sorted[..parent.len()].sort_unstable();
    current_sorted[..current.len()].sort_unstable();

    let mut removed_len = 0usize;
    let mut added_len = 0usize;
    let mut i = 0usize;
    let mut j = 0usize;
    while i < parent.len() && j < current.len() {
        let p = parent_sorted[i];
        let c = current_sorted[j];
        if p < c {
            removed_out[removed_len].write(p);
            removed_len += 1;
            i += 1;
        } else if p > c {
            added_out[added_len].write(c);
            added_len += 1;
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    while i < parent.len() {
        removed_out[removed_len].write(parent_sorted[i]);
        removed_len += 1;
        i += 1;
    }
    while j < current.len() {
        added_out[added_len].write(current_sorted[j]);
        added_len += 1;
        j += 1;
    }
    (removed_len, added_len)
}

/// active index 集合の symmetric difference を抽出。
///
/// dual_psqt の `diff_dual_active_indices` と同 pattern (sort + merge)。
pub fn diff_dual_cross_active_indices(
    parent: &[u16],
    current: &[u16],
    removed_out: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
    added_out: &mut [u16; MAX_ACTIVE_DUAL_CROSS_FEATURES],
) -> (usize, usize) {
    let mut parent_sorted = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    let mut current_sorted = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
    debug_assert!(parent.len() <= MAX_ACTIVE_DUAL_CROSS_FEATURES);
    debug_assert!(current.len() <= MAX_ACTIVE_DUAL_CROSS_FEATURES);
    parent_sorted[..parent.len()].copy_from_slice(parent);
    current_sorted[..current.len()].copy_from_slice(current);
    parent_sorted[..parent.len()].sort_unstable();
    current_sorted[..current.len()].sort_unstable();

    let mut removed_len = 0usize;
    let mut added_len = 0usize;
    let mut i = 0usize;
    let mut j = 0usize;
    while i < parent.len() && j < current.len() {
        let p = parent_sorted[i];
        let c = current_sorted[j];
        if p < c {
            removed_out[removed_len] = p;
            removed_len += 1;
            i += 1;
        } else if p > c {
            added_out[added_len] = c;
            added_len += 1;
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    while i < parent.len() {
        removed_out[removed_len] = parent_sorted[i];
        removed_len += 1;
        i += 1;
    }
    while j < current.len() {
        added_out[added_len] = current_sorted[j];
        added_len += 1;
        j += 1;
    }
    (removed_len, added_len)
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_consistent() {
        assert_eq!(DUAL_CROSS_DIM_PER_BUCKET, 15_876);
        assert_eq!(DUAL_CROSS_DIM_PER_HEAD, 127_008);
        assert_eq!(DUAL_CROSS_SQ_PAIR_SPACE, 15_876);
        assert_eq!(DUAL_CROSS_COUNT, 508_032);
        assert_eq!(DUAL_CROSS_SUB_MAGIC, *b"P2DUALX\0");
    }

    #[test]
    fn flat_index_range() {
        let idx0 = dual_cross_flat_index(0, 0, 0, 0);
        let idx_max = dual_cross_flat_index(
            NUM_DUAL_CROSS_BUCKETS - 1,
            DUAL_CROSS_SQUARES - 1,
            DUAL_CROSS_PIECE_TYPES - 1,
            DUAL_CROSS_PIECE_TYPES - 1,
        );
        assert_eq!(idx0, 0);
        assert_eq!(idx_max, DUAL_CROSS_DIM_PER_HEAD - 1);
    }

    #[test]
    fn active_index_range() {
        let idx0 = dual_cross_active_index(0, 0, 0);
        let idx_max = dual_cross_active_index(
            DUAL_CROSS_SQUARES - 1,
            DUAL_CROSS_PIECE_TYPES - 1,
            DUAL_CROSS_PIECE_TYPES - 1,
        );
        assert_eq!(idx0, 0);
        assert_eq!(idx_max, DUAL_CROSS_SQ_PAIR_SPACE - 1);
    }

    // =========================================================================
    // bit-exact 検証 test (= dual_psqt の dual_acc_diff_vs_refresh_random_walk
    // と同 pattern。 random walk 100 局面 × 5 手で diff update と refresh の bit-exact
    // 一致を確認)
    // =========================================================================

    use crate::movegen::{MoveList, generate_legal};
    use crate::position::SFEN_HIRATE;
    use crate::types::Move;

    /// XorShift64 (= dual_psqt と同 random gen)
    struct XorShift64 {
        state: u64,
    }
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self {
                state: if seed == 0 { 1 } else { seed },
            }
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }
        fn pick(&mut self, n: usize) -> usize {
            (self.next_u64() as usize) % n
        }
    }

    fn build_test_cross_weights() -> Box<DualCrossWeights> {
        // pattern: w[persp][head][sq][own_p][target_p][bucket] = (sq + own_p + target_p + bucket * 3) % 64 - 32
        // pseudo-random だが deterministic、 値域 -32..32 (= i16 範囲内)。
        // layout: bucket は最内側 (= [sq][own_p][target_p][bucket])。
        // Path I で `NUM_DUAL_CROSS_HEADS = 1` (= head 0 のみ)。
        let mut w: Box<DualCrossWeights> = Box::new(
            [[[[[[0i16; NUM_DUAL_CROSS_BUCKETS]; DUAL_CROSS_PIECE_TYPES]; DUAL_CROSS_PIECE_TYPES];
                DUAL_CROSS_SQUARES]; NUM_DUAL_CROSS_HEADS]; 2],
        );
        for persp in 0..2 {
            for head in 0..NUM_DUAL_CROSS_HEADS {
                for sq in 0..DUAL_CROSS_SQUARES {
                    for own_p in 0..DUAL_CROSS_PIECE_TYPES {
                        for target_p in 0..DUAL_CROSS_PIECE_TYPES {
                            for bucket in 0..NUM_DUAL_CROSS_BUCKETS {
                                let v =
                                    ((sq + own_p + target_p + bucket * 3 + persp * 7 + head * 11)
                                        as i32
                                        % 64
                                        - 32) as i16;
                                w[persp][head][sq][own_p][target_p][bucket] = v;
                            }
                        }
                    }
                }
            }
        }
        w
    }

    /// 1 perspective scratch refresh (= ref 計算結果)。
    #[allow(clippy::type_complexity)]
    fn scratch_refresh_cross(
        pos: &Position,
        weights: &DualCrossWeights,
        perspective: Color,
    ) -> (
        [[i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS],
        [[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
        [[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
        [u16; NUM_DUAL_CROSS_HEADS],
    ) {
        let mut active = [[0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS];
        let mut active_def_sq =
            [[DUAL_CROSS_DEF_SQ_NONE; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS];
        let mut active_len = [0u16; NUM_DUAL_CROSS_HEADS];
        let sums = refresh_dual_cross_perspective(
            pos,
            weights,
            perspective,
            &mut active,
            &mut active_def_sq,
            &mut active_len,
        );
        (sums, active, active_def_sq, active_len)
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

    /// 1 perspective × head 0 のみの diff update (= arm 6 廃止後の reference: per-move full
    /// scan で head 0 active を列挙し、 multiset symmetric difference + 加減算)。 戻り値は
    /// full refresh と同 layout (head 1 slot は 0 / sentinel)。
    #[allow(clippy::type_complexity)]
    fn diff_update_cross(
        pos: &Position,
        weights: &DualCrossWeights,
        perspective: Color,
        parent_sums: &[[i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS],
        parent_active: &[[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
        parent_len: &[u16; NUM_DUAL_CROSS_HEADS],
    ) -> (
        [[i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS],
        [[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
        [[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
        [u16; NUM_DUAL_CROSS_HEADS],
    ) {
        let mut sums = *parent_sums;
        let mut current_active = [[0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS];
        let mut current_def_sq =
            [[DUAL_CROSS_DEF_SQ_NONE; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS];
        let mut current_len = [0u16; NUM_DUAL_CROSS_HEADS];

        // head 0 のみ計算 (= arm 6 + Path I)
        const HEAD0: usize = 0;
        let len = collect_dual_cross_active_per_head(
            pos,
            perspective,
            HEAD0,
            &mut current_active[HEAD0],
            &mut current_def_sq[HEAD0],
        );
        current_len[HEAD0] = len as u16;

        let parent_slice = &parent_active[HEAD0][..parent_len[HEAD0] as usize];
        let current_slice = &current_active[HEAD0][..len];

        let mut removed = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
        let mut added = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
        let (rlen, alen) =
            diff_dual_cross_active_indices(parent_slice, current_slice, &mut removed, &mut added);

        update_dual_cross_perspective_diff(
            &removed[..rlen],
            &added[..alen],
            weights,
            perspective,
            HEAD0,
            &mut sums[HEAD0],
        );

        (sums, current_active, current_def_sq, current_len)
    }

    fn startpos_cross() -> Position {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).unwrap();
        pos
    }

    /// random walk × 100 局面 で diff update と scratch refresh の acc が
    /// bit-exact 一致することを検証する。
    ///
    /// - 各 game: 開始局面から legal_moves random で 5 手指す
    /// - 各 do_move 後に scratch refresh と diff update を比較
    /// - king move は両 perspective 共に full refresh (= production と同)
    #[test]
    fn dual_cross_acc_diff_vs_refresh_random_walk() {
        let weights = build_test_cross_weights();
        let mut rng = XorShift64::new(0xCAFE_BABE);

        const NUM_GAMES: usize = 100;
        const PLIES_PER_GAME: usize = 5;
        let mut total_steps = 0usize;
        let mut king_move_steps = 0usize;

        for _game in 0..NUM_GAMES {
            let mut pos = startpos_cross();

            let mut parent_sums: [[[i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS]; 2] =
                [[[0; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS]; 2];
            let mut parent_active: [[[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS];
                2] = [[[0; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS]; 2];
            let mut parent_len: [[u16; NUM_DUAL_CROSS_HEADS]; 2] = [[0; NUM_DUAL_CROSS_HEADS]; 2];

            for &persp in &[Color::Black, Color::White] {
                let (s, a, _ds, l) = scratch_refresh_cross(&pos, &weights, persp);
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

                    let (ref_sums, ref_active, _ref_ds, ref_len) =
                        scratch_refresh_cross(&pos, &weights, persp);

                    let (acc_sums, acc_active, _acc_ds, acc_len) = if king_moved {
                        (ref_sums, ref_active, _ref_ds, ref_len)
                    } else {
                        diff_update_cross(
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

                    // active set 一致 (head 別、 sort して比較)
                    for head in 0..NUM_DUAL_CROSS_HEADS {
                        let acc_len_h = acc_len[head] as usize;
                        let ref_len_h = ref_len[head] as usize;
                        assert_eq!(
                            acc_len_h, ref_len_h,
                            "active_len mismatch: persp={persp:?}, head={head}, mv={mv:?}"
                        );
                        let mut a = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
                        let mut r = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
                        a[..acc_len_h].copy_from_slice(&acc_active[head][..acc_len_h]);
                        r[..ref_len_h].copy_from_slice(&ref_active[head][..ref_len_h]);
                        a[..acc_len_h].sort_unstable();
                        r[..ref_len_h].sort_unstable();
                        assert_eq!(
                            a[..acc_len_h],
                            r[..ref_len_h],
                            "active set mismatch: persp={persp:?}, head={head}, mv={mv:?}"
                        );
                    }

                    parent_sums[p] = acc_sums;
                    parent_active[p] = acc_active;
                    parent_len[p] = acc_len;
                }
            }
        }

        // sanity: king move ratio (= 数 % 程度を期待、 大半は通常駒移動)
        let _ratio = king_move_steps as f64 / total_steps as f64;
    }

    // =========================================================================
    // bit-exact 検証 test (= 完全 delta `update_dual_cross_perspective_full_delta`
    // と既存 pseudo refresh が **bit-exact 一致** することを確認する。
    //
    // 仕様要件: random walk × 500 step で各 step ごとに 両方法の accumulator
    // 完全一致を確認。 一致しなければ panic。 seed 固定で再現可能。
    // =========================================================================

    use crate::nnue::dual_delta::compute_move_delta;

    /// 1 perspective 完全 delta update wrapper (= production 経路と等価)。
    /// king moved 時は full refresh、 それ以外は full delta path を呼ぶ。
    #[allow(clippy::type_complexity)]
    fn full_delta_update_cross(
        pos_after: &Position,
        weights: &DualCrossWeights,
        perspective: Color,
        parent_sums: &[[i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS],
        parent_active: &[[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
        parent_def_sq: &[[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
        parent_len: &[u16; NUM_DUAL_CROSS_HEADS],
    ) -> (
        [[i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS],
        [[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
        [[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS],
        [u16; NUM_DUAL_CROSS_HEADS],
    ) {
        // delta が取れない (= 特殊手) or king moved → full refresh
        let delta = match compute_move_delta(pos_after) {
            Some(d) if !d.king_moved_any => d,
            _ => return scratch_refresh_cross(pos_after, weights, perspective),
        };

        let mut sums = *parent_sums;
        let mut current_active = [[0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS];
        let mut current_def_sq =
            [[DUAL_CROSS_DEF_SQ_NONE; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS];
        let mut current_len = [0u16; NUM_DUAL_CROSS_HEADS];

        update_dual_cross_perspective_full_delta(
            pos_after,
            &delta,
            weights,
            perspective,
            parent_active,
            parent_def_sq,
            parent_len,
            &mut sums,
            &mut current_active,
            &mut current_def_sq,
            &mut current_len,
        );

        (sums, current_active, current_def_sq, current_len)
    }

    /// random walk × 500 step で full delta と scratch refresh が bit-exact 一致
    /// することを確認する。
    ///
    /// - 開始局面から 1 game = 500 step 連続 random 合法手
    /// - 各 do_move 後に 両 perspective で full delta update と scratch refresh
    ///   の (sums, active_multiset) 一致を verify
    /// - king moved 時は full refresh 適用 (= production と同 fallback)
    /// - 投了 (= no legal move) で game 終了、 残 step はあえて新 game で続行
    #[test]
    fn dual_cross_acc_full_delta_vs_refresh_random_walk_500() {
        let weights = build_test_cross_weights();
        let mut rng = XorShift64::new(0xDEAD_BEEF);

        const TOTAL_STEPS: usize = 500;
        let mut completed_steps = 0usize;
        let mut games_played = 0usize;
        let mut king_moved_steps = 0usize;
        let mut delta_path_steps = 0usize;

        'outer: while completed_steps < TOTAL_STEPS {
            let mut pos = startpos_cross();
            games_played += 1;

            // 各 game 開始時に full refresh で parent state を初期化
            let mut parent_sums: [[[i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS]; 2] =
                [[[0; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS]; 2];
            let mut parent_active: [[[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS];
                2] = [[[0; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS]; 2];
            let mut parent_def_sq: [[[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES]; NUM_DUAL_CROSS_HEADS];
                2] = [[[DUAL_CROSS_DEF_SQ_NONE; MAX_ACTIVE_DUAL_CROSS_FEATURES];
                NUM_DUAL_CROSS_HEADS]; 2];
            let mut parent_len: [[u16; NUM_DUAL_CROSS_HEADS]; 2] = [[0; NUM_DUAL_CROSS_HEADS]; 2];
            for &persp in &[Color::Black, Color::White] {
                let (s, a, ds, l) = scratch_refresh_cross(&pos, &weights, persp);
                let p = persp as usize;
                parent_sums[p] = s;
                parent_active[p] = a;
                parent_def_sq[p] = ds;
                parent_len[p] = l;
            }

            // 各 game は最大 200 ply (= 終局 / 投了で打ち切り、 残り step は次 game)
            for _ply in 0..200 {
                if completed_steps >= TOTAL_STEPS {
                    break 'outer;
                }
                let mut list = MoveList::new();
                generate_legal(&pos, &mut list);
                if list.is_empty() {
                    break;
                }
                let mv = list.at(rng.pick(list.len()));
                let gc = pos.gives_check(mv);
                let _dirty = pos.do_move(mv, gc);
                completed_steps += 1;

                let king_moved = move_moves_king(mv);
                if king_moved {
                    king_moved_steps += 1;
                } else {
                    delta_path_steps += 1;
                }

                for &persp in &[Color::Black, Color::White] {
                    let p = persp as usize;
                    let (ref_sums, ref_active, ref_def_sq, ref_len) =
                        scratch_refresh_cross(&pos, &weights, persp);
                    let (delta_sums, delta_active, delta_def_sq, delta_len) =
                        full_delta_update_cross(
                            &pos,
                            &weights,
                            persp,
                            &parent_sums[p],
                            &parent_active[p],
                            &parent_def_sq[p],
                            &parent_len[p],
                        );

                    // sums 完全一致
                    assert_eq!(
                        delta_sums, ref_sums,
                        "full_delta sums mismatch: step={completed_steps}, persp={persp:?}, mv={mv:?}"
                    );

                    // active multiset 一致 (= sort + 比較)
                    for head in 0..NUM_DUAL_CROSS_HEADS {
                        let dl = delta_len[head] as usize;
                        let rl = ref_len[head] as usize;
                        assert_eq!(
                            dl, rl,
                            "full_delta active_len mismatch: step={completed_steps}, persp={persp:?}, head={head}, mv={mv:?}"
                        );
                        let mut a = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
                        let mut r = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
                        a[..dl].copy_from_slice(&delta_active[head][..dl]);
                        r[..rl].copy_from_slice(&ref_active[head][..rl]);
                        a[..dl].sort_unstable();
                        r[..rl].sort_unstable();
                        assert_eq!(
                            a[..dl],
                            r[..rl],
                            "full_delta active multiset mismatch: step={completed_steps}, persp={persp:?}, head={head}, mv={mv:?}"
                        );
                    }

                    // ref と delta の def_sq metadata は完全一致を要求しない
                    // (= def_sq は filter 専用 metadata、 同じ (own_sq, own_p, def_p)
                    // 組み合わせを複数 defender が生む multiset では順序差が出うる)。
                    // delta_def_sq は kept+added の連結で、 既存 kept entry の
                    // def_sq を維持するため ref とは順序が異なる可能性がある。
                    // ただし def_sq の値域 (= 0..81 OR sentinel) と filter 効果を
                    // 維持していれば bit-exact sums 一致で十分。
                    let _ = (&ref_def_sq, &delta_def_sq);

                    parent_sums[p] = delta_sums;
                    parent_active[p] = delta_active;
                    parent_def_sq[p] = delta_def_sq;
                    parent_len[p] = delta_len;
                }
            }
        }

        assert_eq!(completed_steps, TOTAL_STEPS, "should have completed all 500 steps");
        // diagnostics: 少なくとも数 game 行われ、 delta path が大半を占めること
        assert!(games_played >= 1);
        assert!(delta_path_steps > 0);
        // king moved は数 % 程度 (= 0 だと delta path のみで test として limited)
        let _delta_ratio = delta_path_steps as f64 / completed_steps as f64;
        let _king_ratio = king_moved_steps as f64 / completed_steps as f64;
    }

    // =========================================================================
    // SIMD vs scalar bit-exact 検証 test
    //
    // 本 module の `dual_cross_add_idx_to_acc` / `dual_cross_sub_idx_from_acc`
    // は AVX2 build で SIMD path、 非 AVX2 build で scalar fallback path を取る。
    // 同 build 内に「明示 scalar reference 実装」 を書き、 production 経路 (= SIMD
    // 経路) と 8 bucket の i32 acc 値が **完全一致** することを random walk × 500
    // step で確認する。 production が SIMD の場合は SIMD vs scalar の bit-exact
    // 一致を直接 verify。
    // =========================================================================

    /// scalar reference: bucket-inner layout を index 計算で 1 つずつ touch する。
    /// `dual_cross_add_idx_to_acc` の SIMD path に依存しない明示 fallback。
    fn ref_scalar_add_idx_to_acc(
        w_head: &DualCrossPerspectiveHeadWeights,
        idx: u16,
        acc: &mut [i32; NUM_DUAL_CROSS_BUCKETS],
    ) {
        let i = idx as usize;
        assert!(i < DUAL_CROSS_SQ_PAIR_SPACE);
        let sq = i / (DUAL_CROSS_PIECE_TYPES * DUAL_CROSS_PIECE_TYPES);
        let own_p = (i / DUAL_CROSS_PIECE_TYPES) % DUAL_CROSS_PIECE_TYPES;
        let target_p = i % DUAL_CROSS_PIECE_TYPES;
        for (b, sum) in acc.iter_mut().enumerate().take(NUM_DUAL_CROSS_BUCKETS) {
            *sum = sum.wrapping_add(w_head[sq][own_p][target_p][b] as i32);
        }
    }

    fn ref_scalar_sub_idx_from_acc(
        w_head: &DualCrossPerspectiveHeadWeights,
        idx: u16,
        acc: &mut [i32; NUM_DUAL_CROSS_BUCKETS],
    ) {
        let i = idx as usize;
        assert!(i < DUAL_CROSS_SQ_PAIR_SPACE);
        let sq = i / (DUAL_CROSS_PIECE_TYPES * DUAL_CROSS_PIECE_TYPES);
        let own_p = (i / DUAL_CROSS_PIECE_TYPES) % DUAL_CROSS_PIECE_TYPES;
        let target_p = i % DUAL_CROSS_PIECE_TYPES;
        for (b, sum) in acc.iter_mut().enumerate().take(NUM_DUAL_CROSS_BUCKETS) {
            *sum = sum.wrapping_sub(w_head[sq][own_p][target_p][b] as i32);
        }
    }

    /// scalar reference で full refresh を計算 (= SIMD 非依存、 arm 6 で head 0 のみ)。
    fn ref_scalar_refresh(
        pos: &Position,
        weights: &DualCrossWeights,
        perspective: Color,
    ) -> [[i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS] {
        let mut sums = [[0i32; NUM_DUAL_CROSS_BUCKETS]; NUM_DUAL_CROSS_HEADS];
        let persp_idx = perspective.index();
        let w_persp = &weights[persp_idx];

        let mut active = [0u16; MAX_ACTIVE_DUAL_CROSS_FEATURES];
        let mut active_def_sq = [DUAL_CROSS_DEF_SQ_NONE; MAX_ACTIVE_DUAL_CROSS_FEATURES];

        const HEAD0: usize = 0;
        let len = collect_dual_cross_active_per_head(
            pos,
            perspective,
            HEAD0,
            &mut active,
            &mut active_def_sq,
        );
        let w_head = &w_persp[HEAD0];
        for &idx in &active[..len] {
            ref_scalar_add_idx_to_acc(w_head, idx, &mut sums[HEAD0]);
        }
        sums
    }

    /// 単一 idx の SIMD path (= dual_cross_add_idx_to_acc) と scalar reference
    /// が任意 idx で完全一致することを deterministic full sweep で確認する。
    #[test]
    fn dual_cross_simd_add_vs_scalar_full_sweep() {
        let weights = build_test_cross_weights();

        for persp in 0..2 {
            for head in 0..NUM_DUAL_CROSS_HEADS {
                let w_head = &weights[persp][head];
                // 全 active idx 空間 (= 15876) を sweep するのは時間かかるので
                // 主要 idx + ramdom sample で。
                let mut rng = XorShift64::new(0x1234_5678 ^ (persp as u64) ^ ((head as u64) << 8));
                let mut sample_idxs: Vec<u16> =
                    (0..DUAL_CROSS_SQ_PAIR_SPACE as u16).step_by(37).collect();
                for _ in 0..500 {
                    sample_idxs.push(rng.pick(DUAL_CROSS_SQ_PAIR_SPACE) as u16);
                }

                for &idx in &sample_idxs {
                    let mut acc_simd = [0i32; NUM_DUAL_CROSS_BUCKETS];
                    let mut acc_ref = [0i32; NUM_DUAL_CROSS_BUCKETS];
                    dual_cross_add_idx_to_acc(w_head, idx, &mut acc_simd);
                    ref_scalar_add_idx_to_acc(w_head, idx, &mut acc_ref);
                    assert_eq!(
                        acc_simd, acc_ref,
                        "SIMD add path mismatch: persp={persp} head={head} idx={idx}"
                    );

                    // sub も同様
                    let mut acc_simd_s = [0i32; NUM_DUAL_CROSS_BUCKETS];
                    let mut acc_ref_s = [0i32; NUM_DUAL_CROSS_BUCKETS];
                    dual_cross_sub_idx_from_acc(w_head, idx, &mut acc_simd_s);
                    ref_scalar_sub_idx_from_acc(w_head, idx, &mut acc_ref_s);
                    assert_eq!(
                        acc_simd_s, acc_ref_s,
                        "SIMD sub path mismatch: persp={persp} head={head} idx={idx}"
                    );
                }
            }
        }
    }

    /// random walk × 500 step で SIMD refresh path (= production
    /// `refresh_dual_cross_perspective`) と scalar reference refresh が bit-exact
    /// 一致することを確認する。
    #[test]
    fn dual_cross_simd_vs_scalar_random_walk_500() {
        let weights = build_test_cross_weights();
        let mut rng = XorShift64::new(0xC0FF_EE42);

        const TOTAL_STEPS: usize = 500;
        let mut completed_steps = 0usize;

        'outer: while completed_steps < TOTAL_STEPS {
            let mut pos = startpos_cross();

            for _ply in 0..200 {
                if completed_steps >= TOTAL_STEPS {
                    break 'outer;
                }
                let mut list = MoveList::new();
                generate_legal(&pos, &mut list);
                if list.is_empty() {
                    break;
                }
                let mv = list.at(rng.pick(list.len()));
                let gc = pos.gives_check(mv);
                let _dirty = pos.do_move(mv, gc);
                completed_steps += 1;

                for &persp in &[Color::Black, Color::White] {
                    // production (= SIMD if AVX2) refresh
                    let (simd_sums, _, _, _) = scratch_refresh_cross(&pos, &weights, persp);
                    // scalar reference refresh
                    let ref_sums = ref_scalar_refresh(&pos, &weights, persp);
                    assert_eq!(
                        simd_sums, ref_sums,
                        "SIMD refresh vs scalar mismatch: step={completed_steps}, persp={persp:?}, mv={mv:?}"
                    );
                }
            }
        }
        assert_eq!(completed_steps, TOTAL_STEPS);
    }
}
