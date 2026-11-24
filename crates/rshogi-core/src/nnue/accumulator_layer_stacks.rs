//! AccumulatorLayerStacks - LayerStacksアーキテクチャ用アキュムレータ
//!
//! LayerStacks の Feature Transformer は各視点で L1 次元を出力する。
//! L1 は const generics で 1536 / 768 等を切り替え可能。
//! 既存の Accumulator（256次元、HalfKP用）とは別に管理する。

use super::accumulator::{DirtyPiece, IndexList, MAX_PATH_LENGTH};
use super::bona_piece::BonaPiece;
#[cfg(feature = "halfkav2-hm")]
use super::bona_piece_halfka_hm::is_friend_king_bp;
#[cfg(feature = "nnue-psqt")]
use super::constants::NUM_LAYER_STACK_BUCKETS;
#[cfg(all(
    feature = "nnue-dual-psqt",
    any(
        not(feature = "nnue-scalar-threat"),
        feature = "nnue-scalar-threat-coexist"
    )
))]
use super::dual_cross::{MAX_ACTIVE_DUAL_CROSS_FEATURES, NUM_DUAL_CROSS_BUCKETS};
#[cfg(feature = "nnue-dual-psqt")]
use super::dual_psqt::{MAX_ACTIVE_DUAL_FEATURES, NUM_DUAL_PSQT_BUCKETS, NUM_DUAL_PSQT_HEADS};
#[cfg(feature = "nnue-kingring-psqt")]
use super::kingring::{KINGRING_NUM_BUCKETS, KINGRING_NUM_SQ, KINGRING_SLOT_SENTINEL};
#[cfg(feature = "nnue-pairshape-psqt")]
use super::pairshape::{PAIRSHAPE_MAX_ACTIVE, PAIRSHAPE_NUM_BUCKETS};
use super::piece_list::PieceNumber;
#[cfg(feature = "nnue-pin-aware")]
use super::pin_features::{MAX_ACTIVE_PIN_FEATURES, NUM_PIN_BUCKETS};
use crate::types::{Color, MAX_PLY, Square};

/// LayerStacks用アキュムレータ（L1次元）
#[repr(C, align(64))]
#[derive(Clone)]
pub struct AccumulatorLayerStacks<const L1: usize> {
    /// 各視点の累積値 [perspective][dimension]
    /// perspective: 0 = Black, 1 = White
    pub accumulation: [[i16; L1]; 2],

    /// Threat アキュムレータ [perspective][dimension]
    /// Threat weights (i8) の累積値。評価時に piece accumulation と加算して SCReLU に入力する。
    #[cfg(feature = "nnue-threat")]
    pub threat_accumulation: [[i16; L1]; 2],

    /// KZT-FT アキュムレータ [perspective][dimension] (= feature `nnue-kzt-ft`)。
    ///
    /// KZT (King Zone Threat / 持駒打ち込み脅威) sparse feature を FT 入力として L1
    /// 隠れ層へ流す版。 KZT-FT weights (i8) の累積値。 threat_accumulation と同流儀で、
    /// 評価時に piece accumulation と加算して SCReLU に入力する。
    /// `kzt_accumulation[p]` = attacker=p の単一側 feature 総和 (= us/them は network の
    /// perspective 連結で分離)。
    #[cfg(feature = "nnue-kzt-ft")]
    pub kzt_accumulation: [[i16; L1]; 2],

    /// Full Threat scalar residual の差分 cache `[perspective][bucket]`。
    #[cfg(feature = "nnue-scalar-threat")]
    pub scalar_threat_accumulation:
        [[i32; crate::nnue::scalar_threat::NUM_SCALAR_THREAT_BUCKETS]; 2],

    /// PSQT アキュムレータ [perspective][bucket]
    /// 各駒の PSQT 重みを視点ごとに累積する。
    #[cfg(feature = "nnue-psqt")]
    pub psqt_accumulation: [[i32; NUM_LAYER_STACK_BUCKETS]; 2],

    /// Pin アキュムレータ [perspective][bucket]
    ///
    /// `pin_accumulation[persp][bucket]` =
    ///   `sum_{idx in active_pin_indices(persp)} W[bucket][persp][idx]`。
    /// pin head は単一 scalar の affine head なので、 dense vector を持つ必要は
    /// なく per-bucket × per-perspective の i32 scalar で十分。
    ///
    /// pin 状態が do_move/undo_move で出入りするたびに `±W[b][persp][idx]` を
    /// 加減算する (= 8 bucket × N active indices)。
    #[cfg(feature = "nnue-pin-aware")]
    pub pin_accumulation: [[i32; NUM_PIN_BUCKETS]; 2],

    /// Pin sparse active index cache [perspective][slot]
    ///
    /// `pin_accumulation` を構成する active pin feature index 集合を保存する。
    /// 差分更新時に親 accumulator の cache と現在の active 集合を比較して
    /// `removed` / `added` を抽出するために使う。 stack alloc。
    #[cfg(feature = "nnue-pin-aware")]
    pub pin_active_indices: [[u16; MAX_ACTIVE_PIN_FEATURES]; 2],

    /// 各 perspective での `pin_active_indices` の有効長 (= 実際の active 数)。
    #[cfg(feature = "nnue-pin-aware")]
    pub pin_active_len: [u8; 2],

    /// Dual PSQT (King-Attacker) アキュムレータ
    /// `dual_accumulation[perspective][head][bucket]`
    ///   = sum_{idx in dual_active_indices[persp][head]} W[persp][head][bucket][...]
    ///
    /// per-perspective × per-head × per-bucket の i32 scalar。 head 0 = attack、
    /// head 1 = defend。 do_move/undo_move ごとに `±W` を加減算する。
    #[cfg(feature = "nnue-dual-psqt")]
    pub dual_accumulation: [[[i32; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS]; 2],

    /// Dual sparse active index cache `[perspective][head][slot]`
    ///
    /// `dual_accumulation` を構成する active feature index 集合を head ごとに保存。
    /// 差分更新時に親 acc の cache と現在の active 集合を比較して `removed` / `added`
    /// を抽出するために使う。 stack alloc (= persp 2 × head 2 × MAX 32 = 128 entry)。
    #[cfg(feature = "nnue-dual-psqt")]
    pub dual_active_indices: [[[u16; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS]; 2],

    /// 各 perspective × head での `dual_active_indices` の有効長。
    #[cfg(feature = "nnue-dual-psqt")]
    pub dual_active_len: [[u8; NUM_DUAL_PSQT_HEADS]; 2],

    /// Dual Cross (D2: piece_threat_relation) アキュムレータ (= head 0 のみ、 arm 6)
    /// `dual_cross_accumulation[perspective][bucket]`
    ///   = sum_{idx in dual_cross_active_indices[persp]}
    ///     W_cross[persp][0][bucket][sq][own_p][target_p]
    ///
    /// arm 6 で head 1 (= defended) を production path から完全削除し、 accumulator
    /// 側 layout からも head 次元を縮小 (= Path E、 NPS 改善目的)。 ckpt format 互換性
    /// のため weight 配列の head 次元 ([`NUM_DUAL_CROSS_HEADS`] = 2) は維持する。
    #[cfg(all(
        feature = "nnue-dual-psqt",
        any(
            not(feature = "nnue-scalar-threat"),
            feature = "nnue-scalar-threat-coexist"
        )
    ))]
    pub dual_cross_accumulation: [[i32; NUM_DUAL_CROSS_BUCKETS]; 2],

    /// Dual cross sparse active index cache `[perspective][slot]` (= head 0 のみ)
    ///
    /// 各 entry は `sq * 196 + own_p * 14 + target_p` (= 0..15876、 u16 で表現)。
    /// `MAX_ACTIVE_DUAL_CROSS_FEATURES` = 96。
    #[cfg(all(
        feature = "nnue-dual-psqt",
        any(
            not(feature = "nnue-scalar-threat"),
            feature = "nnue-scalar-threat-coexist"
        )
    ))]
    pub dual_cross_active_indices: [[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; 2],

    /// Dual cross active の **並列 metadata** (= head 0 用 sentinel 配列)
    ///
    /// 旧 arm では head 1 (defended) の defender 生 board sq を保持していたが、
    /// arm 6 で head 1 廃止、 現在は **head 0 で常に sentinel** (= 81 =
    /// [`crate::nnue::dual_cross::DUAL_CROSS_DEF_SQ_NONE`]) のみが入る dead metadata。
    /// 既存 helper API (= `update_dual_cross_perspective_full_delta` 等) 互換性のため
    /// 残存 (= 関数 signature に存在)、 将来削除候補。
    #[cfg(all(
        feature = "nnue-dual-psqt",
        any(
            not(feature = "nnue-scalar-threat"),
            feature = "nnue-scalar-threat-coexist"
        )
    ))]
    pub dual_cross_active_def_sq: [[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES]; 2],

    /// 各 perspective での `dual_cross_active_indices` の有効長 (= u16、 max 96、 head 0 のみ)。
    #[cfg(all(
        feature = "nnue-dual-psqt",
        any(
            not(feature = "nnue-scalar-threat"),
            feature = "nnue-scalar-threat-coexist"
        )
    ))]
    pub dual_cross_active_len: [u16; 2],

    /// `dual_cross_*` 系 state が stale (= non-PV node で update を skip した
    /// 過去差分が反映されていない可能性) かどうか。
    ///
    /// arm 7 Path J + Refresh (= 2026-05-27): non-PV node で
    /// `update_dual_cross_accumulator` が prev → acc を memcpy する際、
    /// prev の `dual_cross_stale` を引き継ぐか新規 true を立てる。 PV node 復帰時に
    /// `update_and_evaluate_layer_stacks_cached` が この flag を見て full refresh を
    /// 1 度だけ走らせ、 stale 値が eval に伝播するのを防ぐ (= 棋力大敗の元凶を断つ)。
    ///
    /// 完全な仕様:
    /// - 初期化: false (= 全 0 で fresh)
    /// - 正規 update / refresh 完了直後: false (= 最新値が反映されている)
    /// - non-PV skip update 直後: true (= prev からの memcpy で diff 未適用)
    #[cfg(all(
        feature = "nnue-dual-psqt",
        any(
            not(feature = "nnue-scalar-threat"),
            feature = "nnue-scalar-threat-coexist"
        )
    ))]
    pub dual_cross_stale: bool,

    /// every-N ply update スキップ用 連続スキップ回数カウンター (= arm 7 path J 拡張)。
    ///
    /// `DUAL_CROSS_UPDATE_INTERVAL` 環境変数で N > 1 のとき、 PV node の
    /// `update_dual_cross_accumulator` 経路で 1..(N-1) 回連続で memcpy skip を許容し、
    /// N 回目で full refresh して stale 値を解消する。 PV node 上で意図的に stale な
    /// dual_cross 値を eval に流すことで NPS overhead を削減する設計の trade-off
    /// (= eval drift と棋力ロスの balance を selfplay で検証)。
    ///
    /// 仕様:
    /// - 初期化: 0
    /// - skip 1 手 (= memcpy from prev): `prev.count + 1` を新 acc に書く
    /// - count + 1 == N: full refresh + count = 0
    /// - non-PV skip path (= `non_pv_dual_skip`) では本 counter は触らない (= 既存 stale
    ///   flag で別管理)
    /// - N == 1 (= default): 旧挙動 (= 毎 ply update)、 本 field は常に 0 のまま
    #[cfg(all(
        feature = "nnue-dual-psqt",
        any(
            not(feature = "nnue-scalar-threat"),
            feature = "nnue-scalar-threat-coexist"
        )
    ))]
    pub dual_cross_update_skip_count: u8,

    /// Dual Cross Geometry残差の差分cache。
    #[cfg(feature = "nnue-dual-cross-geometry")]
    pub dual_cross_geometry_accumulation: [[i32; 8]; 2],
    #[cfg(feature = "nnue-dual-cross-geometry")]
    pub dual_cross_geometry_active_indices: [[u16; MAX_ACTIVE_DUAL_CROSS_FEATURES]; 2],
    #[cfg(feature = "nnue-dual-cross-geometry")]
    pub dual_cross_geometry_active_source_squares: [[u8; MAX_ACTIVE_DUAL_CROSS_FEATURES]; 2],
    #[cfg(feature = "nnue-dual-cross-geometry")]
    pub dual_cross_geometry_active_len: [u16; 2],

    /// KingRing accumulator (= arm 11、 2026-05-28)。
    /// `kingring_accumulation[perspective][bucket] = sum_{ring_idx in 0..24, slot != SENTINEL}
    ///   W[bucket][persp][ring_sq][piece][state]`。
    ///
    /// 玉周辺 24 sq × 13 piece × 4 state を sparse encode し、 1 sq 1 entry の最大
    /// 24 active で 8 bucket 全部に i32 add する。 PSQT 同様 evaluate 側で scalar
    /// 加算するだけで NPS 影響 5% 以内目標。
    #[cfg(feature = "nnue-kingring-psqt")]
    pub kingring_accumulation: [[i32; KINGRING_NUM_BUCKETS]; 2],

    /// KingRing **ring_idx 固定 slot** active cache `[perspective][ring_idx]` (= 2026-05-29)。
    ///
    /// `slots[p][ring_idx] = encoded_idx` (= 0..1248) または `KINGRING_SLOT_SENTINEL`
    /// (= u16::MAX、 空 marker)。 ring_idx 位置が固定なので、 diff update は old/new slot を
    /// 1 対 1 cmp するだけで済む (= removed/added buffer + active_len 不要)。
    ///
    /// king が動いた perspective は ring 全体が shift するため slot diff の意味がなくなり、
    /// network 側で full refresh される。
    #[cfg(feature = "nnue-kingring-psqt")]
    pub kingring_active_indices: [[u16; KINGRING_NUM_SQ]; 2],

    /// PairShape accumulator (= arm 12、 2026-05-29)。
    /// `pairshape_accumulation[perspective][bucket] = sum_{idx in active(persp)}
    ///   W[bucket][persp][pair_type][shape]`。
    ///
    /// 12 pair_type × 16 shape の sparse encode、 sorted multiset として保持。
    /// pair_type ごとに `pieces_pt(pt)` ∩ own を iterate して `(pair_type, shape)` を
    /// 集計、 PSQT 同様 evaluate 側で scalar 加算する。
    #[cfg(feature = "nnue-pairshape-psqt")]
    pub pairshape_accumulation: [[i32; PAIRSHAPE_NUM_BUCKETS]; 2],

    /// PairShape sparse active index cache `[perspective][slot]` (= sorted multiset、 u16)。
    ///
    /// 各 entry は `pair_type_idx * 16 + shape_idx` (= 0..192)。 同 idx の重複は
    /// 多 pair の `(pair_type, shape)` が同一なケースで発生 (= multiset)。
    #[cfg(feature = "nnue-pairshape-psqt")]
    pub pairshape_active_indices: [[u16; PAIRSHAPE_MAX_ACTIVE]; 2],

    /// 各 perspective での `pairshape_active_indices` の有効長 (= u16、 max ~256)。
    #[cfg(feature = "nnue-pairshape-psqt")]
    pub pairshape_active_len: [u16; 2],

    /// 計算済みフラグ
    pub computed_accumulation: bool,

    /// スコア計算済みフラグ（差分更新時にリセット）
    pub computed_score: bool,
}

impl<const L1: usize> AccumulatorLayerStacks<L1> {
    /// 新規作成
    pub fn new() -> Self {
        Self {
            accumulation: [[0; L1]; 2],
            #[cfg(feature = "nnue-threat")]
            threat_accumulation: [[0; L1]; 2],
            #[cfg(feature = "nnue-kzt-ft")]
            kzt_accumulation: [[0; L1]; 2],
            #[cfg(feature = "nnue-scalar-threat")]
            scalar_threat_accumulation: [[0; crate::nnue::scalar_threat::NUM_SCALAR_THREAT_BUCKETS];
                2],
            #[cfg(feature = "nnue-psqt")]
            psqt_accumulation: [[0; NUM_LAYER_STACK_BUCKETS]; 2],
            #[cfg(feature = "nnue-pin-aware")]
            pin_accumulation: [[0; NUM_PIN_BUCKETS]; 2],
            #[cfg(feature = "nnue-pin-aware")]
            pin_active_indices: [[0; MAX_ACTIVE_PIN_FEATURES]; 2],
            #[cfg(feature = "nnue-pin-aware")]
            pin_active_len: [0; 2],
            #[cfg(feature = "nnue-dual-psqt")]
            dual_accumulation: [[[0; NUM_DUAL_PSQT_BUCKETS]; NUM_DUAL_PSQT_HEADS]; 2],
            #[cfg(feature = "nnue-dual-psqt")]
            dual_active_indices: [[[0; MAX_ACTIVE_DUAL_FEATURES]; NUM_DUAL_PSQT_HEADS]; 2],
            #[cfg(feature = "nnue-dual-psqt")]
            dual_active_len: [[0; NUM_DUAL_PSQT_HEADS]; 2],
            #[cfg(all(
                feature = "nnue-dual-psqt",
                any(
                    not(feature = "nnue-scalar-threat"),
                    feature = "nnue-scalar-threat-coexist"
                )
            ))]
            dual_cross_accumulation: [[0; NUM_DUAL_CROSS_BUCKETS]; 2],
            #[cfg(all(
                feature = "nnue-dual-psqt",
                any(
                    not(feature = "nnue-scalar-threat"),
                    feature = "nnue-scalar-threat-coexist"
                )
            ))]
            dual_cross_active_indices: [[0; MAX_ACTIVE_DUAL_CROSS_FEATURES]; 2],
            #[cfg(all(
                feature = "nnue-dual-psqt",
                any(
                    not(feature = "nnue-scalar-threat"),
                    feature = "nnue-scalar-threat-coexist"
                )
            ))]
            dual_cross_active_def_sq: [[crate::nnue::dual_cross::DUAL_CROSS_DEF_SQ_NONE;
                MAX_ACTIVE_DUAL_CROSS_FEATURES]; 2],
            #[cfg(all(
                feature = "nnue-dual-psqt",
                any(
                    not(feature = "nnue-scalar-threat"),
                    feature = "nnue-scalar-threat-coexist"
                )
            ))]
            dual_cross_active_len: [0; 2],
            #[cfg(all(
                feature = "nnue-dual-psqt",
                any(
                    not(feature = "nnue-scalar-threat"),
                    feature = "nnue-scalar-threat-coexist"
                )
            ))]
            dual_cross_stale: false,
            #[cfg(all(
                feature = "nnue-dual-psqt",
                any(
                    not(feature = "nnue-scalar-threat"),
                    feature = "nnue-scalar-threat-coexist"
                )
            ))]
            dual_cross_update_skip_count: 0,
            #[cfg(feature = "nnue-dual-cross-geometry")]
            dual_cross_geometry_accumulation: [[0; 8]; 2],
            #[cfg(feature = "nnue-dual-cross-geometry")]
            dual_cross_geometry_active_indices: [[0; MAX_ACTIVE_DUAL_CROSS_FEATURES]; 2],
            #[cfg(feature = "nnue-dual-cross-geometry")]
            dual_cross_geometry_active_source_squares: [[0; MAX_ACTIVE_DUAL_CROSS_FEATURES]; 2],
            #[cfg(feature = "nnue-dual-cross-geometry")]
            dual_cross_geometry_active_len: [0; 2],
            #[cfg(feature = "nnue-kingring-psqt")]
            kingring_accumulation: [[0; KINGRING_NUM_BUCKETS]; 2],
            #[cfg(feature = "nnue-kingring-psqt")]
            kingring_active_indices: [[KINGRING_SLOT_SENTINEL; KINGRING_NUM_SQ]; 2],
            #[cfg(feature = "nnue-pairshape-psqt")]
            pairshape_accumulation: [[0; PAIRSHAPE_NUM_BUCKETS]; 2],
            #[cfg(feature = "nnue-pairshape-psqt")]
            pairshape_active_indices: [[0; PAIRSHAPE_MAX_ACTIVE]; 2],
            #[cfg(feature = "nnue-pairshape-psqt")]
            pairshape_active_len: [0; 2],
            computed_accumulation: false,
            computed_score: false,
        }
    }

    /// 指定視点の累積値を取得
    #[inline]
    pub fn get(&self, perspective: usize) -> &[i16; L1] {
        debug_assert!(perspective < 2);
        // SAFETY: perspective は Color::Black(0) または Color::White(1) であり、
        //         accumulation は [_; 2] なので常に範囲内。
        unsafe { self.accumulation.get_unchecked(perspective) }
    }

    /// 指定視点の累積値を取得（可変）
    #[inline]
    pub fn get_mut(&mut self, perspective: usize) -> &mut [i16; L1] {
        debug_assert!(perspective < 2);
        // SAFETY: 同上。
        unsafe { self.accumulation.get_unchecked_mut(perspective) }
    }

    /// 指定視点の Threat 累積値を取得
    #[cfg(feature = "nnue-threat")]
    #[inline]
    pub fn get_threat(&self, perspective: usize) -> &[i16; L1] {
        debug_assert!(perspective < 2);
        unsafe { self.threat_accumulation.get_unchecked(perspective) }
    }

    /// 指定視点の Threat 累積値を取得（可変）
    #[cfg(feature = "nnue-threat")]
    #[inline]
    pub fn get_threat_mut(&mut self, perspective: usize) -> &mut [i16; L1] {
        debug_assert!(perspective < 2);
        unsafe { self.threat_accumulation.get_unchecked_mut(perspective) }
    }

    /// 指定視点の KZT-FT 累積値を取得
    #[cfg(feature = "nnue-kzt-ft")]
    #[inline]
    pub fn get_kzt(&self, perspective: usize) -> &[i16; L1] {
        debug_assert!(perspective < 2);
        // SAFETY: perspective は 0 (Black) / 1 (White) で kzt_accumulation は [_; 2]。
        unsafe { self.kzt_accumulation.get_unchecked(perspective) }
    }

    /// 指定視点の KZT-FT 累積値を取得（可変）
    #[cfg(feature = "nnue-kzt-ft")]
    #[inline]
    pub fn get_kzt_mut(&mut self, perspective: usize) -> &mut [i16; L1] {
        debug_assert!(perspective < 2);
        // SAFETY: 同上。
        unsafe { self.kzt_accumulation.get_unchecked_mut(perspective) }
    }

    /// 指定視点・bucket の Pin 累積値を取得
    #[cfg(feature = "nnue-pin-aware")]
    #[inline]
    pub fn pin_bucket(&self, perspective: usize, bucket: usize) -> i32 {
        debug_assert!(perspective < 2);
        debug_assert!(bucket < NUM_PIN_BUCKETS);
        self.pin_accumulation[perspective][bucket]
    }
}

impl<const L1: usize> Default for AccumulatorLayerStacks<L1> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// AccumulatorCacheLayerStacks - Finny Tables（玉位置×視点キャッシュ）
// =============================================================================

/// AccumulatorCaches のキャッシュエントリ（Finny Tables、Stockfish 風 piece_list 差分）
///
/// 各玉位置×視点ごとに、最後に計算したアキュムレータ値と、その時点の
/// `PieceList` (perspective 固有の `piece_list_fb` / `piece_list_fw`) を保持。
/// refresh 時に現在の `PieceList` との slot-wise 比較で差分駒を抽出し、
/// 差分のみ add/sub でアキュムレータを更新することで、
/// `append_active_indices` + `sort_unstable` のオーバーヘッドを回避する。
#[repr(C, align(64))]
struct AccCacheEntry<const L1: usize> {
    /// キャッシュされたアキュムレータ値
    accumulation: [i16; L1],
    /// キャッシュ時点の `PieceList`（perspective 固有の fb または fw 配列）
    ///
    /// `PieceNumber::NB = 40` 固定長。BonaPiece::ZERO は slot 未使用を表す。
    piece_list: [BonaPiece; PieceNumber::NB],
    /// 有効フラグ
    valid: bool,
}

impl<const L1: usize> AccCacheEntry<L1> {
    /// 無効な初期状態で作成
    fn new_invalid() -> Self {
        Self {
            accumulation: [0; L1],
            piece_list: [BonaPiece::ZERO; PieceNumber::NB],
            valid: false,
        }
    }
}

/// 玉位置×視点ごとのアキュムレータキャッシュ（Finny Tables）
///
/// 81マス × 2視点 = 162 エントリ。
/// 玉が移動して full refresh が必要な場合に、前回同じ玉位置で計算した
/// アキュムレータとの差分のみを適用することで高速化する。
pub struct AccumulatorCacheLayerStacks<const L1: usize> {
    /// [king_sq][perspective] のキャッシュエントリ
    entries: Box<[[AccCacheEntry<L1>; 2]; Square::NUM]>,
}

impl<const L1: usize> AccumulatorCacheLayerStacks<L1> {
    /// 新規作成（全エントリ無効）
    pub fn new() -> Self {
        // Box で確保（162 × エントリサイズはスタックに収まらないため）
        let entries: Vec<[AccCacheEntry<L1>; 2]> = (0..Square::NUM)
            .map(|_| [AccCacheEntry::new_invalid(), AccCacheEntry::new_invalid()])
            .collect();
        // SAFETY: Vec の長さが Square::NUM であることを保証
        let boxed: Box<[[AccCacheEntry<L1>; 2]]> = entries.into_boxed_slice();
        // SAFETY: Square::NUM == 81 なので配列サイズと一致する
        let ptr = Box::into_raw(boxed) as *mut [[AccCacheEntry<L1>; 2]; Square::NUM];
        let entries = unsafe { Box::from_raw(ptr) };
        Self { entries }
    }

    /// 全エントリを無効化
    pub fn invalidate(&mut self) {
        for sq_entries in self.entries.iter_mut() {
            for entry in sq_entries.iter_mut() {
                entry.valid = false;
            }
        }
    }

    /// キャッシュからの差分で refresh を実行（Stockfish 風 piece_list 差分方式）
    ///
    /// キャッシュが有効な場合、現在の `PieceList` と cache の `PieceList` を
    /// slot-wise に比較し、変化した slot のみ `idx_fn` で feature index を
    /// 算出して add/sub を適用する。
    /// キャッシュが無効な場合は biases から full refresh し、cache を更新する。
    ///
    /// # 引数
    ///
    /// - `king_sq`: この視点の玉位置（cache key）
    /// - `perspective`: 視点（cache key）
    /// - `piece_list`: 現在の perspective 固有 `PieceList`（`piece_list_fb` / `piece_list_fw`）
    /// - `biases`: Feature Transformer のバイアス
    /// - `accumulation`: 更新先のアキュムレータ値
    /// - `idx_fn`: BonaPiece (非 ZERO) → feature index の変換
    /// - `add_fn`: 重み加算関数
    /// - `sub_fn`: 重み減算関数
    pub(crate) fn refresh_or_cache<FI, FA, FS>(
        &mut self,
        king_sq: Square,
        perspective: Color,
        piece_list: &[BonaPiece; PieceNumber::NB],
        biases: &[i16; L1],
        accumulation: &mut [i16; L1],
        idx_fn: FI,
        add_fn: FA,
        sub_fn: FS,
    ) where
        FI: Fn(BonaPiece) -> usize,
        FA: Fn(&mut [i16; L1], usize),
        FS: Fn(&mut [i16; L1], usize),
    {
        let entry = &mut self.entries[king_sq.raw() as usize][perspective as usize];

        if entry.valid {
            crate::nnue::stats::count_cache_hit!();
            // キャッシュのアキュムレータ値をコピー
            accumulation.copy_from_slice(&entry.accumulation);

            // 40 slot 比較して変化した slot のみ差分適用
            let mut diff_count = 0usize;
            for (cached_bp, &current_bp) in entry.piece_list.iter().copied().zip(piece_list.iter())
            {
                if cached_bp != current_bp {
                    if cached_bp != BonaPiece::ZERO {
                        // HalfKAv2_hm: friend_king feature は K bucket と完全相関で無情報なので skip
                        #[cfg(feature = "halfkav2-hm")]
                        let skip_cached = is_friend_king_bp(cached_bp);
                        #[cfg(not(feature = "halfkav2-hm"))]
                        let skip_cached = false;
                        if !skip_cached {
                            sub_fn(accumulation, idx_fn(cached_bp));
                            diff_count += 1;
                        }
                    }
                    if current_bp != BonaPiece::ZERO {
                        #[cfg(feature = "halfkav2-hm")]
                        let skip_current = is_friend_king_bp(current_bp);
                        #[cfg(not(feature = "halfkav2-hm"))]
                        let skip_current = false;
                        if !skip_current {
                            add_fn(accumulation, idx_fn(current_bp));
                            diff_count += 1;
                        }
                    }
                }
            }
            crate::nnue::stats::count_refresh_diff!(diff_count);
        } else {
            crate::nnue::stats::count_cache_miss!();
            // キャッシュ無効 → バイアスから full refresh
            accumulation.copy_from_slice(biases);
            for &bp in piece_list.iter() {
                if bp != BonaPiece::ZERO {
                    // HalfKAv2_hm: friend_king feature は K bucket と完全相関で無情報なので skip
                    #[cfg(feature = "halfkav2-hm")]
                    if is_friend_king_bp(bp) {
                        continue;
                    }
                    add_fn(accumulation, idx_fn(bp));
                }
            }
        }

        // キャッシュを更新
        entry.accumulation.copy_from_slice(accumulation);
        entry.piece_list.copy_from_slice(piece_list);
        entry.valid = true;
    }
}

impl<const L1: usize> Default for AccumulatorCacheLayerStacks<L1> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// DirtyPiece - 駒の変更情報（LayerStacks用、accumulator.rsから再エクスポート）
// =============================================================================

// DirtyPiece は accumulator.rs で定義済み

// =============================================================================
// StackEntryLayerStacks - スタックエントリ
// =============================================================================

/// スタックエントリ（LayerStacks用）
pub struct StackEntryLayerStacks<const L1: usize> {
    /// アキュムレータ
    pub accumulator: AccumulatorLayerStacks<L1>,
    /// 変更された駒の情報
    pub dirty_piece: DirtyPiece,
    /// 直前のエントリインデックス（差分計算用）
    pub previous: Option<usize>,
    /// progress8kpabs の重み付き和（差分更新用）
    #[cfg(feature = "nnue-progress-diff")]
    pub progress_sum: f32,
    /// progress_sum 計算済みフラグ
    #[cfg(feature = "nnue-progress-diff")]
    pub computed_progress: bool,
    /// 2026-06-06: Activity-Risk Interaction Head accumulator (= 中盤楽観バイアス補正用)。
    /// dirty=true で次回 evaluate 時 lazy refresh、 false なら cached features 利用。
    #[cfg(feature = "nnue-activity-risk-psqt")]
    pub activity_risk_accum: super::activity_risk::ActivityRiskAccumulator,
}

impl<const L1: usize> StackEntryLayerStacks<L1> {
    pub fn new() -> Self {
        Self {
            accumulator: AccumulatorLayerStacks::new(),
            dirty_piece: DirtyPiece::default(),
            previous: None,
            #[cfg(feature = "nnue-progress-diff")]
            progress_sum: 0.0,
            #[cfg(feature = "nnue-progress-diff")]
            computed_progress: false,
            #[cfg(feature = "nnue-activity-risk-psqt")]
            activity_risk_accum: super::activity_risk::ActivityRiskAccumulator {
                dirty: true,
                ..Default::default()
            },
        }
    }
}

impl<const L1: usize> Default for StackEntryLayerStacks<L1> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// AccumulatorStackLayerStacks - スタック管理
// =============================================================================

/// アキュムレータスタック（LayerStacks用）
pub struct AccumulatorStackLayerStacks<const L1: usize> {
    /// スタックエントリ
    entries: Box<[StackEntryLayerStacks<L1>]>,
    /// 現在のインデックス
    current: usize,
}

impl<const L1: usize> AccumulatorStackLayerStacks<L1> {
    const STACK_SIZE: usize = (MAX_PLY as usize) + 16;

    /// 新規作成
    pub fn new() -> Self {
        let entries: Vec<StackEntryLayerStacks<L1>> =
            (0..Self::STACK_SIZE).map(|_| StackEntryLayerStacks::new()).collect();

        Self {
            entries: entries.into_boxed_slice(),
            current: 0,
        }
    }

    /// 現在のエントリを取得
    #[inline]
    pub fn current(&self) -> &StackEntryLayerStacks<L1> {
        debug_assert!(self.current < self.entries.len());
        // SAFETY: current は push/pop の対でインクリメント/デクリメントされ、
        //         do_move と undo_move の対称呼び出しにより 0 <= current < STACK_SIZE が保証される。
        unsafe { self.entries.get_unchecked(self.current) }
    }

    /// 現在のエントリを取得（可変）
    #[inline]
    pub fn current_mut(&mut self) -> &mut StackEntryLayerStacks<L1> {
        debug_assert!(self.current < self.entries.len());
        // SAFETY: 同上。do_move/undo_move の対称呼び出しで current は常に範囲内。
        unsafe { self.entries.get_unchecked_mut(self.current) }
    }

    /// 現在のインデックスを取得
    #[inline]
    pub fn current_index(&self) -> usize {
        self.current
    }

    /// 指定インデックスのエントリを取得
    #[inline]
    pub(crate) fn entry_at(&self, index: usize) -> &StackEntryLayerStacks<L1> {
        debug_assert!(index < self.entries.len());
        // SAFETY: index は previous チェーンまたは find_usable_accumulator 由来で常に
        //         current 以下の有効なインデックス（STACK_SIZE 未満）。
        unsafe { self.entries.get_unchecked(index) }
    }

    /// スタックをプッシュ
    #[inline]
    pub fn push(&mut self) {
        let prev = self.current;
        self.current += 1;
        debug_assert!(self.current < Self::STACK_SIZE);
        // SAFETY: current < STACK_SIZE は上の debug_assert で検証。
        //         push は do_move ごとに 1 回呼ばれ、pop と対になるため
        //         current は常に STACK_SIZE 未満。
        let entry = unsafe { self.entries.get_unchecked_mut(self.current) };
        entry.previous = Some(prev);
        entry.accumulator.computed_accumulation = false;
        entry.accumulator.computed_score = false;
        entry.dirty_piece = DirtyPiece::default();
        #[cfg(feature = "nnue-progress-diff")]
        {
            entry.computed_progress = false;
        }
        // 2026-06-06: activity_risk accumulator は dirty=true で lazy refresh path 採用。
        // 真 incremental update は set_current_dirty_piece + DirtyPiece info 抽出 + shield/pressure
        // delta apply で実現可能、 ただし ExtBonaPiece 逆引き必要で別 step。
        #[cfg(feature = "nnue-activity-risk-psqt")]
        {
            entry.activity_risk_accum.dirty = true;
        }
    }

    /// スタックをポップ
    #[inline]
    pub fn pop(&mut self) {
        debug_assert!(self.current > 0);
        self.current -= 1;
    }

    /// 前回と現在のアキュムレータを同時に取得（clone不要）
    ///
    /// `split_at_mut`を使用して、prev_idx の accumulator への不変参照と
    /// 現在の accumulator への可変参照を同時に返す。
    #[inline]
    pub fn get_prev_and_current_accumulators(
        &mut self,
        prev_idx: usize,
    ) -> (&AccumulatorLayerStacks<L1>, &mut AccumulatorLayerStacks<L1>) {
        let cur_idx = self.current;
        debug_assert!(prev_idx < cur_idx, "prev_idx ({prev_idx}) must be < cur_idx ({cur_idx})");
        debug_assert!(cur_idx < self.entries.len());
        let (left, right) = self.entries.split_at_mut(cur_idx);
        // SAFETY: prev_idx < cur_idx（上の debug_assert で検証）かつ left の長さは cur_idx。
        //         right は少なくとも 1 要素を持つ（cur_idx < entries.len() を保証）。
        unsafe {
            (
                &left.get_unchecked(prev_idx).accumulator,
                &mut right.get_unchecked_mut(0).accumulator,
            )
        }
    }

    /// source_idx (= 過去エントリ) の accumulator と current を同時取得 (= path F NPS
    /// 最適化: `forward_update_incremental` の全 accumulator clone を回避)。
    ///
    /// `get_prev_and_current_accumulators` と同様 `split_at_mut` で borrow を分離するが、
    /// source_idx は任意の past index (= path 長 1+ の祖先) で良い。
    #[inline]
    pub(crate) fn get_source_and_current_accumulators(
        &mut self,
        source_idx: usize,
    ) -> (&AccumulatorLayerStacks<L1>, &mut AccumulatorLayerStacks<L1>) {
        let cur_idx = self.current;
        debug_assert!(
            source_idx < cur_idx,
            "source_idx ({source_idx}) must be < cur_idx ({cur_idx})"
        );
        debug_assert!(cur_idx < self.entries.len());
        let (left, right) = self.entries.split_at_mut(cur_idx);
        // SAFETY: source_idx < cur_idx (debug_assert で検証)、 right[0] が current。
        unsafe {
            (
                &left.get_unchecked(source_idx).accumulator,
                &mut right.get_unchecked_mut(0).accumulator,
            )
        }
    }

    /// スタックをリセット
    #[inline]
    pub fn reset(&mut self) {
        self.current = 0;
        self.entries[0].accumulator.computed_accumulation = false;
        self.entries[0].accumulator.computed_score = false;
        self.entries[0].previous = None;
        #[cfg(feature = "nnue-progress-diff")]
        {
            self.entries[0].computed_progress = false;
        }
    }

    /// 祖先を辿って使用可能なアキュムレータを探す
    ///
    /// ## 実装方針
    ///
    /// アキュムレータの差分更新における祖先探索には複数のアプローチがある:
    ///
    /// - **YaneuraOu方式**: 1手前のみをチェック（シンプルだが差分更新の機会を逃す）
    /// - **Stockfish方式**: スタック全体を探索し、各ステップで玉移動をチェック
    ///
    /// このプロジェクトでは、HalfKP側（accumulator.rs）と同じロジックを採用している。
    /// 最大8手前まで探索し、各ステップで玉移動があれば即座に打ち切る方式である。
    /// この方式により、1手前限定より多くの差分更新機会を得つつ、玉移動時の
    /// 無駄な探索を早期に打ち切ることでNPS向上が観測されている。
    ///
    /// ## 戻り値
    ///
    /// `Some((計算済みエントリのインデックス, 経由する局面数))` - 玉移動がない範囲で
    /// 計算済み祖先が見つかった場合。`None` - 使用可能な祖先が見つからない場合。
    pub fn find_usable_accumulator(&self) -> Option<(usize, usize)> {
        // representative 4局面 x 2 rounds の search-only A/B では
        // MAX_DEPTH=4 が MAX_DEPTH=1 比で +2.15% だったため維持する。
        const MAX_DEPTH: usize = 4;

        debug_assert!(self.current < self.entries.len());
        // SAFETY: current は do_move/undo_move の対称呼び出しで常に範囲内。
        let current = unsafe { self.entries.get_unchecked(self.current) };

        // 現局面で玉が動いていたら差分更新不可
        if current.dirty_piece.king_moved[0] || current.dirty_piece.king_moved[1] {
            return None;
        }

        // 直前局面をチェック（depth=1から開始）
        let mut prev_idx = current.previous?;
        let mut depth = 1;

        loop {
            debug_assert!(prev_idx < self.entries.len());
            // SAFETY: prev_idx は previous チェーンを辿った有効なインデックス（STACK_SIZE 未満）。
            let prev = unsafe { self.entries.get_unchecked(prev_idx) };

            // 計算済みなら成功
            if prev.accumulator.computed_accumulation {
                return Some((prev_idx, depth));
            }

            // 探索上限に達した
            if depth >= MAX_DEPTH {
                return None;
            }

            // さらに前の局面へ（ルートに達したらNone）
            let next_prev_idx = prev.previous?;

            // 玉が動いていたら打ち切り（早期終了による最適化）
            if prev.dirty_piece.king_moved[0] || prev.dirty_piece.king_moved[1] {
                return None;
            }

            prev_idx = next_prev_idx;
            depth += 1;
        }
    }

    /// 指定インデックスから現在位置までのパスを収集
    ///
    /// 戻り値:
    /// - Some(path): source_idx に到達できた場合、source側から適用する順のインデックス列
    /// - None: パスが途切れた場合、または MAX_PATH_LENGTH を超えた場合
    pub fn collect_path(&self, source_idx: usize) -> Option<IndexList<MAX_PATH_LENGTH>> {
        self.collect_path_internal(source_idx)
    }

    fn collect_path_internal(&self, source_idx: usize) -> Option<IndexList<MAX_PATH_LENGTH>> {
        let mut path = IndexList::new();
        let mut idx = self.current;

        while idx != source_idx {
            // パス長が上限を超えたら失敗
            if !path.push(idx) {
                return None;
            }
            match self.entries[idx].previous {
                Some(prev) => idx = prev,
                None => return None,
            }
        }

        path.reverse();
        Some(path)
    }
}

impl<const L1: usize> Default for AccumulatorStackLayerStacks<L1> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// LayerStacksAccStack - L1 サイズ dispatch enum
// =============================================================================

/// LayerStacks アキュムレータスタックの L1 サイズ dispatch enum
///
/// Cargo feature `layerstacks-1536x16x32` / `layerstacks-1536x32x32`
/// / `layerstacks-768x16x32` / `layerstacks-512x16x32` で有効なバリアントが制御される。
pub enum LayerStacksAccStack {
    #[cfg(feature = "layerstacks-1536x16x32")]
    L1536x16x32(AccumulatorStackLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x32x32")]
    L1536x32x32(AccumulatorStackLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x8x96")]
    L1536x8x96(AccumulatorStackLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x16x96")]
    L1536x16x96(AccumulatorStackLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x32x96")]
    L1536x32x96(AccumulatorStackLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x48x96")]
    L1536x48x96(AccumulatorStackLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x64x96")]
    L1536x64x96(AccumulatorStackLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x16x48")]
    L1536x16x48(AccumulatorStackLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1024x16x48")]
    L1024x16x48(AccumulatorStackLayerStacks<1024>),
    #[cfg(feature = "layerstacks-1024x8x64")]
    L1024x8x64(AccumulatorStackLayerStacks<1024>),
    #[cfg(feature = "layerstacks-768x32x96")]
    L768x32x96(AccumulatorStackLayerStacks<768>),
    #[cfg(feature = "layerstacks-768x16x32")]
    L768x16x32(AccumulatorStackLayerStacks<768>),
    #[cfg(feature = "layerstacks-768x8x32")]
    L768x8x32(AccumulatorStackLayerStacks<768>),
    #[cfg(feature = "layerstacks-512x16x32")]
    L512x16x32(AccumulatorStackLayerStacks<512>),
}

/// LayerStacks dispatch match の網羅性を確保するマクロ
///
/// 全 feature が無効の場合（WASM ビルド等）は空 enum のためインスタンスが存在しないが、
/// コンパイルを通すために wildcard arm を生成する。
macro_rules! ls_match {
    ($val:expr, $pat:ident => $body:expr) => {
        match $val {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32($pat) => $body,
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32($pat) => $body,
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96($pat) => $body,
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96($pat) => $body,
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96($pat) => $body,
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96($pat) => $body,
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96($pat) => $body,
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48($pat) => $body,
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48($pat) => $body,
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64($pat) => $body,
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96($pat) => $body,
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32($pat) => $body,
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32($pat) => $body,
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32($pat) => $body,
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    };
}

impl LayerStacksAccStack {
    /// L1 サイズを取得
    pub fn l1_size(&self) -> usize {
        match self {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32(_) => 1536,
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32(_) => 1536,
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48(_) => 1536,
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48(_) => 1024,
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64(_) => 1024,
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96(_) => 768,
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32(_) => 768,
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32(_) => 768,
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32(_) => 512,
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    }

    /// アーキテクチャ寸法 (L1, L2, L3) を返す
    pub fn architecture_dims(&self) -> (usize, usize, usize) {
        match self {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32(_) => (1536, 16, 32),
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32(_) => (1536, 32, 32),
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96(_) => (1536, 8, 96),
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96(_) => (1536, 16, 96),
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96(_) => (1536, 32, 96),
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96(_) => (1536, 48, 96),
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96(_) => (1536, 64, 96),
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48(_) => (1536, 16, 48),
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48(_) => (1024, 16, 48),
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64(_) => (1024, 8, 64),
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96(_) => (768, 32, 96),
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32(_) => (768, 16, 32),
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32(_) => (768, 8, 32),
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32(_) => (512, 16, 32),
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    }

    /// スタックをリセット
    #[inline]
    pub fn reset(&mut self) {
        ls_match!(self, s => s.reset());
    }

    /// do_move 時にスタックをプッシュ
    #[inline]
    pub fn push(&mut self) {
        ls_match!(self, s => s.push());
    }

    /// undo_move 時にスタックをポップ
    #[inline]
    pub fn pop(&mut self) {
        ls_match!(self, s => s.pop());
    }

    /// 現在のエントリの dirty_piece を設定
    #[inline]
    pub fn set_current_dirty_piece(&mut self, dirty: DirtyPiece) {
        ls_match!(self, s => s.current_mut().dirty_piece = dirty);
    }
}

// =============================================================================
// LayerStacksAccCache - L1 サイズ dispatch enum
// =============================================================================

/// LayerStacks アキュムレータキャッシュの L1 サイズ dispatch enum
pub enum LayerStacksAccCache {
    #[cfg(feature = "layerstacks-1536x16x32")]
    L1536x16x32(AccumulatorCacheLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x32x32")]
    L1536x32x32(AccumulatorCacheLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x8x96")]
    L1536x8x96(AccumulatorCacheLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x16x96")]
    L1536x16x96(AccumulatorCacheLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x32x96")]
    L1536x32x96(AccumulatorCacheLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x48x96")]
    L1536x48x96(AccumulatorCacheLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x64x96")]
    L1536x64x96(AccumulatorCacheLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1536x16x48")]
    L1536x16x48(AccumulatorCacheLayerStacks<1536>),
    #[cfg(feature = "layerstacks-1024x16x48")]
    L1024x16x48(AccumulatorCacheLayerStacks<1024>),
    #[cfg(feature = "layerstacks-1024x8x64")]
    L1024x8x64(AccumulatorCacheLayerStacks<1024>),
    #[cfg(feature = "layerstacks-768x32x96")]
    L768x32x96(AccumulatorCacheLayerStacks<768>),
    #[cfg(feature = "layerstacks-768x16x32")]
    L768x16x32(AccumulatorCacheLayerStacks<768>),
    #[cfg(feature = "layerstacks-768x8x32")]
    L768x8x32(AccumulatorCacheLayerStacks<768>),
    #[cfg(feature = "layerstacks-512x16x32")]
    L512x16x32(AccumulatorCacheLayerStacks<512>),
}

impl LayerStacksAccCache {
    /// この cache の L1 サイズを返す
    ///
    /// ネットワーク reload で L1 バリアントが変わったときに、旧 cache を
    /// 使い回してしまう事故を防ぐため、`prepare_search` 側で
    /// `network.l1_size()` と比較するために使う。
    pub fn l1_size(&self) -> usize {
        match self {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32(_) => 1536,
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32(_) => 1536,
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96(_) => 1536,
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48(_) => 1536,
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48(_) => 1024,
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64(_) => 1024,
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96(_) => 768,
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32(_) => 768,
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32(_) => 768,
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32(_) => 512,
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    }

    /// アーキテクチャ寸法 (L1, L2, L3) を返す
    pub fn architecture_dims(&self) -> (usize, usize, usize) {
        match self {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32(_) => (1536, 16, 32),
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32(_) => (1536, 32, 32),
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96(_) => (1536, 8, 96),
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96(_) => (1536, 16, 96),
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96(_) => (1536, 32, 96),
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96(_) => (1536, 48, 96),
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96(_) => (1536, 64, 96),
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48(_) => (1536, 16, 48),
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48(_) => (1024, 16, 48),
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64(_) => (1024, 8, 64),
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96(_) => (768, 32, 96),
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32(_) => (768, 16, 32),
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32(_) => (768, 8, 32),
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32(_) => (512, 16, 32),
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    }

    /// 全エントリを無効化
    pub fn invalidate(&mut self) {
        match self {
            #[cfg(feature = "layerstacks-1536x16x32")]
            Self::L1536x16x32(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-1536x32x32")]
            Self::L1536x32x32(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-1536x8x96")]
            Self::L1536x8x96(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-1536x16x96")]
            Self::L1536x16x96(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-1536x32x96")]
            Self::L1536x32x96(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-1536x48x96")]
            Self::L1536x48x96(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-1536x64x96")]
            Self::L1536x64x96(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-1536x16x48")]
            Self::L1536x16x48(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-1024x16x48")]
            Self::L1024x16x48(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-1024x8x64")]
            Self::L1024x8x64(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-768x32x96")]
            Self::L768x32x96(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-768x16x32")]
            Self::L768x16x32(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-768x8x32")]
            Self::L768x8x32(c) => c.invalidate(),
            #[cfg(feature = "layerstacks-512x16x32")]
            Self::L512x16x32(c) => c.invalidate(),
            #[cfg(not(any(
                feature = "layerstacks-1536x16x32",
                feature = "layerstacks-1536x32x32",
                feature = "layerstacks-1536x8x96",
                feature = "layerstacks-1536x16x96",
                feature = "layerstacks-1536x32x96",
                feature = "layerstacks-1536x48x96",
                feature = "layerstacks-1536x64x96",
                feature = "layerstacks-1536x16x48",
                feature = "layerstacks-1024x16x48",
                feature = "layerstacks-1024x8x64",
                feature = "layerstacks-768x32x96",
                feature = "layerstacks-768x16x32",
                feature = "layerstacks-768x8x32",
                feature = "layerstacks-512x16x32"
            )))]
            _ => unreachable!("no LayerStacks variant enabled"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::constants::NNUE_PYTORCH_L1;

    /// テスト用の具体的な L1 サイズ
    const TEST_L1: usize = NNUE_PYTORCH_L1; // 1536

    #[test]
    fn test_accumulator_new() {
        let acc = AccumulatorLayerStacks::<TEST_L1>::new();
        assert!(!acc.computed_accumulation);
        assert_eq!(acc.accumulation[0].len(), TEST_L1);
    }

    #[test]
    fn test_stack_push_pop() {
        let mut stack = AccumulatorStackLayerStacks::<TEST_L1>::new();
        assert_eq!(stack.current_index(), 0);

        stack.push();
        assert_eq!(stack.current_index(), 1);
        assert_eq!(stack.current().previous, Some(0));

        stack.pop();
        assert_eq!(stack.current_index(), 0);
    }

    #[test]
    fn test_cache_new_is_invalid() {
        let cache = AccumulatorCacheLayerStacks::<TEST_L1>::new();
        // 全エントリが無効であることを確認
        for sq in 0..Square::NUM {
            // SAFETY: sq は 0..81 の範囲内であることが Square::NUM により保証
            let king_sq = unsafe { Square::from_u8_unchecked(sq as u8) };
            for perspective in [Color::Black, Color::White] {
                let entry = &cache.entries[king_sq.raw() as usize][perspective as usize];
                assert!(!entry.valid);
            }
        }
    }

    #[test]
    fn test_cache_invalidate() {
        let mut cache = AccumulatorCacheLayerStacks::<TEST_L1>::new();
        // エントリを有効にする
        cache.entries[0][0].valid = true;
        cache.entries[40][1].valid = true;

        cache.invalidate();

        // 全エントリが無効になっていることを確認
        assert!(!cache.entries[0][0].valid);
        assert!(!cache.entries[40][1].valid);
    }

    /// refresh_or_cache: cold start → full refresh → cache に保存
    #[test]
    fn test_refresh_or_cache_cold_start() {
        let mut cache = AccumulatorCacheLayerStacks::<TEST_L1>::new();
        let king_sq = Square::SQ_55;
        let perspective = Color::Black;

        let mut biases = [0i16; TEST_L1];
        biases[0] = 100;
        biases[1] = 200;

        // ダミーの piece_list: slot 0-2 に非 ZERO BonaPiece
        let mut piece_list = [BonaPiece::ZERO; PieceNumber::NB];
        piece_list[0] = BonaPiece(5);
        piece_list[1] = BonaPiece(10);
        piece_list[2] = BonaPiece(15);

        let mut accumulation = [0i16; TEST_L1];
        cache.refresh_or_cache(
            king_sq,
            perspective,
            &piece_list,
            &biases,
            &mut accumulation,
            |bp| bp.0 as usize,
            |acc, idx| acc[0] = acc[0].wrapping_add(idx as i16),
            |acc, idx| acc[0] = acc[0].wrapping_sub(idx as i16),
        );

        // biases[0] + 5 + 10 + 15 = 130
        assert_eq!(accumulation[0], 130);
        assert_eq!(accumulation[1], 200);

        let entry = &cache.entries[king_sq.raw() as usize][perspective as usize];
        assert!(entry.valid);
        assert_eq!(entry.piece_list[0], BonaPiece(5));
        assert_eq!(entry.piece_list[1], BonaPiece(10));
        assert_eq!(entry.piece_list[2], BonaPiece(15));
    }

    /// refresh_or_cache: 2 回目のキャッシュヒット → piece_list 差分更新
    #[test]
    fn test_refresh_or_cache_hit() {
        let mut cache = AccumulatorCacheLayerStacks::<TEST_L1>::new();
        let king_sq = Square::SQ_55;
        let perspective = Color::Black;
        let biases = [0i16; TEST_L1];

        // 1 回目: slot 0-2 = [5, 10, 15]
        let mut pl1 = [BonaPiece::ZERO; PieceNumber::NB];
        pl1[0] = BonaPiece(5);
        pl1[1] = BonaPiece(10);
        pl1[2] = BonaPiece(15);
        let mut acc1 = [0i16; TEST_L1];
        cache.refresh_or_cache(
            king_sq,
            perspective,
            &pl1,
            &biases,
            &mut acc1,
            |bp| bp.0 as usize,
            |acc, idx| acc[0] = acc[0].wrapping_add(idx as i16),
            |acc, idx| acc[0] = acc[0].wrapping_sub(idx as i16),
        );
        assert_eq!(acc1[0], 30);

        // 2 回目: slot 2 の 15 → 20 に変化
        let mut pl2 = pl1;
        pl2[2] = BonaPiece(20);
        let mut acc2 = [0i16; TEST_L1];
        cache.refresh_or_cache(
            king_sq,
            perspective,
            &pl2,
            &biases,
            &mut acc2,
            |bp| bp.0 as usize,
            |acc, idx| acc[0] = acc[0].wrapping_add(idx as i16),
            |acc, idx| acc[0] = acc[0].wrapping_sub(idx as i16),
        );
        // hit: 30 - 15 + 20 = 35
        assert_eq!(acc2[0], 35);
    }

    /// refresh_or_cache: slot 消滅 (capture)
    #[test]
    fn test_refresh_or_cache_slot_disappears() {
        let mut cache = AccumulatorCacheLayerStacks::<TEST_L1>::new();
        let king_sq = Square::SQ_55;
        let perspective = Color::Black;
        let biases = [0i16; TEST_L1];

        let mut pl1 = [BonaPiece::ZERO; PieceNumber::NB];
        pl1[0] = BonaPiece(5);
        pl1[1] = BonaPiece(10);
        let mut acc1 = [0i16; TEST_L1];
        cache.refresh_or_cache(
            king_sq,
            perspective,
            &pl1,
            &biases,
            &mut acc1,
            |bp| bp.0 as usize,
            |acc, idx| acc[0] = acc[0].wrapping_add(idx as i16),
            |acc, idx| acc[0] = acc[0].wrapping_sub(idx as i16),
        );
        assert_eq!(acc1[0], 15);

        // slot 1 が ZERO になった
        let mut pl2 = pl1;
        pl2[1] = BonaPiece::ZERO;
        let mut acc2 = [0i16; TEST_L1];
        cache.refresh_or_cache(
            king_sq,
            perspective,
            &pl2,
            &biases,
            &mut acc2,
            |bp| bp.0 as usize,
            |acc, idx| acc[0] = acc[0].wrapping_add(idx as i16),
            |acc, idx| acc[0] = acc[0].wrapping_sub(idx as i16),
        );
        // hit: 15 - 10 = 5
        assert_eq!(acc2[0], 5);
    }
}
