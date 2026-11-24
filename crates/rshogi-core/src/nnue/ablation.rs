//! Plan C feature ablation flags (NPS 計測専用、 棋力評価不可)
//!
//! `PLAN_C_ABLATE` 環境変数 (csv) で起動時に各 head の eval 寄与を skip する。
//!
//! 使用例:
//! ```bash
//! PLAN_C_ABLATE=threat,pin ./rshogi-usi
//! PLAN_C_ABLATE=threat ./bench_nnue_eval ...
//! ```
//!
//! 受理 token: `psqt` / `pin` / `threat` / `dual` / `kingring` / `pairshape` / `all`
//!
//! ## skip の意味
//!
//! - **psqt**: evaluate() で psqt_value 加算を 0 に。 accumulator psqt 更新は維持
//!   (= 計測対象外、 更新コストは psqt_accumulation のみ)
//! - **pin**: evaluate() で `pin_head_eval` 呼び出しを skip
//! - **threat**: accumulator update の `threat_features::for_each_active_threat_index` /
//!   `add_threat_weights` / `sub_threat_weights` を全て skip。 transformed 計算で
//!   `+ threat_accumulation[bucket]` を 0 ベクトル相当に
//! - **dual**: evaluate() で `compute_dual_king_attacker` 呼び出しを skip
//!   (= King-Attacker 設計、 旧 dual_psqt_head_eval は 2026-05-26 削除)
//! - **kingring**: evaluate() で kingring_value 加算を skip + accumulator 更新も skip
//!   (= arm 11、 NPS 比較用)
//!
//! ## 注意
//!
//! ckpt 読み込み (= section parse + memory 確保) は変更しない。 read path は維持して
//! eval/update path の hot loop だけを skip する。 棋力評価には使用不可 (= head 寄与が
//! 0 になるので強度低下)。

use std::cell::Cell;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

static SKIP_PSQT_EVAL: AtomicBool = AtomicBool::new(false);
static SKIP_PIN_EVAL: AtomicBool = AtomicBool::new(false);
static SKIP_THREAT_UPDATE: AtomicBool = AtomicBool::new(false);
static SKIP_DUAL_EVAL: AtomicBool = AtomicBool::new(false);
static SKIP_KINGRING_EVAL: AtomicBool = AtomicBool::new(false);
static SKIP_PAIRSHAPE_EVAL: AtomicBool = AtomicBool::new(false);

/// `DUAL_CROSS_UPDATE_INTERVAL` env var で指定された更新間隔 N (= 1..255、 default 1)。
///
/// 1 = 旧挙動 (= 毎 ply update)、 2 以上で (N-1) 回 skip → 1 回 refresh のパターンを
/// PV node 限定で適用する (= arm 7 path J 拡張、 NPS と棋力 trade-off 計測)。
static DUAL_CROSS_UPDATE_INTERVAL: AtomicU8 = AtomicU8::new(1);
static DUAL_CROSS_INTERVAL_INIT: Once = Once::new();

/// `DUAL_CROSS_ENDGAME_ONLY` env var で指定された終盤限定モード (= default false)。
///
/// true の間、 ply 1..=100 (= opening + middle) では `update_dual_cross_accumulator`
/// が delta update を skip (= prev → acc memcpy + stale 化) し、 `evaluate_with_bucket`
/// 側も `dual_cross_value` 加算を 0 に gate する (= PSQT only 相当の高速 eval)。
/// ply 101.. (= endgame) のみ通常の dual_cross update + eval を走らせる。
///
/// 動機: arm 6 で全 phase delta update は NPS -27% (= 局面複雑度で endgame の delta
/// コストが重い)。 ponkotsuWCSC35 学習の dual_cross は endgame 寄与が主で opening
/// 寄与は薄いと仮定し、 opening/middle を skip して NPS を 10-15pp 回復する。
static DUAL_CROSS_ENDGAME_ONLY: AtomicBool = AtomicBool::new(false);
static DUAL_CROSS_ENDGAME_ONLY_INIT: Once = Once::new();

/// `DUAL_CROSS_USE_ENDGAME_DETECTOR` env var で指定された position-based 終盤限定モード
/// (= default false)。
///
/// true の間、 既存の ply-based phase 判定 (= `is_endgame_ply`) ではなく、 探索側 USI が
/// `p2_search::phase::GamePhase` を `dual_cross::set_current_phase_idx` で thread-local
/// に書き込んだ override 値を見て gate を判断する (= position-based、 駒数 + eval +
/// mobility 等 多変量)。 override が `GamePhase::Endgame` (= 2) のとき通常の dual_cross
/// update + eval が走り、 Opening/Middle (= 0/1) のときは skip 経路に入る。
///
/// 動機: a85f sub-agent の `DUAL_CROSS_ENDGAME_ONLY` (= ply-based) で vs tatara 2W-4L
/// (= 33%、 +120 Elo vs default) を確認。 ply は 戦型/速度依存で endgame node を
/// 取り逃す/取りすぎがある。 既存 EndgameDetector で position-based 判定すれば 精度
/// 高い endgame gate になり、 NPS と eval 精度の trade-off が改善されると仮定。
///
/// 注意: 探索側 (= USI go コマンド) が `set_current_phase_idx` を呼ばない限り override
/// は default 1 (= middle) のため、 本 flag を有効化したまま set_current_phase_idx を
/// 呼ばないと「常に skip」 状態になる。 USI 側で go コマンド入口に組込済。
static DUAL_CROSS_USE_ENDGAME_DETECTOR: AtomicBool = AtomicBool::new(false);
static DUAL_CROSS_USE_ENDGAME_DETECTOR_INIT: Once = Once::new();

thread_local! {
    /// non-PV node 中の dual_cross update / eval を skip する thread-local 制御フラグ。
    ///
    /// 探索側 (= `eval_helpers.rs` / `qsearch.rs` / `alpha_beta.rs`) が `nnue_evaluate`
    /// を呼ぶ直前に `pv_node` の **否定** を書き込み、 呼び出し後に false に戻す。
    /// このフラグが true の間、
    ///
    /// - `update_dual_cross_accumulator` は 差分計算を skip し prev → acc を memcpy
    /// - `evaluate_with_bucket` は `dual_cross_value` を 0 に
    ///
    /// する。 副作用として 非 PV chain 上で dual_cross accumulator は stale 化するが、
    /// arm 6 path I 設計で PV node の eval 精度を主軸に保ち non-PV では PSQT only
    /// 相当の高速 eval を狙う。 PV node 復帰時は引き続き stale な prev に基づくため
    /// 完全な数値復元は行わない (= 意図的 eval drift)。
    pub(crate) static NON_PV_DUAL_SKIP: Cell<bool> = const { Cell::new(false) };

    /// `LOG_KINGRING_VALUE` 計測用の thread-local request フラグ。
    ///
    /// USI `eval` コマンド (= `cmd_eval`) など、 1 SFEN につき 1 回だけ evaluate を
    /// 呼ぶ caller がこのフラグを ON にして evaluate 関数を呼び、 呼び出し終了後に
    /// OFF へ戻す。 `evaluate_with_bucket` は本フラグが ON のときだけ
    /// `info string KINGRING_VALUE ...` 行を stdout に書き出す。
    ///
    /// 探索 hot path (= 数百万回呼ばれる alpha-beta / qsearch evaluate) ではこの
    /// フラグは常に OFF のため、 1 thread_local load + branch のみで影響ゼロ。
    /// `LOG_KINGRING_VALUE=1` env var が立っていないときは API helper 自体が早期
    /// 返却して overhead 0。
    pub(crate) static LOG_KINGRING_REQUEST: Cell<bool> = const { Cell::new(false) };
}

/// `LOG_KINGRING_VALUE` env var (= "1" / "true" 受理) で global enable する flag。
///
/// 起動時 1 度だけ env を読んで `AtomicBool` に格納し、 以降は Relaxed load のみ。
/// 本 flag が OFF の場合、 [`log_kingring_value_enabled`] は常に false を返し
/// thread-local request も無視される (= overhead 0)。
static LOG_KINGRING_VALUE_ENABLED: AtomicBool = AtomicBool::new(false);
static LOG_KINGRING_VALUE_INIT: Once = Once::new();

static INIT: Once = Once::new();

/// 起動時 1 度だけ env var を読んで flag をセット
fn init() {
    INIT.call_once(|| {
        if let Ok(s) = std::env::var("PLAN_C_ABLATE") {
            for tok in s.split(',').map(|t| t.trim()) {
                match tok {
                    "" => {}
                    "psqt" => SKIP_PSQT_EVAL.store(true, Ordering::Relaxed),
                    "pin" => SKIP_PIN_EVAL.store(true, Ordering::Relaxed),
                    "threat" => SKIP_THREAT_UPDATE.store(true, Ordering::Relaxed),
                    "dual" => SKIP_DUAL_EVAL.store(true, Ordering::Relaxed),
                    "kingring" => SKIP_KINGRING_EVAL.store(true, Ordering::Relaxed),
                    "pairshape" => SKIP_PAIRSHAPE_EVAL.store(true, Ordering::Relaxed),
                    "all" => {
                        SKIP_PSQT_EVAL.store(true, Ordering::Relaxed);
                        SKIP_PIN_EVAL.store(true, Ordering::Relaxed);
                        SKIP_THREAT_UPDATE.store(true, Ordering::Relaxed);
                        SKIP_DUAL_EVAL.store(true, Ordering::Relaxed);
                        SKIP_KINGRING_EVAL.store(true, Ordering::Relaxed);
                        SKIP_PAIRSHAPE_EVAL.store(true, Ordering::Relaxed);
                    }
                    other => {
                        eprintln!(
                            "info string PLAN_C_ABLATE: unknown token '{other}' (accepts: psqt,pin,threat,dual,kingring,pairshape,all)"
                        );
                    }
                }
            }
            let active: Vec<&str> = [
                ("psqt", SKIP_PSQT_EVAL.load(Ordering::Relaxed)),
                ("pin", SKIP_PIN_EVAL.load(Ordering::Relaxed)),
                ("threat", SKIP_THREAT_UPDATE.load(Ordering::Relaxed)),
                ("dual", SKIP_DUAL_EVAL.load(Ordering::Relaxed)),
                ("kingring", SKIP_KINGRING_EVAL.load(Ordering::Relaxed)),
                ("pairshape", SKIP_PAIRSHAPE_EVAL.load(Ordering::Relaxed)),
            ]
            .iter()
            .filter_map(|(name, on)| if *on { Some(*name) } else { None })
            .collect();
            if !active.is_empty() {
                eprintln!("info string PLAN_C_ABLATE active: {}", active.join(","));
            }
        }
    });
}

/// PSQT 評価寄与を skip するか (eval 加算のみ、 accumulator update は維持)
#[inline]
pub fn skip_psqt_eval() -> bool {
    init();
    SKIP_PSQT_EVAL.load(Ordering::Relaxed)
}

/// Pin 評価寄与を skip するか (`pin_head_eval` 呼び出し全 skip)
#[inline]
pub fn skip_pin_eval() -> bool {
    init();
    SKIP_PIN_EVAL.load(Ordering::Relaxed)
}

/// Threat accumulator update を skip するか (refresh + incremental update 両方)
#[inline]
pub fn skip_threat_update() -> bool {
    init();
    SKIP_THREAT_UPDATE.load(Ordering::Relaxed)
}

/// Dual PSQT 評価寄与を skip するか
#[inline]
pub fn skip_dual_eval() -> bool {
    init();
    SKIP_DUAL_EVAL.load(Ordering::Relaxed)
}

/// KingRing PSQT scalar head の評価寄与 / accumulator 更新を skip するか
#[inline]
pub fn skip_kingring_eval() -> bool {
    init();
    SKIP_KINGRING_EVAL.load(Ordering::Relaxed)
}

/// Piece-Pair Shape PSQT scalar head の評価寄与 / accumulator 更新を skip するか
#[inline]
pub fn skip_pairshape_eval() -> bool {
    init();
    SKIP_PAIRSHAPE_EVAL.load(Ordering::Relaxed)
}

/// non-PV node 中の dual_cross skip フラグを取得する。
#[inline]
pub fn non_pv_dual_skip() -> bool {
    NON_PV_DUAL_SKIP.with(|c| c.get())
}

/// `DUAL_CROSS_UPDATE_INTERVAL` env var の現在値を返す (= 1..255、 default 1)。
///
/// 起動時 1 度だけ env を読んで `AtomicU8` に格納し、 以降は Relaxed load のみ。
/// 値が 0 または非数値の場合は 1 を採用する (= 旧挙動 fallback)。 255 を超える指定も
/// 飽和して 255 に clamp。
///
/// PV node の `update_dual_cross_accumulator` のみが本値を参照する。 N == 1 では
/// 完全に旧挙動と一致する (= 全 ply で delta update、 skip path に入らない)。
#[inline]
pub fn dual_cross_update_interval() -> u8 {
    DUAL_CROSS_INTERVAL_INIT.call_once(|| {
        let v = std::env::var("DUAL_CROSS_UPDATE_INTERVAL")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .map(|n| if n == 0 { 1 } else { n.min(255) as u8 })
            .unwrap_or(1);
        DUAL_CROSS_UPDATE_INTERVAL.store(v, Ordering::Relaxed);
        if v > 1 {
            eprintln!("info string DUAL_CROSS_UPDATE_INTERVAL={v} (every-N skip mode)");
        }
    });
    DUAL_CROSS_UPDATE_INTERVAL.load(Ordering::Relaxed)
}

/// `DUAL_CROSS_ENDGAME_ONLY` env var の現在値を返す (= default false)。
///
/// 起動時 1 度だけ env を読んで `AtomicBool` に格納し、 以降は Relaxed load のみ。
/// 受理値: `1` / `true` (大文字小文字無視) で有効化、 それ以外 (= 未設定含む) は無効。
///
/// true のとき、 `update_dual_cross_accumulator` と `evaluate_with_bucket` 内の
/// dual_cross 経路は ply 1..=100 で完全 skip され、 ply 101.. でのみ寄与する。
#[inline]
pub fn dual_cross_endgame_only() -> bool {
    DUAL_CROSS_ENDGAME_ONLY_INIT.call_once(|| {
        if let Ok(s) = std::env::var("DUAL_CROSS_ENDGAME_ONLY")
            && (s == "1" || s.eq_ignore_ascii_case("true"))
        {
            DUAL_CROSS_ENDGAME_ONLY.store(true, Ordering::Relaxed);
            eprintln!("info string DUAL_CROSS_ENDGAME_ONLY=1 (ply 101+ only)");
        }
    });
    DUAL_CROSS_ENDGAME_ONLY.load(Ordering::Relaxed)
}

/// `DUAL_CROSS_USE_ENDGAME_DETECTOR` env var の現在値を返す (= default false)。
///
/// 起動時 1 度だけ env を読んで `AtomicBool` に格納し、 以降は Relaxed load のみ。
/// 受理値: `1` / `true` (大文字小文字無視) で有効化、 それ以外 (= 未設定含む) は無効。
///
/// true のとき、 探索側 が `dual_cross::set_current_phase_idx` で thread-local に
/// 書き込んだ phase index (= 0=Opening / 1=Middle / 2=Endgame) を見て gate を判断する。
/// override == 2 (= Endgame) のとき dual_cross 経路は通常通り、 それ以外で skip。
#[inline]
pub fn dual_cross_use_endgame_detector() -> bool {
    DUAL_CROSS_USE_ENDGAME_DETECTOR_INIT.call_once(|| {
        if let Ok(s) = std::env::var("DUAL_CROSS_USE_ENDGAME_DETECTOR")
            && (s == "1" || s.eq_ignore_ascii_case("true"))
        {
            DUAL_CROSS_USE_ENDGAME_DETECTOR.store(true, Ordering::Relaxed);
            eprintln!("info string DUAL_CROSS_USE_ENDGAME_DETECTOR=1 (position-based phase gate)");
        }
    });
    DUAL_CROSS_USE_ENDGAME_DETECTOR.load(Ordering::Relaxed)
}

/// `LOG_KINGRING_VALUE` env var が起動時に有効化されたかを返す。
///
/// 起動時 1 度だけ env を読む。 値が `1` / `true` (大文字小文字無視) で有効化、
/// それ以外 (= 未設定含む) は無効。 無効時、 `evaluate_with_bucket` 内の
/// kingring 計測 log path は完全に dead branch で overhead 0。
#[inline]
pub fn log_kingring_value_enabled() -> bool {
    LOG_KINGRING_VALUE_INIT.call_once(|| {
        if let Ok(s) = std::env::var("LOG_KINGRING_VALUE")
            && (s == "1" || s.eq_ignore_ascii_case("true"))
        {
            LOG_KINGRING_VALUE_ENABLED.store(true, Ordering::Relaxed);
            eprintln!(
                "info string LOG_KINGRING_VALUE=1 (kingring scalar logging on USI 'eval' command)"
            );
        }
    });
    LOG_KINGRING_VALUE_ENABLED.load(Ordering::Relaxed)
}

/// 現 thread で kingring scalar log が要求されているか (=  USI `eval` 経路内か)。
///
/// `evaluate_with_bucket` がこの関数を呼んで true なら 1 行 `info string` を出力する。
/// `LOG_KINGRING_VALUE` env が無効の場合は早期 false (= overhead 1 atomic load + branch)。
#[inline]
pub fn log_kingring_request_active() -> bool {
    if !log_kingring_value_enabled() {
        return false;
    }
    LOG_KINGRING_REQUEST.with(|c| c.get())
}

/// USI `eval` コマンドが evaluate を呼ぶ前後で thread-local request flag を ON/OFF する guard。
///
/// `LOG_KINGRING_VALUE` env が無効でも生成可能 (= no-op、 overhead 1 thread_local touch)。
/// drop 時に前値へ復元するため reentrant 呼び出しに対応 (= 入れ子の eval でも log 1 行)。
pub struct LogKingringRequestGuard {
    prev: bool,
    was_enabled: bool,
}

impl LogKingringRequestGuard {
    /// log request を ON にして guard を返す。 drop で前値復元。
    ///
    /// env が無効なら no-op guard (= drop で何もしない)。
    #[inline]
    pub fn new() -> Self {
        let was_enabled = log_kingring_value_enabled();
        if !was_enabled {
            return Self {
                prev: false,
                was_enabled,
            };
        }
        let prev = LOG_KINGRING_REQUEST.with(|c| {
            let p = c.get();
            c.set(true);
            p
        });
        Self { prev, was_enabled }
    }
}

impl Default for LogKingringRequestGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LogKingringRequestGuard {
    #[inline]
    fn drop(&mut self) {
        if self.was_enabled {
            LOG_KINGRING_REQUEST.with(|c| c.set(self.prev));
        }
    }
}

/// 一時的に non-PV dual skip フラグを設定し、 drop で前値に戻す guard。
///
/// 例外安全に flag を巻き戻すため `Drop` で前値を復元する。 thread-local の
/// `Cell` を 1 read + 1 write のみ touch するため overhead 微小。
pub struct NonPvDualSkipGuard {
    prev: bool,
}

impl NonPvDualSkipGuard {
    /// `value` を thread-local に書き込み、 前値を guard に保存する。
    #[inline]
    pub fn new(value: bool) -> Self {
        let prev = NON_PV_DUAL_SKIP.with(|c| {
            let p = c.get();
            c.set(value);
            p
        });
        Self { prev }
    }
}

impl Drop for NonPvDualSkipGuard {
    #[inline]
    fn drop(&mut self) {
        NON_PV_DUAL_SKIP.with(|c| c.set(self.prev));
    }
}
