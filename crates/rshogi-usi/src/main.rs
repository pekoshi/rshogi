//! USIプロトコルエンジン
//!
//! 将棋GUIとの通信を行うUSIプロトコル実装。

use std::io::{self, BufRead, Write};
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use anyhow::{Result, anyhow};
use rshogi_core::eval::{
    DEFAULT_PASS_RIGHT_VALUE_EARLY, DEFAULT_PASS_RIGHT_VALUE_LATE, MaterialLevel, disable_material,
    is_material_enabled, set_eval_hash_enabled, set_material_level, set_pass_move_bonus,
    set_pass_right_value_phased,
};
use rshogi_core::nnue::dual_cross::{init_dual_cross_profile, print_dual_cross_profile};
use rshogi_core::nnue::{
    AccumulatorStackVariant, LayerStackBucketMode, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, clear_nnue,
    evaluate_dispatch, get_network, init_nnue, parse_layer_stack_bucket_mode,
    parse_nnue_architecture, print_nnue_stats, reset_layer_stack_progress_kpabs_weights,
    set_bucket_interp, set_fv_scale_override, set_layer_stack_bucket_mode,
    set_layer_stack_progress_kpabs_weights, set_nnue_architecture_override,
};
use rshogi_core::position::Position;
use rshogi_core::search::{
    DEFAULT_DRAW_VALUE_BLACK, DEFAULT_DRAW_VALUE_WHITE, LimitsType, Search, SearchInfo,
    SearchResult, SearchTuneParams,
};
use rshogi_core::types::{EnteringKingRule, Move};
use serde_json::json;

// ─── PUCT hybrid (puct-hybrid feature) ───
//
// 序中盤 PUCT + 終盤 αβ の hybrid 構成。 NNUE-B Transformer が未訓練のため
// stub evaluator (value=0、 すべての局面で引き分け扱い) で pipeline 整備のみを行う。
// 実 NNUE-B 統合は別 session (Phase B-3 NNUE-B Transformer 訓練後)。
#[cfg(feature = "puct-hybrid")]
use std::sync::atomic::AtomicI32;
#[cfg(feature = "puct-hybrid")]
use std::time::Instant;

#[cfg(feature = "puct-hybrid")]
use p2_nnue::evaluator::NnueEvaluator;
#[cfg(feature = "puct-hybrid")]
use p2_nnue::features::P2NnueInput;
#[cfg(feature = "puct-hybrid")]
use p2_search::puct::{PuctBackend, PuctLimits};
#[cfg(feature = "puct-hybrid")]
use p2_search::{PuctConfig, PuctEngine};

use p2_search::mate_probe::{MateResult, mate_probe_1ply, mate_probe_3ply};

/// PUCT 用 stub evaluator
///
/// NNUE-B Transformer が未訓練のため、 すべての局面で value=0 (引き分け) を返す。
/// PuctEngine は internally `value / VALUE_SCALE` で [-1.0, 1.0] winrate に変換するため
/// 0 → 0.0 で安定。 strong play は期待しない、 pipeline 動作確認のみ。
///
/// 実 NNUE-B 統合は別 session で `NnueEvaluator` を本物の Transformer 推論に置き換える。
#[cfg(feature = "puct-hybrid")]
struct StubPuctEvaluator;

#[cfg(feature = "puct-hybrid")]
impl NnueEvaluator for StubPuctEvaluator {
    fn evaluate(&self, _input: &P2NnueInput) -> i32 {
        // value=0 = winrate 0.0 (引き分け)。 一様な policy と組み合わせて UCB が機能する。
        0
    }

    fn backend_name(&self) -> &'static str {
        "stub-puct"
    }
}

/// エンジン名
const ENGINE_NAME: &str = "Shogi Engine";
/// エンジンバージョン
const ENGINE_VERSION: &str = "0.1.0";
/// エンジン作者
const ENGINE_AUTHOR: &str = "sh11235";
/// 探索スレッド用のスタックサイズ（SearchWorkerが大きいため増やす）
const SEARCH_STACK_SIZE: usize = 64 * 1024 * 1024;

fn load_progress_coeff_kpabs(path: &str) -> Result<Box<[f32]>, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read LS_PROGRESS_COEFF '{path}': {e}"))?;
    let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f64>();
    if bytes.len() != expected {
        return Err(format!(
            "progress.bin size mismatch: got {} bytes, expected {}",
            bytes.len(),
            expected
        ));
    }

    let weights: Vec<f32> = bytes
        .chunks_exact(size_of::<f64>())
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("chunk size is checked")) as f32)
        .collect();

    Ok(weights.into_boxed_slice())
}

/// USIエンジンの状態
struct UsiEngine {
    /// 探索エンジン
    search: Option<Search>,
    /// 現在の局面
    position: Position,
    /// 置換表サイズ（USI_Hashで変更）
    tt_size_mb: usize,
    /// 評価ハッシュサイズ（EvalHashで変更）
    eval_hash_size_mb: usize,
    /// EvalHash使用フラグ（UseEvalHashで変更）
    use_eval_hash: bool,
    /// MultiPV値
    multi_pv: usize,
    /// Skill Level オプション
    skill_options: rshogi_core::search::SkillOptions,
    /// 探索スレッドのハンドル
    search_thread: Option<thread::JoinHandle<(Search, SearchResult)>>,
    /// 探索停止用のフラグ（探索スレッドと共有）
    stop_flag: Option<Arc<AtomicBool>>,
    /// ponderhit通知フラグ
    ponderhit_flag: Option<Arc<AtomicBool>>,
    /// bestmove出力抑制フラグ（cmd_go内部でcmd_stopする際に使用）
    suppress_bestmove: Arc<AtomicBool>,
    /// Stochastic_Ponder オプションのミラー
    stochastic_ponder: bool,
    /// 直近の position コマンド文字列（Stochastic_Ponder の再始動用）
    last_position_cmd: Option<String>,
    /// 直近の go コマンド文字列（Stochastic_Ponder の再始動用）
    last_go_cmd: Option<String>,
    /// EvalFile の明示指定状態
    /// None: 未指定（eval/nn.bin 自動ロード対象）
    /// Some(true): 明示指定されロード成功
    /// Some(false): 明示指定されたがロード失敗
    eval_file_explicit: Option<bool>,
    /// 最後に指定された EvalFile パス（NNUE_ARCHITECTURE 変更時の再読込用）
    eval_file_path: Option<String>,
    /// SPSAParamsFile の明示指定パス（setoption で設定）
    spsa_params_file: Option<String>,
    /// SPSA params ファイルの読み込み済みフラグ
    spsa_params_loaded: bool,
    /// Large Pages使用メッセージの出力済みフラグ
    large_pages_reported: bool,
    // --- 有限パス権（Finite Pass Rights）関連 ---
    /// パス権ルール有効化フラグ
    pass_rights_enabled: bool,
    /// 初期パス権数（デフォルト2）
    initial_pass_count: u8,
    /// パス権評価値（序盤）
    pass_right_value_early: i32,
    /// パス権評価値（終盤）
    pass_right_value_late: i32,
    // --- PUCT hybrid foundation (B-1、 観測層のみ。 search backend は αβ のまま) ---
    /// 局面進行 phase 追跡 (序盤/中盤/終盤)
    phase_tracker: p2_search::PhaseTracker,
    /// 終盤検出器 (αβ vs PUCT のルーティング判定用、 PUCT 未統合時は info 出力のみ)
    endgame_detector: p2_search::EndgameDetector,
    // --- Midgame policy hook (αβ root move ordering を Transformer policy で補完) ---
    /// Midgame policy 設定 (USI option `UseMidgamePolicy` 等で更新)
    midgame_policy_config: p2_midgame_policy::MidgamePolicyConfig,
    /// Midgame policy model のパス (`MidgamePolicyModel`)。 `isready` で load 試行。
    midgame_policy_model_path: Option<String>,
    /// Midgame policy hook 本体。 `isready` で初期化、 evaluator 未設定なら `is_active()` で false。
    midgame_policy_hook: Option<p2_midgame_policy::MidgamePolicyHook>,
    // --- Mate engine midgame integration (αβ engine 用、 USI option で制御) ---
    /// Root mate search を有効化 (= cmd_go 冒頭で自詰み 3 ply 検索)
    use_root_mate_search: bool,
    /// Opponent mate check を有効化 (= 各 root move を do_move + mate_probe_1ply で頓死検知)
    use_opponent_mate_check: bool,
    /// Opponent mate check で試行する root move の上限 (= legal の最初 N moves のみ check、 overhead 抑制)
    opponent_mate_max_moves: usize,
    // --- MateSolverThread (= 別 thread dfpn、 main search 完全不変) ---
    /// MateSolverThread を有効化 (= Threads >= 2 時のみ実起動)
    use_mate_solver_thread: bool,
    /// 現在の Threads option 値 (= MateSolverThread の gate 用)
    num_threads: usize,
    /// 現在走行中の MateSolverThread (= cmd_go で start、 cmd_stop で stop)
    mate_solver_thread: Option<p2_search::mate_solver_thread::MateSolverThread>,
    // --- PUCT hybrid engine (puct-hybrid feature、 default OFF) ---
    /// PUCT hybrid 経路を使う (false なら既存 αβ 経路のみ)
    #[cfg(feature = "puct-hybrid")]
    use_puct: bool,
    /// 終盤切替閾値 (endgame_score がこの cp 換算値以上なら αβ 経路、 未満なら PUCT)
    /// 既存 EndgameDetector の `use_alpha_beta` 判定とは独立した USI 公開値。
    /// 実装的には `endgame_score >= threshold/1000.0` で αβ 側にルート。
    #[cfg(feature = "puct-hybrid")]
    puct_endgame_threshold: i32,
    /// NNUE-B ONNX model path (未訓練のため現状は受け取るだけ、 stub evaluator を使う)
    #[cfg(feature = "puct-hybrid")]
    puct_network_path: Option<String>,
    /// 現在走行中の PUCT 探索の停止 flag (cmd_stop で立てる)
    #[cfg(feature = "puct-hybrid")]
    puct_stop_flag: Option<Arc<AtomicBool>>,
    /// 現在走行中の PUCT 探索スレッド
    #[cfg(feature = "puct-hybrid")]
    puct_thread: Option<thread::JoinHandle<()>>,
    /// PUCT 経路用 bestmove score (info 表示用、 cp 換算)
    #[cfg(feature = "puct-hybrid")]
    puct_last_score_cp: Arc<AtomicI32>,
}

impl UsiEngine {
    /// 新しいUSIエンジンを作成
    fn new() -> Self {
        let tt_size_mb = 256;
        let eval_hash_size_mb = 256;
        let use_eval_hash = true;

        // グローバルフラグをデフォルト値で初期化
        // （USI GUIがsetoptionを送らない場合に備える）
        set_eval_hash_enabled(use_eval_hash);

        Self {
            // EvalHash は最初の `go` 直前まで遅延確保する。
            // selfplay のように起動直後に setoption でサイズを下げるケースで、
            // 先に既定 256MB を確保してしまう無駄を避ける。
            search: Some(Search::new_with_eval_hash(tt_size_mb, 0)),
            position: Position::new(),
            tt_size_mb,
            eval_hash_size_mb,
            use_eval_hash,
            multi_pv: 1,
            skill_options: rshogi_core::search::SkillOptions::default(),
            search_thread: None,
            stop_flag: None,
            ponderhit_flag: None,
            suppress_bestmove: Arc::new(AtomicBool::new(false)),
            stochastic_ponder: false,
            last_position_cmd: None,
            last_go_cmd: None,
            eval_file_explicit: None,
            eval_file_path: None,
            spsa_params_file: None,
            spsa_params_loaded: false,
            large_pages_reported: false,
            pass_rights_enabled: false,
            initial_pass_count: 2,
            pass_right_value_early: DEFAULT_PASS_RIGHT_VALUE_EARLY,
            pass_right_value_late: DEFAULT_PASS_RIGHT_VALUE_LATE,
            phase_tracker: p2_search::PhaseTracker::new(),
            endgame_detector: p2_search::EndgameDetector::new(),
            midgame_policy_config: p2_midgame_policy::MidgamePolicyConfig::default(),
            midgame_policy_model_path: None,
            midgame_policy_hook: None,
            use_root_mate_search: true,
            use_opponent_mate_check: true,
            opponent_mate_max_moves: 16,
            use_mate_solver_thread: true,
            num_threads: 1,
            mate_solver_thread: None,
            #[cfg(feature = "puct-hybrid")]
            use_puct: false,
            #[cfg(feature = "puct-hybrid")]
            puct_endgame_threshold: 800,
            #[cfg(feature = "puct-hybrid")]
            puct_network_path: None,
            #[cfg(feature = "puct-hybrid")]
            puct_stop_flag: None,
            #[cfg(feature = "puct-hybrid")]
            puct_thread: None,
            #[cfg(feature = "puct-hybrid")]
            puct_last_score_cp: Arc::new(AtomicI32::new(0)),
        }
    }

    /// USIコマンドを処理
    fn process_command(&mut self, line: &str) -> Result<bool> {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            return Ok(true);
        }

        match tokens[0] {
            "usi" => {
                self.cmd_usi();
            }
            "isready" => {
                self.cmd_isready();
            }
            "setoption" => {
                self.cmd_setoption(&tokens);
            }
            "usinewgame" => {
                self.cmd_usinewgame();
            }
            "position" => {
                self.last_position_cmd = Some(line.to_string());
                self.cmd_position(&tokens);
            }
            "go" => {
                self.last_go_cmd = Some(line.to_string());
                self.cmd_go(&tokens);
            }
            "stop" => {
                self.cmd_stop();
            }
            "ponderhit" => {
                self.cmd_ponderhit();
            }
            "quit" => {
                self.cmd_stop();
                // NNUE統計を出力（nnue-stats feature有効時のみ実際に出力）
                print_nnue_stats();
                // dual_cross profiling: DUAL_CROSS_PROFILE=1 時のみ stderr に出力
                print_dual_cross_profile();
                return Ok(false);
            }
            "gameover" => {
                self.cmd_stop();
            }
            // デバッグ用コマンド
            "d" | "display" => {
                self.cmd_display();
            }
            "eval" => {
                let diagnostics = tokens.get(1).is_some_and(|s| *s == "diag");
                self.cmd_eval(diagnostics);
            }
            _ => {
                // 未知のコマンドは無視
            }
        }

        Ok(true)
    }

    /// usiコマンド: エンジン情報を出力
    fn cmd_usi(&self) {
        println!("id name {ENGINE_NAME} {ENGINE_VERSION}");
        println!("id author {ENGINE_AUTHOR}");
        println!();
        // オプション（将来的に追加）
        println!("option name USI_Hash type spin default 256 min 1 max 4096");
        println!("option name Threads type spin default 1 min 1 max 512");
        println!("option name USI_Ponder type check default false");
        println!("option name Stochastic_Ponder type check default false");
        println!("option name MultiPV type spin default 1 min 1 max 500");
        println!("option name NetworkDelay type spin default 120 min 0 max 10000");
        println!("option name NetworkDelay2 type spin default 1120 min 0 max 10000");
        println!("option name MinimumThinkingTime type spin default 2000 min 1000 max 100000");
        println!("option name SlowMover type spin default 100 min 1 max 1000");
        println!("option name MaxMovesToDraw type spin default 100000 min 0 max 100000");
        println!(
            "option name DrawValueBlack type spin default {DEFAULT_DRAW_VALUE_BLACK} min -30000 max 30000"
        );
        println!(
            "option name DrawValueWhite type spin default {DEFAULT_DRAW_VALUE_WHITE} min -30000 max 30000"
        );
        println!("option name DefenseMateProbeNodes type spin default 0 min 0 max 10000000");
        println!("option name EvalHash type spin default 256 min 0 max 4096");
        println!("option name UseEvalHash type check default true");
        println!("option name Skill Level type spin default 20 min 0 max 20");
        println!("option name UCI_LimitStrength type check default false");
        println!("option name UCI_Elo type spin default 0 min 0 max 4000");
        println!(
            "option name MaterialLevel type combo default none var none var 1 var 2 var 3 var 4 var 7 var 8 var 9"
        );
        println!("option name EvalFile type string default eval/nn.bin");
        println!(
            "option name EnteringKingRule type combo default CSARule27 var NoEnteringKing var CSARule24 var CSARule24H var CSARule27 var CSARule27H var TryRule"
        );
        // NNUEEvalScale option は v16 (shogi2344) 系 専用だったため 2026-05-12 削除。
        // LayerStacks 系の eval scale 調整は FV_SCALE option を使用。
        // FV_SCALE: 0=自動判定 (ckpt 値、 LayerStacks=24)、 1以上=指定値でオーバーライド。
        // **default = 16** (Suisho5 / YaneuraOu HalfKP 同 scale で cp 表示を揃える、 2026-05-23)。
        println!("option name FV_SCALE type spin default 16 min 0 max 100");
        #[cfg(feature = "nnue-attack-relay-v2")]
        println!("option name ARH_V2_OUTPUT_SCALE type spin default 4 min 0 max 256");
        #[cfg(feature = "nnue-scalar-threat")]
        println!("option name SCALAR_THREAT_OUTPUT_DIVISOR type spin default 8 min 1 max 256");
        #[cfg(feature = "nnue-king-danger-psqt")]
        println!("option name KING_DANGER_OUTPUT_SCALE type spin default 4 min 0 max 256");
        #[cfg(feature = "nnue-king-danger-psqt")]
        println!("option name KING_DANGER_ANALYTIC_COEF type spin default 0 min 0 max 200");
        #[cfg(feature = "nnue-king-danger-psqt")]
        println!("option name KING_DANGER_SLIDER_ANALYTIC_COEF type spin default 0 min 0 max 200");
        #[cfg(feature = "nnue-king-danger-psqt")]
        println!(
            "option name KING_DANGER_SLIDER_DEF_ANALYTIC_COEF type spin default 0 min 0 max 200"
        );
        #[cfg(feature = "nnue-attack-dev-psqt")]
        println!("option name ATTACK_DEV_OUTPUT_SCALE type spin default 4 min 0 max 256");
        #[cfg(feature = "nnue-slider-xray-psqt")]
        println!("option name SLIDER_XRAY_OUTPUT_SCALE type spin default 4 min 0 max 256");
        #[cfg(feature = "nnue-kzt-psqt")]
        println!("option name KZT_OUTPUT_SCALE type spin default 4 min 0 max 256");
        #[cfg(feature = "nnue-kzt-psqt")]
        println!("option name KZT_CLAMP_RAW type spin default 0 min 0 max 8192");
        #[cfg(feature = "nnue-odh")]
        println!("option name ODH_OUTPUT_SCALE type spin default 0 min 0 max 256");
        #[cfg(feature = "nnue-odh")]
        println!("option name ODH_THRESHOLD type spin default 100 min 0 max 3000");
        #[cfg(feature = "nnue-odh")]
        println!("option name ODH_CAP type spin default 800 min 0 max 30000");
        #[cfg(feature = "nnue-eval-bias-corr")]
        println!("option name EVAL_BIAS_CORR_SCALE type spin default 0 min 0 max 256");
        #[cfg(feature = "nnue-eval-bias-corr")]
        println!("option name EVAL_BIAS_CORR_CLAMP type spin default 300 min 0 max 30000");
        #[cfg(feature = "nnue-eval-bias-corr")]
        println!("option name EVAL_BIAS_CORR_FILE type string default <empty>");
        #[cfg(feature = "nnue-king-mobility-psqt")]
        println!("option name KING_MOBILITY_OUTPUT_SCALE type spin default 4 min 0 max 256");
        #[cfg(feature = "nnue-drop-pressure-psqt")]
        println!("option name DROP_PRESSURE_OUTPUT_SCALE type spin default 4 min 0 max 256");
        println!(
            "option name LS_BUCKET_MODE type combo default {} var progress8kpabs",
            LayerStackBucketMode::Progress8KPAbs.as_str()
        );
        println!("option name LS_PROGRESS_COEFF type string default <empty>");
        println!(
            "option name LS_BUCKET_EVAL_SCALE type string default 100,100,100,100,100,100,100,100"
        );
        for i in 0..8 {
            println!("option name LS_BUCKET_EVAL_SCALE_{i} type spin default 100 min 50 max 200");
        }
        println!("option name LS_BUCKET_INTERP type spin default 0 min 0 max 40");
        println!(
            "option name NNUE_ARCHITECTURE type combo default auto var auto var halfkp var halfka_hm var halfka var layerstacks var layerstacks-psqt"
        );
        // 有限パス権（Finite Pass Rights）オプション
        println!("option name PassRights type check default false");
        println!("option name InitialPassCount type spin default 2 min 0 max 10");
        println!("option name PassMoveBonus type spin default 0 min -1000 max 1000");
        println!(
            "option name PassRightValueEarly type spin default {DEFAULT_PASS_RIGHT_VALUE_EARLY} min 0 max 500"
        );
        println!(
            "option name PassRightValueLate type spin default {DEFAULT_PASS_RIGHT_VALUE_LATE} min 0 max 500"
        );
        println!("option name SPSAParamsFile type string default <auto>");
        // --- Phase B-1 history coefficient sweep (2026-06-05) ---
        println!("option name HistChCoeff type spin default 1 min 0 max 16");
        println!("option name HistMainCoeff type spin default 2 min 0 max 16");
        println!("option name HistPawnCoeff type spin default 2 min 0 max 16");
        println!("option name HistLpCoeff type spin default 8 min 0 max 32");
        // --- 敵玉 drop move-ordering bonus mode (2026-06-16, 害診断後の狭い版切替) ---
        // 0=off / 1=line-only / 2=danger-gated(default) / 3=full(旧dropmo)
        println!("option name KingDropBonusMode type spin default 2 min 0 max 3");
        // mode2 の danger gate 閾値 (= 自軍駒が敵玉5x5内に何枚集まれば zone bonus 有効か)
        println!("option name KingDropDangerMin type spin default 2 min 1 max 8");
        // --- Midgame policy (αβ root ordering を Transformer policy で補完、 default OFF) ---
        println!("option name UseMidgamePolicy type check default false");
        println!("option name MidgamePolicyMinDepth type spin default 8 min 0 max 64");
        println!("option name MidgamePolicyTopK type spin default 4 min 0 max 32");
        println!("option name MidgamePolicyAlpha type spin default 32 min 0 max 1024");
        println!("option name MidgamePolicyCacheEntries type spin default 1024 min 16 max 65536");
        println!("option name MidgamePolicyModel type string default <empty>");
        for spec in SearchTuneParams::option_specs() {
            println!(
                "option name {} type spin default {} min {} max {}",
                spec.usi_name, spec.default, spec.min, spec.max
            );
        }
        // --- PUCT hybrid engine (puct-hybrid feature 有効時のみ公開) ---
        println!("option name UseRootMateSearch type check default true");
        println!("option name UseOpponentMateCheck type check default true");
        println!("option name OpponentMateMaxMoves type spin default 16 min 1 max 256");
        // MateSolverThread (= 別 thread dfpn、 Threads >= 2 時のみ実起動、 main search 完全不変)
        println!("option name UseMateSolverThread type check default true");
        #[cfg(feature = "puct-hybrid")]
        {
            println!("option name UsePuct type check default false");
            println!("option name PuctEndgameThreshold type spin default 800 min 0 max 5000");
            println!("option name PuctNetworkPath type string default <empty>");
        }
        println!("usiok");
    }

    /// isreadyコマンド: 準備完了を通知
    /// YaneuraOu準拠: isready 受信時にTTをクリアする
    fn cmd_isready(&mut self) {
        if let Some(search) = self.search.as_mut() {
            search.clear_tt();
        }
        // EvalFile の状態を確認し、必要なら NNUE をロード
        match self.eval_file_explicit {
            Some(false) => {
                // EvalFile が明示指定されたがロード失敗 → 致命的エラー
                // eval/nn.bin への暗黙フォールバックはしない
                panic!(
                    "EvalFile was explicitly set but failed to load. \
                     Fix the path or remove the setoption."
                );
            }
            Some(true) => {
                // EvalFile が明示指定されロード成功 → 何もしない
            }
            None if !is_material_enabled() && get_network().is_none() => {
                // EvalFile 未指定 + Material 未指定 + NNUE 未ロード → eval/nn.bin を自動ロード
                const DEFAULT_EVAL_FILE: &str = "eval/nn.bin";
                if std::path::Path::new(DEFAULT_EVAL_FILE).exists() {
                    match init_nnue(DEFAULT_EVAL_FILE) {
                        Ok(()) => {
                            let payload = json!({
                                "type": "info",
                                "message": format!("NNUE auto-loaded: {DEFAULT_EVAL_FILE}"),
                            });
                            eprintln!("info string {payload}");
                        }
                        Err(e) => {
                            panic!("Failed to load default NNUE file {DEFAULT_EVAL_FILE}: {e}");
                        }
                    }
                } else {
                    panic!(
                        "No NNUE file loaded and {DEFAULT_EVAL_FILE} not found. \
                         Use 'setoption name EvalFile value <path>' or \
                         'setoption name MaterialLevel value <n>'."
                    );
                }
            }
            None => {
                // EvalFile 未指定だが Material 有効 or NNUE 既ロード → 何もしない
            }
        }
        // progress8kpabs モードで LS_PROGRESS_COEFF 未指定の場合はエラー
        // NNUE が未ロード（Material 評価）または LayerStacks 以外のアーキ（v16 等）はスキップ
        if let Some(net) = get_network() {
            use rshogi_core::nnue::{
                LayerStackBucketMode, get_layer_stack_bucket_mode,
                get_layer_stack_progress_kpabs_weights,
            };
            if net.is_layer_stacks()
                && get_layer_stack_bucket_mode() == LayerStackBucketMode::Progress8KPAbs
                && get_layer_stack_progress_kpabs_weights().iter().all(|&w| w == 0.0)
            {
                panic!(
                    "LS_BUCKET_MODE=progress8kpabs requires LS_PROGRESS_COEFF to be set. \
                     All weights are zero (progress.bin not loaded). \
                     Use 'setoption name LS_PROGRESS_COEFF value <path>' before isready."
                );
            }
        }
        self.maybe_load_spsa_params();
        self.maybe_report_large_pages();
        println!("readyok");
    }

    /// SPSA params ファイルの自動/明示読み込み。
    /// 優先順位: 1. SPSAParamsFile で明示指定 2. バイナリ同ディレクトリの spsa.params 3. なし
    fn maybe_load_spsa_params(&mut self) {
        if self.spsa_params_loaded {
            return;
        }
        self.spsa_params_loaded = true;

        let path = if let Some(ref explicit) = self.spsa_params_file {
            std::path::PathBuf::from(explicit)
        } else {
            // バイナリと同じディレクトリの spsa.params を探す
            let exe_dir =
                std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()));
            match exe_dir {
                Some(dir) => dir.join("spsa.params"),
                None => return,
            }
        };

        if !path.exists() {
            if self.spsa_params_file.is_some() {
                eprintln!("info string Warning: SPSAParamsFile not found: {}", path.display());
            }
            return;
        }

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("info string Warning: failed to read {}: {e}", path.display());
                return;
            }
        };

        let mut applied = 0usize;
        let mut clamped = 0usize;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            // .params format: name,type,value,min,max,c_end,r_end [// comment] [[[NOT USED]]]
            let val_part = trimmed.split("//").next().unwrap_or(trimmed);
            let val_part = val_part.replace("[[NOT USED]]", "");
            let cols: Vec<&str> = val_part.split(',').map(str::trim).collect();
            if cols.len() < 3 {
                continue;
            }
            let name = cols[0];
            let type_name = cols[1];
            let value_str = cols[2];

            let parsed = if type_name.eq_ignore_ascii_case("int") {
                match value_str.parse::<f64>() {
                    Ok(v) => v.round() as i32,
                    Err(_) => continue,
                }
            } else {
                match value_str.parse::<i32>() {
                    Ok(v) => v,
                    Err(_) => continue,
                }
            };

            if let Some(search) = self.search.as_mut()
                && let Some(result) = search.set_search_tune_option(name, parsed)
            {
                applied += 1;
                if result.clamped {
                    clamped += 1;
                }
            }
        }

        if applied > 0 {
            eprintln!(
                "info string SPSA params loaded: {} parameters from {} (clamped: {})",
                applied,
                path.display(),
                clamped
            );
        }
    }

    fn maybe_report_large_pages(&mut self) {
        if self.large_pages_reported {
            return;
        }

        let Some(search) = self.search.as_ref() else {
            return;
        };
        if !search.tt_uses_large_pages() {
            return;
        }

        // Windows: VirtualAlloc with MEM_LARGE_PAGES
        // Linux: madvise(MADV_HUGEPAGE) によるhugepageヒント
        let payload = json!({
            "type": "info",
            "message": "Large Pages are used.",
        });
        println!("info string {}", payload);
        self.large_pages_reported = true;
    }

    /// setoptionコマンド: オプション設定
    fn cmd_setoption(&mut self, tokens: &[&str]) {
        // 探索中の設定変更は避ける
        self.wait_for_search();

        // setoption name <name> value <value>
        let mut name = String::new();
        let mut value = String::new();
        let mut parsing_name = false;
        let mut parsing_value = false;

        for token in tokens.iter().skip(1) {
            match *token {
                "name" => {
                    parsing_name = true;
                    parsing_value = false;
                }
                "value" => {
                    parsing_name = false;
                    parsing_value = true;
                }
                _ => {
                    if parsing_name {
                        if !name.is_empty() {
                            name.push(' ');
                        }
                        name.push_str(token);
                    } else if parsing_value {
                        if !value.is_empty() {
                            value.push(' ');
                        }
                        value.push_str(token);
                    }
                }
            }
        }

        // オプションを適用
        if name.starts_with("SPSA_") {
            let parsed = match value.parse::<i32>() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("info string Warning: invalid SPSA value '{}'", value);
                    return;
                }
            };
            if let Some(search) = self.search.as_mut()
                && let Some(result) = search.set_search_tune_option(name.as_str(), parsed)
            {
                if result.clamped {
                    eprintln!(
                        "info string Warning: {}={} is out of range, clamped to {} ({}..{})",
                        name, parsed, result.applied, result.min, result.max
                    );
                }
                return;
            }
        }

        match name.as_str() {
            "SPSAParamsFile" => {
                if value == "<auto>" || value == "<empty>" || value.is_empty() {
                    self.spsa_params_file = None;
                } else {
                    self.spsa_params_file = Some(value.to_string());
                }
                // 明示指定時は再読み込みを強制
                self.spsa_params_loaded = false;
            }
            "USI_Hash" => {
                if let Ok(size) = value.parse::<usize>() {
                    if let Some(search) = self.search.as_mut() {
                        search.resize_tt(size);
                        self.tt_size_mb = size;
                    }
                    self.maybe_report_large_pages();
                }
            }
            "Threads" => {
                if let Ok(num) = value.parse::<usize>() {
                    self.num_threads = num.max(1);
                    if let Some(search) = self.search.as_mut() {
                        search.set_num_threads(num);
                    }
                }
            }
            "NetworkDelay" => {
                if let Ok(v) = value.parse::<i64>()
                    && let Some(search) = self.search.as_mut()
                {
                    let mut opts = search.time_options();
                    opts.network_delay = v;
                    search.set_time_options(opts);
                }
            }
            "NetworkDelay2" => {
                if let Ok(v) = value.parse::<i64>()
                    && let Some(search) = self.search.as_mut()
                {
                    let mut opts = search.time_options();
                    opts.network_delay2 = v;
                    search.set_time_options(opts);
                }
            }
            "MinimumThinkingTime" => {
                if let Ok(v) = value.parse::<i64>()
                    && let Some(search) = self.search.as_mut()
                {
                    let mut opts = search.time_options();
                    opts.minimum_thinking_time = v;
                    search.set_time_options(opts);
                }
            }
            "SlowMover" => {
                if let Ok(v) = value.parse::<i32>()
                    && let Some(search) = self.search.as_mut()
                {
                    let mut opts = search.time_options();
                    opts.slow_mover = v;
                    search.set_time_options(opts);
                }
            }
            "USI_Ponder" => {
                if let Ok(v) = value.parse::<bool>()
                    && let Some(search) = self.search.as_mut()
                {
                    let mut opts = search.time_options();
                    opts.usi_ponder = v;
                    search.set_time_options(opts);
                }
            }
            "Stochastic_Ponder" => {
                if let Ok(v) = value.parse::<bool>() {
                    self.stochastic_ponder = v;
                    if let Some(search) = self.search.as_mut() {
                        let mut opts = search.time_options();
                        opts.stochastic_ponder = v;
                        search.set_time_options(opts);
                    }
                }
            }
            "Skill Level" => {
                if let Ok(v) = value.parse::<i32>()
                    && let Some(search) = self.search.as_mut()
                {
                    let mut opts = self.skill_options;
                    opts.skill_level = v.clamp(0, 20);
                    self.skill_options = opts;
                    search.set_skill_options(opts);
                }
            }
            "UCI_LimitStrength" => {
                if let Ok(v) = value.parse::<bool>()
                    && let Some(search) = self.search.as_mut()
                {
                    let mut opts = self.skill_options;
                    opts.uci_limit_strength = v;
                    self.skill_options = opts;
                    search.set_skill_options(opts);
                }
            }
            "UCI_Elo" => {
                if let Ok(v) = value.parse::<i32>()
                    && let Some(search) = self.search.as_mut()
                {
                    let mut opts = self.skill_options;
                    opts.uci_elo = v;
                    self.skill_options = opts;
                    search.set_skill_options(opts);
                }
            }
            "EvalHash" => {
                if let Ok(size) = value.parse::<usize>()
                    && let Some(search) = self.search.as_mut()
                {
                    search.resize_eval_hash(size);
                    self.eval_hash_size_mb = size;
                }
            }
            "UseEvalHash" => {
                let v = value == "true" || value == "1";
                self.use_eval_hash = v;
                set_eval_hash_enabled(v);
            }
            "MaxMovesToDraw" => {
                if let Ok(v) = value.parse::<i32>()
                    && let Some(search) = self.search.as_mut()
                {
                    search.set_max_moves_to_draw(v);
                }
            }
            "DrawValueBlack" => {
                if let Ok(v) = value.parse::<i32>()
                    && let Some(search) = self.search.as_mut()
                {
                    search.set_draw_value_black(v);
                }
            }
            "DrawValueWhite" => {
                if let Ok(v) = value.parse::<i32>()
                    && let Some(search) = self.search.as_mut()
                {
                    search.set_draw_value_white(v);
                }
            }
            "DefenseMateProbeNodes" => {
                if let Ok(v) = value.parse::<u32>()
                    && let Some(search) = self.search.as_mut()
                {
                    search.set_defense_mate_probe_nodes(v);
                }
            }
            "LS_BUCKET_EVAL_SCALE" => {
                // カンマ区切り8整数 (%)、各要素 1..=400 にclamp、不正入力は無視してdefault維持
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() == 8 {
                    let parsed: Vec<Option<i32>> =
                        parts.iter().map(|s| s.trim().parse::<i32>().ok()).collect();
                    if parsed.iter().all(|v| v.is_some()) {
                        let mut scale = [100i32; 8];
                        for (i, v) in parsed.iter().enumerate() {
                            scale[i] = v.unwrap().clamp(1, 400);
                        }
                        if let Some(search) = self.search.as_mut() {
                            search.set_bucket_eval_scale(scale);
                        }
                    } else {
                        eprintln!(
                            "info string Warning: LS_BUCKET_EVAL_SCALE parse error for '{value}', ignored"
                        );
                    }
                } else {
                    eprintln!(
                        "info string Warning: LS_BUCKET_EVAL_SCALE requires 8 comma-separated values, got '{value}', ignored"
                    );
                }
            }
            name if name.starts_with("LS_BUCKET_EVAL_SCALE_")
                && name
                    .strip_prefix("LS_BUCKET_EVAL_SCALE_")
                    .and_then(|s| s.parse::<usize>().ok())
                    .is_some_and(|idx| idx < 8) =>
            {
                // SPSA tune 用 per-bucket eval scale (%)、各 index 単独設定。
                let idx = name
                    .strip_prefix("LS_BUCKET_EVAL_SCALE_")
                    .and_then(|s| s.parse::<usize>().ok())
                    .expect("guard checked");
                match value.trim().parse::<i32>() {
                    Ok(v) => {
                        if let Some(search) = self.search.as_mut() {
                            search.set_bucket_eval_scale_index(idx, v);
                        }
                    }
                    Err(_) => {
                        eprintln!("info string Warning: {name} parse error for '{value}', ignored");
                    }
                }
            }
            "LS_BUCKET_INTERP" => {
                // bucket 境界補間の遷移ゾーン幅 ε (%)、0..=40 にclamp、0 = OFF (bit-exact無変化)。
                // global atomic を直接更新する (= FV_SCALE と同様、search 未生成でも反映)。
                match value.trim().parse::<i32>() {
                    Ok(v) => {
                        set_bucket_interp(v.clamp(0, 40));
                        eprintln!("info string LS_BUCKET_INTERP: {}", v.clamp(0, 40));
                    }
                    Err(_) => {
                        eprintln!(
                            "info string Warning: LS_BUCKET_INTERP parse error for '{value}', ignored"
                        );
                    }
                }
            }
            "MultiPV" => {
                if let Ok(v) = value.parse::<usize>() {
                    self.multi_pv = v;
                }
            }
            "MaterialLevel" => {
                if value == "none" {
                    disable_material();
                } else if let Ok(v) = value.parse::<u8>() {
                    if let Some(level) = MaterialLevel::from_value(v) {
                        set_material_level(level);
                    } else {
                        eprintln!("info string Warning: Invalid MaterialLevel value {v}, ignored");
                    }
                } else {
                    eprintln!("info string Warning: MaterialLevel parse error for '{value}'");
                }
            }
            "EnteringKingRule" => {
                if let Some(rule) = EnteringKingRule::from_usi(&value) {
                    // search は new() で常に Some だが、既存パターンに合わせて防御的にチェック
                    if let Some(search) = self.search.as_mut() {
                        search.set_entering_king_rule(rule);
                    }
                } else {
                    eprintln!("info string Warning: unknown EnteringKingRule '{value}'");
                }
            }
            "EvalFile" => {
                if value.is_empty() || value == "<empty>" {
                    // 空 → 明示指定を解除し isready の自動ロードに戻す
                    clear_nnue();
                    self.eval_file_explicit = None;
                    self.eval_file_path = None;
                } else {
                    // パス指定: ロード試行し、結果を記録
                    self.eval_file_path = Some(value.to_string());
                    match init_nnue(&value) {
                        Ok(()) => {
                            self.eval_file_explicit = Some(true);
                            let payload = json!({
                                "type": "info",
                                "message": format!("NNUE loaded: {value}"),
                            });
                            eprintln!("info string {payload}");
                        }
                        Err(e) => {
                            self.eval_file_explicit = Some(false);
                            eprintln!("info string Error loading NNUE file: {e}");
                        }
                    }
                }
            }
            "FV_SCALE" => {
                if let Ok(v) = value.parse::<i32>() {
                    set_fv_scale_override(v);
                    if v == 0 {
                        eprintln!("info string FV_SCALE: auto-detect");
                    } else {
                        eprintln!("info string FV_SCALE: {v}");
                    }
                }
            }
            #[cfg(feature = "nnue-attack-relay-v2")]
            "ARH_V2_OUTPUT_SCALE" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::attack_relay_v2::set_output_scale(v);
                    eprintln!("info string ARH_V2_OUTPUT_SCALE: {v}");
                }
            }
            #[cfg(feature = "nnue-scalar-threat")]
            "SCALAR_THREAT_OUTPUT_DIVISOR" => {
                if let Ok(v) = value.parse::<i32>() {
                    let v = v.clamp(1, 256);
                    rshogi_core::nnue::scalar_threat::set_output_divisor(v);
                    eprintln!("info string SCALAR_THREAT_OUTPUT_DIVISOR: {v}");
                }
            }
            #[cfg(feature = "nnue-king-danger-psqt")]
            "KING_DANGER_OUTPUT_SCALE" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::king_danger::set_output_scale(v);
                    eprintln!("info string KING_DANGER_OUTPUT_SCALE: {v}");
                }
            }
            #[cfg(feature = "nnue-king-danger-psqt")]
            "KING_DANGER_ANALYTIC_COEF" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::king_danger::set_analytic_coef(v);
                    eprintln!("info string KING_DANGER_ANALYTIC_COEF: {v}");
                }
            }
            #[cfg(feature = "nnue-king-danger-psqt")]
            "KING_DANGER_SLIDER_ANALYTIC_COEF" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::king_danger::set_slider_analytic_coef(v);
                    eprintln!("info string KING_DANGER_SLIDER_ANALYTIC_COEF: {v}");
                }
            }
            #[cfg(feature = "nnue-king-danger-psqt")]
            "KING_DANGER_SLIDER_DEF_ANALYTIC_COEF" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::king_danger::set_slider_def_analytic_coef(v);
                    eprintln!("info string KING_DANGER_SLIDER_DEF_ANALYTIC_COEF: {v}");
                }
            }
            #[cfg(feature = "nnue-attack-dev-psqt")]
            "ATTACK_DEV_OUTPUT_SCALE" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::attack_dev::set_output_scale(v);
                    eprintln!("info string ATTACK_DEV_OUTPUT_SCALE: {v}");
                }
            }
            #[cfg(feature = "nnue-slider-xray-psqt")]
            "SLIDER_XRAY_OUTPUT_SCALE" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::slider_xray::set_output_scale(v);
                    eprintln!("info string SLIDER_XRAY_OUTPUT_SCALE: {v}");
                }
            }
            #[cfg(feature = "nnue-kzt-psqt")]
            "KZT_OUTPUT_SCALE" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::kzt::set_output_scale(v);
                    eprintln!("info string KZT_OUTPUT_SCALE: {v}");
                }
            }
            #[cfg(feature = "nnue-odh")]
            "ODH_OUTPUT_SCALE" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::odh::set_output_scale(v);
                    eprintln!("info string ODH_OUTPUT_SCALE: {v}");
                }
            }
            #[cfg(feature = "nnue-odh")]
            "ODH_THRESHOLD" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::odh::set_threshold(v);
                    eprintln!("info string ODH_THRESHOLD: {v}");
                }
            }
            #[cfg(feature = "nnue-odh")]
            "ODH_CAP" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::odh::set_cap(v);
                    eprintln!("info string ODH_CAP: {v}");
                }
            }
            #[cfg(feature = "nnue-eval-bias-corr")]
            "EVAL_BIAS_CORR_SCALE" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::eval_bias_corr::set_output_scale(v);
                    eprintln!("info string EVAL_BIAS_CORR_SCALE: {v}");
                }
            }
            #[cfg(feature = "nnue-eval-bias-corr")]
            "EVAL_BIAS_CORR_CLAMP" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::eval_bias_corr::set_clamp(v);
                    eprintln!("info string EVAL_BIAS_CORR_CLAMP: {v}");
                }
            }
            #[cfg(feature = "nnue-eval-bias-corr")]
            "EVAL_BIAS_CORR_FILE" => {
                let path = value.trim();
                if !path.is_empty() && path != "<empty>" {
                    match rshogi_core::nnue::eval_bias_corr::load_from_file(path) {
                        Ok(n) => {
                            eprintln!("info string EVAL_BIAS_CORR_FILE: {n} 行ロード ({path})")
                        }
                        Err(e) => eprintln!("info string EVAL_BIAS_CORR_FILE エラー: {e}"),
                    }
                }
            }
            #[cfg(feature = "nnue-kzt-psqt")]
            "KZT_CLAMP_RAW" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::kzt::set_clamp_raw(v);
                    eprintln!("info string KZT_CLAMP_RAW: {v}");
                }
            }
            #[cfg(feature = "nnue-king-mobility-psqt")]
            "KING_MOBILITY_OUTPUT_SCALE" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::king_mobility::set_output_scale(v);
                    eprintln!("info string KING_MOBILITY_OUTPUT_SCALE: {v}");
                }
            }
            #[cfg(feature = "nnue-drop-pressure-psqt")]
            "DROP_PRESSURE_OUTPUT_SCALE" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::nnue::drop_pressure::set_output_scale(v);
                    eprintln!("info string DROP_PRESSURE_OUTPUT_SCALE: {v}");
                }
            }
            "HistChCoeff" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::search::set_hist_ch_coeff(v);
                    eprintln!("info string HistChCoeff: {v}");
                }
            }
            "HistMainCoeff" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::search::set_hist_main_coeff(v);
                    eprintln!("info string HistMainCoeff: {v}");
                }
            }
            "HistPawnCoeff" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::search::set_hist_pawn_coeff(v);
                    eprintln!("info string HistPawnCoeff: {v}");
                }
            }
            "HistLpCoeff" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::search::set_hist_lp_coeff(v);
                    eprintln!("info string HistLpCoeff: {v}");
                }
            }
            "KingDropBonusMode" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::search::set_king_drop_bonus_mode(v);
                    eprintln!("info string KingDropBonusMode: {v}");
                }
            }
            "KingDropDangerMin" => {
                if let Ok(v) = value.parse::<i32>() {
                    rshogi_core::search::set_king_drop_danger_min(v);
                    eprintln!("info string KingDropDangerMin: {v}");
                }
            }
            "NNUE_ARCHITECTURE" => match parse_nnue_architecture(&value) {
                Some(mode) => {
                    set_nnue_architecture_override(mode);
                    // EvalFile が指定済みなら、現在ロード済みか失敗済みかに関係なく再試行する。
                    // arch_str 不整合が原因でロード失敗していた場合、architecture override
                    // 変更後の再試行で成功する可能性がある。再試行しても失敗した場合は
                    // Some(false) のまま維持され、isready の panic 安全策は保持される。
                    if let Some(ref path) = self.eval_file_path {
                        let was_loaded = get_network().is_some();
                        match init_nnue(path) {
                            Ok(()) => {
                                self.eval_file_explicit = Some(true);
                                let action = if was_loaded {
                                    "reloaded"
                                } else {
                                    "retried and loaded"
                                };
                                eprintln!(
                                    "info string NNUE_ARCHITECTURE: {} ({} {})",
                                    value, action, path
                                );
                            }
                            Err(e) => {
                                self.eval_file_explicit = Some(false);
                                let action = if was_loaded {
                                    "reload failed"
                                } else {
                                    "retry failed"
                                };
                                eprintln!(
                                    "info string NNUE_ARCHITECTURE: {} ({}: {})",
                                    value, action, e
                                );
                            }
                        }
                    } else if get_network().is_some() {
                        // EvalFile 未指定で自動ロード済み → クリアして isready に任せる
                        clear_nnue();
                        self.eval_file_explicit = None;
                        eprintln!(
                            "info string NNUE_ARCHITECTURE: {} (NNUE cleared, will reload on isready)",
                            value
                        );
                    } else {
                        eprintln!("info string NNUE_ARCHITECTURE: {}", value);
                    }
                }
                None => {
                    eprintln!(
                        "info string Warning: invalid NNUE_ARCHITECTURE '{}', expected auto, halfkp, halfka_hm, halfka, layerstacks or layerstacks-psqt",
                        value
                    );
                }
            },
            "LS_BUCKET_MODE" => match parse_layer_stack_bucket_mode(&value) {
                Some(mode) => {
                    set_layer_stack_bucket_mode(mode);
                    eprintln!("info string LS_BUCKET_MODE: {}", mode.as_str());
                }
                None => {
                    eprintln!(
                        "info string Warning: invalid LS_BUCKET_MODE '{}', expected progress8kpabs",
                        value
                    );
                }
            },
            "LS_PROGRESS_COEFF" => {
                if value.is_empty() || value == "<empty>" {
                    reset_layer_stack_progress_kpabs_weights();
                    eprintln!("info string LS_PROGRESS_COEFF: reset to built-in default");
                } else {
                    match load_progress_coeff_kpabs(&value) {
                        Ok(weights) => match set_layer_stack_progress_kpabs_weights(weights) {
                            Ok(()) => {
                                eprintln!("info string LS_PROGRESS_COEFF loaded (kpabs): {value}");
                            }
                            Err(err) => {
                                eprintln!("info string Warning: {err}");
                            }
                        },
                        Err(err) => {
                            eprintln!("info string Warning: {err}");
                        }
                    }
                }
            }
            "PassRights" => {
                let v = value == "true" || value == "1";
                self.pass_rights_enabled = v;
                eprintln!("info string PassRights: {}", if v { "enabled" } else { "disabled" });
            }
            "InitialPassCount" => {
                if let Ok(v) = value.parse::<u8>() {
                    self.initial_pass_count = v.clamp(0, 10);
                    eprintln!("info string InitialPassCount: {}", self.initial_pass_count);
                }
            }
            "PassMoveBonus" => {
                if let Ok(v) = value.parse::<i32>() {
                    let clamped = v.clamp(-1000, 1000);
                    set_pass_move_bonus(clamped);
                    eprintln!("info string PassMoveBonus: {clamped}");
                }
            }
            "PassRightValueEarly" => {
                if let Ok(v) = value.parse::<i32>() {
                    self.pass_right_value_early = v.clamp(0, 500);
                    set_pass_right_value_phased(
                        self.pass_right_value_early,
                        self.pass_right_value_late,
                    );
                    eprintln!("info string PassRightValueEarly: {}", self.pass_right_value_early);
                }
            }
            "PassRightValueLate" => {
                if let Ok(v) = value.parse::<i32>() {
                    self.pass_right_value_late = v.clamp(0, 500);
                    set_pass_right_value_phased(
                        self.pass_right_value_early,
                        self.pass_right_value_late,
                    );
                    eprintln!("info string PassRightValueLate: {}", self.pass_right_value_late);
                }
            }
            // ----------------------------------------------------------------
            // Midgame policy hook (αβ root ordering を Transformer policy で補完)
            // 詳細は docs/midgame_policy_design.md
            // ----------------------------------------------------------------
            "UseMidgamePolicy" => {
                let enabled = matches!(value.to_ascii_lowercase().as_str(), "true" | "on" | "1");
                self.midgame_policy_config.enabled = enabled;
                self.refresh_midgame_policy_hook();
                eprintln!("info string UseMidgamePolicy: {enabled}");
            }
            "MidgamePolicyMinDepth" => {
                if let Ok(v) = value.parse::<i32>() {
                    self.midgame_policy_config.min_depth = v.clamp(0, 64);
                    if let Some(h) = self.midgame_policy_hook.as_mut() {
                        h.config_mut().min_depth = self.midgame_policy_config.min_depth;
                    }
                    eprintln!(
                        "info string MidgamePolicyMinDepth: {}",
                        self.midgame_policy_config.min_depth
                    );
                }
            }
            "MidgamePolicyTopK" => {
                if let Ok(v) = value.parse::<usize>() {
                    self.midgame_policy_config.top_k = v.min(32);
                    if let Some(h) = self.midgame_policy_hook.as_mut() {
                        h.config_mut().top_k = self.midgame_policy_config.top_k;
                    }
                    eprintln!(
                        "info string MidgamePolicyTopK: {}",
                        self.midgame_policy_config.top_k
                    );
                }
            }
            "MidgamePolicyAlpha" => {
                if let Ok(v) = value.parse::<i32>() {
                    let clamped = v.clamp(0, 1024);
                    self.midgame_policy_config.policy_alpha = clamped as f32;
                    if let Some(h) = self.midgame_policy_hook.as_mut() {
                        h.config_mut().policy_alpha = self.midgame_policy_config.policy_alpha;
                    }
                    eprintln!("info string MidgamePolicyAlpha: {clamped}");
                }
            }
            "MidgamePolicyCacheEntries" => {
                if let Ok(v) = value.parse::<usize>() {
                    self.midgame_policy_config.cache_entries = v.clamp(16, 65536);
                    // cache サイズ変更時は hook を再構築 (cache を作り直すため)。
                    self.refresh_midgame_policy_hook();
                    eprintln!(
                        "info string MidgamePolicyCacheEntries: {}",
                        self.midgame_policy_config.cache_entries
                    );
                }
            }
            "MidgamePolicyModel" => {
                if value == "<empty>" || value.is_empty() {
                    self.midgame_policy_model_path = None;
                } else {
                    self.midgame_policy_model_path = Some(value.to_string());
                }
                // model 切替は重い候補 (candle load) のため、 evaluator 構築は
                // isready 受信時または UseMidgamePolicy=true 時に試行する。
                self.refresh_midgame_policy_hook();
                eprintln!(
                    "info string MidgamePolicyModel: {}",
                    self.midgame_policy_model_path.as_deref().unwrap_or("<empty>")
                );
            }
            // --- Mate engine 中盤統合 ---
            "UseRootMateSearch" => {
                let v = value == "true" || value == "1";
                self.use_root_mate_search = v;
                eprintln!(
                    "info string UseRootMateSearch: {}",
                    if v { "enabled" } else { "disabled" }
                );
            }
            "UseOpponentMateCheck" => {
                let v = value == "true" || value == "1";
                self.use_opponent_mate_check = v;
                eprintln!(
                    "info string UseOpponentMateCheck: {}",
                    if v { "enabled" } else { "disabled" }
                );
            }
            "OpponentMateMaxMoves" => {
                if let Ok(v) = value.parse::<usize>() {
                    self.opponent_mate_max_moves = v.clamp(1, 256);
                    eprintln!("info string OpponentMateMaxMoves: {}", self.opponent_mate_max_moves);
                }
            }
            "UseMateSolverThread" => {
                let v = value == "true" || value == "1";
                self.use_mate_solver_thread = v;
                eprintln!(
                    "info string UseMateSolverThread: {}",
                    if v { "enabled" } else { "disabled" }
                );
            }
            // --- PUCT hybrid engine (puct-hybrid feature 有効時のみ受付) ---
            #[cfg(feature = "puct-hybrid")]
            "UsePuct" => {
                let v = value == "true" || value == "1";
                self.use_puct = v;
                eprintln!("info string UsePuct: {}", if v { "enabled" } else { "disabled" });
            }
            #[cfg(feature = "puct-hybrid")]
            "PuctEndgameThreshold" => {
                if let Ok(v) = value.parse::<i32>() {
                    self.puct_endgame_threshold = v.clamp(0, 5000);
                    eprintln!("info string PuctEndgameThreshold: {}", self.puct_endgame_threshold);
                }
            }
            #[cfg(feature = "puct-hybrid")]
            "PuctNetworkPath" => {
                if value.is_empty() || value == "<empty>" {
                    self.puct_network_path = None;
                } else {
                    self.puct_network_path = Some(value.to_string());
                }
                // NNUE-B Transformer 未訓練のため現状は受け取るだけ。
                eprintln!(
                    "info string PuctNetworkPath: {} (stub evaluator 使用、 実 load は未実装)",
                    self.puct_network_path.as_deref().unwrap_or("<empty>")
                );
            }
            _ => {
                // 未知のオプションは無視
            }
        }
    }

    /// USI option 変更後に hook の状態を再構築する。
    ///
    /// - hook が未生成なら新規作成 (evaluator=None で deactivated)
    /// - cache_entries が変わった場合は hook 全体を作り直す
    /// - model 設定がある場合はここで evaluator load を試みる (後続セッションで
    ///   実装。 現状は dummy evaluator を生成しない)
    fn refresh_midgame_policy_hook(&mut self) {
        // 既存の hook が同じ cache サイズなら config だけ更新で済む。
        // 違う場合は再構築。
        let need_rebuild = match &self.midgame_policy_hook {
            None => true,
            Some(h) => h.config().cache_entries != self.midgame_policy_config.cache_entries,
        };
        if need_rebuild {
            // 注: evaluator load は本タスクでは TODO (実 model 配置 + adapter
            // feature の組み合わせで別 session に取り組む)。 hook 自体は OK。
            self.midgame_policy_hook = Some(p2_midgame_policy::MidgamePolicyHook::new(
                self.midgame_policy_config.clone(),
                None,
            ));
        } else if let Some(h) = self.midgame_policy_hook.as_mut() {
            *h.config_mut() = self.midgame_policy_config.clone();
        }
    }

    /// usinewgameコマンド: 新しい対局の開始
    fn cmd_usinewgame(&mut self) {
        self.cmd_stop();

        if let Some(search) = self.search.as_mut() {
            search.clear_tt();
            search.clear_histories(); // YaneuraOu準拠：履歴統計もクリア
        }
        self.position = Position::new();
    }

    /// positionコマンド: 局面設定
    ///
    /// 拡張形式: `position [sfen <sfen> | startpos] [passrights <black> <white>] [moves <move1> ...]`
    fn cmd_position(&mut self, tokens: &[&str]) {
        Self::apply_position_tokens(
            &mut self.position,
            tokens,
            self.pass_rights_enabled,
            self.initial_pass_count,
        );
    }

    fn apply_position_tokens(
        position: &mut Position,
        tokens: &[&str],
        pass_rights_enabled: bool,
        initial_pass_count: u8,
    ) {
        // position [sfen <sfen> | startpos] [passrights <black> <white>] [moves <move1> <move2> ...]
        let mut idx = 1;
        if idx >= tokens.len() {
            return;
        }

        // 局面の設定
        if tokens[idx] == "startpos" {
            position.set_hirate();
            idx += 1;
        } else if tokens[idx] == "sfen" {
            idx += 1;
            // SFENを収集（movesまたはpassrightsの前まで）
            let mut sfen_parts = Vec::new();
            while idx < tokens.len() && tokens[idx] != "moves" && tokens[idx] != "passrights" {
                sfen_parts.push(tokens[idx]);
                idx += 1;
            }
            let sfen = sfen_parts.join(" ");
            if let Err(e) = position.set_sfen(&sfen) {
                eprintln!("info string Error parsing SFEN: {e}");
                return;
            }
        }

        // パス権の設定（passrights キーワード）
        // 形式: passrights <black_count> <white_count>
        if idx < tokens.len() && tokens[idx] == "passrights" {
            idx += 1;
            if pass_rights_enabled {
                // 先手のパス権数
                let black_pass = if idx < tokens.len() {
                    tokens[idx].parse::<u8>().unwrap_or(initial_pass_count)
                } else {
                    initial_pass_count
                };
                idx += 1;

                // 後手のパス権数
                let white_pass = if idx < tokens.len() {
                    tokens[idx].parse::<u8>().unwrap_or(initial_pass_count)
                } else {
                    initial_pass_count
                };
                idx += 1;

                // パス権を設定
                position.enable_pass_rights(black_pass, white_pass);
            } else {
                // パス権が無効な場合は値を読み飛ばす
                idx += 2;
            }
        } else if pass_rights_enabled {
            // passrights キーワードがないがパス権が有効な場合、デフォルト値を設定
            position.enable_pass_rights(initial_pass_count, initial_pass_count);
        }

        // 指し手の適用
        if idx < tokens.len() && tokens[idx] == "moves" {
            idx += 1;
            while idx < tokens.len() {
                if let Some(mv) = Move::from_usi(tokens[idx]) {
                    // SEGV-FIX-20260531: 不正 move (= from に駒なし / 駒打ち先に駒あり 等)
                    // を弾いて defensive に skip。 不正な position コマンドを
                    // tournament 等が送りつけるケース (= USI 仕様外だが実害発生) で
                    // do_move 内部の stale state aliasing → search 内 SEGV を防止する。
                    if !mv.is_pass() && !mv.is_drop() && position.piece_on(mv.from()).is_none() {
                        eprintln!(
                            "info string Error: illegal move {token} (no piece at from square); aborting position moves",
                            token = tokens[idx]
                        );
                        break;
                    }
                    if mv.is_drop() && position.piece_on(mv.to()).is_some() {
                        eprintln!(
                            "info string Error: illegal drop {token} (target square occupied); aborting position moves",
                            token = tokens[idx]
                        );
                        break;
                    }
                    // PASS の場合は gives_check は false
                    let gives_check = if mv.is_pass() {
                        false
                    } else {
                        position.gives_check(mv)
                    };
                    position.do_move(mv, gives_check);
                } else {
                    eprintln!("info string Error parsing move: {token}", token = tokens[idx]);
                    break;
                }
                idx += 1;
            }
            // SEGV-FIX-20260531: do_move 中で NNUE モードは board_effects update を
            // skip して dirty フラグを立てるが、 main thread の `detector` 経路で
            // `board_effect()` が呼ばれた時に dirty=true だと panic / 不正値返却に
            // なる。 position コマンド処理の末尾で 1 度だけ強制再計算して dirty を
            // 解消する。 search 中の clone Position は別経路 (= dirty 維持で OK)。
            position.recompute_board_effects();
        }
    }

    fn stochastic_ponder_position(&self) -> Option<Position> {
        let line = self.last_position_cmd.as_deref()?;
        let mut owned: Vec<&str> = line.split_whitespace().collect();
        if owned.len() < 3 {
            return None;
        }

        if let Some(moves_idx) = owned.iter().position(|token| *token == "moves") {
            if owned.len() > moves_idx + 1 {
                owned.pop();
            }
        } else {
            return None;
        }

        let mut position = Position::new();
        Self::apply_position_tokens(
            &mut position,
            &owned,
            self.pass_rights_enabled,
            self.initial_pass_count,
        );
        Some(position)
    }

    /// goコマンド: 探索開始
    fn cmd_go(&mut self, tokens: &[&str]) {
        // 既存の探索を停止（bestmove出力を抑制する）
        // GUIがstopを送らずにposition+goを送ってきた場合、前のponder探索の
        // bestmoveがstdoutに出力されるとGUIが混乱する（YaneuraOu準拠）
        self.stop_search_silently();

        // ★ PUCT hybrid foundation (観測層): phase + endgame 判定して info string で報告。
        // puct-hybrid feature 有効 + UsePuct=true なら hybrid 経路で routing する。
        let signal_score;
        let signal_use_alpha_beta;
        let game_phase;
        {
            let signal = self.endgame_detector.evaluate(&mut self.position);
            let phase = p2_search::phase::determine_phase(signal.score, &mut self.phase_tracker);
            println!(
                "info string phase={:?} endgame_score={:.3} use_alpha_beta={} reason={:?}",
                phase, signal.score, signal.use_alpha_beta, signal.reason
            );
            signal_score = signal.score;
            signal_use_alpha_beta = signal.use_alpha_beta;
            game_phase = phase;
            // `DUAL_CROSS_USE_ENDGAME_DETECTOR=1` 経路で NNUE 側 gate が参照する thread-local
            // override に position-based phase index を書き込む。 `GamePhase` 順
            // (Opening=0 / Middlegame=1 / Endgame=2) を そのまま使う。 flag 無効時 (= default)
            // でも overhead は cell 書き 1 回で negligible。
            let phase_idx: u8 = match phase {
                p2_search::phase::GamePhase::Opening => 0,
                p2_search::phase::GamePhase::Middlegame => 1,
                p2_search::phase::GamePhase::Endgame => 2,
            };
            rshogi_core::nnue::dual_cross::set_current_phase_idx(phase_idx);
        }

        // 制限を解析
        let mut limits = self.parse_go_options(tokens);

        // ─── Mate engine 中盤統合 (Plan B+C) ───
        // - Opening phase は skip (= 序盤頓死は稀、 overhead 無駄)
        // - budget = byoyomi の 2% × NPS 100k で max_nodes 制限
        // - 3 ply mate は維持 (= 中盤詰み筋検知の本命)
        let in_opening = matches!(game_phase, p2_search::phase::GamePhase::Opening);
        let mut mate_excluded: Vec<rshogi_core::types::Move> = Vec::new();
        if !in_opening && (self.use_root_mate_search || self.use_opponent_mate_check) {
            // budget = byoyomi の 1%、 min 5ms max 30ms 程度
            let stm_idx = self.position.side_to_move() as usize;
            let byoyomi_ms = limits.byoyomi[stm_idx].max(500) as u32;
            let budget_ms = (byoyomi_ms / 100).clamp(5, 30);
            // NPS ~100k/s for mate_probe → budget_ms * 100 nodes
            let budget_nodes = (budget_ms * 100).clamp(300, 3000);

            let mut probe_pos = self.position.clone();
            if self.use_root_mate_search {
                let result = mate_probe_3ply(&mut probe_pos, budget_nodes);
                if let MateResult::MateFound { best_move, ply, .. } = result {
                    // 連続王手千日手ガード (= phantom mate 防止、 2026-06-25)
                    // mate_probe_3ply は履歴を持たない clone 上で探索するため、
                    // 「局所的には詰みに見えるが、 その王手で 4 回目の同一局面に達し
                    //  連続王手の千日手で *攻め方が負け* る」筋を区別できない。
                    // game3 (vs Suisho11) では root mate probe が phantom mate を返し続け、
                    // 連続王手を継続して千日手規則で負けた。
                    // ここで履歴付きの実局面 (self.position) 上で best_move を指し、
                    // 千日手状態を確認する。 Lose / Draw なら詰みではないので採用しない。
                    let mut hist_pos = self.position.clone();
                    let gives_check = hist_pos.gives_check(best_move);
                    let _ = hist_pos.do_move(best_move, gives_check);
                    // best_move を指した後の手番は相手（受け手）。
                    // この王手で連続王手の千日手が成立し攻め方（我々）が負ける場合、
                    // 受け手手番から見た repetition_state は Win（受け手が勝ち）になる。
                    // 通常の千日手（引き分け）は Draw。 どちらも「詰みではない」ので採用しない。
                    let rep = hist_pos.repetition_state(1);
                    if matches!(
                        rep,
                        rshogi_core::types::RepetitionState::Win
                            | rshogi_core::types::RepetitionState::Draw
                    ) {
                        println!(
                            "info string root mate rejected (repetition {:?}): best={}",
                            rep,
                            best_move.to_usi()
                        );
                    } else {
                        println!(
                            "info string root mate found: ply={} best={}",
                            ply,
                            best_move.to_usi()
                        );
                        println!("bestmove {}", best_move.to_usi());
                        return;
                    }
                }
            }
            if self.use_opponent_mate_check {
                let mut legal = rshogi_core::movegen::MoveList::new();
                rshogi_core::movegen::generate_legal(&probe_pos, &mut legal);
                let check_count = legal.len().min(self.opponent_mate_max_moves);
                let _ = budget_nodes; // budget は root mate_search 専用
                for i in 0..check_count {
                    let mv = legal[i];
                    let gives_check = probe_pos.gives_check(mv);
                    let _ = probe_pos.do_move(mv, gives_check);
                    // 1 ply mate (= 即詰、 v4 確定設定。 3 ply は v5 で NPS -9% overhead、 +70 Elo に劣化)
                    let opp = mate_probe_1ply(&mut probe_pos);
                    probe_pos.undo_move(mv);
                    if matches!(opp, MateResult::MateFound { .. }) {
                        mate_excluded.push(mv);
                    }
                }
                if !mate_excluded.is_empty() {
                    println!(
                        "info string opponent mate detected: {} of {} moves excluded",
                        mate_excluded.len(),
                        legal.len()
                    );
                    if limits.search_moves.is_empty() {
                        for i in 0..legal.len() {
                            let mv = legal[i];
                            if !mate_excluded.contains(&mv) {
                                limits.search_moves.push(mv);
                            }
                        }
                        if limits.search_moves.is_empty() {
                            println!("info string all moves lead to mate, fallback to full search");
                            for i in 0..legal.len() {
                                limits.search_moves.push(legal[i]);
                            }
                        }
                    } else {
                        limits.search_moves.retain(|m| !mate_excluded.contains(m));
                    }
                }
            }
        }

        // ─── PUCT hybrid 分岐 ───
        //
        // 序中盤 (endgame_score < threshold/1000.0) で UsePuct=true なら PUCT engine、
        // それ以外は既存 αβ 経路。 EndgameDetector の `use_alpha_beta=true` (絶対終盤)
        // も αβ にルート。
        #[cfg(feature = "puct-hybrid")]
        {
            if self.use_puct && !signal_use_alpha_beta {
                let threshold = self.puct_endgame_threshold as f32 / 1000.0;
                if signal_score < threshold {
                    self.cmd_go_puct(limits);
                    return;
                }
                println!(
                    "info string puct route skipped: endgame_score={:.3} >= threshold={:.3}",
                    signal_score, threshold
                );
            }
        }
        #[cfg(not(feature = "puct-hybrid"))]
        {
            // signal_score / signal_use_alpha_beta は puct-hybrid 無効時は未使用なので
            // warn を避けるため bind を消費する。
            let _ = (signal_score, signal_use_alpha_beta);
        }

        // Stochastic_Ponder では 1 手戻した局面から先読みする（YaneuraOu 準拠）
        let mut pos = if self.stochastic_ponder && limits.ponder {
            self.stochastic_ponder_position().unwrap_or_else(|| self.position.clone())
        } else {
            self.position.clone()
        };

        // ─── MateSolverThread 起動 (= 別 thread dfpn、 main search 完全不変) ───
        // Threads >= 2 + UseMateSolverThread=true + Opening 以外 + ponder 以外 で起動。
        // Ponder 中は局面確定していないので skip (= 通常 go で根詰みを掘る)。
        let (mate_found_arc, mate_move_arc) = if self.use_mate_solver_thread
            && self.num_threads >= 2
            && !in_opening
            && !limits.ponder
        {
            // 既存 mate solver thread が残っていれば停止 (= 念のため)
            if let Some(prev) = self.mate_solver_thread.take() {
                prev.request_stop();
                prev.join();
            }
            let probe_pos = self.position.clone();
            let mate = p2_search::mate_solver_thread::MateSolverThread::start(
                probe_pos,
                p2_search::mate_solver_thread::DEFAULT_MATE_SOLVER_MAX_NODES,
                p2_search::mate_solver_thread::DEFAULT_MATE_SOLVER_MAX_DEPTH,
                p2_search::mate_solver_thread::DEFAULT_MATE_SOLVER_TT_MB,
            );
            let mf = mate.mate_found_flag();
            let mm = mate.mate_move_flag();
            self.mate_solver_thread = Some(mate);
            (Some(mf), Some(mm))
        } else {
            (None, None)
        };

        let mut search = self
            .search
            .take()
            .unwrap_or_else(|| Search::new_with_eval_hash(self.tt_size_mb, self.eval_hash_size_mb));
        if search.eval_hash_size_mb() != self.eval_hash_size_mb {
            search.resize_eval_hash(self.eval_hash_size_mb);
        }
        search.set_skill_options(self.skill_options);
        // stop/ponderhitフラグをリセット（スレッド生成前に行い、go()内での競合を防ぐ）
        search.reset_flags();
        let stop_flag = search.stop_flag();
        let ponderhit_flag = search.ponderhit_flag();
        self.stop_flag = Some(stop_flag.clone());
        self.ponderhit_flag = Some(ponderhit_flag.clone());

        let suppress_flag = Arc::clone(&self.suppress_bestmove);
        let builder = thread::Builder::new().stack_size(SEARCH_STACK_SIZE);
        self.search_thread = Some(
            builder
                .spawn(move || {
                    let result = search.go(
                        &mut pos,
                        limits,
                        Some(|info: &SearchInfo| {
                            println!("{}", info.to_usi_string());
                            std::io::stdout().flush().ok();
                        }),
                    );

                    // 探索統計レポートを出力（search-stats feature有効時のみ内容あり）
                    if !result.stats_report.is_empty() {
                        for line in result.stats_report.lines() {
                            println!("info string {line}");
                        }
                        std::io::stdout().flush().ok();
                    }

                    // bestmove出力（suppress_bestmoveが立っていない場合のみ）
                    // cmd_goから内部的にstopされた場合は抑制される
                    if !suppress_flag.load(Ordering::SeqCst) {
                        // MateSolverThread が詰みを発見済なら bestmove を override する。
                        // main search の result は変えない (= TT / 学習 side-effect 無し)、
                        // あくまで出力 layer での swap のみ。
                        let mate_override = if let (Some(mf), Some(mm)) =
                            (&mate_found_arc, &mate_move_arc)
                        {
                            p2_search::mate_solver_thread::MateSolverThread::take_mate_from(mf, mm)
                        } else {
                            None
                        };

                        let best_move = mate_override.unwrap_or(result.best_move);
                        if mate_override.is_some() {
                            println!("info string mate solver found mate: {}", best_move.to_usi());
                        }

                        let best_usi = if best_move != Move::NONE {
                            best_move.to_usi()
                        } else {
                            "resign".to_string()
                        };

                        // mate override 時は ponder move を捨てる (= ponder 先が無意味になるため)
                        if mate_override.is_none() && result.ponder_move != Move::NONE {
                            println!("bestmove {best_usi} ponder {}", result.ponder_move.to_usi());
                        } else {
                            println!("bestmove {best_usi}");
                        }
                        std::io::stdout().flush().ok();
                    }

                    (search, result)
                })
                .expect("failed to spawn search thread"),
        );
    }

    /// PUCT hybrid 経路の探索開始
    ///
    /// 序中盤 PUCT engine で探索し、 bestmove を出力する。
    /// 現状は stub evaluator (value=0) を使うため strong play は期待しない。
    /// 本実装は pipeline 整備のみで、 NNUE-B Transformer 統合は別 session。
    #[cfg(feature = "puct-hybrid")]
    fn cmd_go_puct(&mut self, limits: LimitsType) {
        let pos = self.position.clone();

        // PuctConfig (default)。 phase-aware や hybrid は default off で素朴な PUCT 動作。
        let config = PuctConfig::default();

        // stub evaluator (value=0)
        let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(StubPuctEvaluator);

        // PUCT 探索の時間/ノード上限を抽出。 limits.movetime / limits.nodes 優先。
        // byoyomi / btime+binc の動的時間管理は本 stub では行わない (簡易固定上限)。
        let max_time_ms: u64 = if limits.movetime > 0 {
            limits.movetime as u64
        } else if limits.byoyomi[0] > 0 {
            limits.byoyomi[0] as u64
        } else {
            // 時間指定なし → 3 秒 default (USI proxy 経由 / `go infinite` 時の保険)
            3_000
        };
        let max_nodes: u64 = if limits.nodes > 0 { limits.nodes } else { 0 };

        let mut engine = PuctEngine::new_direct(config, evaluator);
        let stop_flag = engine.stop_flag();
        self.puct_stop_flag = Some(stop_flag.clone());

        let suppress_flag = Arc::clone(&self.suppress_bestmove);
        let score_cp_cell = Arc::clone(&self.puct_last_score_cp);

        let builder = thread::Builder::new().stack_size(SEARCH_STACK_SIZE);
        let handle = builder
            .spawn(move || {
                let start = Instant::now();
                let puct_limits = PuctLimits {
                    max_nodes,
                    max_time_ms,
                    search_mode: p2_search::SearchMode::Normal,
                    game_phase: None,
                    endgame_score: 0.0,
                };

                let mut work_pos = pos;
                let result = engine.search(&work_pos, &puct_limits);
                let _ = &mut work_pos; // do_move/undo_move は engine 内部で完結

                // info 出力 (cp 換算は winrate * 600 を粗近似)
                let elapsed_ms = start.elapsed().as_millis().max(1) as u64;
                // elapsed_ms は max(1) で保証されるが、 clippy の警告回避に checked_div を使う
                let nps = (result.nodes * 1000).checked_div(elapsed_ms).unwrap_or(0);
                let cp = (result.value * 600.0) as i32;
                score_cp_cell.store(cp, Ordering::SeqCst);
                println!(
                    "info nodes {} time {} nps {} score cp {} string backend=puct value={:.4}",
                    result.nodes, elapsed_ms, nps, cp, result.value
                );
                if !result.pv.is_empty() {
                    let pv: Vec<String> = result.pv.iter().map(|m| m.to_usi()).collect();
                    println!("info pv {}", pv.join(" "));
                }
                std::io::stdout().flush().ok();

                if !suppress_flag.load(Ordering::SeqCst) {
                    let best_usi = if result.bestmove != Move::NONE {
                        result.bestmove.to_usi()
                    } else {
                        "resign".to_string()
                    };
                    println!("bestmove {best_usi}");
                    std::io::stdout().flush().ok();
                }
            })
            .expect("failed to spawn puct search thread");
        self.puct_thread = Some(handle);
    }

    /// goオプションを解析
    fn parse_go_options(&self, tokens: &[&str]) -> LimitsType {
        let mut limits = LimitsType::default();
        // YaneuraOu準拠: go受信時点で探索開始時刻を記録し、この時刻を基準に時間管理する
        limits.set_start_time();
        let mut idx = 1;

        while idx < tokens.len() {
            match tokens[idx] {
                "infinite" => {
                    limits.infinite = true;
                }
                "ponder" => {
                    limits.ponder = true;
                }
                "depth" => {
                    idx += 1;
                    if idx < tokens.len() {
                        limits.depth = tokens[idx].parse().unwrap_or(0);
                    }
                }
                "nodes" => {
                    idx += 1;
                    if idx < tokens.len() {
                        limits.nodes = tokens[idx].parse().unwrap_or(0);
                    }
                }
                "movetime" => {
                    idx += 1;
                    if idx < tokens.len() {
                        limits.movetime = tokens[idx].parse().unwrap_or(0);
                    }
                }
                "mate" => {
                    idx += 1;
                    // `go mate` without a value is treated as infinite (YaneuraOu互換)
                    limits.mate = if idx < tokens.len() {
                        match tokens[idx] {
                            "infinite" => i32::MAX,
                            v => v.parse().unwrap_or(0),
                        }
                    } else {
                        i32::MAX
                    };
                }
                "btime" => {
                    idx += 1;
                    if idx < tokens.len() {
                        limits.time[0] = tokens[idx].parse().unwrap_or(0);
                    }
                }
                "wtime" => {
                    idx += 1;
                    if idx < tokens.len() {
                        limits.time[1] = tokens[idx].parse().unwrap_or(0);
                    }
                }
                "binc" => {
                    idx += 1;
                    if idx < tokens.len() {
                        limits.inc[0] = tokens[idx].parse().unwrap_or(0);
                    }
                }
                "winc" => {
                    idx += 1;
                    if idx < tokens.len() {
                        limits.inc[1] = tokens[idx].parse().unwrap_or(0);
                    }
                }
                "byoyomi" => {
                    idx += 1;
                    if idx < tokens.len() {
                        let byoyomi: i64 = tokens[idx].parse().unwrap_or(0);
                        limits.byoyomi[0] = byoyomi;
                        limits.byoyomi[1] = byoyomi;
                    }
                }
                "rtime" => {
                    idx += 1;
                    if idx < tokens.len() {
                        limits.rtime = tokens[idx].parse().unwrap_or(0);
                    }
                }
                "searchmoves" => {
                    // searchmoves <move1> <move2> ...
                    idx += 1;
                    while idx < tokens.len() {
                        // 他のオプションに当たったら終了
                        if matches!(
                            tokens[idx],
                            "infinite"
                                | "ponder"
                                | "depth"
                                | "nodes"
                                | "movetime"
                                | "btime"
                                | "wtime"
                                | "binc"
                                | "winc"
                                | "byoyomi"
                                | "rtime"
                                | "mate"
                        ) {
                            idx -= 1; // 巻き戻して次のループで処理
                            break;
                        }
                        if let Some(mv) = Move::from_usi(tokens[idx]) {
                            if let Some(normalized) = self.position.to_move(mv) {
                                limits.search_moves.push(normalized);
                            } else {
                                eprintln!("warning: invalid searchmoves: {}", tokens[idx]);
                            }
                        }
                        idx += 1;
                    }
                }
                _ => {}
            }
            idx += 1;
        }

        // MultiPVを設定
        limits.multi_pv = self.multi_pv;

        limits
    }

    /// stopコマンド: 探索停止（GUIからの明示的stop — bestmoveは探索スレッドが出力）
    fn cmd_stop(&mut self) {
        if let Some(stop_flag) = &self.stop_flag {
            stop_flag.store(true, Ordering::SeqCst);
        }
        #[cfg(feature = "puct-hybrid")]
        if let Some(stop_flag) = &self.puct_stop_flag {
            stop_flag.store(true, Ordering::SeqCst);
        }
        if let Some(mate) = &self.mate_solver_thread {
            mate.request_stop();
        }
        self.wait_for_search();
    }

    /// 探索を停止するがbestmoveを出力しない（cmd_go内部で使用）
    ///
    /// GUIがstopを送らずにposition+goを送ってきた場合、前のponder探索の
    /// bestmoveを出力するとGUIが混乱する（YaneuraOu準拠）
    fn stop_search_silently(&mut self) {
        self.suppress_bestmove.store(true, Ordering::SeqCst);
        if let Some(stop_flag) = &self.stop_flag {
            stop_flag.store(true, Ordering::SeqCst);
        }
        #[cfg(feature = "puct-hybrid")]
        if let Some(stop_flag) = &self.puct_stop_flag {
            stop_flag.store(true, Ordering::SeqCst);
        }
        if let Some(mate) = &self.mate_solver_thread {
            mate.request_stop();
        }
        self.wait_for_search();
        self.suppress_bestmove.store(false, Ordering::SeqCst);
    }

    /// ponderhitコマンド: 先読みヒットを通知
    fn cmd_ponderhit(&mut self) {
        if self.stochastic_ponder {
            self.restart_after_ponderhit();
            return;
        }

        if let Some(flag) = &self.ponderhit_flag {
            flag.store(true, Ordering::SeqCst);
        }
    }

    /// Stochastic_Ponder の ponderhit 後に通常探索へ切り替える
    fn restart_after_ponderhit(&mut self) {
        self.stop_search_silently();

        if let Some(line) = self.last_position_cmd.clone() {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            self.cmd_position(&tokens);
        }

        if let Some(line) = self.last_go_cmd.clone() {
            let owned: Vec<String> = line
                .split_whitespace()
                .filter(|token| *token != "ponder")
                .map(str::to_owned)
                .collect();
            let tokens: Vec<&str> = owned.iter().map(String::as_str).collect();
            if !tokens.is_empty() {
                self.cmd_go(&tokens);
            }
        }
    }

    /// 探索スレッドの終了を待ち、Searchを取り戻す
    fn wait_for_search(&mut self) {
        if let Some(handle) = self.search_thread.take() {
            match handle.join() {
                Ok((search, _result)) => {
                    self.search = Some(search);
                }
                Err(_) => {
                    eprintln!("info string search thread panicked, resetting Search");
                    let mut search =
                        Search::new_with_eval_hash(self.tt_size_mb, self.eval_hash_size_mb);
                    search.set_skill_options(self.skill_options);
                    self.search = Some(search);
                }
            }
        }
        self.stop_flag = None;
        self.ponderhit_flag = None;

        // PUCT スレッドも join する (cmd_stop / stop_search_silently 経由)
        #[cfg(feature = "puct-hybrid")]
        {
            if let Some(handle) = self.puct_thread.take()
                && handle.join().is_err()
            {
                eprintln!("info string puct search thread panicked");
            }
            self.puct_stop_flag = None;
        }

        // MateSolverThread の join (= stop request 済の前提で blocking wait)
        if let Some(mate) = self.mate_solver_thread.take() {
            mate.request_stop();
            mate.join();
        }
    }

    /// displayコマンド: 現在の局面を表示（デバッグ用）
    fn cmd_display(&self) {
        println!("SFEN: {}", self.position.to_sfen());
        println!("Side to move: {:?}", self.position.side_to_move());
        println!("Game ply: {}", self.position.game_ply());
    }

    /// evalコマンド: 現在の局面の静的評価値を表示（デバッグ用）
    ///
    /// `eval diag` で diagnostics 付き評価（PSQT 含む中間値をログ出力）
    fn cmd_eval(&mut self, diagnostics: bool) {
        let Some(network) = get_network() else {
            println!("info string Error: No NNUE network loaded");
            return;
        };

        // アーキテクチャに応じたアキュムレータスタックを作成
        let mut stack = AccumulatorStackVariant::from_network(&network);

        if diagnostics {
            #[cfg(feature = "diagnostics")]
            {
                use rshogi_core::nnue::{LayerStacksNetwork, NNUENetwork};
                // diagnostics モード: LayerStacks のみ対応
                if let NNUENetwork::LayerStacks(ref net) = *network {
                    // L1 variant に dispatch し、refresh + evaluate_with_diagnostics を実行
                    match net {
                        #[cfg(feature = "layerstacks-1536x16x32")]
                        LayerStacksNetwork::L1536x16x32(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<1536>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-1536x32x32")]
                        LayerStacksNetwork::L1536x32x32(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<1536>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-1536x8x96")]
                        LayerStacksNetwork::L1536x8x96(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<1536>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-1536x16x96")]
                        LayerStacksNetwork::L1536x16x96(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<1536>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-1536x32x96")]
                        LayerStacksNetwork::L1536x32x96(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<1536>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-1536x48x96")]
                        LayerStacksNetwork::L1536x48x96(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<1536>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-1536x64x96")]
                        LayerStacksNetwork::L1536x64x96(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<1536>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-768x32x96")]
                        LayerStacksNetwork::L768x32x96(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<768>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-1536x16x48")]
                        LayerStacksNetwork::L1536x16x48(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<1536>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-1024x16x48")]
                        LayerStacksNetwork::L1024x16x48(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<1024>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-1024x8x64")]
                        LayerStacksNetwork::L1024x8x64(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<1024>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-768x16x32")]
                        LayerStacksNetwork::L768x16x32(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<768>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-768x8x32")]
                        LayerStacksNetwork::L768x8x32(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<768>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[cfg(feature = "layerstacks-512x16x32")]
                        LayerStacksNetwork::L512x16x32(inner) => {
                            let mut acc = rshogi_core::nnue::AccumulatorLayerStacks::<512>::new();
                            inner.refresh_accumulator(&self.position, &mut acc);
                            let value = inner.evaluate_with_diagnostics(&self.position, &acc);
                            println!("info string Static eval (diagnostics): {}", value.raw());
                        }
                        #[allow(unreachable_patterns)]
                        _ => println!(
                            "info string Error: no matching LayerStacks L1 variant compiled"
                        ),
                    }
                } else {
                    println!("info string Error: diagnostics is only supported for LayerStacks");
                }
            }
            #[cfg(not(feature = "diagnostics"))]
            {
                println!("info string Error: build with --features diagnostics to use 'eval diag'");
            }
        } else {
            // LOG_KINGRING_VALUE=1 環境変数有効時、 evaluate 1 回分の kingring scalar 情報を
            // stdout に出力する thread-local request flag を guard で立てる (= drop で復元)。
            // env 無効時は guard は no-op + overhead ゼロ。
            let _kr_log_guard = rshogi_core::nnue::ablation::LogKingringRequestGuard::new();
            let value = evaluate_dispatch(&mut self.position, &mut stack, &mut None);
            println!("info string Static eval: {}", value.raw());
        }
        println!("info string SFEN: {}", self.position.to_sfen());
    }
}

fn main() -> Result<()> {
    thread::Builder::new()
        .name("rshogi-usi-main".to_string())
        .stack_size(SEARCH_STACK_SIZE)
        .spawn(run_usi)?
        .join()
        .map_err(|_| anyhow!("USI main thread panicked"))?
}

fn run_usi() -> Result<()> {
    // ロガー初期化（標準エラー出力）
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Stderr)
        .init();

    // ビットボードテーブルの初期化（ホットパスでの OnceLock atomic check 回避）
    rshogi_core::bitboard::init_bitboard_tables();

    // dual_cross profiling 用の env flag を読み込み (default OFF、 ゼロオーバヘッド)
    init_dual_cross_profile();

    let mut engine = UsiEngine::new();
    let stdin = io::stdin();

    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();

        if !engine.process_command(line)? {
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    // 履歴統計の初期化がスタックを大量に消費するため、別スレッドで実行
    // UsiEngine::new() が NNUE グローバル状態に依存するため、全テストを #[serial] で逐次実行
    const STACK_SIZE: usize = 64 * 1024 * 1024;

    #[test]
    #[serial]
    fn parse_go_mate_sets_limits() {
        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let engine = UsiEngine::new();
                let tokens = vec!["go", "mate", "5"];

                let limits = engine.parse_go_options(&tokens);
                assert_eq!(limits.mate, 5);
                assert!(!limits.use_time_management(), "mate search disables time management");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[serial]
    fn parse_go_mate_without_value_defaults_to_infinite() {
        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let engine = UsiEngine::new();
                let tokens = vec!["go", "mate"];

                let limits = engine.parse_go_options(&tokens);
                assert_eq!(limits.mate, i32::MAX);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[serial]
    fn parse_go_mate_infinite_defaults_to_max() {
        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let engine = UsiEngine::new();
                let tokens = vec!["go", "mate", "infinite"];

                let limits = engine.parse_go_options(&tokens);
                assert_eq!(limits.mate, i32::MAX);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[serial]
    fn stochastic_ponder_position_rewinds_last_move() {
        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let mut engine = UsiEngine::new();
                engine.last_position_cmd = Some("position startpos moves 7g7f 3c3d".to_string());

                let pos = engine.stochastic_ponder_position().expect("stochastic ponder position");
                assert_eq!(
                    pos.to_sfen(),
                    "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2"
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[serial]
    fn setoption_draw_value_updates_search() {
        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let mut engine = UsiEngine::new();
                engine.cmd_setoption(&["setoption", "name", "DrawValueBlack", "value", "123"]);
                engine.cmd_setoption(&["setoption", "name", "DrawValueWhite", "value", "-456"]);

                let search = engine.search.as_ref().expect("search exists");
                assert_eq!(search.draw_value_black(), 123);
                assert_eq!(search.draw_value_white(), -456);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[serial]
    fn setoption_defense_mate_probe_nodes_updates_search() {
        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let mut engine = UsiEngine::new();

                // default は 0 (= 機能 OFF)。
                assert_eq!(
                    engine.search.as_ref().expect("search exists").defense_mate_probe_nodes(),
                    0
                );

                // setoption で値が Search に反映されること。
                engine.cmd_setoption(&[
                    "setoption",
                    "name",
                    "DefenseMateProbeNodes",
                    "value",
                    "50000",
                ]);
                assert_eq!(
                    engine.search.as_ref().expect("search exists").defense_mate_probe_nodes(),
                    50000
                );

                // max を超える値は clamp されること。
                engine.cmd_setoption(&[
                    "setoption",
                    "name",
                    "DefenseMateProbeNodes",
                    "value",
                    "99999999",
                ]);
                assert_eq!(
                    engine.search.as_ref().expect("search exists").defense_mate_probe_nodes(),
                    10_000_000
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[serial]
    fn setoption_layerstack_bucket_updates_globals() {
        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                use rshogi_core::nnue::{
                    LayerStackBucketMode, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
                    get_layer_stack_bucket_mode, get_layer_stack_progress_kpabs_weights,
                    reset_layer_stack_progress_kpabs_weights, set_layer_stack_bucket_mode,
                };

                // テスト開始時に既定値へ戻す
                set_layer_stack_bucket_mode(LayerStackBucketMode::Progress8KPAbs);
                reset_layer_stack_progress_kpabs_weights();

                let mut engine = UsiEngine::new();
                engine.cmd_setoption(&[
                    "setoption",
                    "name",
                    "LS_BUCKET_MODE",
                    "value",
                    "progress8kpabs",
                ]);
                assert_eq!(get_layer_stack_bucket_mode(), LayerStackBucketMode::Progress8KPAbs);

                let tmp_path_bin =
                    std::env::temp_dir().join("rshogi_progress_coeff_kpabs_test.bin");
                let mut bytes = Vec::with_capacity(
                    SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f64>(),
                );
                for i in 0..SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS {
                    let value = if i == 0 {
                        1.25f64
                    } else if i == SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS - 1 {
                        -0.75f64
                    } else {
                        0.0f64
                    };
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                std::fs::write(&tmp_path_bin, bytes).unwrap();
                engine.cmd_setoption(&[
                    "setoption",
                    "name",
                    "LS_PROGRESS_COEFF",
                    "value",
                    tmp_path_bin.to_str().unwrap(),
                ]);
                let kpabs = get_layer_stack_progress_kpabs_weights();
                assert_eq!(kpabs.len(), SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS);
                assert_eq!(kpabs[0], 1.25);
                assert_eq!(kpabs[SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS - 1], -0.75);
                let _ = std::fs::remove_file(tmp_path_bin);

                // 他テストへの影響を避けるため復元
                set_layer_stack_bucket_mode(LayerStackBucketMode::Progress8KPAbs);
                reset_layer_stack_progress_kpabs_weights();
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[serial]
    fn setoption_bucket_eval_scale_updates_search() {
        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let mut engine = UsiEngine::new();

                // 正常系: 8要素カンマ区切り
                engine.cmd_setoption(&[
                    "setoption",
                    "name",
                    "LS_BUCKET_EVAL_SCALE",
                    "value",
                    "80,90,100,110,120,130,140,150",
                ]);
                let search = engine.search.as_ref().expect("search exists");
                let scale = search.bucket_eval_scale();
                assert_eq!(scale, [80, 90, 100, 110, 120, 130, 140, 150]);

                // clamp 確認: 0 → 1, 500 → 400
                engine.cmd_setoption(&[
                    "setoption",
                    "name",
                    "LS_BUCKET_EVAL_SCALE",
                    "value",
                    "0,500,100,100,100,100,100,100",
                ]);
                let search = engine.search.as_ref().expect("search exists");
                let scale = search.bucket_eval_scale();
                assert_eq!(scale[0], 1);
                assert_eq!(scale[1], 400);

                // 不正入力は無視して前回値維持
                engine.cmd_setoption(&[
                    "setoption",
                    "name",
                    "LS_BUCKET_EVAL_SCALE",
                    "value",
                    "bad,input",
                ]);
                let search = engine.search.as_ref().expect("search exists");
                let scale_after = search.bucket_eval_scale();
                // 前回値 (0→1 clamp, 500→400 clamp) が維持されていること
                assert_eq!(scale_after[0], 1);
                assert_eq!(scale_after[1], 400);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[serial]
    fn setoption_bucket_eval_scale_index_updates_search() {
        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let mut engine = UsiEngine::new();

                // 単一 index のみ更新、他は default 100 のまま
                engine.cmd_setoption(&[
                    "setoption",
                    "name",
                    "LS_BUCKET_EVAL_SCALE_3",
                    "value",
                    "120",
                ]);
                let search = engine.search.as_ref().expect("search exists");
                let scale = search.bucket_eval_scale();
                assert_eq!(scale[3], 120);
                for (i, &v) in scale.iter().enumerate() {
                    if i != 3 {
                        assert_eq!(v, 100, "index {i} should stay 100");
                    }
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[serial]
    fn setoption_bucket_interp_updates_global() {
        std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                use rshogi_core::nnue::{get_bucket_interp, set_bucket_interp};

                // テスト開始時に既定値 (OFF) へ戻す
                set_bucket_interp(0);

                let mut engine = UsiEngine::new();

                // 正常系: 20 を設定
                engine.cmd_setoption(&["setoption", "name", "LS_BUCKET_INTERP", "value", "20"]);
                assert_eq!(get_bucket_interp(), 20);

                // clamp 確認: 100 → 40
                engine.cmd_setoption(&["setoption", "name", "LS_BUCKET_INTERP", "value", "100"]);
                assert_eq!(get_bucket_interp(), 40);

                // 0 で OFF
                engine.cmd_setoption(&["setoption", "name", "LS_BUCKET_INTERP", "value", "0"]);
                assert_eq!(get_bucket_interp(), 0);

                // 不正入力は無視して前回値 (0) 維持
                engine.cmd_setoption(&["setoption", "name", "LS_BUCKET_INTERP", "value", "bad"]);
                assert_eq!(get_bucket_interp(), 0);

                // 他テストへの影響を避けるため復元
                set_bucket_interp(0);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
