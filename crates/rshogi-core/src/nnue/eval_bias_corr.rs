//! Eval Bias Correction (EBC) = **correction-history 型の静的評価補正** (2026-07-03)。
//!
//! bias miner (= `eval_bias_miner`) が測った「`search_score - static_eval` の系統ズレ」
//! (= bias) を **局面クラス別に** 補正する runtime-toggle 機構。 ODH ([`crate::nnue::odh`])
//! と同じく合算 `raw_total` に最後に適用するが、 ODH が「正側の楽観を縮めるだけ」なのに
//! 対し、 EBC は **符号込みの table** で優勢の過大評価 (= 下げる) も劣勢の過小評価 (= 上げる)
//! も両方補正できる汎用機構。 学習 graph 不要 (= section save 無し、 engine 内 table)。
//!
//! # class 軸
//!
//! `class = advantage_bucket[7] × phase_bucket[5]` = 35 class。
//!
//! - **advantage_bucket[7]**: 補正適用前の `raw_total` を cp 換算 (= `raw_total / fv_scale`)
//!   した手番側 cp の符号と magnitude で分類。 境界 (cp):
//!   `[-inf,-1200) / [-1200,-600) / [-600,-300) / [-300,+300] / (+300,+600] / (+600,+1200] / (+1200,+inf)`。
//!   → index 0..=6。 ★**中央 [-300,+300] (= index 3) と両端 |cp|>1200 (= index 0/6) は
//!   補正しない** ([`is_correctable`] が false)。 較正良好域 (中央) と危険域 (両端) を避ける設計。
//! - **phase_bucket[5]**: `game_ply` (= 手数) で `<=20 / 21-40 / 41-60 / 61-100 / 101+`
//!   → index 0..=4。 progress bucket でなく手数を使う (= material_phase 非依存で単純)。
//!
//! # 補正 table
//!
//! [`CORR_TABLE`] = `[AtomicI32; 35]` (= `phase_bucket * 7 + advantage_bucket`)。 値は
//! **cp 単位** の bias 補正量。 **初期値は全ゼロ** (= bit-exact、 後で実測 bias 値を注入)。
//! ホットパスは atomic の Relaxed load のみ (= lock/heap 無し)。 起動時に一度
//! [`load_from_str`] / [`load_from_file`] で固定 → 以降は読むだけ。
//!
//! # 適用式
//!
//! ```text
//! cp        = raw_total / fv_scale          // 補正前 cp (手番側)
//! adv       = advantage_bucket(cp)
//! if !is_correctable(adv): delta = 0
//! else:
//!   phase   = phase_bucket(game_ply)
//!   corr_cp = CORR_TABLE[phase * 7 + adv]   // cp 単位、 符号込み
//!   delta_cp = (EVAL_BIAS_CORR_SCALE / 64) * corr_cp
//!   delta_cp = clamp(delta_cp, ±EVAL_BIAS_CORR_CLAMP)   // 1 手あたり過大補正防止
//!   delta   = delta_cp * fv_scale           // cp → raw
//! corrected = raw_total + delta             // network 側で加算
//! ```
//!
//! # 符号規約
//!
//! bias = `search_score - static_eval`。
//! - static **過大** (= 優勢 over-optimism、 search が下方修正) → `corr_cp < 0` (= eval を下げる)。
//! - static **過小** (= 劣勢 over-pessimism、 search が上方修正) → `corr_cp > 0` (= eval を上げる)。
//!
//! table 値に符号を込めるため、 適用は **単純加算** で良い (= メインが符号込みで値を入れる)。
//!
//! # runtime toggle
//!
//! - `EVAL_BIAS_CORR_SCALE` (= spin、 既定 0 = 無効 = bit-exact)。 `SCALE_REF` (= 64) 基準の
//!   線形倍率。 0 の間は class 計算も skip し NPS も温存。
//! - `EVAL_BIAS_CORR_CLAMP` (= spin、 既定 300 cp)。 1 手あたりの `|delta_cp|` 上限。
//! - `EVAL_BIAS_CORR_FILE` (= string)。 `phase,adv,cp` 行の table を起動時に一度ロード。

use std::sync::atomic::{AtomicI32, Ordering};

/// advantage_bucket 数 (= 7)。
pub const EBC_NUM_ADV: usize = 7;
/// phase_bucket 数 (= 5)。
pub const EBC_NUM_PHASE: usize = 5;
/// 補正 table 総エントリ数 (= 35)。
pub const EBC_CLASS_COUNT: usize = EBC_NUM_PHASE * EBC_NUM_ADV;

/// `EVAL_BIAS_CORR_SCALE` の基準値 (= scale=64 で table 設計値、 線形倍率)。
pub const EBC_SCALE_REF: i64 = 64;

/// `EVAL_BIAS_CORR_SCALE` 初期値 (= 0 = 無効 = bit-exact)。
pub const EBC_OUTPUT_SCALE_DEFAULT: i32 = 0;
/// `EVAL_BIAS_CORR_CLAMP` 初期値 (= 300 cp、 1 手あたり過大補正の上限)。
pub const EBC_CLAMP_DEFAULT: i32 = 300;

/// USI option `EVAL_BIAS_CORR_SCALE` で runtime 上書き可能な補正全体倍率。
pub static EVAL_BIAS_CORR_SCALE: AtomicI32 = AtomicI32::new(EBC_OUTPUT_SCALE_DEFAULT);
/// USI option `EVAL_BIAS_CORR_CLAMP` (= cp 単位) で runtime 上書き可能な 1 手 clamp。
pub static EVAL_BIAS_CORR_CLAMP: AtomicI32 = AtomicI32::new(EBC_CLAMP_DEFAULT);

/// 補正 table (= cp 単位、 符号込み、 index = `phase * 7 + adv`)。 初期値全ゼロ。
///
/// ホットパスは Relaxed load のみ (= lock/heap 無し)。 起動時に一度注入 → 以降読むだけ。
// `[AtomicI32; N]` の const 初期化 (= 非 Copy のため inline const で各要素を生成、
// 名前付き interior-mutable const を避ける clippy 準拠イディオム)。
static CORR_TABLE: [AtomicI32; EBC_CLASS_COUNT] = [const { AtomicI32::new(0) }; EBC_CLASS_COUNT];

/// USI option 経由で補正全体倍率を更新する。
pub fn set_output_scale(scale: i32) {
    EVAL_BIAS_CORR_SCALE.store(scale, Ordering::Relaxed);
}

/// USI option 経由で 1 手あたり clamp (= cp) を更新する。
pub fn set_clamp(clamp_cp: i32) {
    EVAL_BIAS_CORR_CLAMP.store(clamp_cp, Ordering::Relaxed);
}

/// `(phase, adv)` の table index (= `phase * 7 + adv`)。
#[inline]
pub const fn class_index(phase: usize, adv: usize) -> usize {
    phase * EBC_NUM_ADV + adv
}

/// 補正 table の 1 セル (= cp) を設定する (= 注入 API)。 範囲外は無視。
pub fn set_corr(phase: usize, adv: usize, corr_cp: i32) {
    if phase < EBC_NUM_PHASE && adv < EBC_NUM_ADV {
        CORR_TABLE[class_index(phase, adv)].store(corr_cp, Ordering::Relaxed);
    }
}

/// 補正 table の 1 セル (= cp) を読む。
#[inline]
pub fn get_corr(phase: usize, adv: usize) -> i32 {
    CORR_TABLE[class_index(phase, adv)].load(Ordering::Relaxed)
}

/// 補正 table を全ゼロにリセットする (= 注入前クリア / test 用)。
pub fn clear_table() {
    for cell in CORR_TABLE.iter() {
        cell.store(0, Ordering::Relaxed);
    }
}

/// `phase,adv,cp` 行のテキストから table を注入する。
///
/// - 各行: `phase,adv,cp` (= カンマ区切り 3 整数)。 空行 / `#` 始まりコメントは無視。
/// - `phase` 0..=4、 `adv` 0..=6。 範囲外行は skip (= 静かに無視)。
/// - 既存 table はクリアせず **上書き** (= 呼び出し前に [`clear_table`] を呼ぶと純注入)。
///
/// # Errors
/// パース不能な非コメント行があれば行番号付きで `Err` を返す (= 起動時に検出)。
pub fn load_from_str(text: &str) -> Result<usize, String> {
    let mut loaded = 0usize;
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split(',');
        let phase = parse_field(it.next(), lineno, "phase")?;
        let adv = parse_field(it.next(), lineno, "adv")?;
        let cp = parse_field(it.next(), lineno, "cp")?;
        if it.next().is_some() {
            return Err(format!("line {}: 余分なフィールド ('{line}')", lineno + 1));
        }
        if !(0..EBC_NUM_PHASE as i32).contains(&phase) || !(0..EBC_NUM_ADV as i32).contains(&adv) {
            // 範囲外は skip (= 将来 bucket 数変更でも壊れない)。
            continue;
        }
        set_corr(phase as usize, adv as usize, cp);
        loaded += 1;
    }
    Ok(loaded)
}

/// 1 フィールドを i32 パースする (= [`load_from_str`] ヘルパ)。
fn parse_field(field: Option<&str>, lineno: usize, name: &str) -> Result<i32, String> {
    field
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("line {}: {name} 欠落", lineno + 1))?
        .parse::<i32>()
        .map_err(|e| format!("line {}: {name} パース失敗 ({e})", lineno + 1))
}

/// ファイルから table を注入する (= USI option `EVAL_BIAS_CORR_FILE` 経路)。
///
/// # Errors
/// ファイル読み込み / パース失敗時に `Err`。
pub fn load_from_file(path: &str) -> Result<usize, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("EVAL_BIAS_CORR_FILE '{path}' 読み込み失敗: {e}"))?;
    load_from_str(&text)
}

/// cp から advantage_bucket (= 0..=6) を返す。
///
/// 境界: `[-inf,-1200) / [-1200,-600) / [-600,-300) / [-300,+300] / (+300,+600] / (+600,+1200] / (+1200,+inf)`。
#[inline]
pub fn advantage_bucket(cp: i32) -> usize {
    if cp < -1200 {
        0
    } else if cp < -600 {
        1
    } else if cp < -300 {
        2
    } else if cp <= 300 {
        3
    } else if cp <= 600 {
        4
    } else if cp <= 1200 {
        5
    } else {
        6
    }
}

/// advantage_bucket が補正対象か (= 中央 index 3 と両端 index 0/6 は非対象)。
#[inline]
pub fn is_correctable(adv: usize) -> bool {
    matches!(adv, 1 | 2 | 4 | 5)
}

/// game_ply (= 手数) から phase_bucket (= 0..=4) を返す。
///
/// 境界 (手数): `<=20 / 21-40 / 41-60 / 61-100 / 101+`。
#[inline]
pub fn phase_bucket(game_ply: i32) -> usize {
    if game_ply <= 20 {
        0
    } else if game_ply <= 40 {
        1
    } else if game_ply <= 60 {
        2
    } else if game_ply <= 100 {
        3
    } else {
        4
    }
}

/// EBC 補正量 (= `raw_total` に **加える** raw 単位の delta) を返す。
///
/// `EVAL_BIAS_CORR_SCALE == 0` (= default) で即 0 → eval bit-exact (= 無効時無害)。
///
/// # Arguments
/// * `raw_total` - EBC 適用前の合算 raw eval (= fv_scale 適用前、 stm 視点、 ODH 適用後)
/// * `game_ply`  - 手数 (= `Position::game_ply()`)
/// * `fv_scale`  - 現在の fv_scale (= cp ⇔ raw 換算用)
///
/// # 仕様
/// - `advantage_bucket` が中央 (index 3) or 両端 (index 0/6) → 0 (= 補正しない)。
/// - `delta_cp = (scale/64) * corr[phase][adv]` を `±clamp` cp に clamp、 raw 換算して返す。
/// - `corr` の符号がそのまま delta の符号 (= static 過大なら負、 過小なら正)。
#[inline]
pub fn apply(raw_total: i32, game_ply: i32, fv_scale: i32) -> i32 {
    let scale = EVAL_BIAS_CORR_SCALE.load(Ordering::Relaxed);
    // scale 0 → 補正なし = bit-exact (= class 計算も skip して NPS 温存)。
    if scale == 0 {
        return 0;
    }
    let fv = fv_scale.max(1);
    let cp = raw_total / fv;
    let adv = advantage_bucket(cp);
    if !is_correctable(adv) {
        return 0;
    }
    let phase = phase_bucket(game_ply);
    let corr_cp = get_corr(phase, adv) as i64;
    if corr_cp == 0 {
        return 0;
    }
    // delta_cp = corr_cp * scale / 64 (= i64 中間で overflow 回避)。
    let delta_cp = corr_cp * (scale as i64) / EBC_SCALE_REF;
    // 1 手あたり ±clamp cp に制限 (= 過大補正で探索木を壊さない)。
    let clamp_cp = EVAL_BIAS_CORR_CLAMP.load(Ordering::Relaxed).max(0) as i64;
    let delta_cp = delta_cp.clamp(-clamp_cp, clamp_cp);
    // cp → raw。 i32 範囲に収める (= 異常設定でも overflow しない)。
    (delta_cp * fv as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // runtime option / table は global static のため、 これを書き換える test を並列実行
    // すると値が race する。 option/table を触る test は本 lock で直列化する。
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock_options() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// option/table を default に戻す (= 他 test への副作用防止)。
    fn reset() {
        set_output_scale(EBC_OUTPUT_SCALE_DEFAULT);
        set_clamp(EBC_CLAMP_DEFAULT);
        clear_table();
    }

    #[test]
    fn test_class_index_range() {
        assert_eq!(class_index(0, 0), 0);
        assert_eq!(class_index(4, 6), EBC_CLASS_COUNT - 1);
        assert_eq!(EBC_CLASS_COUNT, 35);
    }

    #[test]
    fn test_advantage_bucket_boundaries() {
        // 境界値ちょうどの帰属を確認 (= 半開区間の仕様)。
        assert_eq!(advantage_bucket(-5000), 0);
        assert_eq!(advantage_bucket(-1201), 0);
        assert_eq!(advantage_bucket(-1200), 1);
        assert_eq!(advantage_bucket(-601), 1);
        assert_eq!(advantage_bucket(-600), 2);
        assert_eq!(advantage_bucket(-301), 2);
        assert_eq!(advantage_bucket(-300), 3);
        assert_eq!(advantage_bucket(0), 3);
        assert_eq!(advantage_bucket(300), 3);
        assert_eq!(advantage_bucket(301), 4);
        assert_eq!(advantage_bucket(600), 4);
        assert_eq!(advantage_bucket(601), 5);
        assert_eq!(advantage_bucket(1200), 5);
        assert_eq!(advantage_bucket(1201), 6);
        assert_eq!(advantage_bucket(5000), 6);
    }

    #[test]
    fn test_correctable_set() {
        assert!(!is_correctable(0)); // 両端
        assert!(is_correctable(1));
        assert!(is_correctable(2));
        assert!(!is_correctable(3)); // 中央
        assert!(is_correctable(4));
        assert!(is_correctable(5));
        assert!(!is_correctable(6)); // 両端
    }

    #[test]
    fn test_phase_bucket_boundaries() {
        assert_eq!(phase_bucket(1), 0);
        assert_eq!(phase_bucket(20), 0);
        assert_eq!(phase_bucket(21), 1);
        assert_eq!(phase_bucket(40), 1);
        assert_eq!(phase_bucket(41), 2);
        assert_eq!(phase_bucket(60), 2);
        assert_eq!(phase_bucket(61), 3);
        assert_eq!(phase_bucket(100), 3);
        assert_eq!(phase_bucket(101), 4);
        assert_eq!(phase_bucket(500), 4);
    }

    #[test]
    fn test_scale_zero_is_bit_exact() {
        let _g = lock_options();
        reset();
        // 全ゼロ table + scale0 で常に 0 (= 注入があっても scale0 なら不動)。
        set_corr(2, 4, -100);
        for raw in [-20000, -3000, -800, 0, 800, 3000, 20000] {
            for ply in [1, 30, 80, 200] {
                assert_eq!(apply(raw, ply, 14), 0);
            }
        }
        reset();
    }

    #[test]
    fn test_zero_table_is_bit_exact_even_with_scale() {
        let _g = lock_options();
        reset();
        set_output_scale(64);
        // table 全ゼロなら scale>0 でも 0。
        for raw in [-20000, 800 * 14, 20000] {
            assert_eq!(apply(raw, 30, 14), 0);
        }
        reset();
    }

    #[test]
    fn test_center_and_extremes_untouched() {
        let _g = lock_options();
        reset();
        set_output_scale(64);
        let fv = 14;
        // 全 class に補正を入れても、 中央 (adv3) と両端 (adv0/6) は不動。
        for phase in 0..EBC_NUM_PHASE {
            for adv in 0..EBC_NUM_ADV {
                set_corr(phase, adv, -100);
            }
        }
        // adv0 (< -1200cp)、 adv3 (±150 以内)、 adv6 (> 1200cp) は 0。
        assert_eq!(apply(-2000 * fv, 30, fv), 0, "adv0 は不動");
        assert_eq!(apply(0, 30, fv), 0, "adv3 は不動");
        assert_eq!(apply(100 * fv, 30, fv), 0, "adv3 は不動");
        assert_eq!(apply(2000 * fv, 30, fv), 0, "adv6 は不動");
        reset();
    }

    #[test]
    fn test_advantage_negative_correction_lowers_eval() {
        let _g = lock_options();
        reset();
        set_output_scale(64);
        let fv = 14;
        // 中盤 (ply30 = phase1) × 優勢 +600..1200 (= adv5) に -100cp を注入。
        set_corr(1, 5, -100);
        let raw = 800 * fv; // cp=800 → adv5
        assert_eq!(advantage_bucket(800), 5);
        assert_eq!(phase_bucket(30), 1);
        let delta = apply(raw, 30, fv);
        // scale64 → delta_cp = -100、 clamp300 内 → raw = -100*14。
        assert_eq!(delta, -100 * fv);
        assert!(delta < 0, "優勢 static 過大の補正は eval を下げる");
        reset();
    }

    #[test]
    fn test_disadvantage_positive_correction_raises_eval() {
        let _g = lock_options();
        reset();
        set_output_scale(64);
        let fv = 14;
        // 中盤 (ply50 = phase2) × 劣勢 -600..-300 (= adv2) に +80cp を注入。
        set_corr(2, 2, 80);
        let raw = -450 * fv; // cp=-450 → adv2
        assert_eq!(advantage_bucket(-450), 2);
        assert_eq!(phase_bucket(50), 2);
        let delta = apply(raw, 50, fv);
        assert_eq!(delta, 80 * fv);
        assert!(delta > 0, "劣勢 static 過小の補正は eval を上げる");
        reset();
    }

    #[test]
    fn test_scale_linear() {
        let _g = lock_options();
        reset();
        let fv = 14;
        set_corr(1, 4, -60);
        let raw = 400 * fv; // adv4, phase1
        set_output_scale(64);
        let v64 = apply(raw, 30, fv);
        set_output_scale(128);
        let v128 = apply(raw, 30, fv);
        set_output_scale(32);
        let v32 = apply(raw, 30, fv);
        // 基準 64 に対する線形倍率 (= floor 込みで exact)。
        assert_eq!(v64, -60 * fv);
        assert_eq!(v128, -120 * fv);
        assert_eq!(v32, -30 * fv);
        reset();
    }

    #[test]
    fn test_clamp_limits_delta() {
        let _g = lock_options();
        reset();
        set_output_scale(64);
        set_clamp(50); // 1 手 50cp まで
        let fv = 14;
        set_corr(1, 5, -400); // 400cp の補正だが clamp50
        let delta = apply(800 * fv, 30, fv);
        assert_eq!(delta, -50 * fv, "clamp で ±50cp に制限");
        reset();
    }

    #[test]
    fn test_load_from_str() {
        let _g = lock_options();
        reset();
        let text = "\
# phase,adv,cp
1,5,-120
2,2,90
  4 , 4 , -30
99,5,-999
1,3,-50
";
        // 99,5 は範囲外 skip、 1,3 は adv3 も table には入る (= 適用側で無視)。
        let n = load_from_str(text).unwrap();
        assert_eq!(n, 4, "有効 4 行 (範囲外 99 のみ skip)");
        assert_eq!(get_corr(1, 5), -120);
        assert_eq!(get_corr(2, 2), 90);
        assert_eq!(get_corr(4, 4), -30);
        assert_eq!(get_corr(1, 3), -50); // 格納はされるが adv3 は apply で無視
        reset();
    }

    #[test]
    fn test_load_from_str_error() {
        let _g = lock_options();
        reset();
        assert!(load_from_str("1,5,abc").is_err());
        assert!(load_from_str("1,5").is_err());
        assert!(load_from_str("1,5,-10,extra").is_err());
        reset();
    }

    #[test]
    fn test_adv_bucket_matches_raw_total_cp() {
        // apply の内部 adv 分類が raw_total の cp と一致することを確認。
        let _g = lock_options();
        reset();
        set_output_scale(64);
        let fv = 14;
        // 全 adv に一律 +10cp、 phase1 (ply30)。
        for adv in 0..EBC_NUM_ADV {
            set_corr(1, adv, 10);
        }
        // adv1 (-1200..-600): cp=-900。
        assert_eq!(advantage_bucket(-900), 1);
        assert_eq!(apply(-900 * fv, 30, fv), 10 * fv);
        // adv4 (+150..+600): cp=+400。
        assert_eq!(advantage_bucket(400), 4);
        assert_eq!(apply(400 * fv, 30, fv), 10 * fv);
        reset();
    }
}
