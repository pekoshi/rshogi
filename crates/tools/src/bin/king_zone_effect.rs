//! king_zone_effect: 玉zone利き統計の識別性検証ツール
//!
//! NNUE 入力特徴 (tango型 玉zone利き) を本実装する前の識別性検証用。
//!
//! 各局面で attacker = 手番側 (side_to_move)、defender = 相手 (優勢だが
//! 打ち込まれる側) とし、defender の玉を中心とする 5x5 = 25 マスの zone に
//! 対して以下 4 指標を計算する:
//!   - atk_eff:     Σ board_effect(attacker, sq)  相手の玉zoneへの利き総和
//!   - def_eff:     Σ board_effect(defender, sq)  自軍の玉zone守り
//!   - drop_danger: zone の空マスのうち atk_eff>=1 かつ def_eff==0 の数
//!     (= 相手が安全に打ち込める危険マス数)
//!   - atk_hand:    attacker の持駒総数
//!
//! 入力ファイルは 1 行 = `sfen<TAB>tag` 形式。tag ごとに 4 指標の
//! mean / median を集計して表出力し、最後に vuln と ctrl の比率を示す。
//!
//! ★実装注意: board_effect(color, sq) は `layerstack-only` feature では
//! コンパイル時無効化されるため、このツールは `layerstack-only` を付けずに
//! ビルドすること。Position::set_sfen は末尾で recompute_board_effects() を
//! 呼ぶため board_effects は populate 済み (dirty ではない)。

use std::collections::BTreeMap;
use std::env;
use std::fs;

use anyhow::{Context, Result, bail};

use rshogi_core::position::Position;
use rshogi_core::types::{File, PieceType, Rank, Square};

/// 1 局面分の指標。
#[derive(Clone, Copy)]
struct Metrics {
    atk_eff: f64,
    def_eff: f64,
    drop_danger: f64,
    atk_hand: f64,
}

/// 玉を中心とする 5x5 zone の升目を返す。
///
/// zone は玉の筋 ±2 / 段 ±2 の 25 マス。盤外オフセットはスキップ (clamp せず
/// 除外) する。
fn zone_squares(ksq: Square) -> Vec<Square> {
    let kf = ksq.file().index() as i32;
    let kr = ksq.rank().index() as i32;
    let mut v = Vec::with_capacity(25);
    for df in -2..=2 {
        let f = kf + df;
        if !(0..9).contains(&f) {
            continue;
        }
        for dr in -2..=2 {
            let r = kr + dr;
            if !(0..9).contains(&r) {
                continue;
            }
            // f, r は 0..=8 に収まる。
            let file = File::from_u8(f as u8).expect("file in range");
            let rank = Rank::from_u8(r as u8).expect("rank in range");
            v.push(Square::new(file, rank));
        }
    }
    v
}

/// 玉から升目へのチェビシェフ距離 (max(|筋差|, |段差|))。
fn chebyshev(ksq: Square, sq: Square) -> i32 {
    let df = (sq.file().index() as i32 - ksq.file().index() as i32).abs();
    let dr = (sq.rank().index() as i32 - ksq.rank().index() as i32).abs();
    df.max(dr)
}

/// チェビシェフ距離から距離バンド index を返す。
///   0 = band A (cheby <= 2, 従来の 5x5)
///   1 = band B (cheby 3-4, 外側リング)
///   2 = band C (cheby >= 5, 遠方)
fn band_of(cheby: i32) -> usize {
    if cheby <= 2 {
        0
    } else if cheby <= 4 {
        1
    } else {
        2
    }
}

const BAND_NAMES: [&str; 3] = ["A(cheby<=2)", "B(cheby3-4)", "C(cheby>=5)"];

/// defender 玉を中心とする 5x5 zone の 4 指標を計算する。
fn compute_metrics(pos: &Position) -> Metrics {
    let attacker = pos.side_to_move();
    let defender = !attacker;
    let ksq = pos.king_square(defender);

    let mut atk_eff: u32 = 0;
    let mut def_eff: u32 = 0;
    let mut drop_danger: u32 = 0;

    for sq in zone_squares(ksq) {
        let ae = pos.board_effect(attacker, sq) as u32;
        let de = pos.board_effect(defender, sq) as u32;
        atk_eff += ae;
        def_eff += de;

        let empty = pos.piece_on(sq).is_none();
        if empty && ae >= 1 && de == 0 {
            drop_danger += 1;
        }
    }

    // attacker の持駒総数 (歩〜飛の 7 種)。
    let hand = pos.hand(attacker);
    let mut atk_hand: u32 = 0;
    for &pt in &PieceType::HAND_PIECES {
        atk_hand += hand.count(pt);
    }

    Metrics {
        atk_eff: atk_eff as f64,
        def_eff: def_eff as f64,
        drop_danger: drop_danger as f64,
        atk_hand: atk_hand as f64,
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// tag ごとに集計した統計 (mean/median × 4 指標)。
struct Aggregate {
    count: usize,
    atk_eff_mean: f64,
    atk_eff_med: f64,
    def_eff_mean: f64,
    def_eff_med: f64,
    drop_danger_mean: f64,
    drop_danger_med: f64,
    atk_hand_mean: f64,
    atk_hand_med: f64,
}

fn aggregate(rows: &[Metrics]) -> Aggregate {
    let atk_eff: Vec<f64> = rows.iter().map(|m| m.atk_eff).collect();
    let def_eff: Vec<f64> = rows.iter().map(|m| m.def_eff).collect();
    let drop_danger: Vec<f64> = rows.iter().map(|m| m.drop_danger).collect();
    let atk_hand: Vec<f64> = rows.iter().map(|m| m.atk_hand).collect();
    Aggregate {
        count: rows.len(),
        atk_eff_mean: mean(&atk_eff),
        atk_eff_med: median(&atk_eff),
        def_eff_mean: mean(&def_eff),
        def_eff_med: median(&def_eff),
        drop_danger_mean: mean(&drop_danger),
        drop_danger_med: median(&drop_danger),
        atk_hand_mean: mean(&atk_hand),
        atk_hand_med: median(&atk_hand),
    }
}

/// 1 ファイルを読み、各行を (tag -> Metrics 列) に振り分ける。
/// 戻り値は (tag -> Vec<Metrics>, 読み込み総数, パース失敗数)。
fn process_file(path: &str) -> Result<(BTreeMap<String, Vec<Metrics>>, usize, usize)> {
    let text =
        fs::read_to_string(path).with_context(|| format!("入力ファイル読み込み失敗: {path}"))?;
    let mut by_tag: BTreeMap<String, Vec<Metrics>> = BTreeMap::new();
    let mut total = 0usize;
    let mut skipped = 0usize;

    for line in text.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        // `sfen<TAB>tag`。sfen 側にも空白が含まれるので最初の TAB で分割。
        let (sfen, tag) = match line.split_once('\t') {
            Some((s, t)) => (s.trim(), t.trim().to_string()),
            None => {
                // TAB が無ければ tag 無しとして "(untagged)" 扱い。
                (line.trim(), "(untagged)".to_string())
            }
        };

        let mut pos = Position::new();
        if pos.set_sfen(sfen).is_err() {
            skipped += 1;
            continue;
        }
        let m = compute_metrics(&pos);
        by_tag.entry(tag).or_default().push(m);
    }

    Ok((by_tag, total, skipped))
}

fn print_table(label: &str, tag: &str, agg: &Aggregate) {
    println!(
        "{label:<28} tag={tag:<10} n={n:<5} \
         atk_eff[mean={aem:>6.2} med={aemd:>5.1}] \
         def_eff[mean={dem:>6.2} med={demd:>5.1}] \
         drop_danger[mean={ddm:>5.2} med={ddmd:>4.1}] \
         atk_hand[mean={ahm:>5.2} med={ahmd:>4.1}]",
        label = label,
        tag = tag,
        n = agg.count,
        aem = agg.atk_eff_mean,
        aemd = agg.atk_eff_med,
        dem = agg.def_eff_mean,
        demd = agg.def_eff_med,
        ddm = agg.drop_danger_mean,
        ddmd = agg.drop_danger_med,
        ahm = agg.atk_hand_mean,
        ahmd = agg.atk_hand_med,
    );
}

// ============================================================================
//  drop-square 解析モード (`sfen<TAB>drop_sq<TAB>tag`)
// ============================================================================

/// 実打ち込みマス 1 件分の解析結果。
#[derive(Clone, Copy)]
struct DropRow {
    /// drop_sq が defender の 5x5 玉zone に入っているか。
    in_zone: bool,
    /// 玉から drop_sq へのチェビシェフ距離。
    cheby: i32,
    /// board_effect(attacker, drop_sq)。
    atk_at: f64,
    /// board_effect(defender, drop_sq)。
    def_at: f64,
    /// drop_danger 定義 (zone内∧空∧atk>=1∧def==0) を満たすか。
    flagged: bool,
    /// 元 sfen で drop_sq が空マスだったか (打ち込み前局面の想定確認)。
    drop_empty: bool,
    /// 玉zone内の空マス全体の平均 attacker_effect (空マスが無ければ None)。
    zone_empty_avg_atk: Option<f64>,
}

/// 実打ち込みマス drop_sq について 5 指標の素データを計算する。
fn analyze_drop(pos: &Position, drop_sq: Square) -> DropRow {
    let attacker = pos.side_to_move();
    let defender = !attacker;
    let ksq = pos.king_square(defender);

    let cheby = chebyshev(ksq, drop_sq);
    let in_zone = cheby <= 2;
    let atk_at = pos.board_effect(attacker, drop_sq) as f64;
    let def_at = pos.board_effect(defender, drop_sq) as f64;
    let drop_empty = pos.piece_on(drop_sq).is_none();
    let flagged = in_zone && drop_empty && atk_at >= 1.0 && def_at == 0.0;

    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    for sq in zone_squares(ksq) {
        if pos.piece_on(sq).is_none() {
            sum += pos.board_effect(attacker, sq) as f64;
            cnt += 1;
        }
    }
    let zone_empty_avg_atk = if cnt > 0 {
        Some(sum / cnt as f64)
    } else {
        None
    };

    DropRow {
        in_zone,
        cheby,
        atk_at,
        def_at,
        flagged,
        drop_empty,
        zone_empty_avg_atk,
    }
}

/// ファイル先頭の非空行が `sfen<TAB>drop_sq<TAB>...` 形式 (3列以上かつ
/// 2列目が USI square) なら drop-square モードと判定する。
fn is_dropsq_file(text: &str) -> bool {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        return fields.len() >= 3 && Square::from_usi(fields[1].trim()).is_some();
    }
    false
}

/// drop-square ファイルを読み、各行を DropRow に変換する。
/// 戻り値は (rows, 行数, スキップ数)。
fn process_dropsq_file(path: &str) -> Result<(Vec<DropRow>, usize, usize)> {
    let text =
        fs::read_to_string(path).with_context(|| format!("入力ファイル読み込み失敗: {path}"))?;
    let mut rows: Vec<DropRow> = Vec::new();
    let mut total = 0usize;
    let mut skipped = 0usize;

    for line in text.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        // `sfen<TAB>drop_sq<TAB>tag`。sfen 内に空白があるので TAB 区切り。
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 2 {
            skipped += 1;
            continue;
        }
        let sfen = fields[0].trim();
        let dropsq_str = fields[1].trim();

        let mut pos = Position::new();
        if pos.set_sfen(sfen).is_err() {
            skipped += 1;
            continue;
        }
        let drop_sq = match Square::from_usi(dropsq_str) {
            Some(s) => s,
            None => {
                skipped += 1;
                continue;
            }
        };
        rows.push(analyze_drop(&pos, drop_sq));
    }

    Ok((rows, total, skipped))
}

/// drop-square 解析結果を集計して報告する。
fn report_dropsq(path: &str, rows: &[DropRow], total: usize, skipped: usize) {
    let parsed = total - skipped;
    println!(
        "\n=== drop-square 解析: {path}  (行数={total}, パース成功={parsed}, スキップ={skipped}) ==="
    );
    if rows.is_empty() {
        println!("  (有効な行がありません)");
        return;
    }
    let n = rows.len();
    let pct = |c: usize| -> f64 { 100.0 * c as f64 / n as f64 };

    // 1. in_zone 率
    let in_zone_cnt = rows.iter().filter(|r| r.in_zone).count();
    // 3. flagged 率
    let flagged_cnt = rows.iter().filter(|r| r.flagged).count();
    // drop_empty 率 (サニティ)
    let empty_cnt = rows.iter().filter(|r| r.drop_empty).count();

    // 2. drop_sq における attacker/defender 利き
    let atk_at: Vec<f64> = rows.iter().map(|r| r.atk_at).collect();
    let def_at: Vec<f64> = rows.iter().map(|r| r.def_at).collect();

    // 4. drop_sq atk_eff vs 玉zone空マス平均 atk_eff
    let zone_avgs: Vec<f64> = rows.iter().filter_map(|r| r.zone_empty_avg_atk).collect();
    // 玉zone内(=比較可能)の drop_sq のみで per-position 差分も出す。
    let in_zone_pairs: Vec<(f64, f64)> = rows
        .iter()
        .filter(|r| r.in_zone)
        .filter_map(|r| r.zone_empty_avg_atk.map(|z| (r.atk_at, z)))
        .collect();
    let diffs: Vec<f64> = in_zone_pairs.iter().map(|(a, z)| a - z).collect();

    println!("  1) in_zone 率        : {in_zone_cnt}/{n} = {:.1}%", pct(in_zone_cnt));
    println!(
        "  2) drop_sq atk_eff   : mean={:.3} median={:.1}",
        mean(&atk_at),
        median(&atk_at)
    );
    println!(
        "     drop_sq def_eff   : mean={:.3} median={:.1}",
        mean(&def_at),
        median(&def_at)
    );
    println!("  3) flagged 率        : {flagged_cnt}/{n} = {:.1}%", pct(flagged_cnt));
    println!("     (drop_sq が空マスだった件: {empty_cnt}/{n} = {:.1}%)", pct(empty_cnt));
    println!(
        "  4) drop_sq atk_eff mean={:.3}  vs  玉zone空マス平均 atk_eff mean={:.3}",
        mean(&atk_at),
        mean(&zone_avgs)
    );
    if !diffs.is_empty() {
        println!(
            "     [in_zone {} 件] per-position (drop_sq - zone空マス平均) mean={:.3} median={:.1}",
            diffs.len(),
            mean(&diffs),
            median(&diffs)
        );
    }

    // 5. zone 外 drop の chebyshev 距離分布
    let out_rows: Vec<&DropRow> = rows.iter().filter(|r| !r.in_zone).collect();
    println!("  5) 玉zone外 drop の chebyshev 距離分布 ({} 件):", out_rows.len());
    if out_rows.is_empty() {
        println!("     (全 drop が玉zone内)");
    } else {
        let mut dist: BTreeMap<i32, usize> = BTreeMap::new();
        for r in &out_rows {
            *dist.entry(r.cheby).or_default() += 1;
        }
        for (d, c) in &dist {
            let cover = if *d <= 2 { 5 } else { 2 * d + 1 };
            println!("     cheby={d} : {c} 件  (zone を {cover}x{cover} に広げれば捕捉)");
        }
    }
}

// ============================================================================
//  距離バンド解析 (band A/B/C: 玉至近 / 外側リング / 遠方)
// ============================================================================

/// 1 局面分の距離バンド別データ (index 0=A, 1=B, 2=C)。
#[derive(Clone, Copy)]
struct BandRow {
    /// バンドの空マスへの attacker 利き総和。
    atk_eff_band: [f64; 3],
    /// バンドで attacker 利き>=1 の空マス数 (= "支配空マス"数)。
    ctrl_empty_band: [f64; 3],
    /// バンドの空マス総数 (per-empty 平均 baseline 算出用)。
    empty_cnt_band: [f64; 3],
}

/// 全 81 マスを距離バンドに振り分け、空マスの attacker 利きを集計する。
fn compute_band_row(pos: &Position) -> BandRow {
    let attacker = pos.side_to_move();
    let defender = !attacker;
    let ksq = pos.king_square(defender);

    let mut atk_eff = [0.0f64; 3];
    let mut ctrl_empty = [0.0f64; 3];
    let mut empty_cnt = [0.0f64; 3];

    for i in 0..Square::NUM as u8 {
        let sq = Square::from_u8(i).expect("valid square index");
        if pos.piece_on(sq).is_none() {
            let b = band_of(chebyshev(ksq, sq));
            let ae = pos.board_effect(attacker, sq) as f64;
            atk_eff[b] += ae;
            empty_cnt[b] += 1.0;
            if ae >= 1.0 {
                ctrl_empty[b] += 1.0;
            }
        }
    }

    BandRow {
        atk_eff_band: atk_eff,
        ctrl_empty_band: ctrl_empty,
        empty_cnt_band: empty_cnt,
    }
}

/// tag ファイル (`sfen<TAB>tag`) を読み、各行を BandRow に変換する。
fn process_band_file(path: &str) -> Result<(BTreeMap<String, Vec<BandRow>>, usize, usize)> {
    let text =
        fs::read_to_string(path).with_context(|| format!("入力ファイル読み込み失敗: {path}"))?;
    let mut by_tag: BTreeMap<String, Vec<BandRow>> = BTreeMap::new();
    let mut total = 0usize;
    let mut skipped = 0usize;

    for line in text.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        let (sfen, tag) = match line.split_once('\t') {
            Some((s, t)) => (s.trim(), t.trim().to_string()),
            None => (line.trim(), "(untagged)".to_string()),
        };
        let mut pos = Position::new();
        if pos.set_sfen(sfen).is_err() {
            skipped += 1;
            continue;
        }
        by_tag.entry(tag).or_default().push(compute_band_row(&pos));
    }

    Ok((by_tag, total, skipped))
}

/// ctrl 等の baseline: バンド別「空マス 1 マスあたりの平均 attacker 利き」。
/// per-position で atk_eff_band/empty_cnt_band を取り、局面平均する。
fn band_empty_mean_atk(rows: &[BandRow]) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for (b, slot) in out.iter_mut().enumerate() {
        let per: Vec<f64> = rows
            .iter()
            .filter(|r| r.empty_cnt_band[b] > 0.0)
            .map(|r| r.atk_eff_band[b] / r.empty_cnt_band[b])
            .collect();
        *slot = mean(&per);
    }
    out
}

/// バンド別の per-position 集計 (atk_eff_band mean と ctrl_empty_band mean)。
fn band_aggregate(rows: &[BandRow]) -> ([f64; 3], [f64; 3]) {
    let mut atk = [0.0f64; 3];
    let mut ce = [0.0f64; 3];
    for b in 0..3 {
        let a: Vec<f64> = rows.iter().map(|r| r.atk_eff_band[b]).collect();
        let c: Vec<f64> = rows.iter().map(|r| r.ctrl_empty_band[b]).collect();
        atk[b] = mean(&a);
        ce[b] = mean(&c);
    }
    (atk, ce)
}

/// バンド別 vuln vs ctrl 比率テーブルを報告する。
fn report_band_table(band_by_tag: &BTreeMap<String, Vec<BandRow>>) {
    println!("\n=== 距離バンド別 vuln vs ctrl (per-position mean) ===");
    let mut aggs: BTreeMap<String, ([f64; 3], [f64; 3])> = BTreeMap::new();
    for (tag, rows) in band_by_tag {
        let (atk, ce) = band_aggregate(rows);
        aggs.insert(tag.clone(), (atk, ce));
        println!("  [{tag}] n={}", rows.len());
        for b in 0..3 {
            println!(
                "    band {:<12}: atk_eff_band mean={:>7.3}  atk_ctrl_empty_band mean={:>6.3}",
                BAND_NAMES[b], atk[b], ce[b]
            );
        }
    }

    if let (Some((va, vc)), Some((ca, cc))) = (aggs.get("vuln"), aggs.get("ctrl")) {
        let ratio = |a: f64, b: f64| -> String {
            if b.abs() < 1e-9 {
                format!("{a:.3}/0(inf)")
            } else {
                format!("{:.3}x", a / b)
            }
        };
        println!("  --- 比率 (vuln/ctrl) ---");
        for b in 0..3 {
            println!(
                "    band {:<12}: atk_eff_band vuln={:>7.3} ctrl={:>7.3} ratio={:<9}  \
                 atk_ctrl_empty_band vuln={:>6.3} ctrl={:>6.3} ratio={}",
                BAND_NAMES[b],
                va[b],
                ca[b],
                ratio(va[b], ca[b]),
                vc[b],
                cc[b],
                ratio(vc[b], cc[b]),
            );
        }
    } else {
        println!("  (vuln と ctrl の両 tag が揃わないため比率は省略)");
    }
}

/// 実打ち込みマスを in/out zone とバンドで分離報告し、
/// out_zone マスを ctrl 同バンドの空マス平均 attacker 利きと対比する。
fn report_dropsq_bands(rows: &[DropRow], ctrl_baseline: Option<[f64; 3]>) {
    println!("\n=== drop-square in/out 分離 + ctrl baseline 対比 ===");
    if rows.is_empty() {
        println!("  (有効な行がありません)");
        return;
    }
    let mean_atk = |v: &[&DropRow]| mean(&v.iter().map(|r| r.atk_at).collect::<Vec<_>>());
    let mean_def = |v: &[&DropRow]| mean(&v.iter().map(|r| r.def_at).collect::<Vec<_>>());

    let in_rows: Vec<&DropRow> = rows.iter().filter(|r| r.in_zone).collect();
    let out_rows: Vec<&DropRow> = rows.iter().filter(|r| !r.in_zone).collect();

    println!(
        "  in_zone (cheby<=2, {:>2}件): drop_sq atk_eff mean={:.3}  def_eff mean={:.3}",
        in_rows.len(),
        mean_atk(&in_rows),
        mean_def(&in_rows)
    );
    println!(
        "  out_zone(cheby>=3, {:>2}件): drop_sq atk_eff mean={:.3}  def_eff mean={:.3}",
        out_rows.len(),
        mean_atk(&out_rows),
        mean_def(&out_rows)
    );

    println!("  --- out_zone を band別に分解し ctrl 同バンド空マス平均と対比 ---");
    for b in [1usize, 2usize] {
        let band_rows: Vec<&DropRow> = rows.iter().filter(|r| band_of(r.cheby) == b).collect();
        if band_rows.is_empty() {
            continue;
        }
        let atk = mean_atk(&band_rows);
        match ctrl_baseline {
            Some(bl) => println!(
                "    band {:<12} ({:>2}件): drop_sq atk_eff mean={:.3}  vs ctrl空マス平均={:.3}  差={:+.3}",
                BAND_NAMES[b],
                band_rows.len(),
                atk,
                bl[b],
                atk - bl[b]
            ),
            None => println!(
                "    band {:<12} ({:>2}件): drop_sq atk_eff mean={:.3}  (ctrl baseline なし)",
                BAND_NAMES[b],
                band_rows.len(),
                atk
            ),
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        bail!(
            "usage: king_zone_effect <file1> [file2 ...]\n  \
             各ファイルは 1 行 = `sfen<TAB>tag` (集計モード) または\n  \
             `sfen<TAB>drop_sq<TAB>tag` (drop-square 解析モード、自動判別)"
        );
    }

    // tag -> 全ファイル横断の Metrics 列。
    let mut global_by_tag: BTreeMap<String, Vec<Metrics>> = BTreeMap::new();
    let mut tag_files: Vec<&String> = Vec::new();
    let mut dropsq_files: Vec<&String> = Vec::new();
    for path in &args {
        let text = fs::read_to_string(path)
            .with_context(|| format!("入力ファイル読み込み失敗: {path}"))?;
        if is_dropsq_file(&text) {
            dropsq_files.push(path);
        } else {
            tag_files.push(path);
        }
    }

    // ---- tag ファイルの距離バンド集計 (ctrl baseline を先に用意) ----
    let mut global_band_by_tag: BTreeMap<String, Vec<BandRow>> = BTreeMap::new();
    for path in &tag_files {
        let (by_tag, _total, _skipped) = process_band_file(path)?;
        for (tag, rows) in by_tag {
            global_band_by_tag.entry(tag).or_default().extend(rows);
        }
    }
    let ctrl_baseline = global_band_by_tag.get("ctrl").map(|rows| band_empty_mean_atk(rows));

    // ---- drop-square 解析モード ----
    for path in &dropsq_files {
        let (rows, total, skipped) = process_dropsq_file(path)?;
        report_dropsq(path, &rows, total, skipped);
        report_dropsq_bands(&rows, ctrl_baseline);
    }

    if tag_files.is_empty() {
        return Ok(());
    }

    // ---- 従来の tag 集計モード ----
    println!("\n=== per-file / per-tag 集計 ===");
    for path in &tag_files {
        let (by_tag, total, skipped) = process_file(path)?;
        let parsed = total - skipped;
        println!("--- {path}  (行数={total}, パース成功={parsed}, スキップ={skipped}) ---");
        for (tag, rows) in &by_tag {
            let agg = aggregate(rows);
            print_table("(file)", tag, &agg);
            global_by_tag.entry(tag.clone()).or_default().extend(rows.iter().copied());
        }
    }

    println!("\n=== tag 横断集計 (全ファイル合算) ===");
    let mut aggs: BTreeMap<String, Aggregate> = BTreeMap::new();
    for (tag, rows) in &global_by_tag {
        let agg = aggregate(rows);
        print_table("(all)", tag, &agg);
        aggs.insert(tag.clone(), agg);
    }

    // vuln / ctrl の比率 (両方存在する場合のみ)。
    if let (Some(v), Some(c)) = (aggs.get("vuln"), aggs.get("ctrl")) {
        println!("\n=== vuln vs ctrl 比率 (mean, vuln/ctrl) ===");
        let ratio = |a: f64, b: f64| -> String {
            if b.abs() < 1e-9 {
                format!("{a:.3} / 0 (=inf)")
            } else {
                format!("{:.3}x", a / b)
            }
        };
        println!(
            "  atk_eff:     vuln={:.3}  ctrl={:.3}  ratio={}",
            v.atk_eff_mean,
            c.atk_eff_mean,
            ratio(v.atk_eff_mean, c.atk_eff_mean)
        );
        println!(
            "  def_eff:     vuln={:.3}  ctrl={:.3}  ratio={}",
            v.def_eff_mean,
            c.def_eff_mean,
            ratio(v.def_eff_mean, c.def_eff_mean)
        );
        println!(
            "  drop_danger: vuln={:.3}  ctrl={:.3}  ratio={}",
            v.drop_danger_mean,
            c.drop_danger_mean,
            ratio(v.drop_danger_mean, c.drop_danger_mean)
        );
        println!(
            "  atk_hand:    vuln={:.3}  ctrl={:.3}  ratio={}",
            v.atk_hand_mean,
            c.atk_hand_mean,
            ratio(v.atk_hand_mean, c.atk_hand_mean)
        );
    } else {
        println!("\n(vuln と ctrl の両 tag が揃わないため比率は省略)");
    }

    // ---- 距離バンド別 vuln vs ctrl テーブル ----
    report_band_table(&global_band_by_tag);

    Ok(())
}
