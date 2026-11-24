use anyhow::Result;
use rshogi_core::movegen::is_legal_with_pass;
use rshogi_core::types::{Color, Move, RepetitionState};

use rshogi_core::position::Position;

use super::engine::EngineProcess;
use super::position::{ParsedPosition, build_position};
use super::time_control::TimeControl;
use super::types::{EvalLog, GameOutcome, InfoCallback, SearchRequest};

/// 千日手の裁定。`pos` は最後の `do_move` 直後の局面を渡す。
///
/// 🚨2026-10-08 修正: 通常千日手は **同一局面 4 回目の出現** で成立する (競技将棋のルール)。
///   旧コードは `pos.repetition_state(plies_played)` を呼んでいた。`repetition_state` は
///   `rep != 0 && rep < ply` で判定するので、対局手数 (大きな値) を渡すと
///   `rep > 0` (= まだ 2 回目の再来) でも `Draw` になり、**2 回目の再来で対局を打ち切って**いた。
///   実測 (NEWH_FG02_m492_n200000、600 局): 千日手で終わった 53 局が **全局とも**
///   「最終局面の 2 回目の出現」で終了していて、4 回目は 0 局。うち 8 局は我々の評価値が
///   +100cp 以上 (最大 +2047cp) で、勝勢の局を引き分けに変えていた。
///   エンジン側は 4 回目で正しく draw 値を返す (陽性対照で確認済み) ので、器だけの不整合。
///
/// 判定規則は `rshogi-csa-server` の `GameValidator::classify_repetition` と同一:
/// - 通常千日手 (`Draw`): `repetition < 0` (= 4 回目以降) のときだけ成立。
/// - 連続王手の千日手 (`Win` / `Lose`): 反則なので 1 循環目の再来でも成立させる。
/// - `Superior` / `Inferior`: 対局の裁定には使わない。
fn adjudicate_repetition(pos: &Position) -> RepetitionState {
    let st = pos.state();
    if st.repetition == 0 {
        return RepetitionState::None;
    }
    match st.repetition_type {
        // 非決定的な再来 (rep > 0) では打ち切らず対局続行。
        RepetitionState::Draw if st.repetition >= 0 => RepetitionState::None,
        RepetitionState::Superior | RepetitionState::Inferior => RepetitionState::None,
        other => other,
    }
}

/// ゲーム設定
pub struct GameConfig {
    pub max_moves: u32,
    pub timeout_margin_ms: u64,
    /// パス権利の初期値 (先手, 後手)。None の場合はパス権なし。
    pub pass_rights: Option<(u8, u8)>,
    /// Some(n) の場合は `go depth n` を使用（byoyomi より優先）
    pub go_depth: Option<u32>,
    /// 先手に適用するノード数。 Some(n) の場合は `go nodes n` を使用。
    pub go_nodes_black: Option<u64>,
    /// 後手に適用するノード数。 Some(n) の場合は `go nodes n` を使用。
    pub go_nodes_white: Option<u64>,
}

/// 1手ごとに呼ばれるイベント
pub struct MoveEvent {
    pub ply: u32,
    pub side: Color,
    pub sfen_before: String,
    pub move_usi: String,
    /// エンジンが返した生の指し手文字列（合法手に正規化される前）
    pub raw_move_usi: Option<String>,
    pub elapsed_ms: u64,
    pub think_limit_ms: u64,
    pub timed_out: bool,
    pub eval: Option<EvalLog>,
    pub engine_label: String,
}

/// 対局結果
pub struct GameResult {
    pub outcome: GameOutcome,
    pub reason: String,
    pub plies: u32,
}

/// 1局を実行する。
///
/// - `black`, `white`: エンジンプロセス（事前に spawn 済み）
/// - `start_pos`: 開始局面
/// - `tc`: 時間管理（コピーして内部で更新）
/// - `config`: ゲーム設定
/// - `game_id`: ゲーム番号
/// - `on_move`: 1手ごとに呼ばれるコールバック
/// - `info_cb`: info行受信時のコールバック（None ならスキップ）
pub fn run_game(
    black: &mut EngineProcess,
    white: &mut EngineProcess,
    start_pos: &ParsedPosition,
    tc: TimeControl,
    config: &GameConfig,
    game_id: u32,
    on_move: &mut dyn FnMut(&MoveEvent),
    mut info_cb: Option<Box<InfoCallback<'_>>>,
) -> Result<GameResult> {
    let pass_black = config.pass_rights.map(|(b, _)| b);
    let pass_white = config.pass_rights.map(|(_, w)| w);
    let mut pos = build_position(start_pos, pass_black, pass_white)?;
    let mut tc = tc;
    let mut outcome = GameOutcome::InProgress;
    let mut outcome_reason = "max_moves".to_string();
    let mut plies_played = 0u32;
    // 2026-10-07: 棋譜履歴。 開始局面の moves から始めて、 指した手を足していく。
    //   🚨これをエンジンに渡さないと千日手を検出できない (実測 -0.38pt)。
    let root_cmd: String = if start_pos.startpos {
        "position startpos".to_string()
    } else {
        format!("position sfen {}", start_pos.sfen.as_deref().unwrap_or(""))
    };
    let mut played: Vec<String> = start_pos.moves.clone();

    let pass_rights_enabled = config.pass_rights.is_some();

    for ply_idx in 0..config.max_moves {
        plies_played = ply_idx + 1;
        let side = pos.side_to_move();
        let engine = if side == Color::Black {
            &mut *black
        } else {
            &mut *white
        };
        let engine_label = engine.label.clone();
        let sfen_before = pos.to_sfen();
        let think_limit_ms = tc.think_limit_ms(side);
        let pass_rights = if pass_rights_enabled {
            Some((pos.pass_rights(Color::Black), pos.pass_rights(Color::White)))
        } else {
            None
        };
        let pos_cmd = if played.is_empty() {
            root_cmd.clone()
        } else {
            format!("{} moves {}", root_cmd, played.join(" "))
        };
        let req = SearchRequest {
            position_cmd: Some(pos_cmd),
            sfen: &sfen_before,
            time_args: tc.time_args(),
            think_limit_ms,
            timeout_margin_ms: config.timeout_margin_ms,
            game_id,
            ply: plies_played,
            side,
            engine_label: engine_label.clone(),
            pass_rights,
            go_depth: config.go_depth,
            go_nodes: if side == Color::Black {
                config.go_nodes_black
            } else {
                config.go_nodes_white
            },
        };
        let cb = info_cb.as_mut().map(|b| b.as_mut() as &mut dyn FnMut(&str, &SearchRequest<'_>));
        let search = engine.search(&req, cb)?;

        let timed_out = search.timed_out;
        let mut move_usi = search.bestmove.clone().unwrap_or_else(|| "none".to_string());
        let mut raw_move_usi = None;
        let mut terminal = false;
        let elapsed_ms = search.elapsed_ms;
        let eval_log = search.eval.clone();

        if timed_out {
            outcome = if side == Color::Black {
                GameOutcome::WhiteWin
            } else {
                GameOutcome::BlackWin
            };
            outcome_reason = "timeout".to_string();
            terminal = true;
            if search.bestmove.is_none() {
                move_usi = "timeout".to_string();
            }
        } else if let Some(ref mv_str) = search.bestmove {
            raw_move_usi = Some(mv_str.clone());
            match mv_str.as_str() {
                "resign" => {
                    move_usi = mv_str.clone();
                    outcome = if side == Color::Black {
                        GameOutcome::WhiteWin
                    } else {
                        GameOutcome::BlackWin
                    };
                    outcome_reason = "resign".to_string();
                    terminal = true;
                }
                "win" => {
                    move_usi = mv_str.clone();
                    outcome = if side == Color::Black {
                        GameOutcome::BlackWin
                    } else {
                        GameOutcome::WhiteWin
                    };
                    outcome_reason = "win".to_string();
                    terminal = true;
                }
                _ => match Move::from_usi(mv_str) {
                    Some(mv) if is_legal_with_pass(&pos, mv) => {
                        let gives_check = if mv.is_pass() {
                            false
                        } else {
                            pos.gives_check(mv)
                        };
                        pos.do_move(mv, gives_check);
                        tc.update_after_move(side, search.elapsed_ms);
                        played.push(mv_str.clone()); // 2026-10-07: 棋譜履歴に足す
                        move_usi = mv_str.clone();
                        raw_move_usi = None;
                        // 2026-06-05 千日手検出追加 = rshogi-core RepetitionState
                        // (= 過去 SPRT で偽 draw 計上 noise 削減)
                        // 🚨2026-10-08: 旧 `pos.repetition_state(plies_played as i32)` は
                        //   2 回目の再来で Draw を返していた。adjudicate_repetition を参照。
                        let rep = adjudicate_repetition(&pos);
                        match rep {
                            RepetitionState::Draw => {
                                outcome = GameOutcome::Draw;
                                outcome_reason = "sennichite".to_string();
                                terminal = true;
                            }
                            RepetitionState::Win => {
                                // 連続王手 を続けた側が負け、 受けた側 (= 現 side_to_move) が win
                                let win_side = pos.side_to_move();
                                outcome = if win_side == Color::Black {
                                    GameOutcome::BlackWin
                                } else {
                                    GameOutcome::WhiteWin
                                };
                                outcome_reason = "perpetual_check_win".to_string();
                                terminal = true;
                            }
                            RepetitionState::Lose => {
                                let lose_side = pos.side_to_move();
                                outcome = if lose_side == Color::Black {
                                    GameOutcome::WhiteWin
                                } else {
                                    GameOutcome::BlackWin
                                };
                                outcome_reason = "perpetual_check_lose".to_string();
                                terminal = true;
                            }
                            _ => {}
                        }
                    }
                    _ => {
                        outcome = if side == Color::Black {
                            GameOutcome::WhiteWin
                        } else {
                            GameOutcome::BlackWin
                        };
                        outcome_reason = "illegal_move".to_string();
                        terminal = true;
                        move_usi = "illegal".to_string();
                    }
                },
            }
        } else {
            outcome = if side == Color::Black {
                GameOutcome::WhiteWin
            } else {
                GameOutcome::BlackWin
            };
            outcome_reason = "no_bestmove".to_string();
            terminal = true;
        }

        let event = MoveEvent {
            ply: plies_played,
            side,
            sfen_before,
            move_usi,
            raw_move_usi,
            elapsed_ms,
            think_limit_ms,
            timed_out,
            eval: eval_log,
            engine_label,
        };
        on_move(&event);

        if terminal || outcome != GameOutcome::InProgress {
            break;
        }
    }

    if outcome == GameOutcome::InProgress {
        outcome = GameOutcome::Draw;
        outcome_reason = "max_moves".to_string();
    }

    Ok(GameResult {
        outcome,
        reason: outcome_reason,
        plies: plies_played,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rshogi_core::position::{Position, SFEN_HIRATE};

    /// USI の指し手列を適用する。
    fn apply(pos: &mut Position, moves: &[&str]) {
        for s in moves {
            let mv = Move::from_usi(s).unwrap_or_else(|| panic!("不正な USI 手 {s}"));
            assert!(is_legal_with_pass(pos, mv), "非合法手 {s}");
            let gc = pos.gives_check(mv);
            pos.do_move(mv, gc);
        }
    }

    fn hirate() -> Position {
        let mut pos = Position::new();
        pos.set_sfen(SFEN_HIRATE).expect("平手 SFEN");
        pos
    }

    /// 🎯陽性対照: 同一局面 **4 回目** では千日手を成立させる。
    ///   平手から両者の左金を 4 九 ⇔ 4 八 / 4 一 ⇔ 4 二 と 3 循環 (12 手) させると
    ///   平手局面がちょうど 4 回目の出現になる。
    #[test]
    fn adjudicate_repetition_draws_on_fourth_occurrence() {
        let mut pos = hirate();
        let cycle = ["4i4h", "4a4b", "4h4i", "4b4a"];
        for _ in 0..3 {
            apply(&mut pos, &cycle);
        }
        assert_eq!(adjudicate_repetition(&pos), RepetitionState::Draw);
    }

    /// 🚨陰性対照 (今回のバグ本体): 同一局面 **2 回目 / 3 回目** では成立させない。
    ///   旧実装 `pos.repetition_state(plies_played)` はここで Draw を返していたため、
    ///   勝勢の局まで 2 回目の再来で打ち切られていた。
    #[test]
    fn adjudicate_repetition_does_not_draw_before_fourth_occurrence() {
        let cycle = ["4i4h", "4a4b", "4h4i", "4b4a"];
        for cycles in 1..=2 {
            let mut pos = hirate();
            for _ in 0..cycles {
                apply(&mut pos, &cycle);
            }
            // 1 循環 = 平手 2 回目、 2 循環 = 3 回目。 どちらも千日手ではない。
            assert_eq!(
                adjudicate_repetition(&pos),
                RepetitionState::None,
                "{cycles} 循環 (= 同一局面 {} 回目) で千日手にしてはいけない",
                cycles + 1
            );
            // 旧実装の再現: 対局手数を渡すと 2 回目で Draw になってしまうことを固定する。
            assert_eq!(
                pos.repetition_state(pos.game_ply()),
                RepetitionState::Draw,
                "旧実装はここで Draw を返していた (この assert が落ちたら前提が変わった)"
            );
        }
    }
}
