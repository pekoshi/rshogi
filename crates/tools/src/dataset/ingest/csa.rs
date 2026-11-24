//! CSA 棋譜フォーマットの Ingester
//!
//! CSA 棋譜ファイルから全局面を抽出して `RawRecord` に変換する。

use crate::dataset::ingest::Ingester;
use crate::dataset::types::RawRecord;
use anyhow::{Context, Result};
use rshogi_csa::{ParsedMove, SpecialMove, csa_move_to_usi};
use std::path::Path;

/// CSA Ingester
pub struct CsaIngester;

impl Ingester for CsaIngester {
    fn ingest(
        &self,
        path: &Path,
        source_name: &str,
        callback: &mut dyn FnMut(RawRecord) -> bool,
    ) -> Result<u64> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("CSA ファイル読み込み失敗: {}", path.display()))?;

        let mut count = 0u64;

        // 複数棋譜が連結されている場合に対応（"/" 区切り or 空行区切り）
        for game_text in split_csa_games(&content) {
            if let Ok(records) = extract_positions_from_csa(game_text, source_name) {
                for record in records {
                    count += 1;
                    if !callback(record) {
                        return Ok(count);
                    }
                }
            }
        }

        Ok(count)
    }
}

/// CSA テキストを個別の棋譜に分割する
fn split_csa_games(content: &str) -> Vec<&str> {
    // "V2" で始まるブロックごとに分割
    let mut games = Vec::new();
    let mut start = 0;
    let bytes = content.as_bytes();

    for (i, _) in content.match_indices("\nV2") {
        if start < i {
            let segment = content[start..i].trim();
            if !segment.is_empty() {
                games.push(segment);
            }
        }
        start = i + 1; // \n の次
    }

    // 残りの部分
    let segment = content[start..].trim();
    if !segment.is_empty() {
        // V2 で始まらないファイルの場合もそのまま1棋譜として扱う
        if games.is_empty() || !segment.is_empty() {
            games.push(segment);
        }
    }

    if games.is_empty() && !bytes.is_empty() {
        games.push(content.trim());
    }

    games
}

/// 1つの CSA 棋譜から全局面の RawRecord を抽出する
fn extract_positions_from_csa(text: &str, source_name: &str) -> Result<Vec<RawRecord>> {
    let (init_pos, moves, _info) = rshogi_csa::parse_csa_full(text)?;

    // 対局結果を判定する（最終手から）
    let game_result = determine_game_result(&moves);

    let mut records = Vec::new();
    let mut pos = init_pos;
    let mut ply = 1u32;

    for parsed_move in &moves {
        match parsed_move {
            ParsedMove::Normal(csa_mv) => {
                let usi_move = match csa_move_to_usi(&csa_mv.mv, &pos) {
                    Ok(m) => m,
                    Err(_) => {
                        ply += 1;
                        continue;
                    }
                };

                let sfen = pos.to_sfen();

                // 手番から見た結果を計算
                let result_for_side = if pos.side_to_move == rshogi_csa::Color::Black {
                    game_result
                } else {
                    -game_result
                };

                records.push(RawRecord {
                    sfen,
                    best_move: usi_move.clone(),
                    score: 0, // CSA 棋譜には評価値がない
                    game_result: result_for_side,
                    ply,
                    source_name: source_name.to_string(),
                });

                // 局面を進める
                let _ = pos.apply_csa_move(&csa_mv.mv);
                ply += 1;
            }
            ParsedMove::Special(_) => break,
        }
    }

    Ok(records)
}

/// 対局結果を判定する（先手視点: 1=先手勝ち, -1=後手勝ち, 0=引き分け）
fn determine_game_result(moves: &[ParsedMove]) -> i8 {
    for mv in moves.iter().rev() {
        if let ParsedMove::Special(special) = mv {
            return match special {
                SpecialMove::Resign => {
                    // 投了: 直前の手番の反対側が勝ち
                    // （CSAでは投了した側が負け）
                    // moves の長さで手番を推定
                    let normal_count =
                        moves.iter().filter(|m| matches!(m, ParsedMove::Normal(_))).count();
                    if normal_count % 2 == 0 {
                        -1 // 先手が投了
                    } else {
                        1 // 後手が投了
                    }
                }
                SpecialMove::Win => 1, // 入玉宣言勝ち（先手と仮定）
                SpecialMove::Draw | SpecialMove::Sennichite | SpecialMove::Jishogi => 0,
                _ => 0,
            };
        }
    }
    0
}
