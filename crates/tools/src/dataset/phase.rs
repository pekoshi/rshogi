//! フェーズ分類
//!
//! 局面の手数とゲームフェーズ値からフェーズ（序盤・中盤・終盤）を判定する。

use crate::dataset::types::Phase;

/// SFEN 文字列から局面のフェーズを分類する
///
/// ```text
/// opening:  ply <= 40 && game_phase >= 200
/// endgame:  game_phase <= 80 || ply >= 120
/// middle:   else
/// ```
pub fn classify_phase(sfen: &str, ply: u32) -> Phase {
    let game_phase = compute_game_phase_from_sfen(sfen);

    if ply <= 40 && game_phase >= 200 {
        Phase::Opening
    } else if game_phase <= 80 || ply >= 120 {
        Phase::Endgame
    } else {
        Phase::Middle
    }
}

/// SFEN 文字列から駒数ベースのゲームフェーズを計算する
///
/// 駒種ごとの重みで計算する（P2-NNUE の compute_game_phase 相当）:
/// - 飛車/角: 各 48
/// - 金/銀: 各 24
/// - 桂/香: 各 16
/// - 歩: 各 8
///
/// 最大値（平手初期局面）: 2*(48+48+24*2+16*2+8*9) = 2*(48+48+48+32+72) = 496
fn compute_game_phase_from_sfen(sfen: &str) -> i32 {
    let board_part = sfen.split_whitespace().next().unwrap_or("");

    let mut phase = 0i32;

    for ch in board_part.chars() {
        if ch == '+' {
            // 成り駒マーカー（重みは生駒と同じなので無視して続行）
            continue;
        }
        match ch {
            'R' | 'r' => phase += 48,
            'B' | 'b' => phase += 48,
            'G' | 'g' => phase += 24,
            'S' | 's' => phase += 24,
            'N' | 'n' => phase += 16,
            'L' | 'l' => phase += 16,
            'P' | 'p' => phase += 8,
            _ => {}
        }
    }

    // 持ち駒部分のフェーズ加算
    let parts: Vec<&str> = sfen.split_whitespace().collect();
    if parts.len() >= 3 && parts[2] != "-" {
        let hand = parts[2];
        let mut count = 0u32;
        for ch in hand.chars() {
            if ch.is_ascii_digit() {
                count = count * 10 + ch.to_digit(10).unwrap_or(0);
            } else {
                let n = if count == 0 { 1 } else { count };
                let weight = match ch {
                    'R' | 'r' => 48,
                    'B' | 'b' => 48,
                    'G' | 'g' => 24,
                    'S' | 's' => 24,
                    'N' | 'n' => 16,
                    'L' | 'l' => 16,
                    'P' | 'p' => 8,
                    _ => 0,
                };
                phase += (n as i32) * weight;
                count = 0;
            }
        }
    }

    phase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_position_is_opening() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        assert_eq!(classify_phase(sfen, 1), Phase::Opening);
    }

    #[test]
    fn high_ply_is_endgame() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 130";
        assert_eq!(classify_phase(sfen, 130), Phase::Endgame);
    }

    #[test]
    fn game_phase_initial_position() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let phase = compute_game_phase_from_sfen(sfen);
        // 2*R(48) + 2*B(48) + 4*G(24) + 4*S(24) + 4*N(16) + 4*L(16) + 18*P(8)
        // = 96 + 96 + 96 + 96 + 64 + 64 + 144 = 656
        // しかし玉は含まない。上記は盤面の全駒（玉除く）
        assert!(phase > 400);
    }
}
