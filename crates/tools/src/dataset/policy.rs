//! Policy target 生成
//!
//! 局面の全合法手リストを生成し、onehot / soft の policy target を構築する。

use crate::dataset::types::{PolicyTarget, PolicyType};
use rshogi_core::movegen::{MoveList, generate_legal};
use rshogi_core::position::Position;

/// 合法手リストを USI 形式で生成する
pub fn generate_legal_moves_usi(pos: &Position) -> Vec<String> {
    let mut list = MoveList::new();
    generate_legal(pos, &mut list);
    list.iter().map(|mv| mv.to_usi()).collect()
}

/// Onehot policy target を生成する
///
/// best_move に 1.0、他に 0.0 を割り当てる。
pub fn generate_onehot_policy(legal_moves: &[String], best_move: &str) -> Option<PolicyTarget> {
    if legal_moves.is_empty() {
        return None;
    }

    let probs: Vec<f32> =
        legal_moves.iter().map(|m| if m == best_move { 1.0 } else { 0.0 }).collect();

    // best_move が合法手に含まれない場合は None
    if !probs.iter().any(|&p| p > 0.0) {
        return None;
    }

    Some(PolicyTarget {
        target_type: PolicyType::Onehot,
        probs,
    })
}

/// Soft policy target を生成する（将来の MultiPV 用）
///
/// delta_cp から softmax(-delta/temperature) で確率分布を生成する。
pub fn generate_soft_policy(
    legal_moves: &[String],
    move_scores: &[(String, i32)],
    temperature: f64,
) -> Option<PolicyTarget> {
    if legal_moves.is_empty() || move_scores.is_empty() {
        return None;
    }

    let best_score = move_scores.iter().map(|(_, s)| *s).max().unwrap_or(0);

    let mut probs: Vec<f32> = legal_moves
        .iter()
        .map(|m| {
            let score = move_scores
                .iter()
                .find(|(mv, _)| mv == m)
                .map(|(_, s)| *s)
                .unwrap_or(best_score - 1000);
            let delta = (score - best_score) as f64;
            (delta / temperature).exp() as f32
        })
        .collect();

    // 正規化
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    Some(PolicyTarget {
        target_type: PolicyType::Soft,
        probs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onehot_basic() {
        let moves = vec!["7g7f".to_string(), "2g2f".to_string(), "3g3f".to_string()];
        let policy = generate_onehot_policy(&moves, "7g7f").unwrap();
        assert_eq!(policy.target_type, PolicyType::Onehot);
        assert_eq!(policy.probs, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn onehot_missing_move_returns_none() {
        let moves = vec!["7g7f".to_string(), "2g2f".to_string()];
        assert!(generate_onehot_policy(&moves, "9i9h").is_none());
    }
}
