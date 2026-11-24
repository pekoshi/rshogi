//! MCTS ↔ NNUE 評価ワーカー間のメッセージ型

use crate::features::P2NnueInput;

/// NNUE 評価リクエスト
///
/// MCTS 探索スレッド（プロデューサー）が評価ワーカーに送信する。
#[repr(C)]
pub struct EvalRequest {
    /// リクエスト識別子（応答との対応付け）
    pub request_id: u64,
    /// ノード識別子
    pub node_id: u32,
    /// NNUE 入力特徴量（~8195 bytes インライン）
    pub nnue_input: P2NnueInput,
}

/// NNUE 評価結果
///
/// 評価ワーカー（コンシューマー）が MCTS スレッドに返す。
#[repr(C)]
pub struct EvalResult {
    /// 対応するリクエストの識別子
    pub request_id: u64,
    /// ノード識別子
    pub node_id: u32,
    /// 評価値
    pub value: i32,
}
