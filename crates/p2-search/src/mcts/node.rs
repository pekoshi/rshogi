use rshogi_core::types::Move;

/// MCTS ノード（arena/index ベース）
///
/// ポインタを使わず `u32` index で親子を参照する。
/// 未設定の index は `INVALID` で表す。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MctsNode {
    /// この手
    pub mv: Move,
    /// 親ノード index（INVALID = ルート）
    pub parent: u32,
    /// 最初の子の index（INVALID = 未展開）
    pub first_child: u32,
    /// 子ノード数
    pub num_children: u16,
    /// 終端フラグ（合法手なし）
    pub is_terminal: bool,
    /// 訪問回数 N
    pub visit_count: u32,
    /// Virtual loss 加算数
    pub virtual_loss_count: u32,
    /// 累積値 W
    pub value_sum: f32,
    /// Policy prior P
    pub prior: f32,
    /// Tactical PUCT Stage 2: 候補手の tactical_bonus - risk_penalty
    /// (expand 時に position + weights から計算してキャッシュ、 select で加算)
    /// enable_tactical_puct=false なら 0.0 のまま
    pub tactical_score: f32,
    /// Stage 3: risk-aware backup 用の min/max/squared sum
    /// value_min, value_max, value_sq_sum (variance = sq_sum/n - (sum/n)^2)
    pub value_min: f32,
    pub value_max: f32,
    pub value_sq_sum: f32,
}

impl MctsNode {
    /// 無効な index（未設定を表す sentinel）
    pub const INVALID: u32 = u32::MAX;

    /// 新規ノードの初期値
    pub fn new(mv: Move, parent: u32) -> Self {
        Self {
            mv,
            parent,
            first_child: Self::INVALID,
            num_children: 0,
            is_terminal: false,
            visit_count: 0,
            virtual_loss_count: 0,
            value_sum: 0.0,
            prior: 0.0,
            tactical_score: 0.0,
            value_min: f32::INFINITY,
            value_max: f32::NEG_INFINITY,
            value_sq_sum: 0.0,
        }
    }
}

impl MctsNode {
    /// risk-aware Q: Q_avg - λ·variance、 λ は小さく (0.1〜0.3) 推奨
    ///
    /// variance = sq_sum/n - (sum/n)^2
    /// q_risk = avg - lambda * variance
    pub fn q_risk(&self, lambda: f32) -> f32 {
        let n = self.visit_count + self.virtual_loss_count;
        if n == 0 {
            return 0.0;
        }
        let n_f = n as f32;
        let mean = self.value_sum / n_f;
        if n < 2 {
            return mean;
        }
        let var = (self.value_sq_sum / n_f - mean * mean).max(0.0);
        mean - lambda * var
    }
}

/// Arena ベースのノード確保器
///
/// 初期化時に一括確保し、ホットパスでのヒープ割り当てを回避する。
pub struct NodeArena {
    nodes: Box<[MctsNode]>,
    len: u32,
}

impl NodeArena {
    /// 指定容量で arena を確保
    pub fn new(capacity: u32) -> Self {
        let default_node = MctsNode::new(Move::NONE, MctsNode::INVALID);
        Self {
            nodes: vec![default_node; capacity as usize].into_boxed_slice(),
            len: 0,
        }
    }

    /// 1 ノード確保し、その index を返す
    pub fn alloc(&mut self) -> Option<u32> {
        if (self.len as usize) >= self.nodes.len() {
            return None;
        }
        let idx = self.len;
        self.nodes[idx as usize] = MctsNode::new(Move::NONE, MctsNode::INVALID);
        self.len += 1;
        Some(idx)
    }

    /// 連続 n ノードを確保し、最初の index を返す
    pub fn alloc_batch(&mut self, n: u16) -> Option<u32> {
        let n32 = n as u32;
        if self.len + n32 > self.nodes.len() as u32 {
            return None;
        }
        let first = self.len;
        for i in 0..n32 {
            self.nodes[(first + i) as usize] = MctsNode::new(Move::NONE, MctsNode::INVALID);
        }
        self.len += n32;
        Some(first)
    }

    /// 使用済み数を 0 にリセット（メモリ解放なし）
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// 使用済みノード数
    pub fn len(&self) -> u32 {
        self.len
    }

    /// arena が空か
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 容量
    pub fn capacity(&self) -> u32 {
        self.nodes.len() as u32
    }

    /// index からノードへの参照
    pub fn get(&self, idx: u32) -> &MctsNode {
        &self.nodes[idx as usize]
    }

    /// index からノードへの可変参照
    pub fn get_mut(&mut self, idx: u32) -> &mut MctsNode {
        &mut self.nodes[idx as usize]
    }
}
