use rshogi_core::position::Position;
use rshogi_core::search::{LimitsType, Search, SearchInfo, SearchResult};

/// αβ 探索アダプター
///
/// 既存の `Search::go()` を薄くラップする。αβ の再実装は行わない。
pub struct AlphaBetaAdapter {
    search: Search,
}

impl AlphaBetaAdapter {
    pub fn new(tt_size_mb: usize) -> Self {
        Self {
            search: Search::new(tt_size_mb),
        }
    }

    /// αβ 探索を実行
    pub fn search<F>(
        &mut self,
        pos: &mut Position,
        limits: LimitsType,
        on_info: Option<F>,
    ) -> SearchResult
    where
        F: FnMut(&SearchInfo),
    {
        self.search.go(pos, limits, on_info)
    }

    /// 探索を中断
    pub fn stop(&self) {
        self.search.stop();
    }

    /// 内部の Search への可変参照（設定変更用）
    pub fn inner_mut(&mut self) -> &mut Search {
        &mut self.search
    }
}
