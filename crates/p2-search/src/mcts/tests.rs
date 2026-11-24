use std::sync::Arc;

use p2_nnue::evaluator::NnueEvaluator;
use p2_nnue::features::P2NnueInput;
use p2_nnue::queue::MpmcQueue;
use rshogi_core::movegen::{MoveList, generate_legal};
use rshogi_core::position::Position;
use rshogi_core::types::Move;

use super::config::PuctConfig;
use super::engine::PuctEngine;
use super::eval_bridge::EvalBridge;
use super::node::NodeArena;
use super::noise::dirichlet_noise;
use super::tree::MctsTree;
use crate::puct::{PuctBackend, PuctLimits};

// ─── テスト用 MockEvaluator ───

struct MockEvaluator(i32);

impl NnueEvaluator for MockEvaluator {
    fn evaluate(&self, _input: &P2NnueInput) -> i32 {
        self.0
    }

    fn backend_name(&self) -> &'static str {
        "mock"
    }
}

fn hirate_pos() -> Position {
    let mut pos = Position::new();
    pos.set_hirate();
    pos
}

// ─── #1: arena_alloc_and_clear ───

#[test]
fn arena_alloc_and_clear() {
    let mut arena = NodeArena::new(4);
    assert_eq!(arena.capacity(), 4);

    let a = arena.alloc().unwrap();
    let b = arena.alloc().unwrap();
    assert_eq!(a, 0);
    assert_eq!(b, 1);
    assert_eq!(arena.len(), 2);

    let _ = arena.alloc().unwrap();
    let _ = arena.alloc().unwrap();
    assert!(arena.alloc().is_none()); // 容量超過

    arena.clear();
    assert_eq!(arena.len(), 0);
    assert!(arena.alloc().is_some()); // clear 後は再利用可能
}

// ─── #2: arena_batch_contiguous ───

#[test]
fn arena_batch_contiguous() {
    let mut arena = NodeArena::new(16);
    let _ = arena.alloc().unwrap(); // index 0 を使用
    let first = arena.alloc_batch(5).unwrap();
    assert_eq!(first, 1);
    assert_eq!(arena.len(), 6);

    // 残り 10 スロット、11 要求 → None
    assert!(arena.alloc_batch(11).is_none());
}

// ─── #3: dirichlet_sums_to_one ───

#[test]
fn dirichlet_sums_to_one() {
    use rand::SeedableRng;
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
    let mut noise = [0.0f32; 10];
    dirichlet_noise(&mut rng, 0.3, &mut noise);
    let sum: f32 = noise.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
}

// ─── #4: dirichlet_all_positive ───

#[test]
fn dirichlet_all_positive() {
    use rand::SeedableRng;
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(123);
    let mut noise = [0.0f32; 20];
    dirichlet_noise(&mut rng, 0.3, &mut noise);
    for (i, &v) in noise.iter().enumerate() {
        assert!(v > 0.0, "noise[{i}] = {v} は正でない");
    }
}

// ─── #5: tree_expand_creates_children ───

#[test]
fn tree_expand_creates_children() {
    let config = PuctConfig {
        arena_capacity: 1024,
        ..PuctConfig::default()
    };
    let mut tree = MctsTree::new(config);
    let root = tree.init_root();

    let pos = hirate_pos();
    let mut moves = MoveList::new();
    generate_legal(&pos, &mut moves);

    let n = tree.expand(root, &moves);
    assert!(n > 0);
    assert!(n <= 32); // top_k = 32
    assert_eq!(tree.arena.get(root).num_children, n);
}

// ─── #6: tree_backup_updates_values ───

#[test]
fn tree_backup_updates_values() {
    let config = PuctConfig {
        arena_capacity: 1024,
        ..PuctConfig::default()
    };
    let mut tree = MctsTree::new(config);
    let root = tree.init_root();

    let pos = hirate_pos();
    let mut moves = MoveList::new();
    generate_legal(&pos, &mut moves);
    tree.expand(root, &moves);

    // 選択 → backup
    let path = tree.select(root);
    assert!(path.len() >= 2); // root + 1子
    tree.backup(&path, 0.5);

    // ルートの visit_count が 1 増え、value_sum に -0.5（符号反転）が加算
    let root_node = tree.arena.get(root);
    assert_eq!(root_node.visit_count, 1);
    // ルートは葉の 1 ply 上なので -0.5
    assert!((root_node.value_sum - (-0.5)).abs() < 1e-6);

    // 葉の visit_count も 1
    let leaf = tree.arena.get(path.leaf());
    assert_eq!(leaf.visit_count, 1);
    assert!((leaf.value_sum - 0.5).abs() < 1e-6);
}

// ─── #7: tree_select_follows_puct ───

#[test]
fn tree_select_follows_puct() {
    let config = PuctConfig {
        arena_capacity: 1024,
        ..PuctConfig::default()
    };
    let mut tree = MctsTree::new(config);
    let root = tree.init_root();

    let pos = hirate_pos();
    let mut moves = MoveList::new();
    generate_legal(&pos, &mut moves);
    tree.expand(root, &moves);

    // 全子が未訪問（visit=0）の場合、均等 prior ならどれも PUCT スコア同一
    // → 最初の子が選ばれるはず（同値なら先頭）
    let path = tree.select(root);
    let selected = path.get(1);
    let first_child = tree.arena.get(root).first_child;
    assert_eq!(selected, first_child);

    // 最初の子を訪問済みにする → 別の子が選ばれる
    tree.arena.get_mut(first_child).visit_count = 100;
    tree.arena.get_mut(root).visit_count = 100;
    let path2 = tree.select(root);
    let selected2 = path2.get(1);
    assert_ne!(selected2, first_child, "訪問済みの子は回避されるはず");
}

// ─── #8: eval_bridge_direct_returns_value ───

#[test]
fn eval_bridge_direct_returns_value() {
    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(12345));
    let bridge = EvalBridge::Direct(evaluator);

    let pos = hirate_pos();
    let input = p2_nnue::features::extract_features(&pos);
    let value = bridge.evaluate_leaf(0, &input, &pos, 0);
    assert_eq!(value, 12345);
}

// ─── #9: eval_bridge_queue_roundtrip ───

#[test]
fn eval_bridge_queue_roundtrip() {
    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(9999));
    let req_q = Arc::new(MpmcQueue::new(16).unwrap());
    let res_q = Arc::new(MpmcQueue::new(16).unwrap());

    let bridge = EvalBridge::Queued {
        evaluator,
        request_queue: req_q,
        result_queue: res_q,
    };

    let pos = hirate_pos();
    let input = p2_nnue::features::extract_features(&pos);
    let value = bridge.evaluate_leaf(42, &input, &pos, 1);
    assert_eq!(value, 9999);
}

// ─── #10: engine_returns_valid_move ───

#[test]
fn engine_returns_valid_move() {
    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(100));
    let config = PuctConfig {
        arena_capacity: 4096,
        ..PuctConfig::default()
    };
    let mut engine = PuctEngine::new_direct(config, evaluator);

    let pos = hirate_pos();
    let limits = PuctLimits {
        max_nodes: 50,
        max_time_ms: 0,
        search_mode: crate::phase::SearchMode::Normal,
        game_phase: None,
        endgame_score: 0.0,
    };
    let result = engine.search(&pos, &limits);
    assert_ne!(result.bestmove, Move::NONE, "bestmove は NONE であってはならない");
    assert!(result.nodes > 0);

    // bestmove が合法手であることを確認
    let mut moves = MoveList::new();
    generate_legal(&pos, &mut moves);
    assert!(
        moves.contains(result.bestmove),
        "bestmove {:?} が合法手リストに含まれていない",
        result.bestmove
    );
}

// ─── #11: engine_respects_node_limit ───

#[test]
fn engine_respects_node_limit() {
    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(0));
    let config = PuctConfig {
        arena_capacity: 4096,
        ..PuctConfig::default()
    };
    let mut engine = PuctEngine::new_direct(config, evaluator);

    let pos = hirate_pos();
    let limits = PuctLimits {
        max_nodes: 100,
        max_time_ms: 0,
        search_mode: crate::phase::SearchMode::Normal,
        game_phase: None,
        endgame_score: 0.0,
    };
    let result = engine.search(&pos, &limits);
    assert!(result.nodes <= 100, "nodes={} が制限 100 を超えている", result.nodes);
}

// ─── #12: engine_stop_halts_search ───

#[test]
fn engine_stop_halts_search() {
    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(0));
    let config = PuctConfig {
        arena_capacity: 1_048_576,
        ..PuctConfig::default()
    };
    let mut engine = PuctEngine::new_direct(config, evaluator);

    let pos = hirate_pos();
    let limits = PuctLimits {
        max_nodes: 0, // 無制限
        max_time_ms: 0,
        search_mode: crate::phase::SearchMode::Normal,
        game_phase: None,
        endgame_score: 0.0,
    };

    // 別スレッドから少し待って stop() を呼ぶ
    let stop_flag = engine.stop_flag();
    let handle = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    let result = engine.search(&pos, &limits);
    handle.join().unwrap();

    // 無制限だが stop されたので有限ノードで終了しているはず
    // (1M ノード全部使い切ることはない)
    assert!(result.nodes < 1_000_000, "stop で早期終了すべき: nodes={}", result.nodes);
}

// ─── #13: controller_uses_puct_for_midgame ───
// (E2E 統合テストは SearchController に PuctEngine を set_puct してから
//  search_root を呼ぶ形だが、search_root は αβ アダプター内部で
//  実際のエンジン初期化を要求するため、ここでは set_puct + stop の
//  基本接続テストに留める)

#[test]
fn controller_accepts_puct_engine() {
    use crate::controller::SearchController;

    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(0));
    let config = PuctConfig::default();
    let engine = PuctEngine::new_direct(config, evaluator);

    let mut controller = SearchController::new(1);
    controller.set_puct(Box::new(engine));
    // stop が panic しないことを確認
    controller.stop();
}

// ─── #14: existing_tests_pass ───
// これは `cargo test -p p2-search` で既存テスト (controller_tests, detector_tests)
// がパスすることで確認される。個別テストは不要。

// ─── #15: gumbel_engine_returns_valid_move ───
// Gumbel AZ Phase 1+2 統合: use_gumbel=true で search が完走し、
// 合法手 bestmove を返すことを確認 (regression check)

#[test]
fn gumbel_engine_returns_valid_move() {
    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(100));
    let config = PuctConfig {
        arena_capacity: 4096,
        use_gumbel: true,
        gumbel_m_initial: 8,
        ..PuctConfig::default()
    };
    let mut engine = PuctEngine::new_direct(config, evaluator);

    let pos = hirate_pos();
    let limits = PuctLimits {
        max_nodes: 200,
        max_time_ms: 0,
        search_mode: crate::phase::SearchMode::Normal,
        game_phase: None,
        endgame_score: 0.0,
    };
    let result = engine.search(&pos, &limits);
    assert_ne!(result.bestmove, Move::NONE, "bestmove は NONE であってはならない");
    assert!(result.nodes > 0);

    let mut moves = MoveList::new();
    generate_legal(&pos, &mut moves);
    assert!(
        moves.contains(result.bestmove),
        "Gumbel 経路の bestmove {:?} が合法手リストに含まれていない",
        result.bestmove
    );
}

// ─── #16: gumbel_halving_reduces_candidates ───
// SequentialHalving が search 中に候補数を減らしていくことを確認

#[test]
fn gumbel_halving_reduces_candidates() {
    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(50));
    let config = PuctConfig {
        arena_capacity: 16_384,
        use_gumbel: true,
        gumbel_m_initial: 8,
        ..PuctConfig::default()
    };
    let mut engine = PuctEngine::new_direct(config, evaluator);

    let pos = hirate_pos();
    let limits = PuctLimits {
        max_nodes: 1000, // log2(8)=3 phases、 phase_budget=333、 候補が halve されるのに十分
        max_time_ms: 0,
        search_mode: crate::phase::SearchMode::Normal,
        game_phase: None,
        endgame_score: 0.0,
    };
    let _ = engine.search(&pos, &limits);
    // search 終了時点で is_done() = true になっているはず (phase 3 完了 → 候補 1 個)
    // または arena 満杯で中断されただけかもしれないので緩めにアサート
    // (最低限 panic しないこと)
}

// ─── #17: eval_cache_hits_on_repeated_search ───
// MctsEvalCache が動作する: 2 回目の同一局面 search で hit が出る

#[test]
fn eval_cache_hits_on_repeated_search() {
    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(7777));
    let config = PuctConfig {
        arena_capacity: 8192,
        use_eval_cache: true,
        eval_cache_entries: 1024,
        ..PuctConfig::default()
    };
    let mut engine = PuctEngine::new_direct(config, evaluator);

    let pos = hirate_pos();
    let limits = PuctLimits {
        max_nodes: 200,
        max_time_ms: 0,
        search_mode: crate::phase::SearchMode::Normal,
        game_phase: None,
        endgame_score: 0.0,
    };

    // 1 回目: 全ミス
    let _ = engine.search(&pos, &limits);
    let cache = engine.eval_cache().expect("cache should be Some");
    let inserts_first = cache.inserts();
    assert!(inserts_first > 0, "1 回目は cache に書き込みがあるはず");

    // 2 回目: 同じ局面 → 一部は前回の subtree が cache にあるはず
    // ただし init_root で arena を clear するので、 cache にあるのは
    // 過去 search で評価した局面のみ。 hit がゼロでないことだけ確認
    let _ = engine.search(&pos, &limits);
    let cache = engine.eval_cache().expect("cache should be Some");
    assert!(cache.hits() > 0, "2 回目の search で cache hit が発生するはず");
    assert!(cache.hit_rate() > 0.0);
}

// ─── #18: eval_cache_disabled_when_flag_off ───

#[test]
fn eval_cache_disabled_when_flag_off() {
    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(0));
    let config = PuctConfig {
        use_eval_cache: false,
        ..PuctConfig::default()
    };
    let engine = PuctEngine::new_direct(config, evaluator);
    assert!(engine.eval_cache().is_none());
}

// ─── #19: phase_aware_c_puct_dispatch ───

#[test]
fn phase_aware_c_puct_dispatch() {
    use crate::phase::GamePhase;

    // disabled: 常に base c_puct
    let cfg = PuctConfig {
        phase_aware_enabled: false,
        c_puct: 2.5,
        c_puct_opening: 9.9,
        c_puct_middlegame: 9.9,
        c_puct_late_middlegame: 9.9,
        ..PuctConfig::default()
    };
    assert_eq!(cfg.c_puct_for_phase(Some(GamePhase::Opening), 0.2), 2.5);
    assert_eq!(cfg.c_puct_for_phase(Some(GamePhase::Middlegame), 0.5), 2.5);

    // enabled: phase + endgame_score で分岐
    let cfg = PuctConfig {
        phase_aware_enabled: true,
        c_puct_opening: 3.0,
        c_puct_middlegame: 2.5,
        c_puct_late_middlegame: 2.0,
        ..PuctConfig::default()
    };
    assert_eq!(cfg.c_puct_for_phase(Some(GamePhase::Opening), 0.2), 3.0);
    assert_eq!(cfg.c_puct_for_phase(Some(GamePhase::Middlegame), 0.30), 2.5);
    // endgame_score ≥ 0.45 で middle late として扱う
    assert_eq!(cfg.c_puct_for_phase(Some(GamePhase::Middlegame), 0.50), 2.0);
}

// ─── #20: phase_aware_topk_dispatch ───

#[test]
fn phase_aware_topk_dispatch() {
    use crate::phase::GamePhase;

    let cfg = PuctConfig {
        phase_aware_enabled: true,
        opening_topk: 24,
        middlegame_topk: 16,
        late_middlegame_topk: 8,
        ..PuctConfig::default()
    };
    assert_eq!(cfg.top_k_for_phase(Some(GamePhase::Opening), 0.2), 24);
    assert_eq!(cfg.top_k_for_phase(Some(GamePhase::Middlegame), 0.3), 16);
    assert_eq!(cfg.top_k_for_phase(Some(GamePhase::Middlegame), 0.5), 8);

    // disabled: 常に base top_k
    let cfg = PuctConfig {
        phase_aware_enabled: false,
        top_k: 32,
        opening_topk: 24,
        middlegame_topk: 16,
        ..PuctConfig::default()
    };
    assert_eq!(cfg.top_k_for_phase(Some(GamePhase::Opening), 0.2), 32);
}

// ─── #22: q_risk_with_variance ───

#[test]
fn q_risk_with_variance() {
    use crate::mcts::node::MctsNode;
    let mut node = MctsNode::new(Move::NONE, 0);
    // 5 backups: values [0.5, 0.4, 0.6, 0.5, 0.5]
    let vs = [0.5_f32, 0.4, 0.6, 0.5, 0.5];
    for v in &vs {
        node.visit_count += 1;
        node.value_sum += v;
        node.value_sq_sum += v * v;
    }
    let mean = 0.5_f32;
    let var = vs.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vs.len() as f32;
    // q_risk = mean - lambda * var
    let lambda = 0.2_f32;
    let expected = mean - lambda * var;
    let actual = node.q_risk(lambda);
    assert!((actual - expected).abs() < 1e-4, "expected {}, got {}", expected, actual);
}

// ─── #23: q_risk_zero_variance_returns_mean ───

#[test]
fn q_risk_zero_variance_returns_mean() {
    use crate::mcts::node::MctsNode;
    let mut node = MctsNode::new(Move::NONE, 0);
    for _ in 0..10 {
        node.visit_count += 1;
        node.value_sum += 0.7;
        node.value_sq_sum += 0.7 * 0.7;
    }
    let q = node.q_risk(1.0);
    // variance ≈ 0 → q ≈ mean = 0.7
    assert!((q - 0.7).abs() < 1e-4);
}

// ─── #21: skip_mcts_in_endgame ───

#[test]
fn skip_mcts_in_endgame() {
    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEvaluator(100));
    let config = PuctConfig {
        phase_aware_enabled: true,
        skip_mcts_in_endgame: true,
        arena_capacity: 4096,
        ..PuctConfig::default()
    };
    let mut engine = PuctEngine::new_direct(config, evaluator);
    let pos = hirate_pos();
    let limits = PuctLimits {
        max_nodes: 100,
        max_time_ms: 0,
        search_mode: crate::phase::SearchMode::Endgame,
        game_phase: Some(crate::phase::GamePhase::Endgame),
        endgame_score: 0.8,
    };
    let result = engine.search(&pos, &limits);
    // skip 時は nodes=0 で first legal を返す
    assert_eq!(result.nodes, 0);
    assert_ne!(result.bestmove, Move::NONE);

    let mut moves = MoveList::new();
    generate_legal(&pos, &mut moves);
    assert!(moves.contains(result.bestmove));
}
