use rshogi_core::position::Position;

use crate::batch_eval::BatchCollector;
use crate::mcts::tree::SelectionPath;

fn hirate_pos() -> Position {
    let mut pos = Position::new();
    pos.set_hirate();
    pos
}

#[test]
fn batch_submit_flush() {
    use p2_nnue::evaluator::NnueEvaluator;
    use p2_nnue::features::{P2NnueInput, extract_features};
    use std::sync::Arc;

    struct MockEval;
    impl NnueEvaluator for MockEval {
        fn evaluate(&self, _: &P2NnueInput) -> i32 {
            42
        }
        fn backend_name(&self) -> &'static str {
            "mock"
        }
    }

    let evaluator: Arc<dyn NnueEvaluator + Send + Sync> = Arc::new(MockEval);

    let mut batch = BatchCollector::new(4);
    assert_eq!(batch.pending_count(), 0);

    let pos = hirate_pos();
    let input = extract_features(&pos);
    let mut path = SelectionPath::new();
    path.push(0);
    path.push(1);

    batch.submit(&path, &input);
    assert_eq!(batch.pending_count(), 1);

    batch.submit(&path, &input);
    batch.submit(&path, &input);
    batch.submit(&path, &input);
    assert_eq!(batch.pending_count(), 4);

    let results = batch.flush(evaluator.as_ref());
    assert_eq!(results.len(), 4);
    assert_eq!(batch.pending_count(), 0);

    // 値が有効範囲内か
    for r in &results {
        assert!(r.value >= -1.0 && r.value <= 1.0);
    }
}

#[test]
fn batch_should_flush_size() {
    let mut batch = BatchCollector::new(4);
    assert!(!batch.should_flush());

    let pos = hirate_pos();
    let input = p2_nnue::features::extract_features(&pos);
    let mut path = SelectionPath::new();
    path.push(0);

    for _ in 0..4 {
        batch.submit(&path, &input);
    }
    assert!(batch.should_flush());
}
