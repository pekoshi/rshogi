use rshogi_core::types::Move;

use crate::tactical_cache::TacticalCache;

#[test]
fn cache_store_probe() {
    let mut cache = TacticalCache::new(64);
    let key: u64 = 0x1234_5678_9ABC_DEF0;
    cache.store(key, TacticalCache::KIND_MATE, 3, Move::NONE);

    let entry = cache.probe(key).unwrap();
    assert_eq!(entry.kind, TacticalCache::KIND_MATE);
    assert_eq!(entry.depth, 3);
}

#[test]
fn cache_miss() {
    let cache = TacticalCache::new(64);
    let key: u64 = 0x1234_5678_9ABC_DEF0;
    assert!(cache.probe(key).is_none());
}

#[test]
fn cache_generation() {
    let mut cache = TacticalCache::new(64);
    let key: u64 = 0x1234_5678_9ABC_DEF0;

    cache.store(key, TacticalCache::KIND_MATE, 1, Move::NONE);
    let entry1 = cache.probe(key).unwrap();
    let gen1 = entry1.generation;

    cache.new_generation();
    cache.store(key, TacticalCache::KIND_NO_MATE, 3, Move::NONE);
    let entry2 = cache.probe(key).unwrap();
    assert_eq!(entry2.kind, TacticalCache::KIND_NO_MATE);
    assert_eq!(entry2.generation, gen1.wrapping_add(1));
}
