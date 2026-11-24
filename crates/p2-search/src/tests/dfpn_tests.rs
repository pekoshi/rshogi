use rshogi_core::position::Position;

use crate::dfpn::basic::BasicDfpnEngine;
use crate::dfpn::{DfpnEngine, DfpnLimits, DfpnResult};

fn pos_from_sfen(sfen: &str) -> Position {
    let mut pos = Position::new();
    pos.set_sfen(sfen).unwrap();
    pos
}

#[test]
fn dfpn_no_mate() {
    rshogi_core::mate::init();
    // 平手局面: 詰みなし
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let limits = DfpnLimits {
        max_nodes: 1_000,
        max_depth: 5,
        tt_size_mb: 1,
    };
    let mut engine = BasicDfpnEngine::new(1);
    let result = engine.search(&mut pos, &limits);
    match result {
        DfpnResult::ProvenMate { .. } => panic!("平手で詰み証明はおかしい"),
        DfpnResult::ProvenNoMate { .. } | DfpnResult::Unknown { .. } => {} // OK
    }
}

#[test]
fn dfpn_node_limit() {
    rshogi_core::mate::init();
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let limits = DfpnLimits {
        max_nodes: 1,
        max_depth: 15,
        tt_size_mb: 1,
    };
    let mut engine = BasicDfpnEngine::new(1);
    let result = engine.search(&mut pos, &limits);
    match result {
        DfpnResult::Unknown { .. } | DfpnResult::ProvenNoMate { .. } => {} // OK
        DfpnResult::ProvenMate { .. } => panic!("ノード1で詰み証明はおかしい"),
    }
}

#[test]
fn dfpn_tt_clear() {
    rshogi_core::mate::init();
    let mut engine = BasicDfpnEngine::new(1);
    // clear_tt がパニックしないことを確認
    engine.clear_tt();
}

#[test]
fn dfpn_depth_limit() {
    rshogi_core::mate::init();
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let limits = DfpnLimits {
        max_nodes: 100_000,
        max_depth: 1,
        tt_size_mb: 1,
    };
    let mut engine = BasicDfpnEngine::new(1);
    let result = engine.search(&mut pos, &limits);
    match result {
        DfpnResult::ProvenMate { .. } => panic!("深さ1で平手から詰み証明はおかしい"),
        DfpnResult::ProvenNoMate { .. } | DfpnResult::Unknown { .. } => {} // OK
    }
}
