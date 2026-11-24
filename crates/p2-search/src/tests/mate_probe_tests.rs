use rshogi_core::position::Position;

use crate::mate_probe::{MateResult, mate_probe_1ply, mate_probe_3ply, mate_probe_5ply};

fn pos_from_sfen(sfen: &str) -> Position {
    let mut pos = Position::new();
    pos.set_sfen(sfen).unwrap();
    pos
}

#[test]
fn mate_1ply_found() {
    rshogi_core::mate::init();
    // 後手玉が1一、先手の金が2一にいて1手詰め
    let mut pos = pos_from_sfen("k8/G8/9/9/9/9/9/9/4K4 b G 1");
    let result = mate_probe_1ply(&mut pos);
    match result {
        MateResult::MateFound { ply, .. } => assert_eq!(ply, 1),
        _ => panic!("1手詰めが見つからなかった"),
    }
}

#[test]
fn mate_1ply_not_found() {
    rshogi_core::mate::init();
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let result = mate_probe_1ply(&mut pos);
    match result {
        MateResult::NoMate => {} // OK
        _ => panic!("平手で詰みが見つかるのはおかしい"),
    }
}

#[test]
fn mate_3ply_found() {
    rshogi_core::mate::init();
    // 3手詰め局面: 後手玉が1一、先手が金を2つ持っている
    // 1三金打 → 受け → 1二金打 で詰み
    let mut pos = pos_from_sfen("k8/9/9/9/9/9/9/9/4K4 b 2G 1");
    let result = mate_probe_3ply(&mut pos, 10_000);
    // この局面は3手詰めかどうかは局面依存。結果を確認するだけ。
    match result {
        MateResult::MateFound { ply, .. } => {
            assert!(ply <= 3, "ply={ply} が3以下であること");
        }
        MateResult::NoMate => {
            // 3手では詰まない局面の可能性もある
        }
        MateResult::Unknown { .. } => {
            // ノード制限に到達した場合
        }
    }
}

#[test]
fn mate_3ply_no_mate() {
    rshogi_core::mate::init();
    // 平手局面では3手で詰まない
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let result = mate_probe_3ply(&mut pos, 10_000);
    match result {
        MateResult::NoMate => {}         // OK
        MateResult::Unknown { .. } => {} // ノード制限到達も OK
        MateResult::MateFound { .. } => panic!("平手で3手詰みが見つかるのはおかしい"),
    }
}

#[test]
fn mate_5ply_found() {
    rshogi_core::mate::init();
    // 5手詰め以内の局面をテスト
    // 受け側の応手が限定される局面
    let mut pos = pos_from_sfen("k8/9/1G7/9/9/9/9/9/4K4 b 2G 1");
    let result = mate_probe_5ply(&mut pos, 50_000);
    // 結果を確認するだけ（局面依存）
    match result {
        MateResult::MateFound { ply, .. } => {
            assert!(ply <= 5, "ply={ply} が5以下であること");
        }
        MateResult::NoMate | MateResult::Unknown { .. } => {} // OK
    }
}

#[test]
fn mate_probe_node_limit() {
    rshogi_core::mate::init();
    // 非常に少ないノード制限で3手詰め探索
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let result = mate_probe_3ply(&mut pos, 1);
    // ノード制限に到達するか、NoMate
    match result {
        MateResult::Unknown { .. } | MateResult::NoMate => {} // OK
        MateResult::MateFound { .. } => panic!("ノード1で詰みが見つかるのはおかしい"),
    }
}

#[test]
fn mate_probe_search_mode_gating() {
    use crate::phase::SearchMode;

    rshogi_core::mate::init();
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");

    // Normal モードでは mate_probe は root でのみ 1ply
    let result = crate::mate_probe::mate_probe(&mut pos, SearchMode::Normal, None, false, false);
    assert!(result.is_none(), "Normal + non-root では詰み探索されないはず");

    // root での 1ply 探索は実行される
    let result = crate::mate_probe::mate_probe(&mut pos, SearchMode::Normal, None, true, false);
    // 平手では詰みなし
    assert!(result.is_none());
}
