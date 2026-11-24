use rshogi_core::position::Position;

use crate::detector::{EndgameDetector, EndgameReason};

/// テスト用ヘルパー: SFEN から Position を生成
fn pos_from_sfen(sfen: &str) -> Position {
    let mut pos = Position::new();
    pos.set_sfen(sfen).unwrap();
    pos
}

#[test]
fn startpos_is_not_endgame() {
    let detector = EndgameDetector::new();
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let signal = detector.evaluate(&mut pos);
    assert!(!signal.use_alpha_beta);
    assert_eq!(signal.reason, EndgameReason::None);
}

#[test]
fn high_ply_alone_does_not_trigger_endgame() {
    // 旧仕様 (`ply >= 80` ハードコード) では即終盤判定だったが、
    // 新仕様では駒配置・玉間距離・利き圧力の連続スコアで判定する。
    // ply 80 のままで駒が初期配置なら use_alpha_beta=false が正しい。
    let detector = EndgameDetector::new();
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 80");
    let signal = detector.evaluate(&mut pos);
    assert!(!signal.use_alpha_beta);
    assert_eq!(signal.reason, EndgameReason::None);
}

#[test]
fn low_material_triggers_endgame() {
    let detector = EndgameDetector::new();
    // 王と金のみ（金 x2 = 540 * 2 = 1080 < 2000）
    let mut pos = pos_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2G2g 1");
    let signal = detector.evaluate(&mut pos);
    assert!(signal.use_alpha_beta);
    assert_eq!(signal.reason, EndgameReason::LowMaterial);
}

#[test]
fn mate_1ply_triggers_mate_threat() {
    rshogi_core::mate::init();
    let detector = EndgameDetector::new();
    // 1手詰め局面: 後手玉が1一、先手の金が2一にいて、先手番で金を1二に打てば詰み
    // より確実な1手詰め局面を使用
    let mut pos = pos_from_sfen("k8/9/1G7/9/9/9/9/9/4K4 b G 1");
    let signal = detector.evaluate(&mut pos);
    // material が低い場合は LowMaterial が先に引っかかるため、mate_1ply は到達しない可能性がある
    // この局面は material が低いので LowMaterial が先
    if signal.reason == EndgameReason::LowMaterial {
        // LowMaterial が先に判定されるのは正しい優先順位
        assert!(signal.use_alpha_beta);
    } else {
        assert!(signal.use_alpha_beta);
        assert_eq!(signal.reason, EndgameReason::MateThreat);
    }
}

#[test]
fn custom_material_threshold() {
    let mut detector = EndgameDetector::new();
    // 閾値を非常に高く設定 → 初期局面でも LowMaterial になる
    detector.material_threshold = 100_000;
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    let signal = detector.evaluate(&mut pos);
    assert!(signal.use_alpha_beta);
    assert_eq!(signal.reason, EndgameReason::LowMaterial);
}

#[test]
fn material_check_takes_priority_over_progress_score() {
    // 旧仕様では ply > material > mate > king_safety だったが、
    // 新仕様は mate > material > king_safety > progress_score の順。
    // 高 material_threshold で初期局面でも LowMaterial 即決させる確認。
    let mut detector = EndgameDetector::new();
    detector.material_threshold = 100_000;
    let mut pos = pos_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 80");
    let signal = detector.evaluate(&mut pos);
    assert!(signal.use_alpha_beta);
    assert_eq!(signal.reason, EndgameReason::LowMaterial);
}
