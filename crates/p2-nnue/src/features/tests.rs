//! V3 では旧 layout テストは廃止 (features 構造が dlshogi-style に変わったため)。
//! 新 features の最低限の sanity test のみ残す。

#[cfg(test)]
mod tests {
    use super::super::extract::extract_features;
    use super::super::feature_types::{
        FEATURES2_START, MAX_FEATURES2_HAND_NUM, P2_FEATURE_DIM, SQUARE_NUM,
    };
    use rshogi_core::position::Position;

    #[test]
    fn test_extract_hirate_runs() {
        let mut pos = Position::new();
        pos.set_hirate();
        let input = extract_features(&pos);
        assert_eq!(input.piece_features.len(), P2_FEATURE_DIM);

        // 平手は王手でない
        let in_check_plane_offset = FEATURES2_START + MAX_FEATURES2_HAND_NUM * SQUARE_NUM;
        for sq in 0..SQUARE_NUM {
            assert_eq!(input.piece_features[in_check_plane_offset + sq], 0);
        }
    }

    #[test]
    fn test_extract_features_some_set() {
        let mut pos = Position::new();
        pos.set_hirate();
        let input = extract_features(&pos);
        let active: usize = input.piece_features.iter().map(|&v| v as usize).sum();
        // 平手では駒 + 利き等で多数の bit が立つ
        assert!(active > 0, "expected some active features in hirate");
    }
}
