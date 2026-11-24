//! `MlProgressPredictor` 推論 latency の criterion bench。
//!
//! ## Measured paths
//!
//! - `predict_features_scalar`: fallback path (純 scalar、 auto-vectorize 任せ)
//! - `predict_features` (AVX2): runtime detection 経由で AVX2 path
//! - `predict_full_position`: `Position` から特徴量抽出を含む end-to-end
//!
//! 実行:
//! ```text
//! cargo bench -p p2-progress-predictor --no-default-features --features cpu --bench predict
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use p2_progress_predictor::{
    features::{compute_features, ProgressFeatures},
    features_v2::{compute_features_v2, ProgressFeaturesV2},
    model::{MlProgressPredictor, ProgressWeights, HIDDEN1, HIDDEN2},
    model_v2::MlProgressPredictorV2,
    FEATURE_DIM,
};
use rshogi_core::position::Position;

/// 適当な非ゼロ重みを生成 (bench で 0 折れ paths が選択されないようにする)。
fn make_bench_weights() -> ProgressWeights {
    let mut w = ProgressWeights::zeros();
    let mut s: u32 = 0x9E3779B1;
    let mut next = || {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        (((s >> 8) & 0xffff) as f32 / 65536.0) - 0.5
    };
    for j in 0..HIDDEN1 {
        for k in 0..FEATURE_DIM {
            w.w1[j][k] = next() * 0.5;
        }
        w.b1[j] = next() * 0.1;
    }
    for j in 0..HIDDEN2 {
        for k in 0..HIDDEN1 {
            w.w2[j][k] = next() * 0.5;
        }
        w.b2[j] = next() * 0.1;
    }
    for k in 0..HIDDEN2 {
        w.w3[0][k] = next();
    }
    w.b3[0] = next() * 0.1;
    w
}

fn make_bench_feats() -> ProgressFeatures {
    let mut f = ProgressFeatures::default();
    let mut s: u32 = 0x12345678;
    for v in f.data.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        *v = ((s >> 8) & 0xffff) as f32 / 65536.0;
    }
    f
}

fn bench_predict(c: &mut Criterion) {
    let predictor = MlProgressPredictor::from_weights(make_bench_weights());
    let feats = make_bench_feats();

    let mut group = c.benchmark_group("progress_predict");

    // scalar fallback path
    group.bench_function("scalar", |b| {
        b.iter(|| {
            let y = predictor.predict_features_scalar(black_box(&feats));
            black_box(y)
        })
    });

    // runtime-dispatch path (AVX2 が detect されれば AVX2 path に入る)
    group.bench_function("dispatch", |b| {
        b.iter(|| {
            let y = predictor.predict_features(black_box(&feats));
            black_box(y)
        })
    });

    // End-to-end (Position → features → predict)
    let mut pos = Position::new();
    pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
        .unwrap();
    group.bench_function("full_position", |b| {
        b.iter(|| {
            let f = compute_features(black_box(&pos), 0);
            let y = predictor.predict_features(&f);
            black_box(y)
        })
    });

    // ============================================================
    // V2 model (48→64→32→16→1) benches
    // ============================================================
    let predictor_v2 = MlProgressPredictorV2::from_random(0xDEADBEEF);
    let feats_v2 = make_bench_feats_v2();

    group.bench_function("v2_scalar", |b| {
        b.iter(|| {
            let y = predictor_v2.predict_features_scalar(black_box(&feats_v2));
            black_box(y)
        })
    });

    group.bench_function("v2_dispatch", |b| {
        b.iter(|| {
            let y = predictor_v2.predict_features(black_box(&feats_v2));
            black_box(y)
        })
    });

    group.bench_function("v2_full_position", |b| {
        b.iter(|| {
            let f = compute_features_v2(black_box(&pos), 0, None);
            let y = predictor_v2.predict_features(&f);
            black_box(y)
        })
    });

    group.finish();
}

/// V2 用の bench feats (48 dim)
fn make_bench_feats_v2() -> ProgressFeaturesV2 {
    let mut f = ProgressFeaturesV2::default();
    let mut s: u32 = 0x9ABCDEF1;
    for v in f.data.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        *v = ((s >> 8) & 0xffff) as f32 / 65536.0;
    }
    f
}

criterion_group!(benches, bench_predict);
criterion_main!(benches);
