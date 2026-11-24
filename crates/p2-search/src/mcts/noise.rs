use rand::Rng;

/// Dirichlet 分布からノイズをサンプリングする
///
/// Gamma(α, 1) を各要素に独立サンプリングし、合計で正規化する。
/// `out` の各要素に結果を書き込む。
pub fn dirichlet_noise(rng: &mut impl Rng, alpha: f32, out: &mut [f32]) {
    let mut sum = 0.0f32;
    for x in out.iter_mut() {
        *x = sample_gamma(rng, alpha);
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in out.iter_mut() {
            *x *= inv;
        }
    }
}

/// Gamma(α, 1) のサンプリング
///
/// α >= 1: Marsaglia & Tsang (2000) の方法
/// α < 1: Ahrens-Dieter 変換 (α < 1 を α + 1 に持ち上げる)
fn sample_gamma(rng: &mut impl Rng, alpha: f32) -> f32 {
    if alpha < 1.0 {
        // α < 1: Gamma(α, 1) = Gamma(α + 1, 1) * U^(1/α)
        let g = sample_gamma_ge1(rng, alpha + 1.0);
        let u: f32 = rng.random();
        g * u.powf(1.0 / alpha)
    } else {
        sample_gamma_ge1(rng, alpha)
    }
}

/// Marsaglia & Tsang の Gamma(α, 1) サンプリング（α >= 1）
fn sample_gamma_ge1(rng: &mut impl Rng, alpha: f32) -> f32 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = sample_standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u: f32 = rng.random();

        // squeeze test
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Box-Muller 変換で標準正規分布をサンプリング
fn sample_standard_normal(rng: &mut impl Rng) -> f32 {
    loop {
        let u1: f32 = rng.random();
        if u1 > f32::EPSILON {
            let u2: f32 = rng.random();
            return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        }
    }
}
