//! Fit the UMAP low-dim affinity parameters `(a, b)` so that
//! `phi(x) = 1 / (1 + a * x^(2b))` approximates the step-then-exponential
//! target curve controlled by `min_dist` and `spread` -- the McInnes &
//! Healy 2018 recipe, via Gauss-Newton on 2-parameter least squares.

/// Fit `(a, b)` so that `phi(x) = 1 / (1 + a * x^(2b))`
/// approximates the target curve
///   `target(x) = 1 if x < min_dist else exp(-(x-min_dist)/spread)`.
/// Gauss-Newton on the 2-parameter least-squares residuals
/// across 200 evenly-spaced x values in `[0, 3 * spread]`.
/// Matches the McInnes scipy curve_fit pre-computation step.
pub(crate) fn fit_ab_params(min_dist: f64, spread: f64) -> (f64, f64) {
    let (xs, ys) = sample_phi_target(min_dist, spread);
    let (mut a, mut b) = (1.0_f64, 1.0_f64);
    for _ in 0..40 {
        let (jtj, jtr) = gauss_newton_normals(&xs, &ys, a, b);
        let det = jtj[0][0] * jtj[1][1] - jtj[0][1] * jtj[1][0];
        if det.abs() < 1e-14 {
            break;
        }
        let da = (jtj[1][1] * jtr[0] - jtj[0][1] * jtr[1]) / det;
        let db = (jtj[0][0] * jtr[1] - jtj[1][0] * jtr[0]) / det;
        a = (a - da).max(1e-4);
        b = (b - db).max(1e-4);
        if da.abs() < 1e-7 && db.abs() < 1e-7 {
            break;
        }
    }
    (a, b)
}

/// Sample the step-then-exponential target curve at 200
/// evenly-spaced x values in `[0, 3 * spread]`. Used by the
/// Gauss-Newton fit; lifted out of `fit_ab_params` so each
/// stays under the function-LOC budget.
fn sample_phi_target(min_dist: f64, spread: f64) -> (Vec<f64>, Vec<f64>) {
    const N: usize = 200;
    let xs: Vec<f64> = (0..N)
        .map(|i| (i as f64 / (N - 1) as f64) * 3.0 * spread)
        .collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| {
            if x < min_dist {
                1.0
            } else {
                (-((x - min_dist) / spread)).exp()
            }
        })
        .collect();
    (xs, ys)
}

/// Accumulate the normal equations `(J^T J, J^T r)` for one
/// Gauss-Newton step on the (a, b) fit. Residuals are
/// `phi(x; a, b) - target(x)`.
fn gauss_newton_normals(xs: &[f64], ys: &[f64], a: f64, b: f64) -> ([[f64; 2]; 2], [f64; 2]) {
    let mut jtj = [[0.0_f64; 2]; 2];
    let mut jtr = [0.0_f64; 2];
    for (k, &x_raw) in xs.iter().enumerate() {
        let x = x_raw.max(1e-10);
        let x2b = x.powf(2.0 * b);
        let denom = 1.0 + a * x2b;
        let r = 1.0 / denom - ys[k];
        let dpa = -x2b / (denom * denom);
        let dpb = -2.0 * a * x2b * x.ln() / (denom * denom);
        jtj[0][0] += dpa * dpa;
        jtj[0][1] += dpa * dpb;
        jtj[1][1] += dpb * dpb;
        jtr[0] += dpa * r;
        jtr[1] += dpb * r;
    }
    jtj[1][0] = jtj[0][1];
    (jtj, jtr)
}
