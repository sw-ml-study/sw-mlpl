//! Saga 33 step 037c: low-dim layout SGD for UMAP with
//! `min_dist`-driven `a, b` curve fit and no tight coord
//! clamp.
//!
//! Optimizes a 2-D embedding `Y [N, 2]` against a fuzzy
//! simplicial edge list `Vec<(i, j, w)>` (output of
//! `umap_simplicial::fuzzy_simplicial_set`). The objective
//! is the cross-entropy between the high-dim fuzzy graph
//! and the low-dim affinity `phi(d) = 1 / (1 + a * d^(2b))`
//! where `a, b` are fitted so that `phi` approximates a
//! step-then-exponential target curve controlled by
//! `min_dist` and `spread` -- the McInnes & Healy 2018
//! recipe. Smaller `min_dist` -> sharper a/b -> tighter
//! clusters; larger `min_dist` -> broader a/b -> looser
//! arrangement.
//!
//! Analytical gradient per edge factors into two terms:
//! - Attractive (pulls connected points together):
//!   `g_attr = -2 * a * b * d_sq^(b-1) / (1 + a * d_sq^b)`
//!   scaled by edge weight `w`.
//! - Repulsive (pushes negative-sample pairs apart):
//!   `g_rep = 2 * b / ((eps + d_sq) * (1 + a * d_sq^b))`
//!
//! Negative sampling: per attractive update we sample
//! `N_NEG` random targets for the repulsive term, which is
//! the load-bearing scalability trick from the original
//! paper (avoids the O(N^2) pairwise sum that t-SNE pays).
//!
//! Learning rate decays linearly with iteration: `alpha =
//! lr0 * (1 - t / total)`. Reference impl default lr0 = 1.0.
//!
//! Coordinate safety bound: a soft +-COORD_BOUND clamp that
//! only catches truly degenerate edges (NaN / inf). Pre-
//! step-037c the bound was +-4, which forced the layout into
//! a small box and produced visible piling against the
//! walls; now the bound is +-100 so the SGD has room to
//! spread.

use mlpl_runtime_core::prng::Xorshift64;

const EPS: f64 = 1e-3;
const LR0: f64 = 1.0;
const N_NEG: usize = 5;
const INIT_SCALE: f64 = 10.0;
const COORD_BOUND: f64 = 100.0;
const SPREAD: f64 = 1.0;

/// Initial `Y [N, 2]`: uniform random in `[-INIT_SCALE,
/// INIT_SCALE]^2`. UMAP's reference impl spectral-inits
/// from the fuzzy graph's normalized Laplacian; the random
/// init here is what scikit falls back to when spectral
/// fails and is good enough for the small-N milestone
/// scope.
pub(crate) fn init_layout(n: usize, seed: f64) -> Vec<f64> {
    let raw_seed = seed as i64 as u64;
    let mut rng = Xorshift64::new(raw_seed.max(1));
    (0..n * 2)
        .map(|_| (rng.next_f64() * 2.0 - 1.0) * INIT_SCALE)
        .collect()
}

/// Run SGD for `iters` epochs over `edges`. Mutates `y` in
/// place. Returns the final loss (for the loss-decreases
/// test). `a, b` are fitted from `min_dist` once at the
/// start.
pub(crate) fn run_layout_sgd(
    y: &mut [f64],
    edges: &[(usize, usize, f64)],
    iters: usize,
    seed: f64,
    min_dist: f64,
) -> f64 {
    let n = y.len() / 2;
    let raw_seed = seed as i64 as u64;
    let mut rng = Xorshift64::new(raw_seed.wrapping_add(0x_A1B2_C3D4));
    let (a, b) = fit_ab_params(min_dist, SPREAD);
    let total = iters.max(1) as f64;
    let mut last_loss = 0.0;
    for t in 0..iters {
        let alpha = LR0 * (1.0 - (t as f64) / total);
        for &(i, j, w) in edges {
            apply_attractive(y, i, j, w, alpha, a, b);
            for _ in 0..N_NEG {
                let kk = (rng.next_f64() * n as f64) as usize % n.max(1);
                if kk != i {
                    apply_repulsive(y, i, kk, alpha, a, b);
                }
            }
        }
        last_loss = cross_entropy_loss(y, edges, a, b);
    }
    last_loss
}

/// Fit `(a, b)` so that `phi(x) = 1 / (1 + a * x^(2b))`
/// approximates the target curve
///   `target(x) = 1 if x < min_dist else exp(-(x-min_dist)/spread)`.
/// Gauss-Newton on the 2-parameter least-squares residuals
/// across 200 evenly-spaced x values in `[0, 3 * spread]`.
/// Matches the McInnes scipy curve_fit pre-computation step.
fn fit_ab_params(min_dist: f64, spread: f64) -> (f64, f64) {
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

/// Single attractive update for edge `(i, j, w)`. Pulls
/// `y[i]` toward `y[j]` and vice versa by `alpha * gradient`
/// scaled by the edge weight. The wide `COORD_BOUND` clamp
/// catches truly degenerate edges that would produce NaN /
/// inf; normal SGD never reaches it.
fn apply_attractive(y: &mut [f64], i: usize, j: usize, w: f64, alpha: f64, a: f64, b: f64) {
    let dx = y[i * 2] - y[j * 2];
    let dy = y[i * 2 + 1] - y[j * 2 + 1];
    let d_sq = dx * dx + dy * dy;
    let safe_d_sq = d_sq.max(EPS);
    let g = w * (-2.0 * a * b * safe_d_sq.powf(b - 1.0)) / (1.0 + a * safe_d_sq.powf(b));
    let (ux, uy) = (alpha * g * dx, alpha * g * dy);
    y[i * 2] = clamp_safe(y[i * 2] + ux);
    y[i * 2 + 1] = clamp_safe(y[i * 2 + 1] + uy);
    y[j * 2] = clamp_safe(y[j * 2] - ux);
    y[j * 2 + 1] = clamp_safe(y[j * 2 + 1] - uy);
}

/// Single repulsive update for negative sample `(i, k)`.
/// Pushes `y[i]` away from `y[k]`; `k` itself is not
/// updated -- UMAP's negative-sampling convention only
/// updates the source point.
fn apply_repulsive(y: &mut [f64], i: usize, k: usize, alpha: f64, a: f64, b: f64) {
    let dx = y[i * 2] - y[k * 2];
    let dy = y[i * 2 + 1] - y[k * 2 + 1];
    let d_sq = dx * dx + dy * dy;
    let g = 2.0 * b / ((EPS + d_sq) * (1.0 + a * d_sq.max(EPS).powf(b)));
    let (ux, uy) = (alpha * g * dx, alpha * g * dy);
    y[i * 2] = clamp_safe(y[i * 2] + ux);
    y[i * 2 + 1] = clamp_safe(y[i * 2 + 1] + uy);
}

fn clamp_safe(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(-COORD_BOUND, COORD_BOUND)
    } else {
        0.0
    }
}

/// Closed-form cross-entropy loss `-sum_e [w * log(q) +
/// (1-w) * log(1-q)]` where `q = phi(d^2)`. Used only for
/// the "loss decreases" test; the SGD itself never reads
/// it.
pub(crate) fn cross_entropy_loss(y: &[f64], edges: &[(usize, usize, f64)], a: f64, b: f64) -> f64 {
    edges
        .iter()
        .map(|&(i, j, w)| {
            let dx = y[i * 2] - y[j * 2];
            let dy = y[i * 2 + 1] - y[j * 2 + 1];
            let d_sq = (dx * dx + dy * dy).max(EPS);
            let q = (1.0 / (1.0 + a * d_sq.powf(b))).clamp(EPS, 1.0 - EPS);
            -(w * q.ln() + (1.0 - w) * (1.0 - q).ln())
        })
        .sum()
}
