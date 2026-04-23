//! Saga 16 step 002: high-dim affinity matrix `P` for
//! t-SNE. Three phases, each a pure function: pairwise
//! squared distances -> per-row perplexity-calibrated
//! conditional probabilities -> symmetrize + clamp.
//!
//! Split out of `tsne_builtin.rs` so each phase is
//! individually testable and the orchestrator reads top-
//! to-bottom.

/// Minimum probability. Values below this are clamped up
/// so `log(q)` never hits -inf inside the gradient
/// computation (the gradient module imports this too).
pub(crate) const MIN_PROB: f64 = 1e-12;

const BISECTION_ITERS: usize = 50;
const BISECTION_TOL: f64 = 1e-5;

/// Orchestrator: pairwise squared distances -> per-row
/// bisection -> symmetrize + clamp.
pub(crate) fn compute_p_matrix(xs: &[f64], n: usize, d: usize, perplexity: f64) -> Vec<f64> {
    let target = perplexity.ln();
    let d2 = compute_pairwise_sqdist(xs, n, d);
    let mut p_cond = vec![0.0_f64; n * n];
    for i in 0..n {
        let row = bisect_perplexity_per_row(&d2[i * n..(i + 1) * n], i, n, target);
        for (j, v) in row.into_iter().enumerate() {
            p_cond[i * n + j] = v;
        }
    }
    symmetrize_and_clamp(&p_cond, n)
}

/// Phase 1: pairwise squared Euclidean distances over
/// `[N, D]` rows, returned as a row-major `[N, N]` flat
/// vector. Upper-triangle + mirror so each pair is
/// computed once.
fn compute_pairwise_sqdist(xs: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0_f64;
            for k in 0..d {
                let diff = xs[i * d + k] - xs[j * d + k];
                s += diff * diff;
            }
            out[i * n + j] = s;
            out[j * n + i] = s;
        }
    }
    out
}

/// Phase 2: for row `i`, binary-search a positive `beta`
/// such that the Shannon entropy of `P_{j|i} = exp(-beta
/// * d2[j]) / Z` matches `target = log(perplexity)`.
/// Returns the fitted row; self-index `i` stays zero.
fn bisect_perplexity_per_row(d2_row: &[f64], i: usize, n: usize, target: f64) -> Vec<f64> {
    let mut beta = 1.0_f64;
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    let mut last_row = vec![0.0_f64; n];
    for _ in 0..BISECTION_ITERS {
        let (row, entropy) = p_cond_row(d2_row, i, n, beta);
        last_row = row;
        let diff = entropy - target;
        if diff.abs() < BISECTION_TOL {
            break;
        }
        if diff > 0.0 {
            // H too high -> distribution too spread;
            // sharpen by increasing beta.
            lo = beta;
            beta = if hi.is_infinite() {
                beta * 2.0
            } else {
                (beta + hi) * 0.5
            };
        } else {
            hi = beta;
            beta = if lo.is_infinite() {
                beta * 0.5
            } else {
                (beta + lo) * 0.5
            };
        }
    }
    last_row
}

/// Inner: one fitted row at a specific beta. Returns
/// `(P_{·|i}, H(P_{·|i}))` with self excluded.
fn p_cond_row(d2_row: &[f64], i: usize, n: usize, beta: f64) -> (Vec<f64>, f64) {
    let mut row = vec![0.0_f64; n];
    let mut z = 0.0_f64;
    for j in 0..n {
        if j == i {
            continue;
        }
        let v = (-beta * d2_row[j]).exp();
        row[j] = v;
        z += v;
    }
    let mut entropy = 0.0_f64;
    if z > 0.0 {
        for (j, r) in row.iter_mut().enumerate() {
            if j == i {
                continue;
            }
            *r /= z;
            if *r > MIN_PROB {
                entropy -= *r * r.ln();
            }
        }
    }
    (row, entropy)
}

/// Phase 3: `P_{ij} = max((P_{j|i} + P_{i|j}) / (2N),
/// MIN_PROB)`. Returns the symmetric joint probability
/// matrix that the gradient descent loop consumes.
fn symmetrize_and_clamp(p_cond: &[f64], n: usize) -> Vec<f64> {
    let mut p = vec![0.0_f64; n * n];
    let norm = 2.0 * n as f64;
    for i in 0..n {
        for j in 0..n {
            let v = (p_cond[i * n + j] + p_cond[j * n + i]) / norm;
            p[i * n + j] = v.max(MIN_PROB);
        }
    }
    p
}
