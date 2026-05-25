//! Saga 33 step 034: low-dim layout SGD for UMAP.
//!
//! Optimizes a 2-D embedding `Y [N, 2]` against a fuzzy
//! simplicial edge list `Vec<(i, j, w)>` (output of
//! `umap_simplicial::fuzzy_simplicial_set`). The objective
//! is the cross-entropy between the high-dim fuzzy graph
//! and the low-dim Student-t affinity
//! `phi(d) = 1 / (1 + a * d^(2b))`. The analytical gradient
//! per edge factors into two terms:
//!
//! - Attractive (pulls connected points together):
//!   `g_attr = -2 * a * b * d^(2(b-1)) / (1 + a * d^(2b))`
//!   scaled by edge weight `w`.
//! - Repulsive (pushes negative-sample pairs apart):
//!   `g_rep = 2 * b / ((eps + d_sq) * (1 + a * d_sq^b))`
//!
//! We use `a = 1, b = 1` (Student-t / Cauchy curve), the
//! same low-D affinity t-SNE uses. The `min_dist` parameter
//! enters via a soft floor: when `d_sq < min_dist^2` the
//! attractive force is zeroed (points are already close
//! enough). McInnes' reference impl fits `a, b` from
//! `(min_dist, spread)` via curve-fit; the simpler clamp
//! here matches the milestone's "small UMAP" scope.
//!
//! Negative sampling: per attractive update we sample
//! `N_NEG` random targets for the repulsive term, which is
//! the load-bearing scalability trick from the original
//! paper (avoids the O(N^2) pairwise sum that t-SNE pays).
//!
//! Learning rate decays linearly with iteration: `alpha =
//! lr0 * (1 - t / total)`. Reference impl default lr0 = 1.0.

use mlpl_runtime_core::prng::Xorshift64;

const A: f64 = 1.0;
const B: f64 = 1.0;
const EPS: f64 = 1e-3;
const LR0: f64 = 1.0;
const N_NEG: usize = 5;
const INIT_SCALE: f64 = 2.0;

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

/// Run SGD for `iters` epochs over `edges`, with per-epoch
/// `N_NEG` negative samples per edge. Mutates `y` in place.
/// Returns the final loss (for the loss-decreases test).
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
    let min_dist_sq = min_dist * min_dist;
    let total = iters.max(1) as f64;
    let mut last_loss = 0.0;
    for t in 0..iters {
        let alpha = LR0 * (1.0 - (t as f64) / total);
        for &(i, j, w) in edges {
            apply_attractive(y, i, j, w, alpha, min_dist_sq);
            for _ in 0..N_NEG {
                let kk = (rng.next_f64() * n as f64) as usize % n.max(1);
                if kk != i {
                    apply_repulsive(y, i, kk, alpha);
                }
            }
        }
        last_loss = cross_entropy_loss(y, edges);
    }
    last_loss
}

/// Single attractive update for edge `(i, j, w)`. Pulls
/// `y[i]` toward `y[j]` and vice versa by alpha * gradient
/// scaled by the edge weight. The `clamp` keeps each
/// coordinate inside `[-INIT_SCALE * 2, INIT_SCALE * 2]`
/// so a degenerate edge cannot blow Y to infinity.
fn apply_attractive(y: &mut [f64], i: usize, j: usize, w: f64, alpha: f64, min_dist_sq: f64) {
    let dx = y[i * 2] - y[j * 2];
    let dy = y[i * 2 + 1] - y[j * 2 + 1];
    let d_sq = dx * dx + dy * dy;
    if d_sq < min_dist_sq {
        return;
    }
    let g = w * (-2.0 * A * B) / (1.0 + A * d_sq);
    let (ux, uy) = (alpha * g * dx, alpha * g * dy);
    y[i * 2] = clamp_coord(y[i * 2] + ux);
    y[i * 2 + 1] = clamp_coord(y[i * 2 + 1] + uy);
    y[j * 2] = clamp_coord(y[j * 2] - ux);
    y[j * 2 + 1] = clamp_coord(y[j * 2 + 1] - uy);
}

/// Single repulsive update for negative sample `(i, k)`.
/// Pushes `y[i]` away from `y[k]`; `k` itself is not
/// updated -- UMAP's negative-sampling convention only
/// updates the source point.
fn apply_repulsive(y: &mut [f64], i: usize, k: usize, alpha: f64) {
    let dx = y[i * 2] - y[k * 2];
    let dy = y[i * 2 + 1] - y[k * 2 + 1];
    let d_sq = dx * dx + dy * dy;
    let g = 2.0 * B / ((EPS + d_sq) * (1.0 + A * d_sq));
    let (ux, uy) = (alpha * g * dx, alpha * g * dy);
    y[i * 2] = clamp_coord(y[i * 2] + ux);
    y[i * 2 + 1] = clamp_coord(y[i * 2 + 1] + uy);
}

fn clamp_coord(v: f64) -> f64 {
    v.clamp(-INIT_SCALE * 2.0, INIT_SCALE * 2.0)
}

/// Closed-form cross-entropy loss `-sum_e [w * log(q) +
/// (1-w) * log(1-q)]` where `q = phi(d^2)`. Used only for
/// the "loss decreases" test; the SGD itself never reads
/// it.
pub(crate) fn cross_entropy_loss(y: &[f64], edges: &[(usize, usize, f64)]) -> f64 {
    edges
        .iter()
        .map(|&(i, j, w)| {
            let dx = y[i * 2] - y[j * 2];
            let dy = y[i * 2 + 1] - y[j * 2 + 1];
            let q = (1.0 / (1.0 + A * (dx * dx + dy * dy).powf(B))).clamp(EPS, 1.0 - EPS);
            -(w * q.ln() + (1.0 - w) * (1.0 - q).ln())
        })
        .sum()
}
