//! Low-dim layout SGD for UMAP: optimizes a 2-D embedding `Y [N, 2]`
//! against a fuzzy simplicial edge list `Vec<(i, j, w)>` (output of
//! `umap_simplicial::fuzzy_simplicial_set`). The objective is the
//! cross-entropy between the high-dim fuzzy graph and the low-dim affinity
//! `phi(d) = 1 / (1 + a * d^(2b))`, with `a, b` fitted from `min_dist` by
//! `umap_ab_fit` and the per-edge gradient split into attractive and
//! repulsive terms in `umap_forces`. Negative sampling (per attractive
//! update, `N_NEG` random repulsive targets) is the scalability trick that
//! avoids the O(N^2) pairwise sum t-SNE pays. Learning rate decays linearly
//! with iteration: `alpha = lr0 * (1 - t / total)`.

use mlpl_runtime_core::prng::Xorshift64;

use crate::umap_ab_fit::fit_ab_params;
use crate::umap_forces::{EPS, apply_attractive, apply_repulsive};

const LR0: f64 = 1.0;
const N_NEG: usize = 5;
const INIT_SCALE: f64 = 10.0;
const SPREAD: f64 = 1.0;

/// Initial `Y [N, 2]`: uniform random in `[-INIT_SCALE,
/// INIT_SCALE]^2`. UMAP's reference impl spectral-inits from the fuzzy
/// graph's normalized Laplacian; the random init here is what scikit
/// falls back to when spectral fails and is good enough for the small-N
/// milestone scope.
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
