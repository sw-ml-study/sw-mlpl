//! Saga 33 step 034: fuzzy simplicial set construction --
//! turn a k-NN edge list into a symmetric weighted graph
//! whose edge weights are local fuzzy-set memberships.
//!
//! Algorithm (McInnes & Healy 2018, section 2.2-2.3):
//!
//! 1. For each source row i, look at its k nearest non-self
//!    neighbors and let rho_i = min dist (so the nearest
//!    neighbor has membership 1.0 -- preserves locality).
//! 2. Binary-search sigma_i so that
//!    `sum_j exp(-max(0, d_ij - rho_i) / sigma_i) = log2(k)`.
//!    This calibrates per-row scale so every point has the
//!    same effective Shannon entropy of memberships.
//! 3. Compute directed memberships
//!    `w_ij = exp(-max(0, d_ij - rho_i) / sigma_i)`.
//! 4. Symmetrize via fuzzy union:
//!    `t_ij = w_ij + w_ji - w_ij * w_ji`.
//!    Encodes "either side's local view says these points
//!    are neighbors."
//!
//! Returns a deduped undirected edge list `Vec<(i, j, w)>`
//! with `i < j` to make the downstream layout SGD
//! straightforward.

const SIGMA_ITERS: usize = 32;
const SIGMA_MIN: f64 = 1e-10;
const SIGMA_MAX: f64 = 1e10;

/// Build the symmetric fuzzy edge list from a `[N*k, 3]`
/// directed (i, j, dist) edge list (as emitted by
/// `knn_graph(X, k)`). Edges with `i > j` are folded into
/// the `i < j` direction via the fuzzy-union formula
/// `t = w + w' - w * w'`; pairs that appear only one way
/// (`w' = 0`) keep weight `t = w`.
pub(crate) fn fuzzy_simplicial_set(knn: &[f64], n: usize, k: usize) -> Vec<(usize, usize, f64)> {
    let target = (k as f64).log2();
    let directed: Vec<(usize, usize, f64)> = (0..n)
        .flat_map(|i| row_memberships(knn, i, k, target))
        .collect();
    symmetrize_pairs(directed)
}

/// Compute the k outgoing memberships from source `i`.
/// Reads row `i`'s k slots from the `[N*k, 3]` knn buffer,
/// runs the binary search for sigma, and returns the
/// `(i, j, w_ij)` triples.
fn row_memberships(knn: &[f64], i: usize, k: usize, target: f64) -> Vec<(usize, usize, f64)> {
    let dists: Vec<f64> = (0..k).map(|p| knn[(i * k + p) * 3 + 2]).collect();
    let rho = dists.first().copied().unwrap_or(0.0);
    let sigma = solve_sigma(&dists, rho, target);
    (0..k)
        .map(|p| {
            let j = knn[(i * k + p) * 3 + 1] as usize;
            let w = (-((dists[p] - rho).max(0.0)) / sigma).exp();
            (i, j, w)
        })
        .collect()
}

/// Binary search for `sigma` so the membership sum from
/// `i` equals `target = log2(k)`. Bracketed by `[SIGMA_MIN,
/// SIGMA_MAX]`; converges in ~32 iterations to <1e-5.
fn solve_sigma(dists: &[f64], rho: f64, target: f64) -> f64 {
    let (mut lo, mut hi) = (SIGMA_MIN, SIGMA_MAX);
    let mut sigma = 1.0;
    for _ in 0..SIGMA_ITERS {
        let s: f64 = dists
            .iter()
            .map(|&d| (-((d - rho).max(0.0)) / sigma).exp())
            .sum();
        if (s - target).abs() < 1e-5 {
            return sigma;
        }
        if s > target {
            hi = sigma;
            sigma = 0.5 * (lo + sigma);
        } else {
            lo = sigma;
            sigma = if hi >= SIGMA_MAX {
                sigma * 2.0
            } else {
                0.5 * (lo + hi)
            };
        }
    }
    sigma
}

/// Fold a directed `(i, j, w)` list into a symmetric
/// undirected `(min, max, w_sym)` list using the fuzzy
/// union `t = w + w' - w * w'`. Output order is canonical
/// (sorted by `(min, max)`) so test assertions are stable.
fn symmetrize_pairs(directed: Vec<(usize, usize, f64)>) -> Vec<(usize, usize, f64)> {
    use std::collections::HashMap;
    let mut pair: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    for (i, j, w) in directed {
        if i == j {
            continue;
        }
        let key = if i < j { (i, j) } else { (j, i) };
        let entry = pair.entry(key).or_insert((0.0, 0.0));
        if i < j {
            entry.0 = entry.0.max(w);
        } else {
            entry.1 = entry.1.max(w);
        }
    }
    let mut out: Vec<(usize, usize, f64)> = pair
        .into_iter()
        .map(|((a, b), (w, w2))| (a, b, w + w2 - w * w2))
        .filter(|&(_, _, t)| t > 0.0)
        .collect();
    out.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    out
}
