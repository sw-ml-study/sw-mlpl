//! Pure PCA computation helpers extracted from `pca_builtin`
//! so the entry-point module stays under the 7-fn ceiling
//! when Phase 1a of the dim-reduction milestone adds the new
//! `pca_components` + `pca_variance_explained` builtins.
//!
//! `extract_components_with_eigenvalues` is the saga 33 step
//! 031 evolution of the original `extract_components` -- now
//! also returns the per-component eigenvalues so the
//! variance-explained builtin can compute ratios without
//! re-running power iteration.

const POWER_ITERS: usize = 50;

/// Column-center `xs [N, D]`: subtract per-column mean from
/// every row. Returns a fresh flat buffer.
pub(crate) fn center_data(xs: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut means = vec![0.0_f64; d];
    for i in 0..n {
        for j in 0..d {
            means[j] += xs[i * d + j];
        }
    }
    let n_f = n as f64;
    for m in &mut means {
        *m /= n_f;
    }
    let mut xc = vec![0.0_f64; n * d];
    for i in 0..n {
        for j in 0..d {
            xc[i * d + j] = xs[i * d + j] - means[j];
        }
    }
    xc
}

/// `Cov = Xc^T Xc / N`, `[D, D]` row-major.
pub(crate) fn compute_cov(xc: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut cov = vec![0.0_f64; d * d];
    let n_f = n as f64;
    for a in 0..d {
        for b in a..d {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += xc[i * d + a] * xc[i * d + b];
            }
            let v = s / n_f;
            cov[a * d + b] = v;
            cov[b * d + a] = v;
        }
    }
    cov
}

/// Extract top-k principal components + their eigenvalues
/// from `cov [D, D]` via power iteration with Gram-Schmidt
/// orthogonalization + deflation. Returns
/// `(V [k, D] row-major, lambdas [k])`. The eigenvalue
/// sequence is in descending order by construction.
pub(crate) fn extract_components_with_eigenvalues(
    mut cov: Vec<f64>,
    d: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut components = vec![0.0_f64; k * d];
    let mut lambdas = vec![0.0_f64; k];
    let mut v = vec![0.0_f64; d];
    let mut next = vec![0.0_f64; d];
    for comp in 0..k {
        seed_basis_vector(&mut v, comp, d, &components);
        for _ in 0..POWER_ITERS {
            for a in 0..d {
                let row = &cov[a * d..a * d + d];
                next[a] = row.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
            }
            orthogonalize(&mut next, &components, comp, d);
            let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-30 {
                for (vi, ni) in v.iter_mut().zip(next.iter()) {
                    *vi = ni / norm;
                }
            }
        }
        let mut lambda = 0.0_f64;
        for a in 0..d {
            let row = &cov[a * d..a * d + d];
            let cv: f64 = row.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
            lambda += v[a] * cv;
        }
        components[comp * d..comp * d + d].copy_from_slice(&v);
        lambdas[comp] = lambda;
        for a in 0..d {
            for b in 0..d {
                cov[a * d + b] -= lambda * v[a] * v[b];
            }
        }
    }
    (components, lambdas)
}

/// Subtract projections of `w` onto the first `comp` rows
/// of `components` (each a unit vector in `[k, D]` row-major
/// layout). In-place.
pub(crate) fn orthogonalize(w: &mut [f64], components: &[f64], comp: usize, d: usize) {
    for prior in 0..comp {
        let pv = &components[prior * d..prior * d + d];
        let dot: f64 = w.iter().zip(pv.iter()).map(|(a, b)| a * b).sum();
        for (wi, pi) in w.iter_mut().zip(pv.iter()) {
            *wi -= dot * pi;
        }
    }
}

/// Seed `v` with a basis vector orthogonal to all previously
/// extracted components. Tries `e_{comp}`, `e_{comp+1}`, ...
/// until Gram-Schmidt leaves a nonzero residual to normalize.
/// (Helper for `extract_components_with_eigenvalues`; kept
/// here rather than inline to keep that fn under 25 LOC.)
fn seed_basis_vector(v: &mut [f64], comp: usize, d: usize, components: &[f64]) {
    for start in 0..d {
        v.iter_mut().for_each(|x| *x = 0.0);
        v[(comp + start) % d] = 1.0;
        orthogonalize(v, components, comp, d);
        let n0 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n0 > 1e-12 {
            for vi in v.iter_mut() {
                *vi /= n0;
            }
            return;
        }
    }
}
