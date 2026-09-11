//! Per-edge SGD force updates for the UMAP low-dim layout. The attractive
//! term pulls connected points together; the repulsive term pushes
//! negative-sample pairs apart. A wide `COORD_BOUND` clamp only catches
//! truly degenerate (NaN / inf) edges; normal SGD never reaches it.

/// Shared distance epsilon (also used by the cross-entropy loss).
pub(crate) const EPS: f64 = 1e-3;
const COORD_BOUND: f64 = 100.0;

/// Single attractive update for edge `(i, j, w)`. Pulls
/// `y[i]` toward `y[j]` and vice versa by `alpha * gradient`
/// scaled by the edge weight.
pub(crate) fn apply_attractive(
    y: &mut [f64],
    i: usize,
    j: usize,
    w: f64,
    alpha: f64,
    a: f64,
    b: f64,
) {
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
pub(crate) fn apply_repulsive(y: &mut [f64], i: usize, k: usize, alpha: f64, a: f64, b: f64) {
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
