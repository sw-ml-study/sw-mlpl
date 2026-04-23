//! Saga 16 step 002: per-step gradient descent for
//! t-SNE. Three pure phases plus one in-place update:
//! compute Q_unnorm + Z -> compute gradient -> apply
//! momentum/lr + center.
//!
//! Each phase is a pure function so the step function
//! reads as a short orchestrator.

use crate::tsne_affinities::MIN_PROB;

const LEARNING_RATE: f64 = 200.0;

/// One gradient descent step. Orchestrator over the
/// three phases. Mutates `y` and `prev_update` in place.
pub(crate) fn tsne_step(
    y: &mut [f64],
    p: &[f64],
    prev_update: &mut [f64],
    n: usize,
    exag: f64,
    momentum: f64,
) {
    let (q_unnorm, z) = compute_q_unnorm(y, n);
    let grad = compute_gradient(p, &q_unnorm, y, n, exag, z);
    apply_update_and_center(y, prev_update, &grad, n, momentum);
}

/// Phase 1: low-dim pairwise Student-t numerators
/// `q_unnorm[i, j] = 1 / (1 + ||Y_i - Y_j||^2)` and their
/// total `Z`. Self-diagonals stay zero.
fn compute_q_unnorm(y: &[f64], n: usize) -> (Vec<f64>, f64) {
    let mut q_unnorm = vec![0.0_f64; n * n];
    let mut z = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = y[i * 2] - y[j * 2];
            let dy = y[i * 2 + 1] - y[j * 2 + 1];
            let qn = 1.0 / (1.0 + dx * dx + dy * dy);
            q_unnorm[i * n + j] = qn;
            q_unnorm[j * n + i] = qn;
            z += 2.0 * qn;
        }
    }
    (q_unnorm, z)
}

/// Phase 2: gradient of the KL divergence w.r.t. `Y`.
/// `dY_i = 4 * sum_j (exag*P_{ij} - Q_{ij}) *
/// Q_{ij}^unnorm * (Y_i - Y_j)`.
fn compute_gradient(
    p: &[f64],
    q_unnorm: &[f64],
    y: &[f64],
    n: usize,
    exag: f64,
    z: f64,
) -> Vec<f64> {
    let z_inv = if z > 0.0 { 1.0 / z } else { 0.0 };
    let mut grad = vec![0.0_f64; n * 2];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let qn = q_unnorm[i * n + j];
            let q = (qn * z_inv).max(MIN_PROB);
            let pq = exag * p[i * n + j] - q;
            let mult = 4.0 * pq * qn;
            grad[i * 2] += mult * (y[i * 2] - y[j * 2]);
            grad[i * 2 + 1] += mult * (y[i * 2 + 1] - y[j * 2 + 1]);
        }
    }
    grad
}

/// Phase 3: momentum update `u = momentum * u_prev - lr *
/// grad; Y += u`, then subtract the per-axis mean so the
/// solution does not drift across iterations.
fn apply_update_and_center(
    y: &mut [f64],
    prev_update: &mut [f64],
    grad: &[f64],
    n: usize,
    momentum: f64,
) {
    for k in 0..n * 2 {
        prev_update[k] = momentum * prev_update[k] - LEARNING_RATE * grad[k];
        y[k] += prev_update[k];
    }
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    for i in 0..n {
        sum_x += y[i * 2];
        sum_y += y[i * 2 + 1];
    }
    let mx = sum_x / n as f64;
    let my = sum_y / n as f64;
    for i in 0..n {
        y[i * 2] -= mx;
        y[i * 2 + 1] -= my;
    }
}
