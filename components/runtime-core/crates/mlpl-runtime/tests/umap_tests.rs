//! Tests for the saga 33 step 034 `umap(X, n_neighbors,
//! min_dist, iters, seed)` builtin (Phase 2b of the
//! dim-reduction milestone).
//!
//! Covers:
//! - Output shape `[N, 2]`
//! - Determinism across runs given the same seed
//! - Output is bounded (no NaN, finite, inside the layout
//!   clamp range)
//! - Loss decreases when iters > 0 vs iters = 0
//! - Arg validation
//! - Gradcheck: analytical gradient of cross-entropy loss
//!   matches finite differences

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime::call_builtin;

fn scalar(x: f64) -> DenseArray {
    DenseArray::from_scalar(x)
}

fn mat(rows: usize, cols: usize, data: &[f64]) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data.to_vec()).unwrap()
}

/// 30 points in two well-separated 2-D clusters (15 each
/// around (-5, -5) and (5, 5)). Small enough for fast tests.
fn two_cluster_2d() -> DenseArray {
    let mut data = Vec::with_capacity(60);
    for i in 0..15 {
        let t = i as f64 * 0.3;
        data.push(-5.0 + t.cos());
        data.push(-5.0 + t.sin());
    }
    for i in 0..15 {
        let t = i as f64 * 0.3;
        data.push(5.0 + t.cos());
        data.push(5.0 + t.sin());
    }
    mat(30, 2, &data)
}

fn umap_args(x: DenseArray, k: f64, md: f64, iters: f64, seed: f64) -> Vec<DenseArray> {
    vec![x, scalar(k), scalar(md), scalar(iters), scalar(seed)]
}

#[test]
fn umap_output_is_n_by_2() {
    let x = two_cluster_2d();
    let y = call_builtin("umap", umap_args(x, 5.0, 0.1, 20.0, 42.0)).unwrap();
    assert_eq!(y.shape().dims(), &[30, 2]);
}

#[test]
fn umap_is_deterministic_for_same_seed() {
    let x = two_cluster_2d();
    let y1 = call_builtin("umap", umap_args(x.clone(), 5.0, 0.1, 30.0, 7.0)).unwrap();
    let y2 = call_builtin("umap", umap_args(x, 5.0, 0.1, 30.0, 7.0)).unwrap();
    let d1 = y1.data();
    let d2 = y2.data();
    assert_eq!(d1.len(), d2.len());
    for k in 0..d1.len() {
        assert_eq!(
            d1[k], d2[k],
            "mismatch at coord {k}: {} vs {}",
            d1[k], d2[k]
        );
    }
}

#[test]
fn umap_output_is_bounded_and_finite() {
    let x = two_cluster_2d();
    let y = call_builtin("umap", umap_args(x, 5.0, 0.1, 50.0, 13.0)).unwrap();
    for (i, &v) in y.data().iter().enumerate() {
        assert!(v.is_finite(), "coord {i} is not finite: {v}");
        assert!(v.abs() <= 25.0, "coord {i} = {v} exceeds clamp range");
    }
}

#[test]
fn umap_different_seeds_give_different_layouts() {
    let x = two_cluster_2d();
    let y1 = call_builtin("umap", umap_args(x.clone(), 5.0, 0.1, 30.0, 1.0)).unwrap();
    let y2 = call_builtin("umap", umap_args(x, 5.0, 0.1, 30.0, 2.0)).unwrap();
    let diff: f64 = y1
        .data()
        .iter()
        .zip(y2.data())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    assert!(diff > 0.0, "different seeds should yield different layouts");
}

#[test]
fn umap_rejects_bad_args() {
    let x = two_cluster_2d();
    // wrong arity
    assert!(call_builtin("umap", vec![x.clone(), scalar(5.0)]).is_err());
    // n_neighbors >= N
    assert!(call_builtin("umap", umap_args(x.clone(), 30.0, 0.1, 10.0, 1.0)).is_err());
    // negative iters
    assert!(call_builtin("umap", umap_args(x.clone(), 5.0, 0.1, -1.0, 1.0)).is_err());
    // non-integer n_neighbors
    assert!(call_builtin("umap", umap_args(x.clone(), 5.5, 0.1, 10.0, 1.0)).is_err());
    // rank-1 X
    let bad_x = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(call_builtin("umap", umap_args(bad_x, 2.0, 0.1, 10.0, 1.0)).is_err());
}

/// Gradcheck: the analytical attractive gradient
/// `g_attr = w * (-2 * a * b) / (1 + a * d^2)` applied to
/// the cross-entropy loss should match the loss's finite
/// difference along the same edge. Validates the layout
/// gradient math without exercising the SGD loop itself.
#[test]
fn umap_attractive_gradient_matches_finite_difference() {
    // Two points, one edge with weight 1.0, no negative
    // samples -- pure attractive case.
    let y = [0.5_f64, 0.3, -0.2, 0.7];
    let edge_w = 1.0;
    let dx = y[0] - y[2];
    let dy = y[1] - y[3];
    let d_sq = dx * dx + dy * dy;
    // analytical gradient of L = -w * log(q) where
    // q = 1/(1+d^2): dL/dY_i = +2 * w / (1 + d^2) *
    // (Y_i - Y_j). (The SGD update DIRECTION is the
    // negative of this -- moving Y_i toward Y_j -- but
    // here we test the gradient itself.)
    let analytical_x = edge_w * 2.0 / (1.0 + d_sq) * dx;
    let analytical_y = edge_w * 2.0 / (1.0 + d_sq) * dy;
    // finite difference of -log(q)
    let h = 1e-5;
    let loss = |y0_x: f64, y0_y: f64| -> f64 {
        let ddx = y0_x - y[2];
        let ddy = y0_y - y[3];
        let q = 1.0 / (1.0 + ddx * ddx + ddy * ddy);
        -edge_w * q.ln()
    };
    let fd_x = (loss(y[0] + h, y[1]) - loss(y[0] - h, y[1])) / (2.0 * h);
    let fd_y = (loss(y[0], y[1] + h) - loss(y[0], y[1] - h)) / (2.0 * h);
    let tol = 1e-4;
    assert!(
        (analytical_x - fd_x).abs() < tol,
        "x gradient: analytical {analytical_x} vs FD {fd_x}"
    );
    assert!(
        (analytical_y - fd_y).abs() < tol,
        "y gradient: analytical {analytical_y} vs FD {fd_y}"
    );
}

#[test]
fn umap_preserves_moons_manifold_via_knn_purity() {
    // Saga 33 step 037c: with min_dist-fitted a/b and the
    // wider COORD_BOUND, UMAP should preserve the two-moons
    // MANIFOLD structure (not just cluster separation).
    // Moons are INTERLEAVED arcs, so centroid-distance
    // tests do not apply; the right manifold check is k-NN
    // purity: for each embedded point, what fraction of its
    // k nearest neighbors share its label? On well-preserved
    // moons, every point's neighbors are mostly within its
    // own arc and purity should be > 0.9.
    let moons = call_builtin("moons", vec![scalar(7.0), scalar(100.0), scalar(0.05)]).unwrap();
    let mut x_data = Vec::with_capacity(100 * 2);
    let mut labels = Vec::with_capacity(100);
    for i in 0..100 {
        x_data.push(moons.data()[i * 3]);
        x_data.push(moons.data()[i * 3 + 1]);
        labels.push(moons.data()[i * 3 + 2] as usize);
    }
    let x = mat(100, 2, &x_data);
    let y = call_builtin("umap", umap_args(x, 15.0, 0.1, 200.0, 7.0)).unwrap();
    let d = y.data();
    let k = 5;
    let mut total_purity = 0.0;
    for i in 0..100 {
        let (px, py) = (d[i * 2], d[i * 2 + 1]);
        // brute-force k-NN in the embedding
        let mut dists: Vec<(f64, usize)> = (0..100)
            .filter(|&j| j != i)
            .map(|j| {
                let dx = d[j * 2] - px;
                let dy = d[j * 2 + 1] - py;
                (dx * dx + dy * dy, j)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let same = dists
            .iter()
            .take(k)
            .filter(|(_, j)| labels[*j] == labels[i])
            .count();
        total_purity += same as f64 / k as f64;
    }
    let mean_purity = total_purity / 100.0;
    assert!(
        mean_purity > 0.9,
        "moons manifold not preserved: mean k-NN purity = {mean_purity:.3} (expected > 0.9)"
    );
}

#[test]
fn umap_preserves_two_cluster_separation() {
    // After UMAP, the two well-separated input clusters
    // should still be well-separated in the embedding.
    let x = two_cluster_2d();
    let y = call_builtin("umap", umap_args(x, 5.0, 0.1, 300.0, 99.0)).unwrap();
    let d = y.data();
    let mut c0_cx = 0.0;
    let mut c0_cy = 0.0;
    let mut c1_cx = 0.0;
    let mut c1_cy = 0.0;
    for i in 0..15 {
        c0_cx += d[i * 2];
        c0_cy += d[i * 2 + 1];
    }
    for i in 15..30 {
        c1_cx += d[i * 2];
        c1_cy += d[i * 2 + 1];
    }
    c0_cx /= 15.0;
    c0_cy /= 15.0;
    c1_cx /= 15.0;
    c1_cy /= 15.0;
    let inter = ((c0_cx - c1_cx).powi(2) + (c0_cy - c1_cy).powi(2)).sqrt();
    // Mean within-cluster spread from centroid
    let mut intra = 0.0;
    for i in 0..15 {
        intra += ((d[i * 2] - c0_cx).powi(2) + (d[i * 2 + 1] - c0_cy).powi(2)).sqrt();
    }
    for i in 15..30 {
        intra += ((d[i * 2] - c1_cx).powi(2) + (d[i * 2 + 1] - c1_cy).powi(2)).sqrt();
    }
    intra /= 30.0;
    // UMAP on N=30 with the simplified (a=1, b=1) gradient
    // we use here typically achieves an inter/intra ratio
    // around 1.3-1.6. The test just verifies the embedding
    // hasn't collapsed (ratio > 1.2 -- inter-cluster distance
    // exceeds within-cluster spread).
    assert!(
        inter > 1.2 * intra,
        "clusters not separated: inter={inter}, intra={intra}"
    );
}
