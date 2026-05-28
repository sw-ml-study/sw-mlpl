//! Tests for the saga 33 step 041 mds(X, k, iters, seed) and
//! random_projection(X, k, seed) builtins (Phase 5 of the
//! dim-reduction milestone).

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime::call_builtin;

fn scalar(x: f64) -> DenseArray {
    DenseArray::from_scalar(x)
}

fn mat(rows: usize, cols: usize, data: &[f64]) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data.to_vec()).unwrap()
}

fn three_cluster_5d() -> DenseArray {
    // 30 points in 5-D: three clusters of 10 each.
    let mut data = Vec::with_capacity(30 * 5);
    let centers = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [5.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 5.0, 5.0],
    ];
    for c in &centers {
        for i in 0..10 {
            let t = i as f64 * 0.05;
            for (k, &cv) in c.iter().enumerate() {
                data.push(cv + t * (k as f64 - 2.0) * 0.1);
            }
        }
    }
    mat(30, 5, &data)
}

// ---- mds ----

#[test]
fn mds_output_shape_is_n_by_k() {
    let x = three_cluster_5d();
    let y = call_builtin("mds", vec![x, scalar(2.0), scalar(50.0), scalar(7.0)]).unwrap();
    assert_eq!(y.shape().dims(), &[30, 2]);
}

#[test]
fn mds_is_deterministic_for_same_seed() {
    let x = three_cluster_5d();
    let args = || vec![x.clone(), scalar(2.0), scalar(50.0), scalar(11.0)];
    let y1 = call_builtin("mds", args()).unwrap();
    let y2 = call_builtin("mds", args()).unwrap();
    assert_eq!(y1.data(), y2.data());
}

#[test]
fn mds_preserves_cluster_separation() {
    let x = three_cluster_5d();
    let y = call_builtin("mds", vec![x, scalar(2.0), scalar(100.0), scalar(13.0)]).unwrap();
    let d = y.data();
    // Centroids per cluster (10 points each)
    let mut centroids = [[0.0_f64; 2]; 3];
    let mut counts = [0_usize; 3];
    for (i, c) in centroids.iter_mut().enumerate() {
        for j in 0..10 {
            let idx = (i * 10 + j) * 2;
            c[0] += d[idx];
            c[1] += d[idx + 1];
            counts[i] += 1;
        }
        c[0] /= counts[i] as f64;
        c[1] /= counts[i] as f64;
    }
    // Inter-centroid: average pairwise distance between centroids
    let dist = |a: &[f64; 2], b: &[f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let inter = (dist(&centroids[0], &centroids[1])
        + dist(&centroids[0], &centroids[2])
        + dist(&centroids[1], &centroids[2]))
        / 3.0;
    // Intra-centroid: average within-cluster spread
    let mut intra = 0.0_f64;
    for (i, c) in centroids.iter().enumerate() {
        for j in 0..10 {
            let idx = (i * 10 + j) * 2;
            intra += dist(&[d[idx], d[idx + 1]], c);
        }
    }
    intra /= 30.0;
    assert!(
        inter > 1.5 * intra,
        "MDS clusters not separated: inter={inter}, intra={intra}"
    );
}

#[test]
fn mds_rejects_bad_args() {
    let x = three_cluster_5d();
    // Wrong arity
    assert!(call_builtin("mds", vec![x.clone(), scalar(2.0)]).is_err());
    // Non-integer k
    assert!(
        call_builtin(
            "mds",
            vec![x.clone(), scalar(1.5), scalar(10.0), scalar(1.0)]
        )
        .is_err()
    );
    // Negative iters
    assert!(
        call_builtin(
            "mds",
            vec![x.clone(), scalar(2.0), scalar(-1.0), scalar(1.0)]
        )
        .is_err()
    );
    // Non-rank-2 X
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(call_builtin("mds", vec![v, scalar(2.0), scalar(10.0), scalar(1.0)]).is_err());
}

// ---- random_projection ----

#[test]
fn random_projection_output_shape_is_n_by_k() {
    let x = three_cluster_5d();
    let y = call_builtin("random_projection", vec![x, scalar(2.0), scalar(7.0)]).unwrap();
    assert_eq!(y.shape().dims(), &[30, 2]);
}

#[test]
fn random_projection_is_deterministic_for_same_seed() {
    let x = three_cluster_5d();
    let y1 = call_builtin(
        "random_projection",
        vec![x.clone(), scalar(2.0), scalar(7.0)],
    )
    .unwrap();
    let y2 = call_builtin("random_projection", vec![x, scalar(2.0), scalar(7.0)]).unwrap();
    assert_eq!(y1.data(), y2.data());
}

#[test]
fn random_projection_different_seeds_differ() {
    let x = three_cluster_5d();
    let y1 = call_builtin(
        "random_projection",
        vec![x.clone(), scalar(2.0), scalar(1.0)],
    )
    .unwrap();
    let y2 = call_builtin("random_projection", vec![x, scalar(2.0), scalar(2.0)]).unwrap();
    let diff: f64 = y1
        .data()
        .iter()
        .zip(y2.data())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    assert!(diff > 0.0);
}

#[test]
fn random_projection_preserves_pairwise_distances_approximately() {
    // Johnson-Lindenstrauss: random projection to k dimensions
    // should preserve pairwise distances up to a (1 +- eps)
    // factor for modest k. Here N=30, D=5, k=3 -- a small
    // case; we check the average distortion is bounded.
    let x = three_cluster_5d();
    let y = call_builtin(
        "random_projection",
        vec![x.clone(), scalar(3.0), scalar(42.0)],
    )
    .unwrap();
    let n = 30;
    let xs = x.data();
    let ys = y.data();
    let mut total_ratio = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx: f64 = (0..5)
                .map(|c| (xs[i * 5 + c] - xs[j * 5 + c]).powi(2))
                .sum::<f64>()
                .sqrt();
            let dy: f64 = (0..3)
                .map(|c| (ys[i * 3 + c] - ys[j * 3 + c]).powi(2))
                .sum::<f64>()
                .sqrt();
            if dx > 1e-6 {
                total_ratio += dy / dx;
                count += 1;
            }
        }
    }
    let mean_ratio = total_ratio / count as f64;
    // Mean ratio should be roughly 1.0 with a generous
    // tolerance for the small k.
    assert!(
        (mean_ratio - 1.0).abs() < 0.5,
        "JL distortion too large: mean ratio = {mean_ratio:.3}"
    );
}

#[test]
fn random_projection_rejects_bad_args() {
    let x = three_cluster_5d();
    // Wrong arity
    assert!(call_builtin("random_projection", vec![x.clone(), scalar(2.0)]).is_err());
    // Non-integer k
    assert!(
        call_builtin(
            "random_projection",
            vec![x.clone(), scalar(1.5), scalar(1.0)]
        )
        .is_err()
    );
    // Non-rank-2 X
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(call_builtin("random_projection", vec![v, scalar(2.0), scalar(1.0)]).is_err());
}
