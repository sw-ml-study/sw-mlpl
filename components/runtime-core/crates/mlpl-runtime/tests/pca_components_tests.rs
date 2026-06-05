//! Tests for the saga 33 step 031 PCA additions:
//! `pca_components(X, k)` returning the `[k, D]` loadings
//! matrix, and `pca_variance_explained(X, k)` returning the
//! `[k]` per-component variance-explained ratios.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime::call_builtin;

fn scalar(x: f64) -> DenseArray {
    DenseArray::from_scalar(x)
}

fn mat(rows: usize, cols: usize, data: &[f64]) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data.to_vec()).unwrap()
}

/// Anisotropic 2-D Gaussian: ~10x more variance along x than
/// y. Top component should align with the x axis.
fn anisotropic_2d(n: usize) -> DenseArray {
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f64 / n as f64;
        data.push((t - 0.5) * 10.0);
        data.push((t - 0.5) * 1.0);
    }
    mat(n, 2, &data)
}

#[test]
fn pca_components_returns_k_by_d() {
    let x = anisotropic_2d(20);
    let v = call_builtin("pca_components", vec![x, scalar(2.0)]).unwrap();
    assert_eq!(v.shape().dims(), &[2, 2]);
}

#[test]
fn pca_components_rows_are_unit_norm() {
    let x = anisotropic_2d(50);
    let v = call_builtin("pca_components", vec![x, scalar(2.0)]).unwrap();
    let data = v.data();
    for row in 0..2 {
        let n: f64 = data[row * 2..row * 2 + 2].iter().map(|v| v * v).sum();
        assert!(
            (n - 1.0).abs() < 1e-6,
            "row {row} not unit norm: {}",
            n.sqrt()
        );
    }
}

#[test]
fn pca_components_rows_are_orthogonal() {
    let x = anisotropic_2d(50);
    let v = call_builtin("pca_components", vec![x, scalar(2.0)]).unwrap();
    let data = v.data();
    let dot: f64 = data[0..2]
        .iter()
        .zip(data[2..4].iter())
        .map(|(a, b)| a * b)
        .sum();
    assert!(dot.abs() < 1e-6, "rows not orthogonal: dot = {dot}");
}

#[test]
fn pca_components_top_axis_aligns_with_data_direction() {
    // Synthetic data lies on y = x/10, so the top component
    // should point along (10, 1) / sqrt(101) ~ (0.995, 0.0995).
    // The x-component should dominate (top_x / top_y ~ 10).
    let x = anisotropic_2d(100);
    let v = call_builtin("pca_components", vec![x, scalar(2.0)]).unwrap();
    let top_x = v.data()[0].abs();
    let top_y = v.data()[1].abs();
    assert!(
        top_x > 0.99 * 0.995 && top_x < 1.01 * 0.995,
        "top_x should be ~0.995, got {top_x}"
    );
    assert!(
        top_y > 0.5 * 0.0995 && top_y < 1.5 * 0.0995,
        "top_y should be ~0.0995, got {top_y}"
    );
}

#[test]
fn pca_variance_explained_shape_and_sum() {
    let x = anisotropic_2d(50);
    let r = call_builtin("pca_variance_explained", vec![x, scalar(2.0)]).unwrap();
    assert_eq!(r.shape().dims(), &[2]);
    let total: f64 = r.data().iter().sum();
    // k == D so the ratios should sum to ~1.0 (within
    // floating-point + power-iteration tolerance).
    assert!(
        (total - 1.0).abs() < 1e-4,
        "variance_explained should sum to 1 when k==D, got {total}"
    );
}

#[test]
fn pca_variance_explained_descending() {
    let x = anisotropic_2d(50);
    let r = call_builtin("pca_variance_explained", vec![x, scalar(2.0)]).unwrap();
    let d = r.data();
    // Power iteration extracts in descending eigenvalue order,
    // so the ratio sequence must be non-increasing.
    for w in d.windows(2) {
        assert!(w[0] >= w[1] - 1e-9, "ratios not descending: {d:?}");
    }
}

#[test]
fn pca_variance_explained_first_dominates_for_anisotropic() {
    // ~10x x-vs-y aspect ratio -> variance along PC1 should
    // be much larger than PC2.
    let x = anisotropic_2d(100);
    let r = call_builtin("pca_variance_explained", vec![x, scalar(2.0)]).unwrap();
    assert!(
        r.data()[0] > 0.9,
        "PC1 should explain >90% variance for 10x anisotropic data, got {}",
        r.data()[0]
    );
}

#[test]
fn pca_components_rejects_bad_args() {
    let x = anisotropic_2d(5);
    // k too large
    let r = call_builtin("pca_components", vec![x.clone(), scalar(99.0)]);
    assert!(r.is_err());
    // Non-integer k
    let r = call_builtin("pca_components", vec![x.clone(), scalar(1.5)]);
    assert!(r.is_err());
    // Wrong arity
    let r = call_builtin("pca_components", vec![x]);
    assert!(r.is_err());
}
