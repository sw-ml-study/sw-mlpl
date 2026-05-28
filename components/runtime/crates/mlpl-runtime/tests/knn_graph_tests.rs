//! Tests for the saga 33 step 033 `knn_graph(X, k)` builtin
//! (Phase 2a of the dim-reduction milestone -- UMAP's input
//! layer).

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime::call_builtin;

fn scalar(x: f64) -> DenseArray {
    DenseArray::from_scalar(x)
}

fn mat(rows: usize, cols: usize, data: &[f64]) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data.to_vec()).unwrap()
}

/// 5 points on the x-axis at 0, 1, 2, 3, 4. Each point's
/// nearest neighbors are its immediate neighbors in index
/// order.
fn line_5() -> DenseArray {
    mat(5, 2, &[0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0])
}

#[test]
fn knn_graph_shape_is_n_times_k_by_3() {
    let x = line_5();
    let g = call_builtin("knn_graph", vec![x, scalar(2.0)]).unwrap();
    assert_eq!(g.shape().dims(), &[10, 3]);
}

#[test]
fn knn_graph_emits_source_neighbor_distance_triples() {
    // Point 0 at (0, 0) has nearest neighbor 1 at distance
    // 1, then 2 at distance 2. The edge list rows should be
    // (0, 1, 1.0), (0, 2, 2.0).
    let x = line_5();
    let g = call_builtin("knn_graph", vec![x, scalar(2.0)]).unwrap();
    let d = g.data();
    assert_eq!(d[0], 0.0); // i = 0
    assert_eq!(d[1], 1.0); // j = 1
    assert!((d[2] - 1.0).abs() < 1e-12, "dist should be 1, got {}", d[2]);
    assert_eq!(d[3], 0.0); // i = 0
    assert_eq!(d[4], 2.0); // j = 2
    assert!((d[5] - 2.0).abs() < 1e-12);
}

#[test]
fn knn_graph_neighbors_sorted_by_distance_per_source() {
    let x = line_5();
    let g = call_builtin("knn_graph", vec![x, scalar(4.0)]).unwrap();
    let d = g.data();
    // Middle point (index 2 at (2, 0)) has nearest pair at
    // distance 1 (indices 1 and 3, ties broken by smaller j
    // = 1 first), then distance 2 (indices 0, then 4).
    let base = 2 * 4 * 3;
    assert_eq!(d[base], 2.0); // i = 2
    assert_eq!(d[base + 1], 1.0); // j = 1 (closer index wins tie at dist 1)
    assert_eq!(d[base + 3 + 1], 3.0); // next: j = 3
    assert_eq!(d[base + 6 + 1], 0.0); // then: j = 0 at dist 2
    assert_eq!(d[base + 9 + 1], 4.0); // finally: j = 4 at dist 2
}

#[test]
fn knn_graph_self_is_excluded() {
    // With N=5 and k=4 the edge list covers EVERY other
    // point per source; we'd have N*k = 20 edges total but
    // no (i, i) self-edges.
    let x = line_5();
    let g = call_builtin("knn_graph", vec![x, scalar(4.0)]).unwrap();
    let d = g.data();
    for row in 0..20 {
        assert_ne!(
            d[row * 3],
            d[row * 3 + 1],
            "edge row {row} has self-loop: {:?}",
            &d[row * 3..row * 3 + 3]
        );
    }
}

#[test]
fn knn_graph_rejects_bad_args() {
    // k too large (k >= N)
    let x = line_5();
    let r = call_builtin("knn_graph", vec![x.clone(), scalar(5.0)]);
    assert!(r.is_err());
    let r = call_builtin("knn_graph", vec![x.clone(), scalar(6.0)]);
    assert!(r.is_err());
    // Non-integer k
    let r = call_builtin("knn_graph", vec![x.clone(), scalar(1.5)]);
    assert!(r.is_err());
    // k = 0
    let r = call_builtin("knn_graph", vec![x.clone(), scalar(0.0)]);
    assert!(r.is_err());
    // Wrong arity
    let r = call_builtin("knn_graph", vec![x]);
    assert!(r.is_err());
}

#[test]
fn knn_graph_distances_are_euclidean_not_squared() {
    // 3-4-5 triangle: point at (3, 4) is distance 5 from origin.
    let x = mat(2, 2, &[0.0, 0.0, 3.0, 4.0]);
    let g = call_builtin("knn_graph", vec![x, scalar(1.0)]).unwrap();
    let d = g.data();
    // Row 0: (0, 1, 5.0) -- not 25.0
    assert!(
        (d[2] - 5.0).abs() < 1e-12,
        "dist should be 5.0 (Euclidean), got {}",
        d[2]
    );
}
