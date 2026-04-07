//! Tests for `argmax` and `blobs` built-ins.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime::call_builtin;

fn scalar(x: f64) -> DenseArray {
    DenseArray::from_scalar(x)
}

fn vec_of(data: &[f64]) -> DenseArray {
    DenseArray::from_vec(data.to_vec())
}

fn mat(rows: usize, cols: usize, data: &[f64]) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data.to_vec()).unwrap()
}

#[test]
fn argmax_scalar_form() {
    // no axis: returns scalar index into flat data
    let out = call_builtin("argmax", vec![vec_of(&[1.0, 3.0, 2.0, 0.5])]).unwrap();
    assert_eq!(out.rank(), 0);
    assert_eq!(out.data()[0], 1.0);
}

#[test]
fn argmax_axis_0_on_matrix() {
    // [[1, 5, 2],
    //  [4, 0, 9]]
    // axis 0 (reduce rows) -> per-column argmax = [1, 0, 1]
    let m = mat(2, 3, &[1.0, 5.0, 2.0, 4.0, 0.0, 9.0]);
    let out = call_builtin("argmax", vec![m, scalar(0.0)]).unwrap();
    assert_eq!(out.shape().dims(), &[3]);
    assert_eq!(out.data(), &[1.0, 0.0, 1.0]);
}

#[test]
fn argmax_axis_1_on_matrix() {
    // axis 1 (reduce cols) -> per-row argmax = [1, 2]
    let m = mat(2, 3, &[1.0, 5.0, 2.0, 4.0, 0.0, 9.0]);
    let out = call_builtin("argmax", vec![m, scalar(1.0)]).unwrap();
    assert_eq!(out.shape().dims(), &[2]);
    assert_eq!(out.data(), &[1.0, 2.0]);
}

#[test]
fn blobs_shape_and_labels() {
    // 3 centers, 20 points each -> 60x3 matrix
    let centers = mat(3, 2, &[0.0, 0.0, 3.0, 3.0, -3.0, 3.0]);
    let out = call_builtin("blobs", vec![scalar(5.0), scalar(20.0), centers]).unwrap();
    assert_eq!(out.shape().dims(), &[60, 3]);
    // label column: third column values
    let mut counts = [0usize; 3];
    for row in 0..60 {
        let label = out.data()[row * 3 + 2];
        assert!((0.0..3.0).contains(&label));
        counts[label as usize] += 1;
    }
    assert_eq!(counts, [20, 20, 20]);
}

#[test]
fn blobs_is_deterministic() {
    let centers = mat(2, 2, &[0.0, 0.0, 5.0, 5.0]);
    let a = call_builtin("blobs", vec![scalar(42.0), scalar(10.0), centers.clone()]).unwrap();
    let b = call_builtin("blobs", vec![scalar(42.0), scalar(10.0), centers]).unwrap();
    assert_eq!(a.data(), b.data());
}

#[test]
fn blobs_points_near_centers() {
    let centers = mat(2, 2, &[0.0, 0.0, 10.0, 10.0]);
    let out = call_builtin("blobs", vec![scalar(1.0), scalar(50.0), centers]).unwrap();
    // For each point, distance to its own center should be small (sigma 0.15).
    let own_centers = [[0.0, 0.0], [10.0, 10.0]];
    for row in 0..100 {
        let x = out.data()[row * 3];
        let y = out.data()[row * 3 + 1];
        let label = out.data()[row * 3 + 2] as usize;
        let (cx, cy) = (own_centers[label][0], own_centers[label][1]);
        let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        assert!(d < 2.0, "point row={row} label={label} d={d} too far");
    }
}
