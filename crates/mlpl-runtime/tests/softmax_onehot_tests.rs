//! Tests for `softmax` and `one_hot` built-ins.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime::call_builtin;

fn scalar(x: f64) -> DenseArray {
    DenseArray::from_scalar(x)
}

fn mat(rows: usize, cols: usize, data: &[f64]) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data.to_vec()).unwrap()
}

#[test]
fn softmax_rows_sum_to_one() {
    let m = mat(2, 3, &[1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
    let out = call_builtin("softmax", vec![m, scalar(1.0)]).unwrap();
    assert_eq!(out.shape().dims(), &[2, 3]);
    for row in 0..2 {
        let s: f64 = out.data()[row * 3..row * 3 + 3].iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "row {row} sum = {s}");
    }
    // Monotonic: larger logit -> larger probability.
    assert!(out.data()[0] < out.data()[1] && out.data()[1] < out.data()[2]);
}

#[test]
fn softmax_is_numerically_stable_for_large_logits() {
    // Without max subtraction, exp(1000) would overflow.
    let m = mat(1, 3, &[1000.0, 1001.0, 1002.0]);
    let out = call_builtin("softmax", vec![m, scalar(1.0)]).unwrap();
    for v in out.data() {
        assert!(v.is_finite(), "got non-finite {v}");
    }
    let s: f64 = out.data().iter().sum();
    assert!((s - 1.0).abs() < 1e-9);
}

#[test]
fn softmax_axis_0_on_matrix() {
    // axis=0: each column sums to 1.
    let m = mat(3, 2, &[0.0, 0.0, 1.0, 2.0, 2.0, 4.0]);
    let out = call_builtin("softmax", vec![m, scalar(0.0)]).unwrap();
    for col in 0..2 {
        let s: f64 = (0..3).map(|r| out.data()[r * 2 + col]).sum();
        assert!((s - 1.0).abs() < 1e-9, "col {col} sum = {s}");
    }
}

#[test]
fn one_hot_shape_and_values() {
    let labels = DenseArray::from_vec(vec![0.0, 2.0, 1.0, 0.0]);
    let out = call_builtin("one_hot", vec![labels, scalar(3.0)]).unwrap();
    assert_eq!(out.shape().dims(), &[4, 3]);
    let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    assert_eq!(out.data(), &expected);
}
