//! NaN/zero/kind-mismatch semantics of `value_equal`, pinned at
//! the unit level (the eval-layer tests cover the builtin).

use mlpl_array::{DenseArray, Shape};
use mlpl_eval_types::Value;
use mlpl_value_structural::value_equal;

fn arr(dims: Vec<usize>, data: Vec<f64>) -> Value {
    Value::Array(DenseArray::new(Shape::new(dims), data).unwrap())
}

#[test]
fn nan_equals_nan_and_kinds_never_error() {
    assert!(value_equal(
        &arr(vec![2], vec![f64::NAN, 1.0]),
        &arr(vec![2], vec![f64::NAN, 1.0])
    ));
    assert!(!value_equal(
        &arr(vec![2], vec![1.0, 2.0]),
        &Value::Str("x".into())
    ));
    assert!(!value_equal(
        &arr(vec![2], vec![1.0, 2.0]),
        &arr(vec![2, 1], vec![1.0, 2.0])
    ));
    assert!(value_equal(
        &arr(vec![1], vec![0.0]),
        &arr(vec![1], vec![-0.0])
    ));
}
