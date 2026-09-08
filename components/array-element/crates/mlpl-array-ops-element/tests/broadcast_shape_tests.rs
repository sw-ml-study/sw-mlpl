//! Regression tests for single-element (scalar / length-1) broadcasting
//! shape rules in `apply_binop`.
//!
//! BUG 1 (demo-abstract-algebra `docs/sw-mlpl-bug-report.md`): an array
//! every one of whose axes has extent 1 collapsed straight to rank 0 the
//! moment a rank-0 scalar was broadcast against it -- `[[0]] * 1`
//! returned shape `[]` instead of `[1, 1]`. Scalar broadcasting must
//! preserve the array operand's rank and shape exactly.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_element::ApplyBinopExt;

fn arr(dims: &[usize], data: &[f64]) -> DenseArray {
    DenseArray::new(Shape::new(dims.to_vec()), data.to_vec()).unwrap()
}

fn shape_of(a: &DenseArray) -> Vec<usize> {
    a.shape().dims().to_vec()
}

fn mul(a: f64, b: f64) -> f64 {
    a * b
}

fn add(a: f64, b: f64) -> f64 {
    a + b
}

#[test]
fn all_unit_shape_survives_scalar_broadcast() {
    let scalar = DenseArray::from_scalar(1.0); // shape []
    // The regression: these all collapsed to [] before the fix.
    assert_eq!(
        shape_of(&arr(&[1], &[0.0]).apply_binop(&scalar, mul).unwrap()),
        vec![1]
    );
    assert_eq!(
        shape_of(&arr(&[1, 1], &[0.0]).apply_binop(&scalar, mul).unwrap()),
        vec![1, 1]
    );
    assert_eq!(
        shape_of(&arr(&[1, 1, 1], &[0.0]).apply_binop(&scalar, mul).unwrap()),
        vec![1, 1, 1]
    );
}

#[test]
fn scalar_broadcast_is_symmetric() {
    let scalar = DenseArray::from_scalar(1.0);
    // 1 * [[0]] preserves the array operand's shape just like [[0]] * 1.
    assert_eq!(
        shape_of(&scalar.apply_binop(&arr(&[1, 1], &[0.0]), mul).unwrap()),
        vec![1, 1]
    );
}

#[test]
fn unequal_rank_single_elements_keep_the_higher_rank() {
    // [[0]] (rank 2) against [1] (rank 1): the higher rank is preserved.
    assert_eq!(
        shape_of(
            &arr(&[1, 1], &[0.0])
                .apply_binop(&arr(&[1], &[1.0]), mul)
                .unwrap()
        ),
        vec![1, 1]
    );
    // And the value is the applied op, not dropped.
    let r = arr(&[1, 1], &[3.0])
        .apply_binop(&DenseArray::from_scalar(4.0), add)
        .unwrap();
    assert_eq!(r.data(), &[7.0]);
}

#[test]
fn equal_shape_and_extent_over_one_are_unchanged() {
    let scalar = DenseArray::from_scalar(1.0);
    // Equal-shape elementwise is untouched.
    assert_eq!(
        shape_of(
            &arr(&[1, 1], &[0.0])
                .apply_binop(&arr(&[1, 1], &[1.0]), mul)
                .unwrap()
        ),
        vec![1, 1]
    );
    // Any axis with extent > 1 was always preserved; still is.
    assert_eq!(
        shape_of(&arr(&[1, 2], &[0.0, 1.0]).apply_binop(&scalar, mul).unwrap()),
        vec![1, 2]
    );
}

#[test]
fn length_one_still_broadcasts_to_a_larger_array() {
    // The intended single-element broadcast feature is intact:
    // [2] * [1, 2, 3] == [2, 4, 6].
    let r = arr(&[1], &[2.0])
        .apply_binop(&arr(&[3], &[1.0, 2.0, 3.0]), mul)
        .unwrap();
    assert_eq!(shape_of(&r), vec![3]);
    assert_eq!(r.data(), &[2.0, 4.0, 6.0]);
    // Scalar against a 2x2 broadcasts to [2, 2].
    let m = DenseArray::from_scalar(10.0)
        .apply_binop(&arr(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), mul)
        .unwrap();
    assert_eq!(shape_of(&m), vec![2, 2]);
    assert_eq!(m.data(), &[10.0, 20.0, 30.0, 40.0]);
}
