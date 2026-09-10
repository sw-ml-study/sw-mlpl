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

// -- C2: general trailing-axis (rank) broadcasting --

#[test]
fn a_lower_rank_operand_broadcasts_against_trailing_axes() {
    // A rank-3 kernel [2,2,2] multiplies rank-5 patches [2,2,2,2,2] by
    // aligning on the trailing [C,kh,kw] block -- the convolution case.
    let patches = arr(
        &[2, 2, 2, 2, 2],
        &(0..32).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let kernel = arr(&[2, 2, 2], &(0..8).map(|i| i as f64).collect::<Vec<_>>());
    let r = patches.apply_binop(&kernel, mul).unwrap();
    assert_eq!(shape_of(&r), vec![2, 2, 2, 2, 2]);
    // spot-check: output[0..8] = patches[0..8] * kernel[0..8]
    let want: Vec<f64> = (0..8).map(|i| (i * i) as f64).collect();
    assert_eq!(&r.data()[0..8], want.as_slice());
}

#[test]
fn both_operands_broadcast_bidirectionally() {
    // [3,1] + [1,4] -> [3,4] (an outer sum), each operand stretched.
    let col = arr(&[3, 1], &[10.0, 20.0, 30.0]);
    let row = arr(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);
    let r = col.apply_binop(&row, add).unwrap();
    assert_eq!(shape_of(&r), vec![3, 4]);
    assert_eq!(
        r.data(),
        &[
            11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0, 31.0, 32.0, 33.0, 34.0
        ]
    );
}

#[test]
fn genuinely_incompatible_shapes_still_error() {
    // 3 vs 4 on an axis where neither is 1 -- not broadcastable.
    assert!(
        arr(&[2, 3], &[0.0; 6])
            .apply_binop(&arr(&[2, 4], &[0.0; 8]), mul)
            .is_err()
    );
}

#[test]
fn labels_align_from_the_right_under_rank_broadcast() {
    // Labeled rank-5 patches * labeled rank-3 kernel: surviving axes keep
    // the wider operand's labels, and the shared trailing labels agree.
    let patches = arr(&[1, 1, 2, 1, 1], &[1.0, 2.0])
        .with_labels(vec![
            Some("oy".into()),
            Some("ox".into()),
            Some("c".into()),
            Some("ky".into()),
            Some("kx".into()),
        ])
        .unwrap();
    let kernel = arr(&[2, 1, 1], &[3.0, 4.0])
        .with_labels(vec![Some("c".into()), Some("ky".into()), Some("kx".into())])
        .unwrap();
    let r = patches.apply_binop(&kernel, mul).unwrap();
    assert_eq!(shape_of(&r), vec![1, 1, 2, 1, 1]);
    let labels: Vec<Option<String>> = r.labels().unwrap().to_vec();
    assert_eq!(
        labels,
        vec![
            Some("oy".into()),
            Some("ox".into()),
            Some("c".into()),
            Some("ky".into()),
            Some("kx".into())
        ]
    );
}
