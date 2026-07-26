//! `RotateExt::rotate` unit tests (Game of Life saga step 1).
//! APL-style cyclic rotate: positive `k` brings element `k` to the
//! front (left/up shift); negative `k` rotates the other way; any
//! magnitude wraps.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_compose::prelude::*;

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn vector_rotate_left_by_one() {
    let v = arr(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let r = v.rotate(1, 0).unwrap();
    assert_eq!(r.data(), &[2.0, 3.0, 4.0, 1.0]);
    assert_eq!(r.shape().dims(), &[4]);
}

#[test]
fn vector_rotate_right_by_negative_one() {
    let v = arr(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let r = v.rotate(-1, 0).unwrap();
    assert_eq!(r.data(), &[4.0, 1.0, 2.0, 3.0]);
}

#[test]
fn rotate_wraps_past_axis_length() {
    let v = arr(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        v.rotate(5, 0).unwrap().data(),
        v.rotate(1, 0).unwrap().data()
    );
    assert_eq!(
        v.rotate(-5, 0).unwrap().data(),
        v.rotate(-1, 0).unwrap().data()
    );
}

#[test]
fn rotate_zero_is_identity() {
    let v = arr(vec![3], vec![7.0, 8.0, 9.0]);
    assert_eq!(v.rotate(0, 0).unwrap().data(), v.data());
}

#[test]
fn matrix_rotate_axis0_shifts_rows_up() {
    let m = arr(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = m.rotate(1, 0).unwrap();
    assert_eq!(r.data(), &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
}

#[test]
fn matrix_rotate_axis1_shifts_cols_left() {
    let m = arr(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = m.rotate(1, 1).unwrap();
    assert_eq!(r.data(), &[2.0, 3.0, 1.0, 5.0, 6.0, 4.0]);
}

#[test]
fn rank3_rotate_leading_axis() {
    let t = arr(vec![2, 2, 2], (1..=8).map(f64::from).collect());
    let r = t.rotate(1, 0).unwrap();
    assert_eq!(r.data(), &[5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn axis_out_of_range_errors() {
    let v = arr(vec![3], vec![1.0, 2.0, 3.0]);
    assert!(v.rotate(1, 1).is_err());
}

#[test]
fn labels_preserved_shape_unchanged() {
    let m = arr(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
        .with_labels(vec![Some("r".into()), Some("c".into())])
        .unwrap();
    let r = m.rotate(1, 0).unwrap();
    assert_eq!(r.labels().unwrap().len(), 2);
}
