//! moe-microscope finding F10: an element-wise op between a fully-labeled
//! array and a partially-labeled one (a `None` axis) must unify per axis --
//! `None` is a wildcard -- instead of raising LabelMismatch on whole-vector
//! inequality. This is what made `sinusoidal_encoding` ([time, dim]) added to
//! an embedding ([time, _]) panic the autograd tape.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_element::ApplyBinopExt;

fn labeled(dims: &[usize], data: &[f64], labels: Vec<Option<String>>) -> DenseArray {
    DenseArray::new(Shape::new(dims.to_vec()), data.to_vec())
        .unwrap()
        .with_labels(labels)
        .unwrap()
}

fn add(a: f64, b: f64) -> f64 {
    a + b
}

#[test]
fn partial_none_axis_unifies_with_a_label() {
    // [time, dim] + [time, None] -> labels [time, dim], no error.
    let a = labeled(
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        vec![Some("time".into()), Some("dim".into())],
    );
    let b = labeled(
        &[2, 2],
        &[10.0, 20.0, 30.0, 40.0],
        vec![Some("time".into()), None],
    );
    let out = a.apply_binop(&b, add).expect("labels unify, no mismatch");
    assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
    assert_eq!(
        out.labels().map(<[_]>::to_vec),
        Some(vec![Some("time".into()), Some("dim".into())])
    );
}

#[test]
fn two_different_explicit_labels_still_conflict() {
    // A genuine conflict (dim vs batch on the same axis) must still error.
    let a = labeled(&[2], &[1.0, 2.0], vec![Some("dim".into())]);
    let b = labeled(&[2], &[3.0, 4.0], vec![Some("batch".into())]);
    assert!(a.apply_binop(&b, add).is_err(), "conflicting labels error");
}
