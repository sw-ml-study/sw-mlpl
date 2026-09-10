//! Unit tests for the shared axis-selection types (`AxisSpec`) and their
//! resolution against a labeled array. Pure -- no interpreter, no builtin.
//! These pin the semantics every axis-selecting builtin will share.

use mlpl_array::{DenseArray, Shape};
use mlpl_axes::{AxisError, AxisNames, AxisSpec};

fn labeled_3d() -> DenseArray {
    // shape [2,2,2] labeled channel / kernel_y / kernel_x
    DenseArray::new(Shape::new(vec![2, 2, 2]), vec![0.0; 8])
        .unwrap()
        .with_labels(vec![
            Some("channel".into()),
            Some("kernel_y".into()),
            Some("kernel_x".into()),
        ])
        .unwrap()
}

fn unlabeled_3d() -> DenseArray {
    DenseArray::new(Shape::new(vec![2, 2, 2]), vec![0.0; 8]).unwrap()
}

#[test]
fn names_resolve_to_positions() {
    let a = labeled_3d();
    let spec = AxisSpec::Names(vec!["kernel_y".into(), "channel".into()]);
    assert_eq!(spec.resolve(&a).unwrap(), vec![1, 0]);
}

#[test]
fn indices_pass_through_when_in_rank() {
    let a = unlabeled_3d();
    let spec = AxisSpec::Indices(vec![2, 0]);
    assert_eq!(spec.resolve(&a).unwrap(), vec![2, 0]);
}

#[test]
fn empty_selection_resolves_to_empty() {
    let a = labeled_3d();
    assert_eq!(
        AxisSpec::Names(vec![]).resolve(&a).unwrap(),
        Vec::<usize>::new()
    );
    assert_eq!(
        AxisSpec::Indices(vec![]).resolve(&a).unwrap(),
        Vec::<usize>::new()
    );
}

#[test]
fn missing_name_errors() {
    let a = labeled_3d();
    let spec = AxisSpec::Names(vec!["nope".into()]);
    assert_eq!(spec.resolve(&a), Err(AxisError::NoAxisNamed("nope".into())));
}

#[test]
fn named_axis_on_unlabeled_array_errors() {
    let a = unlabeled_3d();
    let spec = AxisSpec::Names(vec!["channel".into()]);
    assert_eq!(spec.resolve(&a), Err(AxisError::NamedAxisButNoLabels));
}

#[test]
fn index_out_of_rank_errors() {
    let a = unlabeled_3d(); // rank 3
    let spec = AxisSpec::Indices(vec![5]);
    assert_eq!(
        spec.resolve(&a),
        Err(AxisError::IndexOutOfRank { index: 5, rank: 3 })
    );
}

#[test]
fn duplicate_axis_by_index_errors() {
    let a = unlabeled_3d();
    let spec = AxisSpec::Indices(vec![1, 1]);
    assert_eq!(spec.resolve(&a), Err(AxisError::DuplicateAxis(1)));
}

#[test]
fn duplicate_axis_by_name_errors() {
    let a = labeled_3d();
    let spec = AxisSpec::Names(vec!["channel".into(), "channel".into()]);
    assert_eq!(spec.resolve(&a), Err(AxisError::DuplicateAxis(0)));
}

#[test]
fn axis_names_wraps_optional_labels() {
    // AxisNames is the shared "names to attach" carrier (inner None = positional).
    let names = AxisNames(vec![Some("a".into()), None]);
    assert_eq!(names.0.len(), 2);
    assert_eq!(names.0[1], None);
}
