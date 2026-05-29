//! Saga 78: tests for `viz3d_events` extracted to keep the
//! parent module under the sw-checklist function count limit.
//! Loaded via `#[path]` include so `super::*` still resolves.

use super::*;

#[test]
fn event_serializes() {
    let ev = Stage3dEvent {
        step_idx: 0,
        label: "x = 1 + 2".into(),
        output: build_shape_info("x".into(), vec![], None),
    };
    let json = serde_json::to_string(&ev).unwrap();
    assert!(json.contains("\"step_idx\":0"));
    assert!(json.contains("\"rank\":0"));
}

#[test]
fn shape_scalar() {
    assert_eq!(shape_from_output("3"), (vec![], 1));
}

#[test]
fn shape_vector() {
    assert_eq!(shape_from_output("0 1 2 3 4"), (vec![5], 5));
}

#[test]
fn shape_matrix() {
    let out = "0 1 2 3\n4 5 6 7\n8 9 10 11";
    assert_eq!(shape_from_output(out), (vec![3, 4], 12));
}

#[test]
fn shape_dense_array_summary() {
    let out = "<DenseArray shape=[10, 10] elems=100 first=[0, 1, 2, ...]>";
    assert_eq!(shape_from_output(out), (vec![10, 10], 100));
}

#[test]
fn shape_dense_array_vector() {
    let out = "<DenseArray shape=[100] elems=100 first=[0, 1, 2, ...]>";
    assert_eq!(shape_from_output(out), (vec![100], 100));
}
