//! Primitive-parity tests for `mlpl-rt`.
//!
//! Each primitive's output is asserted against known-good numeric
//! results. The numbers here match what `mlpl_runtime::call_builtin`
//! produces for the same input (hand-verified against step-005's
//! existing eval tests); keeping the check numeric here avoids a
//! dev-dep on `mlpl-runtime` and therefore keeps `mlpl-rt` fully
//! decoupled from the interpreter stack.

use mlpl_rt::{
    DenseArray, Shape, iota, rank, reduce_add, reduce_add_axis, reshape, shape, transpose,
};

#[test]
fn iota_scalar_basis() {
    let v = iota(5);
    assert_eq!(v.shape(), &Shape::vector(5));
    assert_eq!(v.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn iota_zero_is_empty() {
    let v = iota(0);
    assert_eq!(v.shape(), &Shape::vector(0));
    assert!(v.data().is_empty());
}

#[test]
fn shape_matches_dims() {
    let m = DenseArray::new(Shape::new(vec![2, 3]), vec![0.0; 6]).unwrap();
    assert_eq!(shape(&m).data(), &[2.0, 3.0]);
}

#[test]
fn rank_of_scalar_vector_matrix() {
    assert_eq!(rank(&DenseArray::from_scalar(7.0)).data(), &[0.0]);
    assert_eq!(rank(&DenseArray::from_vec(vec![1.0; 4])).data(), &[1.0]);
    let m = DenseArray::new(Shape::new(vec![2, 3]), vec![0.0; 6]).unwrap();
    assert_eq!(rank(&m).data(), &[2.0]);
}

#[test]
fn reshape_preserves_elements() {
    let v = iota(6);
    let m = reshape(&v, &[2, 3]).unwrap();
    assert_eq!(m.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(m.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn reshape_element_count_mismatch_errors() {
    let v = iota(6);
    assert!(reshape(&v, &[2, 2]).is_err());
}

#[test]
fn transpose_matrix() {
    let m = reshape(&iota(6), &[2, 3]).unwrap();
    let t = transpose(&m);
    assert_eq!(t.shape(), &Shape::new(vec![3, 2]));
    assert_eq!(t.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn reduce_add_flat() {
    let r = reduce_add(&iota(5));
    // 0 + 1 + 2 + 3 + 4 = 10
    assert_eq!(r.data(), &[10.0]);
}

#[test]
fn reduce_add_axis_column_sums() {
    // [[0,1,2],[3,4,5]] reduced along axis 0 -> [3, 5, 7]
    let m = reshape(&iota(6), &[2, 3]).unwrap();
    let s = reduce_add_axis(&m, 0).unwrap();
    assert_eq!(s.shape(), &Shape::vector(3));
    assert_eq!(s.data(), &[3.0, 5.0, 7.0]);
}

#[test]
fn reduce_add_axis_drops_label() {
    // Saga 11.5 label propagation: reducing axis 0 keeps axis 1's label.
    let m = reshape(&iota(6), &[2, 3])
        .unwrap()
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap();
    let s = reduce_add_axis(&m, 0).unwrap();
    assert_eq!(s.labels(), Some(&[Some("feat".into())][..]));
}

#[test]
fn transpose_swaps_labels() {
    // Saga 11.5 label propagation through transpose.
    let m = reshape(&iota(6), &[2, 3])
        .unwrap()
        .with_labels(vec![Some("rows".into()), Some("cols".into())])
        .unwrap();
    let t = transpose(&m);
    assert_eq!(
        t.labels(),
        Some(&[Some("cols".into()), Some("rows".into())][..])
    );
}
