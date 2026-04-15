use mlpl_array::{ArrayError, DenseArray, Shape};

// -- Construction --

#[test]
fn new_vector() {
    let arr = DenseArray::new(Shape::vector(3), vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0]);
    assert_eq!(arr.rank(), 1);
}

#[test]
fn new_matrix() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(arr.rank(), 2);
    assert_eq!(arr.elem_count(), 6);
}

#[test]
fn new_data_length_mismatch() {
    let result = DenseArray::new(Shape::vector(3), vec![1.0, 2.0]);
    assert_eq!(
        result,
        Err(ArrayError::DataLengthMismatch {
            expected: 3,
            got: 2
        })
    );
}

#[test]
fn zeros_vector() {
    let arr = DenseArray::zeros(Shape::vector(4));
    assert_eq!(arr.data(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn zeros_scalar() {
    let arr = DenseArray::zeros(Shape::scalar());
    assert_eq!(arr.data(), &[0.0]);
    assert_eq!(arr.rank(), 0);
}

#[test]
fn from_scalar() {
    let arr = DenseArray::from_scalar(42.0);
    assert_eq!(arr.shape(), &Shape::scalar());
    assert_eq!(arr.data(), &[42.0]);
}

#[test]
fn from_vec() {
    let arr = DenseArray::from_vec(vec![10.0, 20.0, 30.0]);
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[10.0, 20.0, 30.0]);
}

// -- Multi-dim indexing --

#[test]
fn get_vector() {
    let arr = DenseArray::from_vec(vec![10.0, 20.0, 30.0]);
    assert_eq!(arr.get(&[0]).unwrap(), &10.0);
    assert_eq!(arr.get(&[2]).unwrap(), &30.0);
}

#[test]
fn get_matrix_row_major() {
    // shape [2, 3], data [1,2,3,4,5,6]
    // row 0: [1, 2, 3], row 1: [4, 5, 6]
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(arr.get(&[0, 0]).unwrap(), &1.0);
    assert_eq!(arr.get(&[0, 2]).unwrap(), &3.0);
    assert_eq!(arr.get(&[1, 0]).unwrap(), &4.0);
    assert_eq!(arr.get(&[1, 2]).unwrap(), &6.0);
}

#[test]
fn get_scalar() {
    let arr = DenseArray::from_scalar(5.0);
    assert_eq!(arr.get(&[]).unwrap(), &5.0);
}

#[test]
fn get_out_of_bounds() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(
        arr.get(&[3]),
        Err(ArrayError::IndexOutOfBounds {
            axis: 0,
            index: 3,
            size: 3
        })
    );
}

#[test]
fn get_rank_mismatch() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(
        arr.get(&[0, 0]),
        Err(ArrayError::RankMismatch {
            expected: 1,
            got: 2
        })
    );
}

#[test]
fn get_empty_array() {
    let arr = DenseArray::zeros(Shape::vector(0));
    assert_eq!(arr.get(&[0]), Err(ArrayError::EmptyArray));
}

// -- Set --

#[test]
fn set_vector() {
    let mut arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    arr.set(&[1], 99.0).unwrap();
    assert_eq!(arr.get(&[1]).unwrap(), &99.0);
}

#[test]
fn set_matrix() {
    let mut arr = DenseArray::new(Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    arr.set(&[1, 0], 99.0).unwrap();
    assert_eq!(arr.get(&[1, 0]).unwrap(), &99.0);
}

#[test]
fn set_out_of_bounds() {
    let mut arr = DenseArray::from_vec(vec![1.0, 2.0]);
    assert_eq!(
        arr.set(&[5], 0.0),
        Err(ArrayError::IndexOutOfBounds {
            axis: 0,
            index: 5,
            size: 2
        })
    );
}

// -- Display --

#[test]
fn display_scalar() {
    let arr = DenseArray::from_scalar(7.5);
    assert_eq!(arr.to_string(), "7.5");
}

#[test]
fn display_vector() {
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(arr.to_string(), "1 2 3");
}

#[test]
fn display_matrix() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(arr.to_string(), "1 2 3\n4 5 6");
}

#[test]
fn display_empty_vector() {
    let arr = DenseArray::zeros(Shape::vector(0));
    assert_eq!(arr.to_string(), "[]");
}

// -- Labels (Saga 11.5 Phase 2) --

#[test]
fn with_labels_matrix() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let labeled = arr
        .with_labels(vec![Some("seq".into()), Some("d_k".into())])
        .unwrap();
    assert_eq!(
        labeled.labels(),
        Some(&[Some("seq".into()), Some("d_k".into())][..])
    );
    // Data and shape unchanged.
    assert_eq!(labeled.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(labeled.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn with_labels_scalar_empty_ok() {
    let arr = DenseArray::from_scalar(7.5);
    let labeled = arr.with_labels(vec![]).unwrap();
    assert_eq!(labeled.labels(), Some(&[][..]));
}

#[test]
fn with_labels_rank_mismatch() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![0.0; 6]).unwrap();
    let result = arr.with_labels(vec![Some("rows".into())]);
    assert_eq!(
        result,
        Err(ArrayError::LabelsRankMismatch { rank: 2, labels: 1 })
    );
}

#[test]
fn with_labels_unlabeled_is_none() {
    // Starting state: fresh arrays have no labels.
    let arr = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(arr.labels(), None);
}

// -- Elementwise label propagation (Saga 11.5 Phase 3) --

#[test]
fn binop_same_labels_propagate() {
    let a = DenseArray::from_vec(vec![1.0, 2.0, 3.0])
        .with_labels(vec![Some("seq".into())])
        .unwrap();
    let b = DenseArray::from_vec(vec![10.0, 20.0, 30.0])
        .with_labels(vec![Some("seq".into())])
        .unwrap();
    let r = a.apply_binop(&b, |x, y| x + y).unwrap();
    assert_eq!(r.labels(), Some(&[Some("seq".into())][..]));
    assert_eq!(r.data(), &[11.0, 22.0, 33.0]);
}

#[test]
fn binop_label_mismatch_errors() {
    let a = DenseArray::from_vec(vec![1.0, 2.0, 3.0])
        .with_labels(vec![Some("seq".into())])
        .unwrap();
    let b = DenseArray::from_vec(vec![10.0, 20.0, 30.0])
        .with_labels(vec![Some("batch".into())])
        .unwrap();
    let r = a.apply_binop(&b, |x, y| x + y);
    assert_eq!(
        r,
        Err(ArrayError::LabelMismatch {
            expected: vec![Some("seq".into())],
            actual: vec![Some("batch".into())],
        })
    );
}

#[test]
fn binop_labeled_plus_unlabeled_adopts_labels() {
    let a = DenseArray::from_vec(vec![1.0, 2.0, 3.0])
        .with_labels(vec![Some("seq".into())])
        .unwrap();
    let b = DenseArray::from_vec(vec![10.0, 20.0, 30.0]);
    let r = a.apply_binop(&b, |x, y| x + y).unwrap();
    assert_eq!(r.labels(), Some(&[Some("seq".into())][..]));
}

#[test]
fn binop_unlabeled_plus_labeled_adopts_labels() {
    let a = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    let b = DenseArray::from_vec(vec![10.0, 20.0, 30.0])
        .with_labels(vec![Some("seq".into())])
        .unwrap();
    let r = a.apply_binop(&b, |x, y| x + y).unwrap();
    assert_eq!(r.labels(), Some(&[Some("seq".into())][..]));
}

#[test]
fn binop_both_unlabeled_stays_unlabeled() {
    let a = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    let b = DenseArray::from_vec(vec![10.0, 20.0, 30.0]);
    let r = a.apply_binop(&b, |x, y| x + y).unwrap();
    assert_eq!(r.labels(), None);
}

#[test]
fn binop_scalar_lhs_preserves_labeled_rhs() {
    let s = DenseArray::from_scalar(2.0);
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0])
        .with_labels(vec![Some("seq".into())])
        .unwrap();
    let r = s.apply_binop(&v, |x, y| x * y).unwrap();
    assert_eq!(r.labels(), Some(&[Some("seq".into())][..]));
    assert_eq!(r.data(), &[2.0, 4.0, 6.0]);
}

#[test]
fn binop_scalar_rhs_preserves_labeled_lhs() {
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0])
        .with_labels(vec![Some("seq".into())])
        .unwrap();
    let s = DenseArray::from_scalar(2.0);
    let r = v.apply_binop(&s, |x, y| x * y).unwrap();
    assert_eq!(r.labels(), Some(&[Some("seq".into())][..]));
}

// -- matmul label propagation (Saga 11.5 Phase 3 cont.) --

#[test]
fn matmul_matching_contraction_axis() {
    // [seq, d] @ [d, heads] -> [seq, heads]
    let a = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6])
        .unwrap()
        .with_labels(vec![Some("seq".into()), Some("d".into())])
        .unwrap();
    let b = DenseArray::new(Shape::new(vec![3, 4]), vec![1.0; 12])
        .unwrap()
        .with_labels(vec![Some("d".into()), Some("heads".into())])
        .unwrap();
    let r = a.matmul(&b).unwrap();
    assert_eq!(r.shape(), &Shape::new(vec![2, 4]));
    assert_eq!(
        r.labels(),
        Some(&[Some("seq".into()), Some("heads".into())][..])
    );
}

#[test]
fn matmul_contraction_axis_mismatch_errors() {
    let a = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6])
        .unwrap()
        .with_labels(vec![Some("seq".into()), Some("d".into())])
        .unwrap();
    let b = DenseArray::new(Shape::new(vec![3, 4]), vec![1.0; 12])
        .unwrap()
        .with_labels(vec![Some("time".into()), Some("heads".into())])
        .unwrap();
    let r = a.matmul(&b);
    assert!(
        matches!(r, Err(ArrayError::LabelMismatch { .. })),
        "expected LabelMismatch, got {r:?}"
    );
}

#[test]
fn matmul_labeled_matrix_unlabeled_matrix() {
    let a = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6])
        .unwrap()
        .with_labels(vec![Some("seq".into()), Some("d".into())])
        .unwrap();
    let b = DenseArray::new(Shape::new(vec![3, 4]), vec![1.0; 12]).unwrap();
    // Mixed: left is labeled, right is not. Inner dim silently lines up;
    // result carries left's outer label, right's axis is None.
    let r = a.matmul(&b).unwrap();
    assert_eq!(r.labels(), Some(&[Some("seq".into()), None][..]));
}

#[test]
fn matmul_matrix_vector_propagates_outer_label() {
    // [m, k] @ [k] -> [m]
    let a = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6])
        .unwrap()
        .with_labels(vec![Some("seq".into()), Some("d".into())])
        .unwrap();
    let b = DenseArray::from_vec(vec![1.0, 2.0, 3.0])
        .with_labels(vec![Some("d".into())])
        .unwrap();
    let r = a.matmul(&b).unwrap();
    assert_eq!(r.shape(), &Shape::vector(2));
    assert_eq!(r.labels(), Some(&[Some("seq".into())][..]));
}

#[test]
fn matmul_both_unlabeled_stays_unlabeled() {
    let a = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6]).unwrap();
    let b = DenseArray::new(Shape::new(vec![3, 4]), vec![1.0; 12]).unwrap();
    let r = a.matmul(&b).unwrap();
    assert_eq!(r.labels(), None);
}

// -- reduce_axis / argmax_axis label propagation --

#[test]
fn reduce_axis_drops_reduced_label() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap();
    let r = arr.reduce_axis(0, 0.0, |a, b| a + b).unwrap();
    // Axis 0 removed; only "feat" remains.
    assert_eq!(r.labels(), Some(&[Some("feat".into())][..]));
}

#[test]
fn reduce_axis_drops_reduced_label_axis1() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap();
    let r = arr.reduce_axis(1, 0.0, |a, b| a + b).unwrap();
    assert_eq!(r.labels(), Some(&[Some("batch".into())][..]));
}

#[test]
fn reduce_axis_unlabeled_stays_none() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6]).unwrap();
    let r = arr.reduce_axis(0, 0.0, |a, b| a + b).unwrap();
    assert_eq!(r.labels(), None);
}

#[test]
fn argmax_axis_drops_reduced_label() {
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 5.0, 2.0, 4.0, 0.0, 3.0])
        .unwrap()
        .with_labels(vec![Some("batch".into()), Some("class".into())])
        .unwrap();
    let r = arr.argmax_axis(1).unwrap();
    assert_eq!(r.labels(), Some(&[Some("batch".into())][..]));
}

#[test]
fn binop_partial_labels_match() {
    // Both sides have `[None, Some("cols")]` -- matches.
    let a = DenseArray::new(Shape::new(vec![2, 3]), vec![0.0; 6])
        .unwrap()
        .with_labels(vec![None, Some("cols".into())])
        .unwrap();
    let b = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6])
        .unwrap()
        .with_labels(vec![None, Some("cols".into())])
        .unwrap();
    let r = a.apply_binop(&b, |x, y| x + y).unwrap();
    assert_eq!(r.labels(), Some(&[None, Some("cols".into())][..]));
}
