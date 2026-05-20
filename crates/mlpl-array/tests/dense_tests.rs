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

#[test]
fn map_preserves_labels() {
    // Elementwise map is 1:1 with identical shape, so axis identity
    // survives. Saga 11.5: unblocks label flow through math builtins
    // (exp, log, sigmoid, tanh_fn, ...) and model DSL activations.
    let arr = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6])
        .unwrap()
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap();
    let mapped = arr.map(|x| x * 2.0);
    assert_eq!(
        mapped.labels(),
        Some(&[Some("batch".into()), Some("feat".into())][..])
    );
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

// -- concat axis-N (saga 30 step 001) --

#[test]
fn concat_rank3_axis2_shape_and_content() {
    // a: [2, 2, 3] = 12 elements, b: [2, 2, 4] = 16 elements, concat along axis 2 -> [2, 2, 7].
    let a_data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let b_data: Vec<f64> = (100..116).map(|i| i as f64).collect();
    let a = DenseArray::new(Shape::new(vec![2, 2, 3]), a_data).unwrap();
    let b = DenseArray::new(Shape::new(vec![2, 2, 4]), b_data).unwrap();
    let r = a.concat(&b, 2).expect("rank-3 concat axis 2 must succeed");
    assert_eq!(r.shape().dims(), &[2, 2, 7]);
    // For outer index (i, j) the row is a[i, j, 0..3] then b[i, j, 0..4].
    // First row: a[0,0,0..3] = 0,1,2; b[0,0,0..4] = 100..103 -> indices 0..7 in flat output.
    assert_eq!(&r.data()[..7], &[0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 103.0]);
    // Second row: a[0,1,0..3] = 3,4,5; b[0,1,0..4] = 104..107.
    assert_eq!(
        &r.data()[7..14],
        &[3.0, 4.0, 5.0, 104.0, 105.0, 106.0, 107.0]
    );
    // Third row: a[1,0,0..3] = 6,7,8; b[1,0,0..4] = 108..111.
    assert_eq!(
        &r.data()[14..21],
        &[6.0, 7.0, 8.0, 108.0, 109.0, 110.0, 111.0]
    );
    // Fourth row: a[1,1,0..3] = 9,10,11; b[1,1,0..4] = 112..115.
    assert_eq!(
        &r.data()[21..28],
        &[9.0, 10.0, 11.0, 112.0, 113.0, 114.0, 115.0]
    );
}

#[test]
fn concat_rank4_axis3_shape_and_content() {
    // a: [2, 1, 2, 2] = 8 elements, b: [2, 1, 2, 1] = 4 elements, concat axis 3 -> [2, 1, 2, 3].
    let a_data: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let b_data: Vec<f64> = (50..54).map(|i| i as f64).collect();
    let a = DenseArray::new(Shape::new(vec![2, 1, 2, 2]), a_data).unwrap();
    let b = DenseArray::new(Shape::new(vec![2, 1, 2, 1]), b_data).unwrap();
    let r = a.concat(&b, 3).expect("rank-4 concat axis 3 must succeed");
    assert_eq!(r.shape().dims(), &[2, 1, 2, 3]);
    // a[0,0,0,0..2]=0,1; b[0,0,0,0..1]=50 -> 0,1,50
    assert_eq!(&r.data()[..3], &[0.0, 1.0, 50.0]);
    // a[0,0,1,0..2]=2,3; b[0,0,1,0..1]=51 -> 2,3,51
    assert_eq!(&r.data()[3..6], &[2.0, 3.0, 51.0]);
    // a[1,0,0,0..2]=4,5; b[1,0,0,0..1]=52 -> 4,5,52
    assert_eq!(&r.data()[6..9], &[4.0, 5.0, 52.0]);
    // a[1,0,1,0..2]=6,7; b[1,0,1,0..1]=53 -> 6,7,53
    assert_eq!(&r.data()[9..12], &[6.0, 7.0, 53.0]);
}

#[test]
fn concat_rank3_axis0_still_works() {
    // Regression guard: the axis-0 path still concatenates whole slabs.
    let a = DenseArray::new(
        Shape::new(vec![2, 2, 2]),
        (0..8).map(|i| i as f64).collect(),
    )
    .unwrap();
    let b = DenseArray::new(
        Shape::new(vec![1, 2, 2]),
        (100..104).map(|i| i as f64).collect(),
    )
    .unwrap();
    let r = a.concat(&b, 0).expect("rank-3 concat axis 0");
    assert_eq!(r.shape().dims(), &[3, 2, 2]);
    assert_eq!(&r.data()[..8], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert_eq!(&r.data()[8..], &[100.0, 101.0, 102.0, 103.0]);
}

#[test]
fn concat_rank3_axis1_still_works() {
    // Regression guard: the axis-1 path still works.
    let a = DenseArray::new(
        Shape::new(vec![2, 2, 2]),
        (0..8).map(|i| i as f64).collect(),
    )
    .unwrap();
    let b = DenseArray::new(
        Shape::new(vec![2, 1, 2]),
        (50..54).map(|i| i as f64).collect(),
    )
    .unwrap();
    let r = a.concat(&b, 1).expect("rank-3 concat axis 1");
    assert_eq!(r.shape().dims(), &[2, 3, 2]);
    // For i=0: a[0,0..2,:]=[0,1,2,3]; b[0,0,:]=[50,51] -> [0,1,2,3,50,51]
    assert_eq!(&r.data()[..6], &[0.0, 1.0, 2.0, 3.0, 50.0, 51.0]);
    // For i=1: a[1,0..2,:]=[4,5,6,7]; b[1,0,:]=[52,53] -> [4,5,6,7,52,53]
    assert_eq!(&r.data()[6..], &[4.0, 5.0, 6.0, 7.0, 52.0, 53.0]);
}
