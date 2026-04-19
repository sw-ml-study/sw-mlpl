//! Primitive-parity tests for `mlpl-rt`.
//!
//! Each primitive's output is asserted against known-good numeric
//! results. The numbers here match what `mlpl_runtime::call_builtin`
//! produces for the same input (hand-verified against step-005's
//! existing eval tests); keeping the check numeric here avoids a
//! dev-dep on `mlpl-runtime` and therefore keeps `mlpl-rt` fully
//! decoupled from the interpreter stack.

use mlpl_rt::{
    DenseArray, Shape, add, argmax, array_lit, cross_entropy, div, exp, iota, log, log_softmax,
    mean, mul, neg, rank, reduce_add, reduce_add_axis, reduce_mul, relu, reshape, shape, sigmoid,
    softmax, sub, tanh, transpose,
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
fn array_lit_scalars_pack_to_vector() {
    let v = array_lit(vec![
        DenseArray::from_scalar(1.0),
        DenseArray::from_scalar(2.0),
        DenseArray::from_scalar(3.0),
    ])
    .unwrap();
    assert_eq!(v.shape(), &Shape::vector(3));
    assert_eq!(v.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn array_lit_empty_is_empty_vector() {
    let v = array_lit(vec![]).unwrap();
    assert_eq!(v.shape(), &Shape::vector(0));
}

#[test]
fn array_lit_rows_stack_into_matrix() {
    // [[1, 2, 3], [4, 5, 6]] -> shape [2, 3].
    let row_a = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    let row_b = DenseArray::from_vec(vec![4.0, 5.0, 6.0]);
    let m = array_lit(vec![row_a, row_b]).unwrap();
    assert_eq!(m.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(m.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
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

// Saga 14 step 002: forward-pass primitives that the MLX backend mirrors.
// Every test here exercises shape, values, and label propagation on a
// small fixture so the mlpl-mlx parity tests have a trusted gold
// standard to compare against.

fn mat23(data: [f64; 6]) -> DenseArray {
    DenseArray::new(Shape::new(vec![2, 3]), data.to_vec()).unwrap()
}

#[test]
fn add_elementwise_matches_broadcast_and_labels() {
    let a = mat23([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap();
    let b = mat23([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    let c = add(&a, &b).unwrap();
    assert_eq!(c.shape(), a.shape());
    assert_eq!(c.data(), &[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
    assert_eq!(c.labels(), a.labels());
}

#[test]
fn sub_elementwise_produces_expected_values() {
    let a = mat23([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
    let b = mat23([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let c = sub(&a, &b).unwrap();
    assert_eq!(c.data(), &[4.0, 3.0, 2.0, 1.0, 0.0, -1.0]);
}

#[test]
fn mul_elementwise_produces_expected_values() {
    let a = mat23([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = mat23([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    let c = mul(&a, &b).unwrap();
    assert_eq!(c.data(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn div_elementwise_produces_expected_values() {
    let a = mat23([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    let b = mat23([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    let c = div(&a, &b).unwrap();
    assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn neg_preserves_labels_and_flips_signs() {
    let a = mat23([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
        .with_labels(vec![Some("r".into()), Some("c".into())])
        .unwrap();
    let c = neg(&a);
    assert_eq!(c.data(), &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    assert_eq!(c.labels(), a.labels());
}

#[test]
fn exp_of_zero_vector_is_ones() {
    let a = DenseArray::from_vec(vec![0.0, 0.0, 0.0]);
    let c = exp(&a);
    assert_eq!(c.data(), &[1.0, 1.0, 1.0]);
}

#[test]
fn log_of_e_vector_is_ones() {
    let e = std::f64::consts::E;
    let a = DenseArray::from_vec(vec![e, e, e]);
    let c = log(&a);
    for &v in c.data() {
        assert!((v - 1.0_f64).abs() < 1e-12);
    }
}

#[test]
fn tanh_of_zero_is_zero() {
    let a = DenseArray::from_vec(vec![0.0, 0.0]);
    let c = tanh(&a);
    assert_eq!(c.data(), &[0.0, 0.0]);
}

#[test]
fn sigmoid_of_zero_is_half() {
    let a = DenseArray::from_vec(vec![0.0, 0.0, 0.0]);
    let c = sigmoid(&a);
    for &v in c.data() {
        assert!((v - 0.5_f64).abs() < 1e-12);
    }
}

#[test]
fn relu_zeros_negatives_and_keeps_positives() {
    let a = DenseArray::from_vec(vec![-2.0, -0.5, 0.0, 0.5, 2.0]);
    let c = relu(&a);
    assert_eq!(c.data(), &[0.0, 0.0, 0.0, 0.5, 2.0]);
}

// Saga 14 step 003: reductions, normalisation, and loss primitives.
//
// These tests pin the CPU-side semantics that the MLX parity tests
// in `mlpl-mlx/tests/parity_tests.rs` compare against, including
// the Saga 11.5 label-propagation rules (reductions drop the
// reduced axis label; softmax/log_softmax preserve labels;
// cross_entropy returns an unlabeled scalar).

#[test]
fn reduce_mul_flat_is_product_of_elements() {
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let r = reduce_mul(&v, None).unwrap();
    assert_eq!(r.data(), &[24.0]);
}

#[test]
fn reduce_mul_axis_drops_reduced_label() {
    let m = reshape(&iota(6), &[2, 3])
        .unwrap()
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap();
    let r = reduce_mul(&m, Some(0)).unwrap();
    // [[0,1,2],[3,4,5]] product along axis 0 -> [0, 4, 10]
    assert_eq!(r.shape(), &Shape::vector(3));
    assert_eq!(r.data(), &[0.0, 4.0, 10.0]);
    assert_eq!(r.labels(), Some(&[Some("feat".into())][..]));
}

#[test]
fn mean_flat_is_average() {
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let r = mean(&v, None).unwrap();
    assert_eq!(r.data(), &[2.5]);
}

#[test]
fn mean_axis_returns_per_axis_average() {
    let m = reshape(&iota(6), &[2, 3]).unwrap();
    // [[0,1,2],[3,4,5]] mean along axis 1 -> [1, 4]
    let r = mean(&m, Some(1)).unwrap();
    assert_eq!(r.shape(), &Shape::vector(2));
    assert_eq!(r.data(), &[1.0, 4.0]);
}

#[test]
fn argmax_flat_returns_index_of_max() {
    let v = DenseArray::from_vec(vec![0.5, -1.0, 4.0, 2.0, 4.0]);
    // Ties go to the first occurrence -> 2.
    let r = argmax(&v, None).unwrap();
    assert_eq!(r.data(), &[2.0]);
}

#[test]
fn argmax_axis_drops_reduced_axis() {
    let m = DenseArray::new(Shape::new(vec![2, 3]), vec![0.0, 5.0, 1.0, 4.0, 2.0, 3.0]).unwrap();
    let r = argmax(&m, Some(1)).unwrap();
    // Row 0: [0,5,1] -> 1; Row 1: [4,2,3] -> 0.
    assert_eq!(r.shape(), &Shape::vector(2));
    assert_eq!(r.data(), &[1.0, 0.0]);
}

#[test]
fn softmax_axis_normalises_per_group_to_one() {
    let m = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
        .unwrap()
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap();
    let s = softmax(&m, 1).unwrap();
    assert_eq!(s.shape(), m.shape());
    assert_eq!(s.labels(), m.labels());
    for row in s.data().chunks(3) {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
}

#[test]
fn log_softmax_equals_log_of_softmax() {
    let m = DenseArray::new(
        Shape::new(vec![2, 3]),
        vec![0.5, -1.0, 2.0, 3.0, 0.0, -0.25],
    )
    .unwrap();
    let s = softmax(&m, 1).unwrap();
    let ls = log_softmax(&m, 1).unwrap();
    for (sv, lsv) in s.data().iter().zip(ls.data().iter()) {
        assert!((sv.ln() - lsv).abs() < 1e-12);
    }
}

#[test]
fn cross_entropy_matches_manual_log_sum_exp() {
    // logits [N=4, V=3], targets [N=4].
    let logits = DenseArray::new(
        Shape::new(vec![4, 3]),
        vec![2.0, 1.0, 0.1, 0.5, 2.5, 0.0, 1.0, 1.0, 1.0, -1.0, -2.0, 3.0],
    )
    .unwrap();
    let targets = DenseArray::from_vec(vec![0.0, 1.0, 2.0, 2.0]);
    let loss = cross_entropy(&logits, &targets).unwrap();
    let mut expected = 0.0;
    for (i, t) in [0usize, 1, 2, 2].iter().enumerate() {
        let row: Vec<f64> = logits.data()[i * 3..(i + 1) * 3].to_vec();
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + row.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        expected += lse - row[*t];
    }
    expected /= 4.0;
    assert_eq!(loss.shape().dims(), &[] as &[usize]);
    assert!(loss.labels().is_none());
    assert!((loss.data()[0] - expected).abs() < 1e-12);
}
