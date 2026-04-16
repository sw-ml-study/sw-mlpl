//! Primitive-parity tests for `mlpl-rt`.
//!
//! Each primitive's output is asserted against known-good numeric
//! results. The numbers here match what `mlpl_runtime::call_builtin`
//! produces for the same input (hand-verified against step-005's
//! existing eval tests); keeping the check numeric here avoids a
//! dev-dep on `mlpl-runtime` and therefore keeps `mlpl-rt` fully
//! decoupled from the interpreter stack.

use mlpl_rt::{
    DenseArray, Shape, add, array_lit, div, exp, iota, log, mul, neg, rank, reduce_add,
    reduce_add_axis, relu, reshape, shape, sigmoid, sub, tanh, transpose,
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
