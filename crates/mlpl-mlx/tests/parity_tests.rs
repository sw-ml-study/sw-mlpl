//! Parity tests: `mlpl_mlx::matmul` agrees with the CPU path in
//! `mlpl_array::DenseArray::matmul` on shape, labels, and
//! numerical values.
//!
//! The MLX path computes in fp32 (GPU / Accelerate dispatch) and
//! casts back to f64 on the way out; the CPU path stays in f64
//! end-to-end. That means the two outputs are NOT bit-for-bit
//! equal -- f32 rounding dominates the difference. For k = 4 and
//! small operand magnitudes, the theoretical relative error is
//! O(k) * fp32_eps ~= 5e-7; we use a 1e-5 absolute tolerance so
//! the test is robust to operand rescaling.
//!
//! All MLX tests are triple-gated behind:
//!   * `target_os = "macos"`,
//!   * `target_arch = "aarch64"`,
//!   * `feature = "mlx"`.
//!
//! On any other configuration `cargo test -p mlpl-mlx` still
//! compiles (the tests simply aren't generated). That matches the
//! Saga 14 invariant that CI on non-Apple hosts stays green.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_mlx::{DenseArray, Shape};

/// Tolerance budget for MLX (fp32) vs CPU (f64) parity. See the
/// module-level docstring for the derivation.
const FP32_TOL: f64 = 1e-5;

fn assert_within_fp32(mlx: &[f64], cpu: &[f64]) {
    assert_eq!(mlx.len(), cpu.len(), "output length mismatch");
    for (i, (m, c)) in mlx.iter().zip(cpu.iter()).enumerate() {
        let diff = (m - c).abs();
        assert!(
            diff <= FP32_TOL,
            "element {i}: mlx={m} cpu={c} diff={diff} tol={FP32_TOL}"
        );
    }
}

#[test]
fn matmul_8x4_mul_4x8_matches_cpu_within_fp32_tolerance() {
    let a_data: Vec<f64> = (0..32).map(|i| ((i as f64) * 0.1) - 1.5).collect();
    let b_data: Vec<f64> = (0..32).map(|i| ((i as f64) * -0.05) + 0.25).collect();
    let a = DenseArray::new(Shape::new(vec![8, 4]), a_data).unwrap();
    let b = DenseArray::new(Shape::new(vec![4, 8]), b_data).unwrap();

    let cpu = a.matmul(&b).unwrap();
    let mlx = mlpl_mlx::matmul(&a, &b).unwrap();

    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn matmul_matrix_vector_matches_cpu() {
    let a = DenseArray::new(Shape::new(vec![3, 4]), (0..12).map(|i| i as f64).collect()).unwrap();
    let b = DenseArray::new(Shape::vector(4), vec![0.5, -0.25, 2.0, 1.0]).unwrap();

    let cpu = a.matmul(&b).unwrap();
    let mlx = mlpl_mlx::matmul(&a, &b).unwrap();

    assert_eq!(mlx.shape(), cpu.shape());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn matmul_propagates_labels_on_both_sides() {
    // Saga 11.5: contraction axis labels agree -> surviving labels
    // are [a_row_label, b_col_label].
    let a = DenseArray::new(Shape::new(vec![2, 3]), (0..6).map(|i| i as f64).collect())
        .unwrap()
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap();
    let b = DenseArray::new(Shape::new(vec![3, 4]), (0..12).map(|i| i as f64).collect())
        .unwrap()
        .with_labels(vec![Some("feat".into()), Some("out".into())])
        .unwrap();

    let mlx = mlpl_mlx::matmul(&a, &b).unwrap();

    assert_eq!(mlx.shape(), &Shape::new(vec![2, 4]));
    assert_eq!(
        mlx.labels(),
        Some(&[Some("batch".into()), Some("out".into())][..])
    );
}

#[test]
fn matmul_rejects_label_mismatch_on_contraction_axis() {
    let a = DenseArray::new(Shape::new(vec![2, 3]), (0..6).map(|i| i as f64).collect())
        .unwrap()
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap();
    let b = DenseArray::new(Shape::new(vec![3, 4]), (0..12).map(|i| i as f64).collect())
        .unwrap()
        .with_labels(vec![Some("time".into()), Some("out".into())])
        .unwrap();

    assert!(mlpl_mlx::matmul(&a, &b).is_err());
}

#[test]
fn matmul_rejects_contraction_dim_mismatch() {
    let a = DenseArray::new(Shape::new(vec![2, 3]), vec![0.0; 6]).unwrap();
    let b = DenseArray::new(Shape::new(vec![5, 4]), vec![0.0; 20]).unwrap();

    assert!(mlpl_mlx::matmul(&a, &b).is_err());
}

// Saga 14 step 002: forward-pass primitives.
//
// Each test below compares `mlpl_mlx::<op>` against `mlpl_rt::<op>`
// on the same fixture. Because the MLX path computes in fp32 and
// casts back to f64 at the boundary, the parity bound is `FP32_TOL`
// (see module-level docstring), not bit-for-bit. Labels are compared
// exactly -- they live in `mlpl-core`, not MLX, so no precision
// round-trip applies.

fn mat23(data: [f64; 6]) -> DenseArray {
    DenseArray::new(Shape::new(vec![2, 3]), data.to_vec()).unwrap()
}

fn mat23_labeled(data: [f64; 6]) -> DenseArray {
    mat23(data)
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap()
}

#[test]
fn add_matches_cpu_on_labeled_matrix() {
    let a = mat23_labeled([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = mat23_labeled([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    let cpu = mlpl_rt::add(&a, &b).unwrap();
    let mlx = mlpl_mlx::add(&a, &b).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn sub_matches_cpu() {
    let a = mat23([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
    let b = mat23([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let cpu = mlpl_rt::sub(&a, &b).unwrap();
    let mlx = mlpl_mlx::sub(&a, &b).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn mul_matches_cpu() {
    let a = mat23([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = mat23([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    let cpu = mlpl_rt::mul(&a, &b).unwrap();
    let mlx = mlpl_mlx::mul(&a, &b).unwrap();
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn div_matches_cpu() {
    let a = mat23([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    let b = mat23([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    let cpu = mlpl_rt::div(&a, &b).unwrap();
    let mlx = mlpl_mlx::div(&a, &b).unwrap();
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn neg_preserves_labels_and_matches_cpu() {
    let a = mat23_labeled([1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
    let cpu = mlpl_rt::neg(&a);
    let mlx = mlpl_mlx::neg(&a);
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn exp_matches_cpu_within_fp32_tolerance() {
    let a = DenseArray::new(
        Shape::new(vec![2, 3]),
        vec![0.0, 0.25, 0.5, -0.5, -0.25, 1.0],
    )
    .unwrap();
    let cpu = mlpl_rt::exp(&a);
    let mlx = mlpl_mlx::exp(&a);
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn log_matches_cpu_within_fp32_tolerance() {
    let a = DenseArray::new(Shape::vector(5), vec![0.25, 0.5, 1.0, 2.0, 4.0]).unwrap();
    let cpu = mlpl_rt::log(&a);
    let mlx = mlpl_mlx::log(&a);
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn tanh_matches_cpu_within_fp32_tolerance() {
    let a = DenseArray::new(Shape::vector(5), vec![-2.0, -0.5, 0.0, 0.5, 2.0]).unwrap();
    let cpu = mlpl_rt::tanh(&a);
    let mlx = mlpl_mlx::tanh(&a);
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn sigmoid_matches_cpu_within_fp32_tolerance() {
    let a = DenseArray::new(Shape::vector(5), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
    let cpu = mlpl_rt::sigmoid(&a);
    let mlx = mlpl_mlx::sigmoid(&a);
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn relu_matches_cpu_on_mixed_signs() {
    let a = DenseArray::new(Shape::vector(5), vec![-2.0, -0.5, 0.0, 0.5, 2.0]).unwrap();
    let cpu = mlpl_rt::relu(&a);
    let mlx = mlpl_mlx::relu(&a);
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn reshape_matches_cpu_on_6_to_2x3() {
    let a = DenseArray::from_vec((0..6).map(|i| i as f64).collect());
    let cpu = mlpl_rt::reshape(&a, &[2, 3]).unwrap();
    let mlx = mlpl_mlx::reshape(&a, &[2, 3]).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn transpose_matches_cpu_and_reverses_labels() {
    let a = mat23_labeled([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let cpu = mlpl_rt::transpose(&a);
    let mlx = mlpl_mlx::transpose(&a);
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    assert_within_fp32(mlx.data(), cpu.data());
}

// Saga 14 step 003: reductions, normalisation, and loss primitives.
//
// These fixtures match the prompt: `[4, 3]` and `[2, 3, 5]` for
// reductions and softmax, and `[B*T = 8, V = 5]` for cross_entropy.
// MLX runs in fp32 -- the same `FP32_TOL` budget applies, except
// the cross_entropy path round-trips through fp32 twice (once for
// the per-row LSE, once when reading it back), so we double the
// tolerance there.

const CE_TOL: f64 = 2.0 * FP32_TOL;

fn mat43() -> DenseArray {
    let data: Vec<f64> = (0..12).map(|i| (i as f64) * 0.5 - 1.0).collect();
    DenseArray::new(Shape::new(vec![4, 3]), data)
        .unwrap()
        .with_labels(vec![Some("batch".into()), Some("feat".into())])
        .unwrap()
}

fn cube235() -> DenseArray {
    let data: Vec<f64> = (0..30).map(|i| ((i as f64) - 15.0) * 0.2).collect();
    DenseArray::new(Shape::new(vec![2, 3, 5]), data)
        .unwrap()
        .with_labels(vec![
            Some("batch".into()),
            Some("time".into()),
            Some("feat".into()),
        ])
        .unwrap()
}

#[test]
fn reduce_mul_flat_matches_cpu_within_fp32_tolerance() {
    let a = mat43();
    let cpu = mlpl_rt::reduce_mul(&a, None).unwrap();
    let mlx = mlpl_mlx::reduce_mul(&a, None).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn reduce_mul_axis_drops_label_and_matches_cpu() {
    let a = mat43();
    let cpu = mlpl_rt::reduce_mul(&a, Some(0)).unwrap();
    let mlx = mlpl_mlx::reduce_mul(&a, Some(0)).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn mean_axis_on_3d_matches_cpu_and_drops_axis_label() {
    let a = cube235();
    let cpu = mlpl_rt::mean(&a, Some(1)).unwrap();
    let mlx = mlpl_mlx::mean(&a, Some(1)).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn mean_flat_returns_scalar_matching_cpu() {
    let a = cube235();
    let cpu = mlpl_rt::mean(&a, None).unwrap();
    let mlx = mlpl_mlx::mean(&a, None).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn argmax_axis_matches_cpu_index_layout() {
    let a = mat43();
    let cpu = mlpl_rt::argmax(&a, Some(1)).unwrap();
    let mlx = mlpl_mlx::argmax(&a, Some(1)).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    // Indices are exact integers, so no fp tolerance is needed.
    assert_eq!(mlx.data(), cpu.data());
}

#[test]
fn argmax_flat_returns_scalar_matching_cpu() {
    let a = mat43();
    let cpu = mlpl_rt::argmax(&a, None).unwrap();
    let mlx = mlpl_mlx::argmax(&a, None).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.data(), cpu.data());
}

#[test]
fn softmax_on_4x3_preserves_labels_and_matches_cpu() {
    let a = mat43();
    let cpu = mlpl_rt::softmax(&a, 1).unwrap();
    let mlx = mlpl_mlx::softmax(&a, 1).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn softmax_on_3d_along_feature_axis_matches_cpu() {
    let a = cube235();
    let cpu = mlpl_rt::softmax(&a, 2).unwrap();
    let mlx = mlpl_mlx::softmax(&a, 2).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn log_softmax_on_4x3_matches_cpu() {
    let a = mat43();
    let cpu = mlpl_rt::log_softmax(&a, 1).unwrap();
    let mlx = mlpl_mlx::log_softmax(&a, 1).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert_eq!(mlx.labels(), cpu.labels());
    assert_within_fp32(mlx.data(), cpu.data());
}

#[test]
fn cross_entropy_on_8x5_matches_cpu_within_fp32_tolerance() {
    // [B*T = 8, V = 5] logits with mixed signs; targets cover every class.
    let logits_data: Vec<f64> = (0..40).map(|i| ((i as f64) - 20.0) * 0.15).collect();
    let logits = DenseArray::new(Shape::new(vec![8, 5]), logits_data).unwrap();
    let targets = DenseArray::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0, 4.0]);
    let cpu = mlpl_rt::cross_entropy(&logits, &targets).unwrap();
    let mlx = mlpl_mlx::cross_entropy(&logits, &targets).unwrap();
    assert_eq!(mlx.shape(), cpu.shape());
    assert!(mlx.labels().is_none());
    assert_eq!(mlx.data().len(), 1);
    assert!(
        (mlx.data()[0] - cpu.data()[0]).abs() <= CE_TOL,
        "cross_entropy mlx={} cpu={} diff={} tol={CE_TOL}",
        mlx.data()[0],
        cpu.data()[0],
        (mlx.data()[0] - cpu.data()[0]).abs()
    );
}
