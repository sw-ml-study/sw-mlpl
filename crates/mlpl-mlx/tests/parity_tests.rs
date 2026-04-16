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
