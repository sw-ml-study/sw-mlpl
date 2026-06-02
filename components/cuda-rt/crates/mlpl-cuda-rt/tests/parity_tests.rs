//! Parity tests: `mlpl-cuda-rt` matmul + shape ops agree with the
//! CPU path on shape and values. candle computes in fp32 on the GPU
//! and casts back to f64, so we compare within an fp32 tolerance.
//! matmul uses the CPU `DenseArray::matmul` as reference; shape ops
//! are checked against hand-computed values.
//!
//! Triple-gated (cuda feature + Linux + `x86_64`); a no-op elsewhere.

#![cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]

use mlpl_array_ops_matmul::prelude::*;
use mlpl_cuda_rt::{DenseArray, Shape};

const FP32_TOL: f64 = 1e-5;

fn assert_within(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() <= FP32_TOL, "elem {i}: cuda={g} want={w}");
    }
}

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn matmul_matches_cpu() {
    let a = arr(
        vec![8, 4],
        (0..32).map(|i| f64::from(i) * 0.1 - 1.5).collect(),
    );
    let b = arr(
        vec![4, 8],
        (0..32).map(|i| f64::from(i) * -0.05 + 0.25).collect(),
    );
    let cpu = a.matmul(&b).unwrap();
    let cuda = mlpl_cuda_rt::matmul(&a, &b).unwrap();
    assert_eq!(cuda.shape(), cpu.shape());
    assert_within(cuda.data(), cpu.data());
}

#[test]
fn matmul_matrix_vector_matches_cpu() {
    let a = arr(vec![3, 4], (0..12).map(f64::from).collect());
    let b = DenseArray::new(Shape::vector(4), vec![0.5, -0.25, 2.0, 1.0]).unwrap();
    let cpu = a.matmul(&b).unwrap();
    let cuda = mlpl_cuda_rt::matmul(&a, &b).unwrap();
    assert_eq!(cuda.shape(), cpu.shape());
    assert_within(cuda.data(), cpu.data());
}

#[test]
fn reshape_preserves_data() {
    let a = arr(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = mlpl_cuda_rt::reshape(&a, &[3, 2]).unwrap();
    assert_eq!(r.shape().dims(), &[3, 2]);
    assert_within(r.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(mlpl_cuda_rt::reshape(&a, &[5]).is_err());
}

#[test]
fn transpose_2x3() {
    let a = arr(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = mlpl_cuda_rt::transpose(&a);
    assert_eq!(t.shape().dims(), &[3, 2]);
    assert_within(t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
