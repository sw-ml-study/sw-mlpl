//! Parity tests: the CUDA elementwise ops agree with hand-computed
//! references within an fp32 tolerance (candle computes in f32 on
//! the GPU). Triple-gated; a no-op off-target.

#![cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]

use mlpl_array::{DenseArray, Shape};

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
fn elementwise_same_shape() {
    let a = arr(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = arr(vec![2, 3], vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
    assert_within(
        mlpl_cuda_elementwise::add(&a, &b).unwrap().data(),
        &[1.5, 3.0, 4.5, 6.0, 7.5, 9.0],
    );
    assert_within(
        mlpl_cuda_elementwise::sub(&a, &b).unwrap().data(),
        &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    );
    assert_within(
        mlpl_cuda_elementwise::mul(&a, &b).unwrap().data(),
        &[0.5, 2.0, 4.5, 8.0, 12.5, 18.0],
    );
    assert_within(
        mlpl_cuda_elementwise::div(&a, &b).unwrap().data(),
        &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    );
}

#[test]
fn scalar_broadcast_and_neg() {
    let a = arr(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let s = DenseArray::from_scalar(10.0);
    let sum = mlpl_cuda_elementwise::add(&a, &s).unwrap();
    assert_eq!(sum.shape().dims(), &[4]);
    assert_within(sum.data(), &[11.0, 12.0, 13.0, 14.0]);
    assert_within(
        mlpl_cuda_elementwise::neg(&a).data(),
        &[-1.0, -2.0, -3.0, -4.0],
    );
}

#[test]
fn activations_match_cpu() {
    let a = arr(vec![5], vec![-1.0, -0.25, 0.0, 0.5, 2.0]);
    assert_within(
        mlpl_cuda_elementwise::exp(&a).data(),
        mlpl_rt::exp(&a).data(),
    );
    assert_within(
        mlpl_cuda_elementwise::relu(&a).data(),
        mlpl_rt::relu(&a).data(),
    );
    assert_within(
        mlpl_cuda_elementwise::sigmoid(&a).data(),
        mlpl_rt::sigmoid(&a).data(),
    );
    assert_within(
        mlpl_cuda_elementwise::tanh(&a).data(),
        mlpl_rt::tanh(&a).data(),
    );
    let pos = arr(vec![3], vec![0.5, 1.0, 4.0]);
    assert_within(
        mlpl_cuda_elementwise::log(&pos).data(),
        mlpl_rt::log(&pos).data(),
    );
}
