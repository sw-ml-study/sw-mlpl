//! Parity tests: the CUDA nn ops agree with the CPU `mlpl-rt` path
//! on shape, labels, and values within an fp32 tolerance (candle
//! computes in f32 on the GPU). Triple-gated; a no-op off-target.

#![cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]

use mlpl_array::{DenseArray, Shape};

const FP32_TOL: f64 = 1e-4;

fn assert_within(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() <= FP32_TOL, "elem {i}: cuda={g} cpu={w}");
    }
}

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn reductions_match_cpu() {
    let a = arr(vec![2, 3], vec![1.0, 5.0, 2.0, 4.0, 3.0, 0.0]);
    // mean over each axis + flat
    assert_within(
        mlpl_cuda_nn::mean(&a, Some(1)).unwrap().data(),
        mlpl_rt::mean(&a, Some(1)).unwrap().data(),
    );
    assert_within(
        mlpl_cuda_nn::mean(&a, None).unwrap().data(),
        mlpl_rt::mean(&a, None).unwrap().data(),
    );
    // argmax over axis 1: row maxes at indices 1 and 0
    assert_within(
        mlpl_cuda_nn::argmax(&a, Some(1)).unwrap().data(),
        mlpl_rt::argmax(&a, Some(1)).unwrap().data(),
    );
    // reduce_mul (CPU-delegated) must match the CPU path exactly
    assert_within(
        mlpl_cuda_nn::reduce_mul(&a, Some(0)).unwrap().data(),
        mlpl_rt::reduce_mul(&a, Some(0)).unwrap().data(),
    );
}

#[test]
fn softmax_and_cross_entropy_match_cpu() {
    let logits = arr(vec![2, 3], vec![1.0, 2.0, 0.5, -1.0, 0.0, 3.0]);
    assert_within(
        mlpl_cuda_nn::softmax(&logits, 1).unwrap().data(),
        mlpl_rt::softmax(&logits, 1).unwrap().data(),
    );
    assert_within(
        mlpl_cuda_nn::log_softmax(&logits, 1).unwrap().data(),
        mlpl_rt::log_softmax(&logits, 1).unwrap().data(),
    );
    let targets = DenseArray::new(Shape::vector(2), vec![1.0, 2.0]).unwrap();
    assert_within(
        mlpl_cuda_nn::cross_entropy(&logits, &targets)
            .unwrap()
            .data(),
        mlpl_rt::cross_entropy(&logits, &targets).unwrap().data(),
    );
}

#[test]
fn axis_out_of_range_errors() {
    let a = arr(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(mlpl_cuda_nn::mean(&a, Some(5)).is_err());
    assert!(mlpl_cuda_nn::softmax(&a, 9).is_err());
}
