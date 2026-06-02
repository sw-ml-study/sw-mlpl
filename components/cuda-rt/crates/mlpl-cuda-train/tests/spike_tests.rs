//! Step 001 GO/NO-GO gate: candle autodiff + Adam on the CUDA GPU.
//! Gated so the suite is a no-op off-target (non-Linux / no `cuda`
//! feature) and stays green everywhere else.

#![cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]

use candle_core::{Device, Tensor};
use mlpl_cuda_train::{cuda_device, grad_at_zero, train_adam};

/// A tiny 4x2 design matrix + target, materialized on the GPU.
fn problem(dev: &Device) -> (Tensor, Tensor) {
    let x = Tensor::from_vec(
        vec![1.0_f32, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
        (4, 2),
        dev,
    )
    .unwrap();
    let y = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (4, 1), dev).unwrap();
    (x, y)
}

#[test]
fn cuda_device_initializes() {
    let dev = cuda_device().expect("CUDA device 0 must initialize");
    assert!(dev.is_cuda(), "device must report CUDA, got {dev:?}");
}

#[test]
fn grad_matches_closed_form_at_zero() {
    let dev = cuda_device().unwrap();
    let (x, y) = problem(&dev);
    // Closed form at w=0: g = -(2/n) transpose(X) @ y.
    let n = 4.0_f32;
    let xt_y = x.t().unwrap().matmul(&y).unwrap();
    let expected: Vec<f32> = xt_y
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|v| -2.0 * v / n)
        .collect();
    let got = grad_at_zero(&x, &y, 2).unwrap();
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-4, "grad {g} != closed form {e}");
    }
}

#[test]
fn adam_reduces_loss_on_gpu() {
    let dev = cuda_device().unwrap();
    let (x, y) = problem(&dev);
    let curve = train_adam(&x, &y, 2, 400, 0.1).unwrap();
    let (first, last) = (curve[0], *curve.last().unwrap());
    assert!(last < first * 0.05, "loss must collapse: {first} -> {last}");
}
