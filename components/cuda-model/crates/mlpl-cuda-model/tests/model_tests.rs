//! Prove the demo model trains on CUDA: candle autograd flows through
//! embed -> `rms_norm` -> (frozen) attention -> `rms_norm` -> `LoRA` head ->
//! `cross_entropy`, and candle's `AdamW` drives the loss down by training
//! ONLY the head `[A, B]` adapters (everything else frozen). Runs on
//! the GPU. Triple-gated; a no-op off-target.

#![cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]

use candle_core::{Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use mlpl_cuda_forward::causal_mask;
use mlpl_cuda_model::{DemoWeights, demo_forward};

// Small varied weights in [-0.4, 0.4], deterministic (no RNG). The
// `(i+seed) % 5` is 0..4, so the u8 -> f32 conversion is lossless and
// cast-free.
fn pat(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| f32::from(u8::try_from((i + seed) % 5).unwrap()) * 0.2 - 0.4)
        .collect()
}

fn mat(seed: usize, rows: usize, cols: usize, device: &Device) -> Tensor {
    Tensor::from_vec(pat(seed, rows * cols), (rows, cols), device).unwrap()
}

// V=d=4, T=3, rank=2. All base weights frozen; gamma=ones, eps=1e-8.
fn fixtures(device: &Device) -> (DemoWeights, Tensor, Tensor) {
    let weights = DemoWeights {
        embed: mat(1, 4, 4, device),
        wq: mat(2, 4, 4, device),
        wk: mat(3, 4, 4, device),
        wv: mat(4, 4, 4, device),
        wo: mat(5, 4, 4, device),
        head_w: mat(6, 4, 4, device),
        head_b: Tensor::from_vec(vec![0.0f32, 0.1, -0.1, 0.0], (1, 4), device).unwrap(),
        gamma: Tensor::from_vec(vec![1.0f32; 4], 4, device).unwrap(),
        mask: causal_mask(3, device).unwrap(),
        scale: 2.0,
        eps: 1e-8,
    };
    // 3 one-hot input tokens [1, 3, 2] and one-hot targets [3, 2, 0].
    // f32 suffix so the literals are F32 (bare floats default to f64).
    let inputs = Tensor::from_vec(
        vec![0.0f32, 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.],
        (3, 4),
        device,
    )
    .unwrap();
    let targets = Tensor::from_vec(
        vec![0.0f32, 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.],
        (3, 4),
        device,
    )
    .unwrap();
    (weights, inputs, targets)
}

#[test]
fn demo_model_trains_on_cuda() {
    let device = Device::new_cuda(0).expect("CUDA device 0");
    let (weights, inputs, targets) = fixtures(&device);
    // Head adapters: A nonzero, B zero -> student starts at the frozen
    // base (lora delta = scale * A@B = 0), so any loss drop is learning.
    let a = Var::from_tensor(&mat(7, 4, 2, &device)).unwrap();
    let zero_b = Tensor::zeros((2, 4), candle_core::DType::F32, &device).unwrap();
    let b = Var::from_tensor(&zero_b).unwrap();

    let params = ParamsAdamW {
        lr: 0.05,
        ..Default::default()
    };
    let mut opt = AdamW::new(vec![a.clone(), b.clone()], params).unwrap();

    let mut curve = Vec::new();
    for _ in 0..200 {
        let adapters = [a.as_tensor().clone(), b.as_tensor().clone()];
        let loss = demo_forward(&weights, &adapters, &inputs, &targets).unwrap();
        curve.push(loss.to_scalar::<f32>().unwrap());
        opt.backward_step(&loss).unwrap();
    }
    let (first, last) = (curve[0], *curve.last().unwrap());
    assert!(
        last < first * 0.9,
        "demo loss {first} -> {last} (should drop)"
    );
}
