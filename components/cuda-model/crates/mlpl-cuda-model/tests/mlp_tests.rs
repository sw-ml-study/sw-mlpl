//! Prove the tic-tac-toe board-policy MLP trains on CUDA: candle
//! autograd flows through lora(linear) -> relu -> lora(linear) ->
//! `cross_entropy`, and candle's `AdamW` drives the loss down by training
//! ONLY the four `LoRA` adapters (both bases frozen). On the GPU.
//! Triple-gated; a no-op off-target.

#![cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use mlpl_cuda_model::{MlpWeights, mlp_forward};

// Small varied weights in [-0.4, 0.4], deterministic (no RNG); cast-free.
fn pat(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| f32::from(u8::try_from((i + seed) % 5).unwrap()) * 0.2 - 0.4)
        .collect()
}

fn mat(seed: usize, rows: usize, cols: usize, device: &Device) -> Tensor {
    Tensor::from_vec(pat(seed, rows * cols), (rows, cols), device).unwrap()
}

#[test]
fn mlp_policy_trains_on_cuda() {
    // in=4, hidden=3, classes=2, rank=2. Bases frozen; each layer's LoRA
    // pair is A nonzero, B zero (so the student starts at the base).
    let device = Device::new_cuda(0).expect("CUDA device 0");
    let w = MlpWeights {
        w1: mat(1, 4, 3, &device),
        b1: Tensor::from_vec(vec![0.1f32, -0.1, 0.0], (1, 3), &device).unwrap(),
        w2: mat(2, 3, 2, &device),
        b2: Tensor::from_vec(vec![0.0f32, 0.2], (1, 2), &device).unwrap(),
        scale1: 2.0,
        scale2: 2.0,
    };
    let zero =
        |r, c| Var::from_tensor(&Tensor::zeros((r, c), DType::F32, &device).unwrap()).unwrap();
    let a1 = Var::from_tensor(&mat(3, 4, 2, &device)).unwrap();
    let b1 = zero(2, 3);
    let a2 = Var::from_tensor(&mat(4, 3, 2, &device)).unwrap();
    let b2 = zero(2, 2);
    // 3 board-like rows -> classes {1, 0, 1} as one-hot [3, 2].
    let x = mat(5, 3, 4, &device);
    let y = Tensor::from_vec(vec![0.0f32, 1., 1., 0., 0., 1.], (3, 2), &device).unwrap();

    let params = ParamsAdamW {
        lr: 0.05,
        ..Default::default()
    };
    let mut opt = AdamW::new(vec![a1.clone(), b1.clone(), a2.clone(), b2.clone()], params).unwrap();
    let mut curve = Vec::new();
    for _ in 0..200 {
        let adapters = [
            a1.as_tensor().clone(),
            b1.as_tensor().clone(),
            a2.as_tensor().clone(),
            b2.as_tensor().clone(),
        ];
        let loss = mlp_forward(&w, &adapters, &x, &y).unwrap();
        curve.push(loss.to_scalar::<f32>().unwrap());
        opt.backward_step(&loss).unwrap();
    }
    let (first, last) = (curve[0], *curve.last().unwrap());
    assert!(
        last < first * 0.9,
        "mlp loss {first} -> {last} (should drop)"
    );
}
