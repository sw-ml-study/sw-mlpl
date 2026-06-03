//! Prove the CUDA training kernel on the GPU: `loss_and_grads` (candle
//! autodiff) + `adam_update` (stateless, externally-persisted moments)
//! drive a tiny regression to its target. Triple-gated; a no-op
//! off-target.

#![cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]

use candle_core::{DType, Device, Tensor};
use mlpl_cuda_train::{AdamHp, adam_update, loss_and_grads};

#[test]
fn loss_and_grads_plus_adam_minimizes() {
    let device = Device::new_cuda(0).expect("CUDA device 0");
    let target = Tensor::from_vec(vec![3.0f32, -1.0], 2, &device).unwrap();
    let mut w = Tensor::zeros(2, DType::F32, &device).unwrap();
    let mut m = w.clone();
    let mut v = w.clone();
    let (mut first, mut last) = (0.0f32, 0.0f32);

    for t in 1..=200 {
        let (loss, grads) = loss_and_grads(std::slice::from_ref(&w), |p| {
            p[0].sub(&target)?.sqr()?.sum_all()
        })
        .unwrap();
        if t == 1 {
            first = loss;
        }
        last = loss;
        let hp = AdamHp {
            lr: 0.1,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
            t,
        };
        let (w_new, m_new, v_new) = adam_update(&w, &grads[0], &m, &v, &hp).unwrap();
        w = w_new;
        m = m_new;
        v = v_new;
    }
    assert!(last < first * 0.01, "regression loss {first} -> {last}");
}
