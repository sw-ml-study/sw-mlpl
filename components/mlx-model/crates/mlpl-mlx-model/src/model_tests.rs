//! Step 009 slice: prove the FULL demo-architecture model trains on
//! MLX -- gradients flow through embed -> rms_norm -> attention ->
//! rms_norm -> lora head -> cross_entropy, and the on-device MlxAdam
//! drives the loss down by training only the LoRA adapters (frozen
//! base). One value_and_grad test -> no serialization lock needed.

use crate::{DemoWeights, demo_forward};
use mlpl_mlx_forward::causal_mask;
use mlpl_mlx_train::{MlxAdam, train_steps};
use mlx_rs::Array;

// Small varied weights in [-0.4, 0.4], deterministic (no RNG).
fn pat(seed: i32, n: usize) -> Vec<f32> {
    (0..n as i32)
        .map(|i| (((i + seed) % 5) as f32) * 0.2 - 0.4)
        .collect()
}

// V=d=4, T=3, rank=2. Base weights frozen; adapters: A nonzero, B zero
// (initial delta zero). 5 linears (wq/wk/wv/wo/head) -> 10 adapters.
fn fixtures() -> (DemoWeights, Vec<Array>, Array, Array) {
    let m44 = |seed| Array::from_slice(&pat(seed, 16), &[4, 4]);
    let g = Array::from_slice(&[1.0f32; 4], &[4]);
    let w = DemoWeights {
        embed: m44(1),
        gamma1: g.clone(),
        wq: m44(2),
        wk: m44(3),
        wv: m44(4),
        wo: m44(5),
        gamma2: g,
        head: m44(6),
        mask: causal_mask(3),
        scale: 2.0,
        eps: 1e-5,
    };
    // tokens 0,1,2 -> targets 1,2,3 (one-hot [T, V]).
    let x = Array::from_slice(&[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.], &[3, 4]);
    let y = Array::from_slice(&[0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.], &[3, 4]);
    (w, adapters(), x, y)
}

// 5 LoRA pairs (wq/wk/wv/wo/head): A nonzero [4,2], B zero [2,4].
fn adapters() -> Vec<Array> {
    let mut ad = Vec::new();
    for k in 0..5 {
        ad.push(Array::from_slice(&pat(k + 7, 8), &[4, 2]));
        ad.push(Array::from_slice(&[0.0f32; 8], &[2, 4]));
    }
    ad
}

#[test]
fn demo_model_trains_on_mlx() {
    let (w, mut ad, x, y) = fixtures();
    let mut adam = MlxAdam::with_lr(0.05);
    let curve = train_steps(&mut ad, &mut adam, 200, |a| {
        Ok(vec![demo_forward(&w, a, &x, &y)?])
    })
    .unwrap();
    let (first, last) = (curve[0], *curve.last().unwrap());
    assert!(
        last < first * 0.9,
        "demo loss {first} -> {last} (should drop)"
    );
}
