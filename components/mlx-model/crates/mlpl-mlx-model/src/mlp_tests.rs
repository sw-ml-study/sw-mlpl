//! Prove the tic-tac-toe board-policy MLP trains on MLX: gradients flow
//! through `lora(linear) -> relu -> lora(linear) -> cross_entropy`, and
//! the on-device MlxAdam drives the loss down by training only the four
//! LoRA adapters (both linears' bases frozen).

use crate::{MlpWeights, mlp_forward};
use mlpl_mlx_train::{MlxAdam, train_steps};
use mlx_rs::Array;

fn pat(seed: i32, n: usize) -> Vec<f32> {
    (0..n as i32)
        .map(|i| (((i + seed) % 5) as f32) * 0.2 - 0.4)
        .collect()
}

// in=4, hidden=3, classes=2, rank=2. Bases frozen; each layer's LoRA
// pair is A nonzero, B zero (so the student starts at the base).
fn fixtures() -> (MlpWeights, Vec<Array>, Array, Array) {
    let w = MlpWeights {
        w1: Array::from_slice(&pat(1, 12), &[4, 3]),
        b1: Array::from_slice(&[0.1f32, -0.1, 0.0], &[1, 3]),
        w2: Array::from_slice(&pat(2, 6), &[3, 2]),
        b2: Array::from_slice(&[0.0f32, 0.2], &[1, 2]),
        scale1: 2.0,
        scale2: 2.0,
    };
    let ad = vec![
        Array::from_slice(&pat(3, 8), &[4, 2]),   // A1
        Array::from_slice(&[0.0f32; 6], &[2, 3]), // B1 (zero)
        Array::from_slice(&pat(4, 6), &[3, 2]),   // A2
        Array::from_slice(&[0.0f32; 4], &[2, 2]), // B2 (zero)
    ];
    // 3 board-like rows -> classes {1, 0, 1} as one-hot [3, 2].
    let x = Array::from_slice(&pat(5, 12), &[3, 4]);
    let y = Array::from_slice(&[0., 1., 1., 0., 0., 1.], &[3, 2]);
    (w, ad, x, y)
}

#[test]
fn mlp_policy_trains_on_mlx() {
    let _mlx = crate::MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (w, mut ad, x, y) = fixtures();
    let mut adam = MlxAdam::with_lr(0.05);
    let curve = train_steps(&mut ad, &mut adam, 200, |a| {
        Ok(vec![mlp_forward(&w, a, &x, &y)?])
    })
    .unwrap();
    let (first, last) = (curve[0], *curve.last().unwrap());
    assert!(
        last < first * 0.9,
        "mlp loss {first} -> {last} (should drop)"
    );
}
