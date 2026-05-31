//! Step 006 slice: prove the LoRA-adapter training mechanism on-device.
//! A frozen base `W` plus trainable low-rank `A`,`B`; `value_and_grad`
//! over BOTH adapters and `MlxAdam` recover a rank-1 target residual.
//! Gradient parity is checked by finite differences (vs MLX autodiff),
//! exercising forward + backward + optimizer all on MLX.

use crate::{MlxAdam, lora_linear, loss_and_grads, train_steps};
use mlx_rs::Array;
use mlx_rs::error::Result;

const IN: usize = 3;
const OUT: usize = 2;

// Returns (X, frozen W, target Y, initial [A, B]). Y = X @ (W + A* B*)
// has a rank-1 residual, so trainable rank-1 adapters can fully fit it.
// A starts small/nonzero, B starts zero -> initial delta is zero.
fn fixtures() -> (Array, Array, Array, Vec<Array>) {
    let x = Array::from_slice(
        &[
            1.0f32, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0,
        ],
        &[4, IN as i32],
    );
    let w = Array::from_slice(
        &[0.5f32, -0.5, 1.0, 0.0, -1.0, 0.5],
        &[IN as i32, OUT as i32],
    );
    let a_star = Array::from_slice(&[1.0f32, -1.0, 0.5], &[IN as i32, 1]);
    let b_star = Array::from_slice(&[2.0f32, -1.0], &[1, OUT as i32]);
    let y = x
        .matmul(w.add(a_star.matmul(&b_star).unwrap()).unwrap())
        .unwrap();
    let init = vec![
        Array::from_slice(&[0.1f32, 0.1, 0.1], &[IN as i32, 1]),
        Array::from_slice(&[0.0f32, 0.0], &[1, OUT as i32]),
    ];
    (x, w, y, init)
}

fn mse(x: &Array, w: &Array, y: &Array, a: &Array, b: &Array) -> Result<Array> {
    lora_linear(x, w, a, b, 1.0)?
        .subtract(y)?
        .square()?
        .mean(None)
}

#[test]
fn value_and_grad_matches_finite_difference() {
    let _mlx = crate::MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (x, w, y, p) = fixtures();
    let (loss0, grads) = loss_and_grads(&p, |q| Ok(vec![mse(&x, &w, &y, &q[0], &q[1])?])).unwrap();
    // Perturb A[0,0] and forward-difference the loss.
    let mut a_pert = p[0].as_slice::<f32>().to_vec();
    a_pert[0] += 1e-3;
    let a2 = Array::from_slice(&a_pert, &[IN as i32, 1]);
    let (loss1, _) = loss_and_grads(&[a2, p[1].clone()], |q| {
        Ok(vec![mse(&x, &w, &y, &q[0], &q[1])?])
    })
    .unwrap();
    let numeric = (loss1 - loss0) / 1e-3;
    let analytic = grads[0].as_slice::<f32>()[0];
    assert!(
        (numeric - analytic).abs() < 2e-2,
        "dL/dA[0,0]: {numeric} vs {analytic}"
    );
}

#[test]
fn on_device_adam_trains_lora_adapters_to_fit() {
    let _mlx = crate::MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (x, w, y, mut p) = fixtures();
    let mut adam = MlxAdam::with_lr(0.1);
    let curve = train_steps(&mut p, &mut adam, 800, |q| {
        Ok(vec![mse(&x, &w, &y, &q[0], &q[1])?])
    })
    .unwrap();
    let (first, last) = (curve[0], *curve.last().unwrap());
    assert!(
        first > 0.1,
        "initial loss {first} should be clearly nonzero"
    );
    assert!(
        last < first * 1e-3,
        "loss {first} -> {last} (should collapse)"
    );
}
