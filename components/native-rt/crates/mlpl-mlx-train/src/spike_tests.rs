//! Step 005 spike: prove approach A end-to-end on a tiny convex
//! problem -- gradients from `value_and_grad` match the analytic
//! gradient (parity vs CPU), and the on-device Adam converges to the
//! closed-form least-squares solution. Forward, backward, AND the
//! optimizer run on MLX.

use crate::{MlxAdam, loss_and_grads};
use mlx_rs::Array;
use mlx_rs::error::Result;

// Fixed 4x2 design matrix and targets y = X @ [2, -1].
// Rows: (1,0)->2, (0,1)->-1, (1,1)->1, (2,1)->3.
fn fixtures() -> (Array, Array) {
    let x = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0], &[4, 2]);
    let y = Array::from_slice(&[2.0f32, -1.0, 1.0, 3.0], &[4, 1]);
    (x, y)
}

// mean((X w - y)^2), a scalar Array, fully on-device.
fn mse(x: &Array, y: &Array, w: &Array) -> Result<Array> {
    x.matmul(w)?.subtract(y)?.square()?.mean(None)
}

#[test]
fn value_and_grad_matches_analytic_gradient_at_zero() {
    let (x, y) = fixtures();
    let params = vec![Array::from_slice(&[0.0f32, 0.0], &[2, 1])];
    let (_loss, grads) = loss_and_grads(&params, |p| Ok(vec![mse(&x, &y, &p[0])?])).unwrap();
    // d/dw mean((Xw-y)^2) = (2/n) X^T (Xw - y); at w=0 that is
    // (2/4) X^T (-y) = 0.5 * [-9, -3] = [-4.5, -1.5].
    let g = grads[0].as_slice::<f32>();
    assert!((g[0] - -4.5).abs() < 1e-4, "grad[0] = {} (want -4.5)", g[0]);
    assert!((g[1] - -1.5).abs() < 1e-4, "grad[1] = {} (want -1.5)", g[1]);
}

#[test]
fn on_device_adam_converges_to_closed_form() {
    let (x, y) = fixtures();
    let mut params = vec![Array::from_slice(&[0.0f32, 0.0], &[2, 1])];
    let mut adam = MlxAdam::with_lr(0.3);
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    for step in 0..400 {
        let (loss, grads) = loss_and_grads(&params, |p| Ok(vec![mse(&x, &y, &p[0])?])).unwrap();
        if step == 0 {
            first_loss = loss;
        }
        last_loss = loss;
        adam.step(&mut params, &grads).unwrap();
    }
    // Loss collapses and w lands on the true [2, -1].
    assert!(
        last_loss < first_loss * 1e-3,
        "loss {first_loss} -> {last_loss}"
    );
    let w = params[0].as_slice::<f32>();
    assert!((w[0] - 2.0).abs() < 0.02, "w[0] = {} (want 2.0)", w[0]);
    assert!((w[1] - -1.0).abs() < 0.02, "w[1] = {} (want -1.0)", w[1]);
}
