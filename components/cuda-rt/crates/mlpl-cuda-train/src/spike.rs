//! Least-squares spike: prove candle autodiff + Adam run ON the
//! CUDA GPU. Minimize `mean((X w - y)^2)`. The analytic gradient at
//! `w = 0` is `-(2/n) transpose(X) @ y`; Adam must drive the loss toward zero.

use candle_core::{Device, Result, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

/// Open CUDA device 0 (the GO/NO-GO signal for the whole CUDA track).
///
/// # Errors
/// Returns an error if the candle CUDA backend cannot initialize
/// device 0 (no GPU, driver/toolkit mismatch, etc.).
pub fn cuda_device() -> Result<Device> {
    Device::new_cuda(0)
}

/// Mean-squared-error loss of `X @ w` against `y` (column vectors).
fn mse(x: &Tensor, w: &Tensor, y: &Tensor) -> Result<Tensor> {
    x.matmul(w)?.sub(y)?.sqr()?.mean_all()
}

/// Gradient of the MSE loss w.r.t. `w` evaluated at `w = 0`, on the
/// GPU. Returned as a host `Vec<f32>` for assertion against the
/// closed form `-(2/n) transpose(X) @ y`.
///
/// # Errors
/// Returns an error if any GPU tensor op or the backward pass fails.
///
/// # Panics
/// Panics if the autograd store has no gradient for `w` (would mean
/// candle failed to record the parameter).
pub fn grad_at_zero(x: &Tensor, y: &Tensor, dim: usize) -> Result<Vec<f32>> {
    let dev = x.device();
    let weights = Var::from_tensor(&Tensor::zeros((dim, 1), candle_core::DType::F32, dev)?)?;
    let loss = mse(x, weights.as_tensor(), y)?;
    let grads = loss.backward()?;
    let grad = grads.get(&weights).expect("w has a gradient");
    grad.flatten_all()?.to_vec1::<f32>()
}

/// Run `steps` of Adam on `w` (init zero) and return the loss curve.
/// All tensors stay resident on `x`'s device.
///
/// # Errors
/// Returns an error if any GPU tensor op or optimizer step fails.
pub fn train_adam(x: &Tensor, y: &Tensor, d: usize, steps: usize, lr: f64) -> Result<Vec<f32>> {
    let dev = x.device();
    let w = Var::from_tensor(&Tensor::zeros((d, 1), candle_core::DType::F32, dev)?)?;
    let params = ParamsAdamW {
        lr,
        ..Default::default()
    };
    let mut opt = AdamW::new(vec![w.clone()], params)?;
    let mut curve = Vec::with_capacity(steps);
    for _ in 0..steps {
        let loss = mse(x, w.as_tensor(), y)?;
        curve.push(loss.to_scalar::<f32>()?);
        opt.backward_step(&loss)?;
    }
    Ok(curve)
}
