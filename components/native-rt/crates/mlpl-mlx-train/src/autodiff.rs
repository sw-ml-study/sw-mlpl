//! On-device gradients via `mlx_rs::transforms::value_and_grad`.

use mlx_rs::Array;
use mlx_rs::error::Result;
use mlx_rs::transforms::value_and_grad_with_argnums;

/// Evaluate `loss_fn(params)` and its gradient w.r.t. every param,
/// entirely on-device. `loss_fn` returns a single scalar `Array`
/// wrapped in a `Vec` (mlx's transform contract). Returns the scalar
/// loss as `f32` plus one gradient `Array` per param (same order).
pub fn loss_and_grads<F>(params: &[Array], loss_fn: F) -> Result<(f32, Vec<Array>)>
where
    F: Fn(&[Array]) -> Result<Vec<Array>>,
{
    let argnums: Vec<i32> = (0..params.len() as i32).collect();
    let mut value_and_grad = value_and_grad_with_argnums(loss_fn, argnums.as_slice());
    let (values, grads) = value_and_grad(params)?;
    Ok((values[0].item::<f32>(), grads))
}
