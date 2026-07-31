//! Initial Engram parameter arrays (split from
//! `model_eval_engram.rs`, saga E3 step 2): the concrete
//! near-identity values -- zero memory, 0.01-scaled randn
//! projections, zero value bias, -2 gate bias.

use mlpl_array::{DenseArray, Shape};

use crate::model_engram_init::EngramDims;
use mlpl_eval_types::EvalError;

/// The five initial parameter arrays with their table roles.
pub(crate) fn engram_param_values(
    dims: &EngramDims,
    seed: f64,
) -> Result<[(usize, DenseArray, &'static str); 5], EvalError> {
    let &EngramDims {
        rows,
        head_dim,
        retrieved,
        hidden,
    } = dims;
    Ok([
        (
            0,
            DenseArray::zeros(Shape::new(vec![rows, head_dim])),
            "memory",
        ),
        (1, scaled_randn(seed + 1.0, retrieved, hidden)?, "W_v"),
        (2, DenseArray::zeros(Shape::new(vec![1, hidden])), "b_v"),
        (3, scaled_randn(seed + 2.0, 2 * hidden, hidden)?, "W_g"),
        (
            4,
            DenseArray::new(Shape::new(vec![1, hidden]), vec![-2.0; hidden])?,
            "b_g",
        ),
    ])
}

/// A `[r, c]` randn draw scaled by 0.01 (the near-identity scale).
fn scaled_randn(seed: f64, r: usize, c: usize) -> Result<DenseArray, EvalError> {
    let init = mlpl_runtime::call_builtin(
        "randn",
        vec![
            DenseArray::from_scalar(seed),
            DenseArray::new(Shape::new(vec![2]), vec![r as f64, c as f64])?,
        ],
    )?;
    let data: Vec<f64> = init.data().iter().map(|v| v * 0.01).collect();
    Ok(DenseArray::new(Shape::new(vec![r, c]), data)?)
}
