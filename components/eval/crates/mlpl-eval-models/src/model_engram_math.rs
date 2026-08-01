//! Pure math helpers for the Engram forward pass (split from
//! `model_apply_engram.rs`, saga E3 step 1): parameter lookup, the
//! flattened-table gather, and the value/gate projections that run
//! through the device-aware dispatch hook.

use mlpl_array::{DenseArray, Shape};
use mlpl_engram_core::{HashSpec, head_offset};

use crate::env_api::EnvVars;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// Look up a parameter array by name.
pub(crate) fn param<'e>(env: &'e Environment, name: &str) -> Result<&'e DenseArray, EvalError> {
    env.get(name)
        .ok_or_else(|| EvalError::UndefinedVariable(name.to_string()))
}

/// Gather the addressed rows into `[T, orders * heads * head_dim]`.
pub(crate) fn gather_retrieved(
    hashes: &[Vec<Vec<u64>>],
    spec: &HashSpec,
    table: &DenseArray,
    dims: (usize, usize),
) -> Result<DenseArray, EvalError> {
    let (t_len, head_dim) = dims;
    let width = spec.ngram_orders.len() * spec.heads_per_ngram * head_dim;
    let mut out = Vec::with_capacity(t_len * width);
    for t_hashes in hashes {
        for (oi, order_hashes) in t_hashes.iter().enumerate() {
            for (head, &local) in order_hashes.iter().enumerate() {
                let row = (head_offset(spec, oi, head) + local) as usize;
                out.extend_from_slice(&table.data()[row * head_dim..(row + 1) * head_dim]);
            }
        }
    }
    Ok(DenseArray::new(Shape::new(vec![t_len, width]), out)?)
}

/// `[X @ W + broadcast(b)]` through the device-aware dispatch hook,
/// following the model_apply_simple linear pattern.
fn linear_via_dispatch(
    env: &Environment,
    x: &DenseArray,
    w: &DenseArray,
    b: &DenseArray,
) -> Result<DenseArray, EvalError> {
    let dispatch = mlpl_eval_env::dispatch_hook::dispatch_or_err;
    let xw = dispatch(env, "matmul", vec![x.clone(), w.clone()])?;
    let n = xw.shape().dims()[0];
    let ones = DenseArray::new(Shape::new(vec![n, 1]), vec![1.0; n])?;
    let b_broadcast = dispatch(env, "matmul", vec![ones, b.clone()])?;
    dispatch(env, "add", vec![xw, b_broadcast])
}

/// `v = r @ W_v + b_v`; `g = sigmoid([h|v] @ W_g + b_g)`;
/// `out = h + g * v` -- projections and elementwise ops through the
/// device-aware dispatch hook (the concat is a plain row splice).
/// Returns `(out, g)`; the gate rides along for `engram_stats`.
pub(crate) fn project_and_gate(
    h: &DenseArray,
    retrieved: &DenseArray,
    env: &Environment,
    names: (&str, &str, &str, &str),
) -> Result<(DenseArray, DenseArray), EvalError> {
    let (w_value, b_value, w_gate, b_gate) = names;
    let dispatch = mlpl_eval_env::dispatch_hook::dispatch_or_err;
    let v = linear_via_dispatch(env, retrieved, param(env, w_value)?, param(env, b_value)?)?;
    let (t_len, hidden) = (h.shape().dims()[0], h.shape().dims()[1]);
    let mut hv = Vec::with_capacity(t_len * 2 * hidden);
    for t in 0..t_len {
        hv.extend_from_slice(&h.data()[t * hidden..(t + 1) * hidden]);
        hv.extend_from_slice(&v.data()[t * hidden..(t + 1) * hidden]);
    }
    let hv = DenseArray::new(Shape::new(vec![t_len, 2 * hidden]), hv)?;
    let pre = linear_via_dispatch(env, &hv, param(env, w_gate)?, param(env, b_gate)?)?;
    let g = mlpl_runtime::call_builtin("sigmoid", vec![pre])?;
    let gv = dispatch(env, "mul", vec![g.clone(), v])?;
    let out = dispatch(env, "add", vec![h.clone(), gv])?;
    Ok((out, g))
}
