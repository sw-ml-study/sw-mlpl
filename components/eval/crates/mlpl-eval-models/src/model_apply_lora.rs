//! Saga 33 step 004: LoRA-adapter forward pass extracted from
//! `model_dispatch.rs`. Computes `x @ W + bias + (alpha/rank) *
//! (x @ A) @ B`, the rank-decomposed update from
//! `lora(linear(...), rank, alpha, seed)`.

use crate::env_api::EnvVars;
use mlpl_array::{DenseArray, Shape};

use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// Named-field inputs for one `apply_linear_lora` call so the
/// helper stays at 3 args (x, inputs, env) and does not trip
/// `clippy::too_many_arguments`.
pub struct LinearLoraInputs<'a> {
    pub w: &'a str,
    pub b: &'a str,
    pub a: &'a str,
    pub b_adapter: &'a str,
    pub rank: usize,
    pub alpha: f64,
}

pub fn apply_linear_lora(
    x: &DenseArray,
    inputs: &LinearLoraInputs<'_>,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let w_arr = env
        .get(inputs.w)
        .ok_or_else(|| EvalError::UndefinedVariable(inputs.w.into()))?;
    let b_arr = env
        .get(inputs.b)
        .ok_or_else(|| EvalError::UndefinedVariable(inputs.b.into()))?;
    let a_arr = env
        .get(inputs.a)
        .ok_or_else(|| EvalError::UndefinedVariable(inputs.a.into()))?;
    let b_adapt_arr = env
        .get(inputs.b_adapter)
        .ok_or_else(|| EvalError::UndefinedVariable(inputs.b_adapter.into()))?;
    let xw = mlpl_eval_env::dispatch_hook::dispatch_or_err(
        env,
        "matmul",
        vec![x.clone(), w_arr.clone()],
    )?;
    let xa = mlpl_eval_env::dispatch_hook::dispatch_or_err(
        env,
        "matmul",
        vec![x.clone(), a_arr.clone()],
    )?;
    let xab = mlpl_eval_env::dispatch_hook::dispatch_or_err(
        env,
        "matmul",
        vec![xa, b_adapt_arr.clone()],
    )?;
    let scale = inputs.alpha / inputs.rank as f64;
    let xab_scaled = mlpl_eval_env::dispatch_hook::dispatch_or_err(
        env,
        "mul",
        vec![xab, DenseArray::from_scalar(scale)],
    )?;
    let n = xw.shape().dims()[0];
    let ones = DenseArray::new(Shape::new(vec![n, 1]), vec![1.0; n])?;
    let b_broadcast =
        mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "matmul", vec![ones, b_arr.clone()])?;
    let sum_wx_and_adapter =
        mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "add", vec![xw, xab_scaled])?;
    mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "add", vec![sum_wx_and_adapter, b_broadcast])
}
