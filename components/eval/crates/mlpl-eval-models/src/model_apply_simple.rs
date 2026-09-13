//! Saga 33 step 004: per-variant `apply_*` helpers for the
//! non-recursive single-tensor `ModelSpec` variants. Each helper
//! reads its parameters from the environment, runs the forward
//! pass through `device::dispatched_call`, and returns the
//! output `DenseArray`.

use crate::env_api::EnvVars;
use mlpl_array::{DenseArray, Shape};

use crate::model_apply_embed::tokens_to_onehot;
use mlpl_eval_core::model::ActKind;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// `apply(Linear{w, b}, x)` = `x @ w + bias_broadcast(b)`.
pub fn apply_linear(
    x: &DenseArray,
    w: &str,
    b: &str,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let w_arr = env
        .get(w)
        .ok_or_else(|| EvalError::UndefinedVariable(w.into()))?;
    let b_arr = env
        .get(b)
        .ok_or_else(|| EvalError::UndefinedVariable(b.into()))?;
    let xw = mlpl_eval_env::dispatch_hook::dispatch_or_err(
        env,
        "matmul",
        vec![x.clone(), w_arr.clone()],
    )?;
    let n = xw.shape().dims()[0];
    let ones = DenseArray::new(Shape::new(vec![n, 1]), vec![1.0; n])?;
    let b_broadcast =
        mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "matmul", vec![ones, b_arr.clone()])?;
    mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "add", vec![xw, b_broadcast])
}

/// `apply(Activation(kind), x)` -- one of tanh / relu / softmax.
/// Softmax is dispatched with `axis = 1` (the trailing axis); the
/// other activations take only the input.
pub fn apply_activation(
    kind: ActKind,
    x: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let name = match kind {
        ActKind::Tanh => "tanh",
        ActKind::Relu => "relu",
        ActKind::Softmax => "softmax",
    };
    let args = if matches!(kind, ActKind::Softmax) {
        vec![x.clone(), DenseArray::from_scalar(1.0)]
    } else {
        vec![x.clone()]
    };
    mlpl_eval_env::dispatch_hook::dispatch_or_err(env, name, args)
}

/// `apply(Embedding{table, vocab, ..}, tokens)` -- gather rows
/// from `table` by lowering to a one-hot matmul.
pub fn apply_embedding(
    x: &DenseArray,
    table: &str,
    vocab: usize,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let t = env
        .get(table)
        .ok_or_else(|| EvalError::UndefinedVariable(table.into()))?;
    let onehot = tokens_to_onehot(x, vocab)?;
    let flat =
        mlpl_eval_env::dispatch_hook::dispatch_or_err(env, "matmul", vec![onehot, t.clone()])?;
    // The lookup is [N, d] over flattened tokens; restore the token shape so a
    // batched [B, T] input returns [B, T, d] (finding F9). Rank-1 is a no-op.
    let d = flat.shape().dims()[1];
    let mut out_dims = x.shape().dims().to_vec();
    out_dims.push(d);
    DenseArray::new(Shape::new(out_dims), flat.data().to_vec())
        .map_err(|e| EvalError::Unsupported(format!("embed: reshape to token shape failed: {e}")))
}
