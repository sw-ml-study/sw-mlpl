//! `apply_engram(e, h, ids)` -- the Engram forward pass (saga E2
//! step 2; decision D3's explicit two-input surface). Composes the
//! frozen addressing contract with the layer's parameters:
//!
//! ```text
//! local  = ngram_hashes(ids)            [T][order][head]
//! global = local + head_offset(o, h)
//! r[t]   = concat over (o, h) of memory[global]   [T, retrieved]
//! v      = r @ W_v + b_v                          [T, hidden]
//! g      = sigmoid([h | v] @ W_g + b_g)           [T, hidden]
//! out    = h + g * v
//! ```
//!
//! A freshly constructed engram (zero memory, zero b_v) yields
//! v == 0, so `out == h` EXACTLY -- the near-identity guarantee.

use mlpl_array::{DenseArray, Shape};
use mlpl_engram_core::{HashSpec, head_offset, ngram_hashes};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env_api::{EnvModels, EnvVars};
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// Evaluate `apply_engram(e, h, ids)`.
///
/// # Errors
/// Bad arity, a non-engram model, shape mismatches between `h`,
/// `ids`, and the spec, or hash-contract violations (fractional or
/// oversized ids).
pub fn eval_apply_engram(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    if args.len() != 3 {
        return Err(EvalError::BadArity {
            func: "apply_engram".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let model_name = match &args[0] {
        Expr::Ident(n, _) => n.clone(),
        _ => {
            return Err(EvalError::Unsupported(
                "apply_engram: first argument must be an engram model identifier".into(),
            ));
        }
    };
    let model = env
        .get_model(&model_name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(model_name.clone()))?;
    let h = mlpl_eval_env::dispatch_hook::eval_or_err(&args[1], env, trace)?.into_array()?;
    let ids = mlpl_eval_env::dispatch_hook::eval_or_err(&args[2], env, trace)?.into_array()?;
    engram_forward(&model, &h, &ids, env)
}

/// The pure forward pass against resolved inputs.
pub(crate) fn engram_forward(
    model: &ModelSpec,
    h: &DenseArray,
    ids: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let ModelSpec::Engram {
        memory,
        w_value,
        b_value,
        w_gate,
        b_gate,
        hidden,
        ngram_orders,
        heads,
        slots,
        head_dim,
        seed,
    } = model
    else {
        return Err(EvalError::Unsupported(
            "apply_engram: model is not an engram layer".into(),
        ));
    };
    let t_len = check_shapes(h, ids, *hidden)?;
    let id_vals: Vec<u64> = ids
        .data()
        .iter()
        .map(|&v| {
            if v < 0.0 || v.fract() != 0.0 {
                Err(EvalError::Unsupported(format!(
                    "apply_engram: ids must be non-negative integers, got {v}"
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect::<Result<_, _>>()?;
    let spec = HashSpec {
        ngram_orders: ngram_orders.clone(),
        heads_per_ngram: *heads,
        slots_per_head: *slots,
        seed: *seed,
    };
    let hashes =
        ngram_hashes(&id_vals, &spec).map_err(|e| EvalError::Unsupported(e.to_string()))?;
    let table = param(env, memory)?;
    let retrieved = gather_retrieved(&hashes, &spec, table, (t_len, *head_dim))?;
    project_and_gate(h, &retrieved, env, (w_value, b_value, w_gate, b_gate))
}

/// `h` must be `[T, hidden]`, `ids` rank-1 `[T]`.
fn check_shapes(h: &DenseArray, ids: &DenseArray, hidden: usize) -> Result<usize, EvalError> {
    if h.rank() != 2 || h.shape().dims()[1] != hidden {
        return Err(EvalError::Unsupported(format!(
            "apply_engram: hidden state must be [T, {hidden}], got shape {:?}",
            h.shape().dims()
        )));
    }
    if ids.rank() != 1 || ids.shape().dims()[0] != h.shape().dims()[0] {
        return Err(EvalError::Unsupported(format!(
            "apply_engram: ids must be rank-1 with the same T as the hidden state \
             ({}), got shape {:?}",
            h.shape().dims()[0],
            ids.shape().dims()
        )));
    }
    Ok(h.shape().dims()[0])
}

/// Look up a parameter array by name.
fn param<'e>(env: &'e Environment, name: &str) -> Result<&'e DenseArray, EvalError> {
    env.get(name)
        .ok_or_else(|| EvalError::UndefinedVariable(name.to_string()))
}

/// Gather the addressed rows into `[T, orders * heads * head_dim]`.
fn gather_retrieved(
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
fn project_and_gate(
    h: &DenseArray,
    retrieved: &DenseArray,
    env: &Environment,
    names: (&str, &str, &str, &str),
) -> Result<DenseArray, EvalError> {
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
    let gv = dispatch(env, "mul", vec![g, v])?;
    dispatch(env, "add", vec![h.clone(), gv])
}
