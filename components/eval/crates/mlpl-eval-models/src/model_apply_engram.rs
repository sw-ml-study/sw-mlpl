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
//! The math helpers live in `model_engram_math` (saga E3 step 1
//! split).

use mlpl_array::DenseArray;
use mlpl_engram_core::{HashSpec, ngram_hashes};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env_api::EnvModels;
use crate::model_engram_math::{gather_retrieved, param, project_and_gate};
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
    let model = resolve_engram_model(&args[0], env)?;
    let h = mlpl_eval_env::dispatch_hook::eval_or_err(&args[1], env, trace)?.into_array()?;
    let ids = mlpl_eval_env::dispatch_hook::eval_or_err(&args[2], env, trace)?.into_array()?;
    engram_forward(&model, &h, &ids, env).map(|(out, _gate)| out)
}

/// The first `apply_engram` argument must be an identifier naming a
/// model bound in the environment.
pub(crate) fn resolve_engram_model(arg: &Expr, env: &Environment) -> Result<ModelSpec, EvalError> {
    let Expr::Ident(name, _) = arg else {
        return Err(EvalError::Unsupported(
            "apply_engram: first argument must be an engram model identifier".into(),
        ));
    };
    env.get_model(name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(name.clone()))
}

/// The pure forward pass against resolved inputs. Returns
/// `(out, gate)`; most callers keep only `out`, `engram_stats`
/// reads the gate.
pub(crate) fn engram_forward(
    model: &ModelSpec,
    h: &DenseArray,
    ids: &DenseArray,
    env: &Environment,
) -> Result<(DenseArray, DenseArray), EvalError> {
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
