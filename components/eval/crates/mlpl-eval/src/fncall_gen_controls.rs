//! Generation-state controls (docs/kv-cache-design.md):
//! `gen_clone` (independent copy), `gen_reset` (drop cached
//! rows back to the prompt), `gen_stats` (cache accounting).
//! Sibling of `fncall_gen` -- kept separate because that module
//! is at the function-count ceiling.

use std::collections::BTreeMap;

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_gen::{state_ident, unknown_state};
use mlpl_eval_core::{AttnKv, GenState, attention_dims};
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "gen_clone" => Some(eval_gen_clone(args, env, trace)),
        "gen_reset" => Some(eval_gen_reset(args, env, trace)),
        "gen_stats" => Some(eval_gen_stats(args, env, trace)),
        _ => None,
    }
}

/// `gen_clone(gs)` -- an independent copy, returned as a value
/// to bind (`gs2 = gen_clone(gs)`); the two states then diverge.
fn eval_gen_clone(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let gs = named_state("gen_clone", args, env, trace)?;
    Ok(Value::GenState(Box::new(gs)))
}

/// `gen_reset(gs)` -- drop every cached K/V row and re-run the
/// prompt, returning the state to just-after-`gen_state`.
/// Returns the token count (the prompt length).
fn eval_gen_reset(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let name = one_ident("gen_reset", args, env, trace)?;
    let mut gs = env
        .gen_states
        .remove(&name)
        .ok_or_else(|| unknown_state(&name))?;
    let outcome = rebuild(&mut gs, env);
    let tokens = gs.tokens;
    env.gen_states.insert(name, gs);
    outcome?;
    #[allow(clippy::cast_precision_loss)]
    Ok(Value::Array(DenseArray::from_scalar(tokens as f64)))
}

/// `gen_stats(gs)` -- `{tokens, layers, kv_rows, kv_values}`.
fn eval_gen_stats(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let gs = named_state("gen_stats", args, env, trace)?;
    let kv_rows: usize = gs.caches.iter().map(|c| c.rows).sum();
    let kv_values: usize = gs.caches.iter().map(|c| c.k.len() + c.v.len()).sum();
    #[allow(clippy::cast_precision_loss)]
    let scalar = |n: usize| Value::Array(DenseArray::from_scalar(n as f64));
    let fields = BTreeMap::from([
        ("tokens".to_string(), scalar(gs.tokens)),
        ("layers".to_string(), scalar(gs.caches.len())),
        ("kv_rows".to_string(), scalar(kv_rows)),
        ("kv_values".to_string(), scalar(kv_values)),
    ]);
    Ok(Value::Record { fields })
}

/// Empty the caches and re-feed the prompt (shared by reset).
fn rebuild(gs: &mut GenState, env: &Environment) -> Result<(), EvalError> {
    gs.caches = attention_dims(&gs.model)
        .into_iter()
        .map(|d_model| AttnKv {
            d_model,
            rows: 0,
            k: Vec::new(),
            v: Vec::new(),
        })
        .collect();
    gs.tokens = 0;
    gs.logits.clear();
    let prompt = gs.prompt.clone();
    for id in prompt {
        mlpl_eval_models::model_apply_cached::feed_state_row(gs, id, env)?;
    }
    Ok(())
}

/// The single bound-name argument.
fn one_ident(
    who: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<String, EvalError> {
    let [arg] = args else {
        return Err(EvalError::BadArity {
            func: who.into(),
            expected: 1,
            got: args.len(),
        });
    };
    let _ = (env, trace);
    state_ident(who, arg)
}

/// Read a bound generation state by name (a clone the caller owns).
fn named_state(
    who: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<GenState, EvalError> {
    let name = one_ident(who, args, env, trace)?;
    env.gen_states
        .get(&name)
        .cloned()
        .ok_or_else(|| unknown_state(&name))
}
