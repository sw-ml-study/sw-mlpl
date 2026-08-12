//! The `gen_*` generation-state family (docs/kv-cache-design.md):
//! `gen_state(model, prompt)` builds the KV cache, `gen_logits(gs)`
//! reads the next position's logits without recompute, and
//! `gen_append(gs, id)` feeds one accepted token. The loop stays
//! visible; only its body gets one complexity class faster.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_core::{AttnKv, GenState, attention_dims};
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "gen_state" => Some(eval_gen_state(args, env, trace)),
        "gen_logits" => Some(eval_gen_logits(args, env)),
        "gen_append" => Some(eval_gen_append(args, env, trace)),
        _ => None,
    }
}

/// `gen_state(model, prompt)` -- run the prompt once through the
/// incremental path, caching every attention layer's K/V rows.
fn eval_gen_state(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [model_arg, prompt_arg] = args else {
        return Err(bad_arity("gen_state", 2, args.len()));
    };
    let model = named_model(model_arg, env)?;
    let prompt = crate::eval::eval_expr(prompt_arg, env, trace)?.into_array()?;
    if prompt.shape().dims().len() > 1 {
        return Err(EvalError::Unsupported(
            "gen_state: prompt must be a token id vector (rank 0 or 1)".into(),
        ));
    }
    let caches = attention_dims(&model)
        .into_iter()
        .map(|d_model| AttnKv {
            d_model,
            rows: 0,
            k: Vec::new(),
            v: Vec::new(),
        })
        .collect();
    let mut gs = GenState {
        model,
        prompt: prompt.data().to_vec(),
        tokens: 0,
        caches,
        logits: Vec::new(),
    };
    for &id in prompt.data() {
        mlpl_eval_models::model_apply_cached::feed_state_row(&mut gs, id, env)?;
    }
    Ok(Value::GenState(Box::new(gs)))
}

/// `gen_logits(gs)` -- the pending next-position logits row (what
/// `last_row(apply(model, seq))` computes), read from the state.
fn eval_gen_logits(args: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    let [state_arg] = args else {
        return Err(bad_arity("gen_logits", 1, args.len()));
    };
    let name = state_ident("gen_logits", state_arg)?;
    let gs = env
        .gen_states
        .get(&name)
        .ok_or_else(|| unknown_state(&name))?;
    if gs.logits.is_empty() {
        return Err(EvalError::Unsupported(
            "gen_logits: the state has no cached positions yet -- \
             gen_state needs a non-empty prompt"
                .into(),
        ));
    }
    Ok(Value::Array(DenseArray::from_vec(gs.logits.clone())))
}

/// `gen_append(gs, ids)` -- feed accepted token id(s): project
/// each row, append to every attention layer's cache, refresh
/// the pending logits. A scalar feeds one token; a rank-1
/// vector feeds several in order (the batched verification
/// hook). Returns the state's new token count.
fn eval_gen_append(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [state_arg, id_arg] = args else {
        return Err(bad_arity("gen_append", 2, args.len()));
    };
    let ids = crate::eval::eval_expr(id_arg, env, trace)?.into_array()?;
    if ids.shape().dims().len() > 1 || ids.data().is_empty() {
        return Err(EvalError::Unsupported(
            "gen_append: the token id(s) must be a scalar or a non-empty rank-1 vector              (a vector is the batched verification hook)"
                .into(),
        ));
    }
    let name = state_ident("gen_append", state_arg)?;
    // Take the state out of the table so the forward pass can
    // read the environment's weights immutably; reinsert after.
    let mut gs = env
        .gen_states
        .remove(&name)
        .ok_or_else(|| unknown_state(&name))?;
    let mut fed = Ok(());
    for &id in ids.data() {
        fed = mlpl_eval_models::model_apply_cached::feed_state_row(&mut gs, id, env);
        if fed.is_err() {
            break;
        }
    }
    let tokens = gs.tokens;
    env.gen_states.insert(name, gs);
    fed?;
    #[allow(clippy::cast_precision_loss)]
    Ok(Value::Array(DenseArray::from_scalar(tokens as f64)))
}

fn bad_arity(func: &str, expected: usize, got: usize) -> EvalError {
    EvalError::BadArity {
        func: func.into(),
        expected,
        got,
    }
}

pub(crate) fn unknown_state(name: &str) -> EvalError {
    EvalError::Unsupported(format!(
        "unknown generation state `{name}` -- bind one first: gs = gen_state(model, prompt)"
    ))
}

pub(crate) fn state_ident(who: &str, arg: &Expr) -> Result<String, EvalError> {
    match arg {
        Expr::Ident(name, _) => Ok(name.clone()),
        _ => Err(EvalError::Unsupported(format!(
            "{who}: the state argument must be a bound gen_state name"
        ))),
    }
}

fn named_model(arg: &Expr, env: &Environment) -> Result<mlpl_eval_core::ModelSpec, EvalError> {
    let Expr::Ident(name, _) = arg else {
        return Err(EvalError::Unsupported(
            "gen_state: the model must be bound to a name (m = chain(...); gen_state(m, ...))"
                .into(),
        ));
    };
    env.models
        .get(name)
        .cloned()
        .ok_or_else(|| EvalError::Unsupported(format!("gen_state: unknown model `{name}`")))
}
