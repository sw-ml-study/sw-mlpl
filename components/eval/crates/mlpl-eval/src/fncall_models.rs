//! FnCall dispatch family: model constructors, activations, and
//! the thin model-output forwarders that wrap an existing
//! `crate::model_*` helper into a `Value::Array` (or
//! `Value::DeviceTensor` for `to_device`).
//!
//! Saga 33 step 023 split this out of `eval::eval_expr` to retire
//! the eval.rs File-LOC FAIL. Each helper here returns
//! `Option<Result<Value, EvalError>>` (see `try_dispatch`) so the
//! caller can chain families with `Option::or_else`.

use mlpl_array::DenseArray;
use mlpl_eval_core::model::ModelSpec;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    if let Some(r) = try_model_ctor(name, args, env) {
        return Some(r.map(Value::Model));
    }
    if let Some(r) = try_activation(name, args) {
        return Some(r);
    }
    if name == "apply_engram" {
        return Some(crate::model_dispatch::eval_apply_engram(args, env, trace).map(Value::Array));
    }
    if name == "engram_stats" {
        return Some(crate::model_dispatch::eval_engram_stats(args, env, trace));
    }
    if name == "to_device" {
        return Some(crate::device::eval_to_device(args, env, trace));
    }
    if name == "param_count" {
        return Some(crate::model_inspect::eval_param_count(args, env).map(Value::Array));
    }
    try_array_forward(name, args, env, trace).map(|r| r.map(Value::Array))
}

fn try_model_ctor(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
) -> Option<Result<ModelSpec, EvalError>> {
    match name {
        "engram" => Some(crate::model_dispatch::eval_engram(args, env)),
        "linear" => Some(crate::model_dispatch::eval_linear(args, env)),
        "embed" => Some(crate::model_dispatch::eval_embedding(args, env)),
        "chain" => Some(crate::model_dispatch::eval_chain(args, env)),
        "residual" => Some(crate::model_dispatch::eval_residual(args, env)),
        "rms_norm" => Some(crate::model_dispatch::eval_rms_norm(args, env)),
        "attention" => Some(crate::model_dispatch::eval_attention(args, env, false)),
        "causal_attention" => Some(crate::model_dispatch::eval_attention(args, env, true)),
        "clone_model" => Some(crate::model_mutate::eval_clone_model(args, env)),
        "lora" => Some(crate::model_lora::eval_lora(args, env)),
        "save_model" => Some(crate::model_io::eval_save_model(args, env)),
        "load_model" => Some(crate::model_io::eval_load_model(args, env)),
        _ => None,
    }
}

fn try_activation(name: &str, args: &[Expr]) -> Option<Result<Value, EvalError>> {
    let kind = crate::model_dispatch::activation_kind(name)?;
    if !args.is_empty() {
        return Some(Err(EvalError::BadArity {
            func: name.into(),
            expected: 0,
            got: args.len(),
        }));
    }
    Some(Ok(Value::Model(ModelSpec::Activation(kind))))
}

fn try_array_forward(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<DenseArray, EvalError>> {
    let r = match name {
        "apply" => crate::model_dispatch::eval_apply(args, env, trace),
        "predict_batch" => crate::model_dispatch::eval_predict_batch(args, env, trace),
        "attention_weights" => crate::model_dispatch::eval_attention_weights(args, env, trace),
        "perturb_params" => crate::model_mutate::eval_perturb_params(args, env),
        "freeze" => crate::model_freeze::eval_freeze(args, env),
        "unfreeze" => crate::model_freeze::eval_unfreeze(args, env),
        "embed_table" => crate::model_inspect::eval_embed_table(args, env),
        "estimate_train" => crate::model_inspect::eval_estimate_train(args, env),
        "calibrate_device" => crate::model_feasibility::eval_calibrate_device(args, env),
        "estimate_hypothetical" => crate::model_feasibility::eval_estimate_hypothetical(args, env),
        "feasible" => crate::model_feasibility::eval_feasible(args, env),
        _ => return None,
    };
    Some(r)
}
