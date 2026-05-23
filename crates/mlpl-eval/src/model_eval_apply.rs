//! Saga 33 step 004: `apply` / `predict_batch` /
//! `attention_weights` dispatchers extracted from
//! `model_dispatch.rs`. All three resolve their first arg to a
//! `ModelSpec` by identifier lookup, then delegate forward work
//! to `model_apply::apply_model` or
//! `model_attn_weights::extract_attn_weights`.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::model_apply::apply_model;
use crate::model_attn_weights::extract_attn_weights;
use mlpl_eval_core::model::ModelSpec;

/// `apply(model_ident, X)`.
pub(crate) fn eval_apply(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "apply".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let model_name = match &args[0] {
        Expr::Ident(n, _) => n.clone(),
        _ => {
            return Err(EvalError::Unsupported(
                "apply: first argument must be a model identifier".into(),
            ));
        }
    };
    let model = env
        .get_model(&model_name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(model_name.clone()))?;
    let x = crate::eval::eval_expr(&args[1], env, trace)?.into_array()?;
    check_device_agreement(&model, &args[1], env)?;
    apply_model(&model, &x, env)
}

/// Saga 29 step 011: `predict_batch(model, X) -> Y`. Forward
/// through the model and return argmax over the trailing axis
/// as integer class labels.
pub(crate) fn eval_predict_batch(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "predict_batch".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let model_name = match &args[0] {
        Expr::Ident(n, _) => n.clone(),
        _ => {
            return Err(EvalError::Unsupported(
                "predict_batch: first argument must be a model identifier".into(),
            ));
        }
    };
    let model = env
        .get_model(&model_name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(model_name.clone()))?;
    let x = crate::eval::eval_expr(&args[1], env, trace)?.into_array()?;
    check_device_agreement(&model, &args[1], env)?;
    let logits = apply_model(&model, &x, env)?;
    let last_axis = logits.shape().dims().len().saturating_sub(1);
    Ok(logits.argmax_axis(last_axis)?)
}

/// `attention_weights(model_ident, X)` -- read-only forward pass
/// that returns the `[T, T]` (single-head) or `[heads, T, T]`
/// attention weight matrix from the first `Attention` layer
/// encountered in the model. Used for visualization.
pub(crate) fn eval_attention_weights(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "attention_weights".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let model_name = match &args[0] {
        Expr::Ident(n, _) => n.clone(),
        _ => {
            return Err(EvalError::Unsupported(
                "attention_weights: first argument must be a model identifier".into(),
            ));
        }
    };
    let model = env
        .get_model(&model_name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(model_name.clone()))?;
    let x = crate::eval::eval_expr(&args[1], env, trace)?.into_array()?;
    extract_attn_weights(&model, &x, env)
}

/// Saga 14 step 005: cross-check that the model's params and the
/// input tensor are on the same device. Raises
/// `EvalError::DeviceMismatch` with a clear message when the
/// user forgot a `to_device` call.
fn check_device_agreement(
    model: &ModelSpec,
    x_expr: &Expr,
    env: &Environment,
) -> Result<(), EvalError> {
    let x_device = match x_expr {
        Expr::Ident(name, _) => env.tensor_device(name).to_string(),
        _ => env.device().to_string(),
    };
    for p in model.params() {
        let p_device = env.tensor_device(&p).to_string();
        if p_device != x_device {
            return Err(EvalError::DeviceMismatch {
                op: "apply".into(),
                expected: p_device,
                actual: x_device,
            });
        }
    }
    Ok(())
}
