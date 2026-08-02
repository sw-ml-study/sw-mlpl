//! Saga 33 step 004: `apply_model` dispatcher extracted from
//! `model_dispatch.rs`. The per-variant arms are split out
//! into `model_apply_simple` (Linear / Activation / Embedding),
//! `model_apply_compose` (Chain / Residual / `RmsNorm`),
//! `model_apply_lora`, and `model_apply_attention`, keeping
//! `apply_model` under the 25-LOC sw-checklist gate.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env_api::EnvModels;
use crate::model_apply_attention::{AttentionArgs, apply_attention};
use crate::model_apply_compose::{apply_chain, apply_residual, apply_rms_norm};
use crate::model_apply_lora::{LinearLoraInputs, apply_linear_lora};
use crate::model_apply_simple::{apply_activation, apply_embedding, apply_linear};
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// Forward-pass dispatcher for the model DSL: maps each
/// `ModelSpec` variant to its per-variant helper. Saga 33
/// step 004 split the 100-LOC dispatcher into per-variant
/// helpers so this function fits the 25-LOC budget.
pub fn apply_model(
    model: &ModelSpec,
    x: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    match model {
        ModelSpec::Linear { w, b } => apply_linear(x, w, b, env),
        ModelSpec::Chain(children) => apply_chain(children, x, env),
        ModelSpec::Activation(kind) => apply_activation(*kind, x, env),
        ModelSpec::Residual(inner) => apply_residual(inner, x, env),
        ModelSpec::RmsNorm { .. } => apply_rms_norm(x),
        ModelSpec::Attention { .. } => apply_attention_spec(model, x, env),
        ModelSpec::Embedding { table, vocab, .. } => apply_embedding(x, table, *vocab, env),
        ModelSpec::LinearLora { .. } => apply_lora_spec(model, x, env),
        ModelSpec::Engram { .. } => Err(EvalError::Unsupported(
            "engram layers need token ids: place the engram inside a chain whose \
             input is the id vector, or call apply_engram(e, h, ids)"
                .into(),
        )),
    }
}

/// Unpack an `Attention` spec into [`AttentionArgs`] and forward.
fn apply_attention_spec(
    model: &ModelSpec,
    x: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let ModelSpec::Attention {
        wq,
        wk,
        wv,
        wo,
        d_model,
        heads,
        causal,
    } = model
    else {
        unreachable!("caller matched Attention");
    };
    let args = AttentionArgs {
        wq,
        wk,
        wv,
        wo,
        d_model: *d_model,
        heads: *heads,
        causal: *causal,
    };
    apply_attention(x, &args, env)
}

/// Unpack a `LinearLora` spec into [`LinearLoraInputs`] and forward.
fn apply_lora_spec(
    model: &ModelSpec,
    x: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    let inputs = LinearLoraInputs::from_spec(model).expect("caller matched LinearLora");
    apply_linear_lora(x, &inputs, env)
}

/// Shared builtin prologue: arity 2, model-identifier first arg,
/// model lookup, and the evaluated input tensor.
pub(crate) fn model_and_input(
    func: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(ModelSpec, DenseArray), EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: func.into(),
            expected: 2,
            got: args.len(),
        });
    }
    let Expr::Ident(name, _) = &args[0] else {
        return Err(EvalError::Unsupported(format!(
            "{func}: first argument must be a model identifier"
        )));
    };
    let model = env
        .get_model(name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(name.clone()))?;
    let x = mlpl_eval_env::dispatch_hook::eval_or_err(&args[1], env, trace)?.into_array()?;
    Ok((model, x))
}
