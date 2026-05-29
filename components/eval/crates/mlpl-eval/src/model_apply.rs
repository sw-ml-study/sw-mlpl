//! Saga 33 step 004: `apply_model` dispatcher extracted from
//! `model_dispatch.rs`. The per-variant arms are split out
//! into `model_apply_simple` (Linear / Activation / Embedding),
//! `model_apply_compose` (Chain / Residual / RmsNorm),
//! `model_apply_lora`, and `model_apply_attention`, keeping
//! `apply_model` under the 25-LOC sw-checklist gate.

use mlpl_array::DenseArray;

use crate::env::Environment;
use crate::model_apply_attention::{AttentionArgs, apply_attention};
use crate::model_apply_compose::{apply_chain, apply_residual, apply_rms_norm};
use crate::model_apply_lora::{LinearLoraInputs, apply_linear_lora};
use crate::model_apply_simple::{apply_activation, apply_embedding, apply_linear};
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_types::EvalError;

/// Forward-pass dispatcher for the model DSL: maps each
/// `ModelSpec` variant to its per-variant helper. Saga 33
/// step 004 split the 100-LOC dispatcher into per-variant
/// helpers so this function fits the 25-LOC budget.
pub(crate) fn apply_model(
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
        ModelSpec::Attention {
            wq,
            wk,
            wv,
            wo,
            d_model,
            heads,
            causal,
        } => apply_attention(
            x,
            &AttentionArgs {
                wq,
                wk,
                wv,
                wo,
                d_model: *d_model,
                heads: *heads,
                causal: *causal,
            },
            env,
        ),
        ModelSpec::Embedding { table, vocab, .. } => apply_embedding(x, table, *vocab, env),
        ModelSpec::LinearLora {
            w,
            b,
            a,
            b_adapter,
            rank,
            alpha,
            ..
        } => apply_linear_lora(
            x,
            &LinearLoraInputs {
                w,
                b,
                a,
                b_adapter,
                rank: *rank,
                alpha: *alpha,
            },
            env,
        ),
    }
}
