//! Tape-side dispatcher for the model DSL. `apply_model_tape`
//! walks a `ModelSpec` tree and emits the matching autograd
//! primitives on the tape, recursing through composite variants
//! (Chain / Residual) and delegating leaf variants to the
//! per-layer helpers in `linear`, `rms_norm`, `embedding`, and
//! `attention`.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor};

use crate::attention::{AttentionInputs, attention_tape};
use crate::error::TapeError;
use crate::layers::{embedding_tape, rms_norm_tape};
use crate::linear::{LinearLoraInputs, linear_lora_tape, linear_tape};
use mlpl_eval_core::model::{ActKind, ModelSpec};

/// Tape-side analogue of model_dispatch's `apply_model`. Walks
/// the model and emits the matching primitive ops on the
/// autograd tape.
pub fn apply_model_tape(
    model: &ModelSpec,
    x: Tensor,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, TapeError> {
    match model {
        ModelSpec::Linear { w, b } => linear_tape(&x, w, b, tape, params),
        ModelSpec::Chain(children) => {
            let mut cur = x;
            for child in children {
                cur = apply_model_tape(child, cur, tape, params)?;
            }
            Ok(cur)
        }
        ModelSpec::Activation(kind) => Ok(match kind {
            ActKind::Tanh => x.tanh(),
            ActKind::Relu => x.relu(),
            ActKind::Softmax => x.softmax(),
        }),
        ModelSpec::Residual(inner) => {
            let inner_out = apply_model_tape(inner, x.clone(), tape, params)?;
            Ok(x.add(&inner_out))
        }
        ModelSpec::RmsNorm { .. } => rms_norm_tape(&x, tape),
        ModelSpec::Embedding { table, vocab, .. } => {
            embedding_tape(&x, table, *vocab, tape, params)
        }
        ModelSpec::Attention { .. } => attention_arm(model, &x, tape, params),
        ModelSpec::LinearLora { .. } => lora_arm(model, &x, tape, params),
    }
}

fn attention_arm(
    model: &ModelSpec,
    x: &Tensor,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, TapeError> {
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
        unreachable!("attention_arm called on non-Attention ModelSpec")
    };
    attention_tape(
        x,
        &AttentionInputs {
            wq,
            wk,
            wv,
            wo,
            d_model: *d_model,
            heads: *heads,
            causal: *causal,
        },
        tape,
        params,
    )
}

fn lora_arm(
    model: &ModelSpec,
    x: &Tensor,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, TapeError> {
    let ModelSpec::LinearLora {
        w,
        b,
        a,
        b_adapter,
        rank,
        alpha,
        ..
    } = model
    else {
        unreachable!("lora_arm called on non-LinearLora ModelSpec")
    };
    linear_lora_tape(
        x,
        &LinearLoraInputs {
            w,
            b,
            a,
            b_adapter,
            rank: *rank,
            alpha: *alpha,
        },
        tape,
        params,
    )
}
