//! In-chain Engram forward (saga E3 step 1): a `ModelSpec::Engram`
//! sitting inside a `Chain` reuses the chain's ORIGINAL input as its
//! token ids -- for embed-first chains that input IS the id vector
//! the embedding consumed -- while the running hidden state supplies
//! `h`. This is the "AfterAttention" insertion point from
//! docs/engram-sagas-plan.md without a new hook type: the chain
//! itself threads the ids.

use mlpl_array::DenseArray;

use crate::model_apply_engram::engram_forward;
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// Forward an in-chain engram: `h` is the running hidden state,
/// `chain_input` doubles as the token ids.
///
/// # Errors
/// Anything `apply_engram` rejects (non-id chain input, hidden-width
/// mismatch, hash-contract violations), tagged with the in-chain
/// context so the message explains where the ids came from.
pub(crate) fn apply_engram_in_chain(
    model: &ModelSpec,
    h: &DenseArray,
    chain_input: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    engram_forward(model, h, chain_input, env)
        .map(|(out, _gate)| out)
        .map_err(engram_chain_context)
}

/// Prefix `Unsupported` messages with the in-chain framing; other
/// error kinds pass through untouched.
fn engram_chain_context(err: EvalError) -> EvalError {
    match err {
        EvalError::Unsupported(msg) => EvalError::Unsupported(format!(
            "engram in chain (the chain input is used as token ids): {msg}"
        )),
        other => other,
    }
}
