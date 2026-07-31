//! `apply_engram` inside `grad()` / the optimizers (saga E2 step
//! 3): resolves the engram model and lowers the forward pass onto
//! the autograd tape via `mlpl_models_tape::engram_tape`, so
//! `train`/`adam` route gradients into the memory table (scatter-
//! ADD: only addressed rows move, duplicate addresses accumulate).

use crate::env_api::*;
use std::collections::HashMap;
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor};
use mlpl_engram_core::HashSpec;
use mlpl_models_tape::EngramInputs;
use mlpl_parser::Expr;

use crate::env::Environment;
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_types::EvalError;

pub(crate) fn call_apply_engram(
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    crate::grad::arity_check(args, 3, "apply_engram")?;
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
    let h = crate::grad::eval_tensor_expr(&args[1], env, tape, params)?;
    let ids = crate::eval::eval_expr(&args[2], env, &mut None)?.into_array()?;
    let inputs = EngramInputs {
        memory: &memory,
        w_value: &w_value,
        b_value: &b_value,
        w_gate: &w_gate,
        b_gate: &b_gate,
        hash: HashSpec {
            ngram_orders,
            heads_per_ngram: heads,
            slots_per_head: slots,
            seed,
        },
        head_dim,
        hidden,
    };
    mlpl_models_tape::engram_tape(&h, &ids, &inputs, tape, params).map_err(EvalError::from)
}
