//! `apply_engram` inside `grad()` / the optimizers (saga E2 step
//! 3): resolves the engram model and lowers the forward pass onto
//! the autograd tape via `mlpl_models_tape::engram_tape`, so
//! `train`/`adam` route gradients into the memory table (scatter-
//! ADD: only addressed rows move, duplicate addresses accumulate).

use crate::env_api::*;
use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::Shape;
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
    // Resolve ids through the traced scope so an index bound to a user-function
    // argument (not a literal or global) is seen and the gradient reaches the
    // memory table -- same scope-resolution fix as gather_rows / F23 (F24).
    let ids = crate::grad_const::eval_const_arg(&args[2], env, params)?;
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

/// `gather_rows(table, indices)` on the tape (finding F4): the native
/// `GatherRows` node copies the addressed rows and its backward is an EXACT
/// scatter-ADD into only those rows (duplicate indices accumulate) -- O(n x d),
/// no one-hot selection matrix -- so a from-scratch embedding / addressing
/// table trains at vocabulary scale. `indices` are concrete integers (not
/// differentiable); the output shape is `indices.shape + [row_dim]`.
pub(crate) fn call_gather_rows(
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    crate::grad::arity_check(args, 2, "gather_rows")?;
    let table = crate::grad::eval_tensor_expr(&args[0], env, tape, params)?;
    let idx = crate::grad_const::eval_const_arg(&args[1], env, params)?;
    let dims = table.value().shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(EvalError::Unsupported(format!(
            "gather_rows: table must be rank 2, got rank {}",
            dims.len()
        )));
    }
    let rows: Vec<usize> = idx
        .data()
        .iter()
        .map(|&v| gather_index(v, dims[0]))
        .collect::<Result<_, _>>()?;
    let mut out_dims = idx.shape().dims().to_vec();
    out_dims.push(dims[1]);
    Ok(table.gather_rows(rows).reshape(Shape::new(out_dims)))
}

/// Validate one gather index: a non-negative integer strictly below `rows`.
fn gather_index(v: f64, rows: usize) -> Result<usize, EvalError> {
    if v < 0.0 || v.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "gather_rows: index {v} is not a non-negative integer"
        )));
    }
    let c = v as usize;
    if c >= rows {
        return Err(EvalError::Unsupported(format!(
            "gather_rows: index {c} out of range for {rows} rows"
        )));
    }
    Ok(c)
}
