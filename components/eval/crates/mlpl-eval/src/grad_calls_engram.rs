//! `apply_engram` inside `grad()` / the optimizers (saga E2 step
//! 3): resolves the engram model and lowers the forward pass onto
//! the autograd tape via `mlpl_models_tape::engram_tape`, so
//! `train`/`adam` route gradients into the memory table (scatter-
//! ADD: only addressed rows move, duplicate addresses accumulate).

use crate::env_api::*;
use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
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

/// `gather_rows(table, indices)` on the tape (finding F4): the addressed rows
/// are gathered by a one-hot selection matmul against the table, whose backward
/// is an EXACT scatter-ADD into the addressed rows (duplicate indices
/// accumulate), so a from-scratch embedding / addressing table trains -- the
/// same seam as the engram memory lookup. `indices` are concrete integers (not
/// differentiable); the output shape is `indices.shape + [row_dim]`.
pub(crate) fn call_gather_rows(
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    crate::grad::arity_check(args, 2, "gather_rows")?;
    let table = crate::grad::eval_tensor_expr(&args[0], env, tape, params)?;
    let idx = crate::eval::eval_expr(&args[1], env, &mut None)?.into_array()?;
    let dims = table.value().shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(EvalError::Unsupported(format!(
            "gather_rows: table must be rank 2, got rank {}",
            dims.len()
        )));
    }
    let sel = Tensor::leaf(Rc::clone(tape), selection_onehot(&idx, dims[0])?, false);
    let mut out_dims = idx.shape().dims().to_vec();
    out_dims.push(dims[1]);
    Ok(sel.matmul(&table).reshape(Shape::new(out_dims)))
}

/// Build a `[n, rows]` one-hot selection matrix from an integer index array
/// (any shape, flattened to `n`). `sel @ table` gathers those rows; the matmul
/// backward `sel^T @ upstream` is the scatter-ADD into the addressed rows.
fn selection_onehot(idx: &DenseArray, rows: usize) -> Result<DenseArray, EvalError> {
    let n = idx.shape().elem_count();
    let mut data = vec![0.0_f64; n * rows];
    for (r, &v) in idx.data().iter().enumerate() {
        let c = gather_index(v, rows)?;
        data[r * rows + c] = 1.0;
    }
    DenseArray::new(Shape::new(vec![n, rows]), data)
        .map_err(|e| EvalError::Unsupported(format!("gather_rows: selection build failed: {e}")))
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
