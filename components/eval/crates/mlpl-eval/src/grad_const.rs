//! Constant / non-differentiable builtin handling inside `grad`. Three related
//! findings share one mechanism -- evaluate eagerly, insert as a non-tracked
//! leaf on the tape: index/mask builtins (F5), constant constructors (F14), and
//! value-independent subexpressions such as shape-derived sizes (F12). Which
//! builtins and subtrees qualify is decided in `grad_purity`.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::DenseArray;
use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::env_api::*;
use crate::grad::eval_tensor_expr;
use mlpl_eval_types::EvalError;

/// Evaluate a non-differentiable argument eagerly -- a `gather_rows` index, a
/// `cross_entropy` target, `rotate`'s shift, `pow`'s exponent, a
/// `transpose_axes` permutation, a constant fold -- in a scope overlaid with
/// the current grad bindings, so a user function's parameters resolve
/// (findings F11, microgpt-mlpl): `start + range(count)` or targets `y` living
/// in the traced scope, not the global env. The overlay runs in an undo-log
/// frame, so it does not leak and costs O(overlaid names), not a copy of
/// every global.
pub(crate) fn eval_const_arg(
    idx: &Expr,
    env: &mut Environment,
    params: &HashMap<String, Tensor>,
) -> Result<DenseArray, EvalError> {
    env.frame_journal.push(Default::default());
    for (name, t) in params {
        env.set(name.clone(), t.value());
    }
    let out =
        crate::eval::eval_expr(idx, env, &mut None).and_then(mlpl_eval_types::Value::into_array);
    env.frame_exit();
    out
}

/// Dispatch a `FnCall` inside `grad`: run it on the tape, and if the tape
/// cannot handle it, fall back to constant-folding when the whole call is
/// value-independent of every parameter (finding F12 -- e.g. a shape-derived
/// size `reduce_mul(shape(...))`). A subtree that differentiably uses a param
/// is NOT folded, so a genuinely non-differentiable op over a param still
/// errors (no silent gradient drop). If the eager fold also fails, the tape
/// error stands.
pub(crate) fn fncall_or_fold(
    expr: &Expr,
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    match crate::grad::eval_tensor_fncall(name, args, env, tape, params) {
        Ok(t) => Ok(t),
        Err(e) => match fold_const_expr(expr, env, tape, params) {
            Some(v) => Ok(Tensor::leaf(Rc::clone(tape), v, false)),
            None => Err(e),
        },
    }
}

/// Fold a param-value-independent expression to a constant `DenseArray` by
/// evaluating it eagerly (finding F12). Returns `None` if the expression
/// differentiably uses a parameter (so it must stay on the tape) or if the
/// eager evaluation fails. Evaluated over the traced-scope overlay, so a
/// shape-derived size of a user function's parameter resolves.
pub(crate) fn fold_const_expr(
    expr: &Expr,
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Option<DenseArray> {
    if crate::grad_purity::differentiably_uses_param(expr, params, env, tape) {
        return None;
    }
    eval_const_arg(expr, env, params).ok()
}

/// Evaluate a stop-gradient builtin from the CURRENT forward values of its
/// arguments (so it tracks the params as they train) and insert the result as
/// a non-tracked constant leaf on the tape.
pub(crate) fn eval_stop_gradient(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    let vals = args
        .iter()
        .map(|a| eval_tensor_expr(a, env, tape, params).map(|t| t.value()))
        .collect::<Result<Vec<DenseArray>, _>>()?;
    let out = mlpl_runtime::call_builtin(name, vals)
        .map_err(|e| EvalError::Unsupported(format!("grad: {name}: {e}")))?;
    Ok(Tensor::leaf(Rc::clone(tape), out, false))
}
