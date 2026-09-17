//! Constant / non-differentiable builtin handling inside `grad`. Three related
//! findings share one mechanism -- evaluate eagerly, insert as a non-tracked
//! leaf on the tape: index/mask builtins (F5), constant constructors (F14), and
//! value-independent subexpressions such as shape-derived sizes (F12). Kept out
//! of `grad.rs` so that module stays under its file-length budget.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::DenseArray;
use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::env_api::*;
use crate::grad::eval_tensor_expr;
use mlpl_eval_types::EvalError;

/// Evaluate a non-differentiable index expression (e.g. a `gather_rows` index)
/// eagerly, in a scope overlaid with the current grad bindings, so index
/// arithmetic over a nested user function's arguments resolves (finding F11):
/// `start + range(count)` where `start`/`count` are function parameters living
/// in the traced scope, not the global env. The overlay is snapshotted and
/// restored so it does not leak.
pub(crate) fn eval_index_expr(
    idx: &Expr,
    env: &mut Environment,
    params: &HashMap<String, Tensor>,
) -> Result<DenseArray, EvalError> {
    let snap = env.snapshot_scope();
    for (name, t) in params {
        env.set(name.clone(), t.value());
    }
    let out =
        crate::eval::eval_expr(idx, env, &mut None).and_then(mlpl_eval_types::Value::into_array);
    env.restore_scope(snap);
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
        Err(e) => match fold_const_expr(expr, env, params) {
            Some(v) => Ok(Tensor::leaf(Rc::clone(tape), v, false)),
            None => Err(e),
        },
    }
}

/// Fold a param-value-independent expression to a constant `DenseArray` by
/// evaluating it eagerly (finding F12). Returns `None` if the expression
/// differentiably uses a parameter (so it must stay on the tape) or if the
/// eager evaluation fails.
pub(crate) fn fold_const_expr(
    expr: &Expr,
    env: &mut Environment,
    params: &HashMap<String, Tensor>,
) -> Option<DenseArray> {
    if differentiably_uses_param(expr, params) {
        return None;
    }
    crate::eval::eval_expr(expr, env, &mut None)
        .and_then(mlpl_eval_types::Value::into_array)
        .ok()
}

/// Whether evaluating `expr` depends on the VALUE of any tracked parameter.
/// A parameter that appears only inside shape-metadata builtins (`shape`,
/// `rank`, `len`, `labels`) does not count -- those read structure, not values
/// -- so a shape-derived size is value-independent and safe to constant-fold.
pub(crate) fn differentiably_uses_param(expr: &Expr, params: &HashMap<String, Tensor>) -> bool {
    match expr {
        Expr::Ident(name, _) => params.contains_key(name),
        Expr::FnCall { name, args, .. } => {
            !matches!(name.as_str(), "shape" | "rank" | "len" | "labels")
                && args.iter().any(|a| differentiably_uses_param(a, params))
        }
        Expr::BinOp { lhs, rhs, .. } => {
            differentiably_uses_param(lhs, params) || differentiably_uses_param(rhs, params)
        }
        Expr::UnaryNeg { operand, .. } => differentiably_uses_param(operand, params),
        Expr::ArrayLit(elems, _) => elems.iter().any(|e| differentiably_uses_param(e, params)),
        Expr::IntLit(..)
        | Expr::FloatLit(..)
        | Expr::StrLit(..)
        | Expr::BuiltinRef(..)
        | Expr::TensorCtor { .. } => false,
        // Unknown / scoped forms: assume they may use a parameter (never
        // fold something we cannot prove is constant).
        _ => true,
    }
}

/// Index / mask builtins that are non-differentiable by nature (they return
/// integer positions or `{0,1}` masks). Inside `grad` they are treated as
/// stop-gradient constants (finding F5), so a top-1 router mask -- e.g.
/// `one_hot(argmax(R, 1), E)` or a `gt`/`eq`/`lt` comparison -- can be computed
/// inside the loss while the gradient flows through the surrounding
/// differentiable ops, never through the mask.
pub(crate) fn is_stop_gradient_builtin(name: &str) -> bool {
    matches!(name, "argmax" | "one_hot" | "eq" | "gt" | "lt" | "argtop_k")
}

/// Constant constructors (finding F14): `fill`, `zeros`, `ones` build an array
/// from shape/value arguments and depend on no parameter, so inside `grad` they
/// are constant leaves on the tape (the same eager-then-leaf treatment as the
/// stop-gradient builtins) -- a loss may scale by a constant mask or add a
/// constant bias built inline.
pub(crate) fn is_const_ctor_builtin(name: &str) -> bool {
    matches!(name, "fill" | "zeros" | "ones")
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
