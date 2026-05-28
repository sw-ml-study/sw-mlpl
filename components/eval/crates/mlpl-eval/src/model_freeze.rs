//! Thin wrappers around `mlpl-models-freeze`'s generic
//! `freeze_inner` / `unfreeze_inner`. The wrappers thread in
//! the eval-loop resolver so `eval.rs` call sites stay
//! signature-compatible with the rest of the dispatch table.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::value::Value;
use mlpl_eval_core::model::ModelSpec;

/// `freeze(m)` entry point: dispatches to the generic body in
/// `mlpl-models-freeze::freeze_inner`, passing the eval-loop
/// resolver closure for non-ident model args.
pub(crate) fn eval_freeze(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    mlpl_models_freeze::freeze_inner(args, env, eval_expr_resolver, "freeze")
}

/// `unfreeze(m)` entry point: mirror of `eval_freeze`.
pub(crate) fn eval_unfreeze(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    mlpl_models_freeze::unfreeze_inner(args, env, eval_expr_resolver, "unfreeze")
}

/// Eval-loop resolver: when the model arg is not a bare
/// identifier, evaluate it via `crate::eval::eval_expr` and
/// unwrap `Value::Model`. The closure is passed into the
/// generic `freeze_inner` / `unfreeze_inner` so they don't
/// need to import this crate.
fn eval_expr_resolver(expr: &Expr, env: &mut Environment) -> Result<ModelSpec, EvalError> {
    match crate::eval::eval_expr(expr, env, &mut None)? {
        Value::Model(m) => Ok(m),
        _ => Err(EvalError::Unsupported(
            "freeze/unfreeze: argument must evaluate to a model".into(),
        )),
    }
}
