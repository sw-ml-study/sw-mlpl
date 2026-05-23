//! Saga 15 step 001: `freeze(m)` / `unfreeze(m)` builtins.
//!
//! Mark every parameter of a model as "frozen" so `adam` and
//! `momentum_sgd` skip the optimizer update for those names.
//! Gradients still flow through frozen parameters (the chain
//! rule does not care about freeze state); only the parameter
//! update is suppressed. Returns a scalar-zero unit, mirroring
//! `to_device` / `perturb_params` for in-place model ops.
//!
//! Saga 33 step 010: rewritten over the capability traits
//! `HasModels + HasFrozen`. The bare-ident model lookup uses
//! `HasModels::get_model` directly; the expression fallback
//! is injected via a closure so the trait body doesn't need
//! to know about `crate::eval::eval_expr`. This is the C+D
//! pattern in action -- the body is generic over `E`, and
//! the caller threads in the eval-loop dependency.
//!
//! See `contracts/eval-contract/freeze.md`.

use mlpl_array::DenseArray;
use mlpl_env_traits::{HasFrozen, HasModels};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::value::Value;
use mlpl_eval_core::model::ModelSpec;

/// `freeze(m)` -- mark every parameter of `m` as frozen.
pub(crate) fn eval_freeze(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    freeze_inner(args, env, eval_expr_resolver, "freeze")
}

/// `unfreeze(m)` -- remove every parameter of `m` from the
/// frozen set. Inverse of `freeze`; idempotent.
pub(crate) fn eval_unfreeze(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    unfreeze_inner(args, env, eval_expr_resolver, "unfreeze")
}

/// Generic over `E`: works against any type implementing the
/// `HasModels + HasFrozen` capability traits. The `resolve`
/// closure is the eval-loop dependency, injected so the body
/// doesn't import `crate::eval`.
pub(crate) fn freeze_inner<E, F>(
    args: &[Expr],
    env: &mut E,
    resolve: F,
    func: &str,
) -> Result<DenseArray, EvalError>
where
    E: HasModels + HasFrozen,
    F: FnOnce(&Expr, &mut E) -> Result<ModelSpec, EvalError>,
{
    let spec = resolve_model(args, env, resolve, func)?;
    for name in spec.params() {
        env.mark_frozen(&name);
    }
    Ok(DenseArray::from_scalar(0.0))
}

pub(crate) fn unfreeze_inner<E, F>(
    args: &[Expr],
    env: &mut E,
    resolve: F,
    func: &str,
) -> Result<DenseArray, EvalError>
where
    E: HasModels + HasFrozen,
    F: FnOnce(&Expr, &mut E) -> Result<ModelSpec, EvalError>,
{
    let spec = resolve_model(args, env, resolve, func)?;
    for name in spec.params() {
        env.unmark_frozen(&name);
    }
    Ok(DenseArray::from_scalar(0.0))
}

/// Validate arity and coerce the argument to a `ModelSpec`.
/// Bare-ident path uses `HasModels::get_model` directly; any
/// other expression flows through the caller-supplied
/// `resolve` closure.
fn resolve_model<E, F>(
    args: &[Expr],
    env: &mut E,
    resolve: F,
    func: &str,
) -> Result<ModelSpec, EvalError>
where
    E: HasModels,
    F: FnOnce(&Expr, &mut E) -> Result<ModelSpec, EvalError>,
{
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: func.into(),
            expected: 1,
            got: args.len(),
        });
    }
    if let Expr::Ident(name, _) = &args[0] {
        return env
            .get_model(name)
            .cloned()
            .ok_or_else(|| EvalError::Unsupported(format!("{func}: '{name}' is not a model")));
    }
    resolve(&args[0], env)
}

/// Default resolver used by the in-mlpl-eval call sites: runs
/// `crate::eval::eval_expr` and unwraps `Value::Model`.
fn eval_expr_resolver(expr: &Expr, env: &mut Environment) -> Result<ModelSpec, EvalError> {
    match crate::eval::eval_expr(expr, env, &mut None)? {
        Value::Model(m) => Ok(m),
        _ => Err(EvalError::Unsupported(
            "freeze/unfreeze: argument must evaluate to a model".into(),
        )),
    }
}
