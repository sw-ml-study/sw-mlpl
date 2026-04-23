//! Saga 15 step 001: `freeze(m)` / `unfreeze(m)` builtins.
//!
//! Mark every parameter of a model as "frozen" so `adam` and
//! `momentum_sgd` skip the optimizer update for those names.
//! Gradients still flow through frozen parameters (the chain
//! rule does not care about freeze state); only the parameter
//! update is suppressed. Returns a scalar-zero unit, mirroring
//! `to_device` / `perturb_params` for in-place model ops.
//!
//! See `contracts/eval-contract/freeze.md`.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::model::ModelSpec;
use crate::value::Value;

/// `freeze(m)` -- mark every parameter of `m` as frozen.
pub(crate) fn eval_freeze(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    let spec = resolve_model(args, env, "freeze")?;
    for name in spec.params() {
        env.mark_frozen(&name);
    }
    Ok(DenseArray::from_scalar(0.0))
}

/// `unfreeze(m)` -- remove every parameter of `m` from the
/// frozen set. Inverse of `freeze`; idempotent.
pub(crate) fn eval_unfreeze(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    let spec = resolve_model(args, env, "unfreeze")?;
    for name in spec.params() {
        env.unmark_frozen(&name);
    }
    Ok(DenseArray::from_scalar(0.0))
}

/// Validate arity and coerce the argument to a `ModelSpec`.
/// Accepts either a bare model identifier (looked up in
/// `env.models`) or any expression that evaluates to
/// `Value::Model`, mirroring `clone_model` / `perturb_params`.
fn resolve_model(args: &[Expr], env: &mut Environment, func: &str) -> Result<ModelSpec, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: func.into(),
            expected: 1,
            got: args.len(),
        });
    }
    if let Expr::Ident(name, _) = &args[0] {
        match env.get_model(name) {
            Some(m) => Ok(m.clone()),
            None => Err(EvalError::Unsupported(format!(
                "{func}: '{name}' is not a model"
            ))),
        }
    } else {
        match crate::eval::eval_expr(&args[0], env, &mut None)? {
            Value::Model(m) => Ok(m),
            _ => Err(EvalError::Unsupported(format!(
                "{func}: argument must evaluate to a model"
            ))),
        }
    }
}
