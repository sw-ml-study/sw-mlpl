//! Thin wrappers around `mlpl-models-feasibility`'s
//! `calibrate_device_inner`, `estimate_hypothetical_inner`,
//! and `feasible_inner`. Threads the eval-loop resolver
//! closures.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::value::Value;

pub(crate) fn eval_calibrate_device(
    args: &[Expr],
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    mlpl_models_feasibility::calibrate_device_inner(args, env, scalar_resolver)
}

pub(crate) fn eval_estimate_hypothetical(
    args: &[Expr],
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    mlpl_models_feasibility::estimate_hypothetical_inner(args, env, name_resolver, scalar_resolver)
}

pub(crate) fn eval_feasible(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    mlpl_models_feasibility::feasible_inner(args, env, array_resolver)
}

fn scalar_resolver(expr: &Expr, env: &mut Environment) -> Result<f64, EvalError> {
    let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported("expected a scalar argument".into()));
    }
    Ok(arr.data()[0])
}

fn name_resolver(expr: &Expr, env: &mut Environment) -> Result<String, EvalError> {
    match crate::eval::eval_expr(expr, env, &mut None)? {
        Value::Str(s) => Ok(s),
        _ => Err(EvalError::Unsupported(
            "estimate_hypothetical: first argument must be a model-name string".into(),
        )),
    }
}

fn array_resolver(expr: &Expr, env: &mut Environment) -> Result<DenseArray, EvalError> {
    crate::eval::eval_expr(expr, env, &mut None)?.into_array()
}
