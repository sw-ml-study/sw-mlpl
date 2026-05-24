//! Thin wrappers around `mlpl-models-mutate`'s
//! `clone_model_inner` + `perturb_params_inner`. Each entry
//! point threads the eval-loop resolver closure so eval.rs
//! call sites stay signature-compatible. Bundles the clone +
//! perturb wrappers in one file to avoid growing mlpl-eval's
//! module count per extracted sub-crate.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::value::Value;
use mlpl_eval_core::model::ModelSpec;

/// `clone_model(m)` entry point.
pub(crate) fn eval_clone_model(
    args: &[Expr],
    env: &mut Environment,
) -> Result<ModelSpec, EvalError> {
    mlpl_models_mutate::clone_model_inner(args, env, clone_resolver)
}

/// `perturb_params(m, family, sigma, seed)` entry point.
pub(crate) fn eval_perturb_params(
    args: &[Expr],
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    mlpl_models_mutate::perturb_params_inner(args, env, scalar_resolver)
}

fn clone_resolver(expr: &Expr, env: &mut Environment) -> Result<ModelSpec, EvalError> {
    match crate::eval::eval_expr(expr, env, &mut None)? {
        Value::Model(m) => Ok(m),
        _ => Err(EvalError::Unsupported(
            "clone_model: argument must evaluate to a model".into(),
        )),
    }
}

fn scalar_resolver(expr: &Expr, env: &mut Environment) -> Result<f64, EvalError> {
    let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(
            "perturb_params: sigma and seed must be scalars".into(),
        ));
    }
    Ok(arr.data()[0])
}
