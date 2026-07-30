//! Thin wrapper around `mlpl-models-tune::lora_inner`. Threads
//! the eval-loop resolver + scalar closures so eval.rs call
//! sites stay signature-compatible.

use mlpl_parser::Expr;

use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

pub fn eval_lora(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    mlpl_models_tune::lora_inner(args, env, model_resolver, scalar_resolver)
}

fn model_resolver(expr: &Expr, env: &mut Environment) -> Result<ModelSpec, EvalError> {
    match mlpl_eval_env::dispatch_hook::eval_or_err(expr, env, &mut None)? {
        Value::Model(m) => Ok(m),
        _ => Err(EvalError::Unsupported(
            "lora: first argument must evaluate to a model".into(),
        )),
    }
}

fn scalar_resolver(expr: &Expr, env: &mut Environment) -> Result<f64, EvalError> {
    let arr = mlpl_eval_env::dispatch_hook::eval_or_err(expr, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(
            "lora: rank, alpha, and seed must be scalars".into(),
        ));
    }
    Ok(arr.data()[0])
}
