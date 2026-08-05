//! Thin wrappers around `mlpl-models-inspect`'s
//! `embed_table_inner` + `estimate_train_inner`. Bundles the
//! eval-loop resolver closures so eval.rs call sites stay
//! signature-compatible.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

pub fn eval_embed_table(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    mlpl_models_inspect::embed_table_inner(args, env, model_resolver)
}

pub fn eval_estimate_train(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    mlpl_models_inspect::estimate_train_inner(args, env, model_resolver, pos_scalar_resolver)
}

/// `param_count(m)`: total trainable parameters across the
/// model's param arrays -- the size axis of a quality-vs-size
/// frontier.
pub fn eval_param_count(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "param_count".into(),
            expected: 1,
            got: args.len(),
        });
    }
    // Bound models are looked up by NAME (bare model identifiers do
    // not evaluate to values); inline constructor expressions fall
    // back to the resolver.
    let spec = match &args[0] {
        Expr::Ident(n, _) if env.models.contains_key(n) => env.models[n].clone(),
        other => model_resolver(other, env)?,
    };
    let total: usize = spec
        .params()
        .iter()
        .map(|p| env.vars.get(p).map_or(0, |a| a.data().len()))
        .sum();
    #[allow(clippy::cast_precision_loss)]
    Ok(DenseArray::from_scalar(total as f64))
}

fn model_resolver(expr: &Expr, env: &mut Environment) -> Result<ModelSpec, EvalError> {
    match mlpl_eval_env::dispatch_hook::eval_or_err(expr, env, &mut None)? {
        Value::Model(m) => Ok(m),
        _ => Err(EvalError::Unsupported(
            "first argument must evaluate to a model".into(),
        )),
    }
}

fn pos_scalar_resolver(expr: &Expr, env: &mut Environment, name: &str) -> Result<f64, EvalError> {
    let arr = mlpl_eval_env::dispatch_hook::eval_or_err(expr, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "estimate_train: {name} must be a scalar, got rank {}",
            arr.rank()
        )));
    }
    let v = arr.data()[0];
    if !v.is_finite() || v <= 0.0 {
        return Err(EvalError::Unsupported(format!(
            "estimate_train: {name} must be positive, got {v}"
        )));
    }
    Ok(v)
}
