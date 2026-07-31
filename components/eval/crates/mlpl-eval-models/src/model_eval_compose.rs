//! Saga 33 step 004: composite model constructors (`chain`,
//! `residual`) and the `activation_kind` lookup that maps
//! `tanh_layer` / `relu_layer` / `softmax_layer` to an
//! `ActKind` discriminant.

use mlpl_parser::Expr;

use mlpl_eval_core::model::{ActKind, ModelSpec};
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

/// `chain(layer_a, layer_b, ...)`. Each argument is either the name
/// of a model already bound in the environment (saga E3 step 1: lets
/// a named engram sit inside a chain) or an expression evaluating to
/// a `Value::Model`.
pub fn eval_chain(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    let mut children = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        children.push(chain_child(i, arg, env)?);
    }
    Ok(ModelSpec::Chain(children))
}

/// Resolve one `chain` argument to a `ModelSpec`.
fn chain_child(i: usize, arg: &Expr, env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if let Expr::Ident(name, _) = arg
        && let Some(m) = crate::env_api::EnvModels::get_model(env, name)
    {
        return Ok(m.clone());
    }
    match mlpl_eval_env::dispatch_hook::eval_or_err(arg, env, &mut None)? {
        Value::Model(m) => Ok(m),
        _ => Err(EvalError::Unsupported(format!(
            "chain: argument {i} did not evaluate to a model"
        ))),
    }
}

/// `residual(inner_model)`. Wraps a single model argument in a
/// skip-connection node.
pub fn eval_residual(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "residual".into(),
            expected: 1,
            got: args.len(),
        });
    }
    match mlpl_eval_env::dispatch_hook::eval_or_err(&args[0], env, &mut None)? {
        Value::Model(m) => Ok(ModelSpec::Residual(Box::new(m))),
        _ => Err(EvalError::Unsupported(
            "residual: argument must evaluate to a model".into(),
        )),
    }
}

/// Parameter-free activation layer constructors. Returns the
/// matching `ActKind` if `name` is recognized.
#[must_use]
pub fn activation_kind(name: &str) -> Option<ActKind> {
    Some(match name {
        "tanh_layer" => ActKind::Tanh,
        "relu_layer" => ActKind::Relu,
        "softmax_layer" => ActKind::Softmax,
        _ => return None,
    })
}
