//! Saga 33 step 004: composite model constructors (`chain`,
//! `residual`) and the `activation_kind` lookup that maps
//! `tanh_layer` / `relu_layer` / `softmax_layer` to an
//! `ActKind` discriminant.

use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::value::Value;
use mlpl_eval_core::model::{ActKind, ModelSpec};

/// `chain(layer_a, layer_b, ...)`. Each argument must evaluate to a
/// `Value::Model`.
pub(crate) fn eval_chain(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    let mut children = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        match crate::eval::eval_expr(arg, env, &mut None)? {
            Value::Model(m) => children.push(m),
            _ => {
                return Err(EvalError::Unsupported(format!(
                    "chain: argument {i} did not evaluate to a model"
                )));
            }
        }
    }
    Ok(ModelSpec::Chain(children))
}

/// `residual(inner_model)`. Wraps a single model argument in a
/// skip-connection node.
pub(crate) fn eval_residual(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "residual".into(),
            expected: 1,
            got: args.len(),
        });
    }
    match crate::eval::eval_expr(&args[0], env, &mut None)? {
        Value::Model(m) => Ok(ModelSpec::Residual(Box::new(m))),
        _ => Err(EvalError::Unsupported(
            "residual: argument must evaluate to a model".into(),
        )),
    }
}

/// Parameter-free activation layer constructors. Returns the
/// matching `ActKind` if `name` is recognized.
#[must_use]
pub(crate) fn activation_kind(name: &str) -> Option<ActKind> {
    Some(match name {
        "tanh_layer" => ActKind::Tanh,
        "relu_layer" => ActKind::Relu,
        "softmax_layer" => ActKind::Softmax,
        _ => return None,
    })
}
