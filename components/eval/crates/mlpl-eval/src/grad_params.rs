//! Parameter names for the optimizer steps: resolving the `params` argument
//! and writing trained values back so they persist across function frames.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::env_api::*;
use mlpl_eval_types::EvalError;

/// Write an optimizer-updated parameter value, persisting it across a
/// user-function frame (finding F21): the optimizer's effect on a named global
/// parameter must survive the frame restore, like an explicit global write, so
/// `adam(...)` inside a `def u:step()` trains the real params, not frame-local
/// copies.
pub(crate) fn set_trained_param(env: &mut Environment, name: &str, value: DenseArray) {
    if env.call_depth > 0 {
        env.global_writes.push((
            name.to_string(),
            mlpl_eval_types::Value::Array(value.clone()),
        ));
    }
    env.set(name.to_string(), value);
}

/// Resolve the optimizer's `params` argument into a flat list of
/// parameter identifiers. Accepts:
///
/// - a single param identifier: `adam(loss, W, ...)`
/// - an array literal of param identifiers: `adam(loss, [W, b], ...)`
/// - a model identifier registered via the Saga 11 model DSL:
///   `adam(loss, M, ...)` walks `ModelSpec::params()` and returns its
///   flat, order-stable parameter list.
pub(crate) fn collect_params(
    arg: &Expr,
    env: &Environment,
    func: &str,
) -> Result<Vec<String>, EvalError> {
    match arg {
        Expr::Ident(n, _) => {
            if let Some(model) = env.get_model(n) {
                Ok(model.params())
            } else {
                Ok(vec![n.clone()])
            }
        }
        Expr::ArrayLit(elems, _) => {
            let mut v = Vec::with_capacity(elems.len());
            for e in elems {
                match e {
                    // Saga 29 step 009: walk model params when the
                    // ArrayLit element resolves to a registered model,
                    // matching the lone-Ident path's behavior. This
                    // is what lets the trained ViT demo write
                    // `adam(loss, [linear_p, attn, classifier], ...)`
                    // and have every model's param list flattened in.
                    Expr::Ident(n, _) => {
                        if let Some(model) = env.get_model(n) {
                            v.extend(model.params());
                        } else {
                            v.push(n.clone());
                        }
                    }
                    _ => {
                        return Err(EvalError::Unsupported(format!(
                            "{func}: params list must contain only identifiers"
                        )));
                    }
                }
            }
            Ok(v)
        }
        _ => Err(EvalError::Unsupported(format!(
            "{func}: second argument must be a param identifier, model identifier, or list"
        ))),
    }
}
