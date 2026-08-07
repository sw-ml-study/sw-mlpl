//! Applicative application over every callable kind (the heart
//! of docs/combinators-design.md): under-application forms a
//! Partial, exact arity executes, and excess arguments apply
//! LEFT-ASSOCIATIVELY to whatever came back -- so
//! `call(f, a, b, c)` is `(((f a) b) c)`.

use crate::env::Environment;
use crate::env_api::*;
use mlpl_eval_types::{EvalError, Value};
use mlpl_trace::Trace;

/// Apply a callable VALUE to already-evaluated arguments.
pub(crate) fn apply_callable(
    f: &Value,
    vals: &[Value],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    match f {
        Value::UserFnRef { name } => apply_user(name, &[], vals, env, trace),
        Value::Partial { name, bound, .. } => apply_user(name, bound, vals, env, trace),
        Value::BuiltinRef { name } => {
            if vals.is_empty() {
                return Err(EvalError::Unsupported(format!(
                    "call: builtin `:{name}` cannot be partially applied -- builtin arity \
                     is not a fixed fact. Wrap it: def u:f(x) {{ {name}(x) }}"
                )));
            }
            let arrs = vals
                .iter()
                .map(|v| v.clone().into_array())
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Array(mlpl_eval_env::dispatch_hook::dispatch_or_err(
                env, name, arrs,
            )?))
        }
        other => Err(EvalError::Unsupported(format!(
            "call: cannot apply a {} -- expected a function reference or partial",
            mlpl_eval_types::value_kind(other)
        ))),
    }
}

/// The applicative table for user functions: bound + new
/// arguments against the definition's arity.
fn apply_user(
    name: &str,
    bound: &[Value],
    new: &[Value],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let arity = env
        .get_fn(name)
        .ok_or_else(|| EvalError::Unsupported(format!("undefined function: {name}")))?
        .params
        .len();
    let mut total: Vec<Value> = bound.to_vec();
    total.extend_from_slice(new);
    if total.len() < arity {
        return Ok(Value::Partial {
            name: name.to_string(),
            arity,
            bound: total,
        });
    }
    let extra = total.split_off(arity);
    let out = crate::eval_user_fn::invoke_user_fn_values(name, &total, env, trace)?;
    if extra.is_empty() {
        Ok(out)
    } else {
        // Left-associative continuation onto the result.
        apply_callable(&out, &extra, env, trace)
    }
}
