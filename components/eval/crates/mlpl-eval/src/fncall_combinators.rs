//! The Result combinators (docs/monads.md rider): `map_ok`,
//! `and_then`, `or_else` -- the error monad gains composition via
//! function references. Function-first argument order matches
//! `reduce(:op, x)`.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value, value_kind};

/// Dispatch one combinator: `name` is `map_ok` / `and_then` /
/// `or_else`, `args` is `(f, r)`.
pub(crate) fn eval_combinator(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    let f = crate::eval::eval_expr(&args[0], env, trace)?;
    let r = crate::eval::eval_expr(&args[1], env, trace)?;
    let Value::Result { ok, payload } = r else {
        return Err(EvalError::Unsupported(format!(
            "{name}: second argument must be a Result (ok/err) -- got {}",
            value_kind(&r)
        )));
    };
    match (name, ok) {
        ("map_ok", true) => Ok(Value::Result {
            ok: true,
            payload: Box::new(apply_ref(name, &f, *payload, env, trace)?),
        }),
        ("and_then", true) => apply_ref(name, &f, *payload, env, trace),
        ("or_else", false) => apply_ref(name, &f, *payload, env, trace),
        _ => Ok(Value::Result { ok, payload }),
    }
}

/// Apply a function reference to one already-evaluated value.
/// User references run the u: path (arity errors name the
/// function); builtin references route through the runtime and
/// need an array payload.
fn apply_ref(
    who: &str,
    f: &Value,
    payload: Value,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    match f {
        Value::UserFnRef { .. } | Value::Partial { .. } => {
            crate::callable_apply::apply_callable(f, &[payload], env, trace)
        }
        Value::BuiltinRef { name } => apply_builtin_ref(who, name, payload),
        other => Err(EvalError::Unsupported(format!(
            "{who}: first argument must be a function reference (`:u:name` or `:name`) \
             -- got {}",
            value_kind(other)
        ))),
    }
}

/// A builtin reference composes over ARRAY payloads only.
fn apply_builtin_ref(who: &str, name: &str, payload: Value) -> Result<Value, EvalError> {
    match payload {
        Value::Array(a) => Ok(Value::Array(mlpl_runtime::call_builtin(name, vec![a])?)),
        other => Err(EvalError::Unsupported(format!(
            "{who}: builtin reference `:{name}` needs an array payload -- got {} \
             (use a `u:` function for non-array payloads)",
            value_kind(&other)
        ))),
    }
}
