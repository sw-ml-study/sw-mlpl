//! `atop(f, g, x...)` = `f(g(x...))` and `over(f, g, x, y)` =
//! `f(g(x), g(y))` -- BQN's composition modifiers as
//! immediate-application builtins over function references.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn eval_compose(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let min = if name == "over" { 4 } else { 3 };
    if args.len() < min || (name == "over" && args.len() != 4) {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: min,
            got: args.len(),
        });
    }
    let f = reference(name, &args[0], env, trace)?;
    let g = reference(name, &args[1], env, trace)?;
    let data = args[2..]
        .iter()
        .map(|a| crate::eval::eval_expr(a, env, trace))
        .collect::<Result<Vec<_>, _>>()?;
    let inner = if name == "over" {
        vec![
            apply_ref(&g, &data[0..1], env, trace)?,
            apply_ref(&g, &data[1..2], env, trace)?,
        ]
    } else {
        vec![apply_ref(&g, &data, env, trace)?]
    };
    apply_ref(&f, &inner, env, trace)
}

fn reference(
    who: &str,
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let v = crate::eval::eval_expr(arg, env, trace)?;
    if matches!(
        v,
        Value::UserFnRef { .. } | Value::BuiltinRef { .. } | Value::Partial { .. }
    ) {
        Ok(v)
    } else {
        Err(EvalError::Unsupported(format!(
            "{who}: the first two arguments must be function references -- got {}",
            mlpl_eval_types::value_kind(&v)
        )))
    }
}

/// Apply a reference to already-evaluated values (arrays go to
/// builtins directly; anything else routes through the user-fn
/// invoke path).
fn apply_ref(
    f: &Value,
    vals: &[Value],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    crate::callable_apply::apply_callable(f, vals, env, trace)
}
