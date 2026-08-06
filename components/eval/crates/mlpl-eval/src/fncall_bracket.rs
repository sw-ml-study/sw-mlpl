//! `bracket(setup, use, teardown)` -- guaranteed-finally over
//! three `:u:` references (docs/finally-design.md). Setup
//! failure skips use AND teardown; after a successful setup the
//! teardown always runs; use's failure stays primary.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn eval_bracket(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [setup, used, teardown] = args else {
        return Err(EvalError::BadArity {
            func: "bracket".into(),
            expected: 3,
            got: args.len(),
        });
    };
    let setup = hook_name("setup", setup, env, trace)?;
    let used = hook_name("use", used, env, trace)?;
    let teardown = hook_name("teardown", teardown, env, trace)?;
    let fixture = match crate::bracket_run::invoke(&setup, &[], env, trace)? {
        Value::Result { ok: false, payload } => {
            return Ok(Value::Result { ok: false, payload });
        }
        Value::Result { ok: true, payload } => *payload,
        plain => plain,
    };
    let primary = crate::bracket_run::invoke(&used, std::slice::from_ref(&fixture), env, trace)?;
    let cleanup = crate::bracket_run::invoke(&teardown, &[fixture], env, trace)?;
    Ok(crate::bracket_run::merge(primary, cleanup))
}

/// Each hook must be a `:u:` reference -- lifecycle hooks are
/// user code, so builtin references are rejected by design.
fn hook_name(
    role: &str,
    arg: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<String, EvalError> {
    match crate::eval::eval_expr(arg, env, trace)? {
        Value::UserFnRef { name } => Ok(name),
        other => Err(EvalError::Unsupported(format!(
            "bracket: {role} must be a `:u:name` user-function reference -- got {}",
            mlpl_eval_types::value_kind(&other)
        ))),
    }
}
