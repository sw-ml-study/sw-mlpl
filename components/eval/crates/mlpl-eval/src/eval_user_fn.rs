use crate::env_api::*;
use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

pub(crate) fn call_user_fn(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let f = env
        .get_fn(name)
        .ok_or_else(|| EvalError::Unsupported(format!("undefined function: {name}")))?
        .clone();
    if args.len() != f.params.len() {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: f.params.len(),
            got: args.len(),
        });
    }
    let evaluated: Vec<Value> = args
        .iter()
        .map(|a| eval_expr(a, env, trace))
        .collect::<Result<_, _>>()?;
    // Fresh local scope per call: snapshot the variable namespaces, run
    // the body, then restore -- so locals (and rebound params) do not
    // leak into the caller or sibling/recursive frames (issue #6 / C1).
    let snapshot = env.snapshot_scope();
    let result = run_body(&f, name, &evaluated, env, trace);
    env.restore_scope(snapshot);
    result
}

/// Invoke a user function with ALREADY-EVALUATED argument values
/// (the combinators' path: the payload is a Value, not an Expr).
/// Same arity/scope/restore semantics as `call_user_fn`.
pub(crate) fn invoke_user_fn_values(
    name: &str,
    values: &[Value],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let f = env
        .get_fn(name)
        .ok_or_else(|| EvalError::Unsupported(format!("undefined function: {name}")))?
        .clone();
    if values.len() != f.params.len() {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: f.params.len(),
            got: values.len(),
        });
    }
    let snapshot = env.snapshot_scope();
    let result = run_body(&f, name, values, env, trace);
    env.restore_scope(snapshot);
    result
}

/// Bind params into the (already-snapshotted) scope and evaluate the
/// body. The caller restores the scope afterwards regardless of outcome.
fn run_body(
    f: &mlpl_eval_state::UserFn,
    name: &str,
    evaluated: &[Value],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    for (param, val) in f.params.iter().zip(evaluated) {
        bind_arg(name, param, val, env)?;
    }
    eval_body(f.body_exprs(), env, trace)
}

fn eval_body(
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let mut last = Value::Array(DenseArray::from_scalar(0.0));
    for stmt in body {
        match eval_expr(stmt, env, trace) {
            Ok(v) => last = v,
            Err(EvalError::ReturnSignal(v)) => return Ok(*v),
            Err(e) => return Err(e),
        }
    }
    Ok(last)
}

/// Bind one evaluated argument into the callee scope. Arrays are
/// the historical case; spike step 011 added Results, strings,
/// and records so error-handling pipelines (`v = r?; ...`) work
/// on function arguments.
fn bind_arg(name: &str, param: &str, val: &Value, env: &mut Environment) -> Result<(), EvalError> {
    env.clear_binding(param);
    match val {
        Value::Array(a) => env.set(param.to_string(), a.clone()),
        Value::Result { ok, payload } => {
            env.set_result(param.to_string(), *ok, (**payload).clone());
        }
        Value::Str(s) => env.set_string(param.to_string(), s.clone()),
        Value::Record { fields } => env.set_record(param.to_string(), fields.clone()),
        Value::UserFnRef { name: t } | Value::BuiltinRef { name: t } => {
            env.set_builtin_ref(param.to_string(), t.clone());
        }
        _ => {
            return Err(EvalError::Unsupported(format!(
                "{name}: argument '{param}' must be an array, Result, string, record, \
                 or function reference"
            )));
        }
    }
    Ok(())
}
