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

/// Bind params into the (already-snapshotted) scope and evaluate the
/// body. The caller restores the scope afterwards regardless of outcome.
fn run_body(
    f: &crate::env_user_fns::UserFn,
    name: &str,
    evaluated: &[Value],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    for (param, val) in f.params.iter().zip(evaluated) {
        match val {
            Value::Array(a) => env.set(param.clone(), a.clone()),
            _ => {
                return Err(EvalError::Unsupported(format!(
                    "{name}: argument '{param}' must be a numeric array"
                )));
            }
        }
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
