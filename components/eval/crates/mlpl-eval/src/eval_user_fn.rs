use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use crate::eval::eval_expr;
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
    let saved: Vec<(String, Option<DenseArray>)> = f
        .params
        .iter()
        .map(|p| (p.clone(), env.get(p).cloned()))
        .collect();
    for (param, val) in f.params.iter().zip(&evaluated) {
        match val {
            Value::Array(a) => env.set(param.clone(), a.clone()),
            _ => {
                return Err(EvalError::Unsupported(format!(
                    "{name}: argument '{param}' must be a numeric array"
                )));
            }
        }
    }
    let result = eval_body(f.body_exprs(), env, trace);
    for (param, old) in &saved {
        match old {
            Some(v) => env.set(param.clone(), v.clone()),
            None => env.remove_var(param),
        }
    }
    result
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
