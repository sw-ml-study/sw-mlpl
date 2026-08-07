//! Higher-order builtins over function references: `each(f, v)`
//! (APL2's f-each: elementwise application, shape preserved) and
//! `table(f, a, b)` (the outer product over `f` -- APL2's
//! jot-dot, BQN's table). Both apply IMMEDIATELY -- no function
//! values are produced.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    match name {
        "each" => Some(eval_each(args, env, trace)),
        "table" => Some(eval_table(args, env, trace)),
        "atop" | "over" => Some(crate::hof_compose::eval_compose(name, args, env, trace)),
        _ => None,
    }
}

fn eval_each(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let (f, rest) = fn_and_args("each", 2, args, env, trace)?;
    let v = rest[0].clone().into_array()?;
    let mut out = Vec::with_capacity(v.data().len());
    for (i, &x) in v.data().iter().enumerate() {
        out.push(apply_scalar("each", &f, &[x], i, env, trace)?);
    }
    Ok(Value::Array(DenseArray::new(
        Shape::new(v.shape().dims().to_vec()),
        out,
    )?))
}

fn eval_table(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let (f, rest) = fn_and_args("table", 3, args, env, trace)?;
    let (a, b) = (rest[0].clone().into_array()?, rest[1].clone().into_array()?);
    if a.rank() != 1 || b.rank() != 1 {
        return Err(EvalError::Unsupported(
            "table: both arguments must be rank-1 vectors".into(),
        ));
    }
    let (m, n) = (a.data().len(), b.data().len());
    let mut out = Vec::with_capacity(m * n);
    for (i, &x) in a.data().iter().enumerate() {
        for &y in b.data() {
            out.push(apply_scalar("table", &f, &[x, y], i, env, trace)?);
        }
    }
    Ok(Value::Array(DenseArray::new(Shape::new(vec![m, n]), out)?))
}

/// Evaluate the leading reference argument and the remaining
/// expressions; loud when `f` is not a reference.
fn fn_and_args(
    who: &str,
    expected: usize,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(Value, Vec<Value>), EvalError> {
    if args.len() != expected {
        return Err(EvalError::BadArity {
            func: who.into(),
            expected,
            got: args.len(),
        });
    }
    let f = crate::eval::eval_expr(&args[0], env, trace)?;
    if !matches!(f, Value::UserFnRef { .. } | Value::BuiltinRef { .. }) {
        return Err(EvalError::Unsupported(format!(
            "{who}: the first argument must be a function reference (`:u:name` or `:name`) -- got {}",
            mlpl_eval_types::value_kind(&f)
        )));
    }
    let rest = args[1..]
        .iter()
        .map(|a| crate::eval::eval_expr(a, env, trace))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((f, rest))
}

/// One scalar application of `f`; the result must be rank-0.
fn apply_scalar(
    who: &str,
    f: &Value,
    xs: &[f64],
    index: usize,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<f64, EvalError> {
    let result = match f {
        Value::UserFnRef { name } => {
            let vals: Vec<Value> = xs
                .iter()
                .map(|&x| Value::Array(DenseArray::from_scalar(x)))
                .collect();
            crate::eval_user_fn::invoke_user_fn_values(name, &vals, env, trace)?
        }
        Value::BuiltinRef { name } => {
            let arrs: Vec<DenseArray> = xs.iter().map(|&x| DenseArray::from_scalar(x)).collect();
            // Env dispatch, not bare call_builtin: elementwise
            // ops like `add` live behind the hook.
            Value::Array(mlpl_eval_env::dispatch_hook::dispatch_or_err(
                env, name, arrs,
            )?)
        }
        _ => unreachable!("checked in fn_and_args"),
    };
    match result {
        Value::Array(a) if a.rank() == 0 => Ok(a.data()[0]),
        other => Err(EvalError::Unsupported(format!(
            "{who}: the function must return a scalar per element -- at index {index} it \
             returned {}",
            mlpl_eval_types::value_kind(&other)
        ))),
    }
}
