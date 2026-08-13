//! The three Result-returning string conversions: `to_number(s)`,
//! `to_int(s)`, `env(name)`. Split out of `result_ops` (per
//! docs/code_metrics.md: split by responsibility) -- each evaluates
//! its single string argument, attempts the conversion, and wraps the
//! outcome in a `Value::Result`.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value, value_kind};

pub(crate) fn eval_string_to_result(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let s = expect_str_arg(name, args, env, trace)?;
    let (ok, payload) = match convert_string(name, &s) {
        Ok(p) => (true, p),
        Err(msg) => (false, Value::Str(msg)),
    };
    Ok(Value::Result {
        ok,
        payload: Box::new(payload),
    })
}

/// Evaluate the single argument to a string, erroring on the wrong
/// arity or a non-string value.
fn expect_str_arg(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<String, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    match eval_expr(&args[0], env, trace)? {
        Value::Str(s) => Ok(s),
        other => Err(EvalError::Unsupported(format!(
            "{name}: expected a string argument, got {}",
            value_kind(&other)
        ))),
    }
}

/// Convert a trimmed string to the target value, or an error message.
fn convert_string(name: &str, s: &str) -> Result<Value, String> {
    match name {
        "to_number" => s
            .trim()
            .parse::<f64>()
            .map(|n| Value::Array(DenseArray::from_scalar(n)))
            .map_err(|_| format!("to_number: cannot parse {s:?} as a number")),
        "to_int" => s
            .trim()
            .parse::<i64>()
            .map(|n| Value::Array(DenseArray::from_scalar(n as f64)))
            .map_err(|_| to_int_error(s)),
        "env" => std::env::var(s)
            .map(Value::Str)
            .map_err(|_| format!("env: {s} not set")),
        _ => unreachable!("dispatcher guard kept us in the to_number/to_int/env set"),
    }
}

/// Distinguish a non-integer number (`3.5`) from an unparseable string.
fn to_int_error(s: &str) -> String {
    if s.trim().parse::<f64>().is_ok() {
        format!("to_int: {s:?} is not an integer")
    } else {
        format!("to_int: cannot parse {s:?} as an integer")
    }
}
