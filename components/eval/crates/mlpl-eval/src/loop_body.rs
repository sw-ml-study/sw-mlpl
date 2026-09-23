//! Shared body evaluation for `repeat` / `train` / `for`. Every statement runs
//! and non-final values are discarded, whatever their kind -- a string-valued
//! statement such as `print("abc")` or `q = "abc"` is as legal inside a loop
//! body as outside one (microgpt-mlpl bug j: the loops used to coerce every
//! statement's value to an array). Only a value the loop CONSUMES must be an
//! array, and that error names the construct.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value, value_kind};

/// Evaluate `body` in order and return the final statement's value (scalar
/// `0` for an empty body).
pub(crate) fn run_body(
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let mut last = Value::Array(DenseArray::from_scalar(0.0));
    for stmt in body {
        last = eval_expr(stmt, env, trace)?;
    }
    Ok(last)
}

/// The array a loop consumes from its body's final value (`train`'s step loss,
/// `for`'s captured row); any other kind is an error naming `construct` and
/// what it expected.
pub(crate) fn consumed_array(
    construct: &str,
    what: &str,
    v: Value,
) -> Result<DenseArray, EvalError> {
    match v {
        Value::Array(a) => Ok(a),
        other => Err(EvalError::Unsupported(format!(
            "{construct}: the body's last statement must be {what}, got a {}",
            value_kind(&other)
        ))),
    }
}
