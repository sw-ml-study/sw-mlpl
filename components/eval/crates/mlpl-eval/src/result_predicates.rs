//! `is_result(x)` -- a TOTAL predicate: `1` when `x` is a Result
//! (`ok(_)` / `err(_)`), `0` for any other value, never raising. The
//! Result accessors (`is_ok`, `unwrap`, ...) require a Result receiver;
//! this is the branch test for values that may or may not be one -- an
//! extension call returns its bare value on success and an `err(_)` on
//! failure, so `if is_result(r) { err_message(r) } else { r.path }`.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn eval_is_result(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [arg] = args else {
        return Err(EvalError::BadArity {
            func: "is_result".into(),
            expected: 1,
            got: args.len(),
        });
    };
    let yes = matches!(eval_expr(arg, env, trace)?, Value::Result { .. });
    Ok(Value::Array(DenseArray::from_scalar(if yes {
        1.0
    } else {
        0.0
    })))
}
