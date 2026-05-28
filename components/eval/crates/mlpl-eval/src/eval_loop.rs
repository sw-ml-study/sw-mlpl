//! Saga 31 step 005: `while cond { body }` loop driver.
//!
//! Evaluates `cond` each iteration; the body re-runs while
//! the condition is truthy (scalar non-zero or `Ok(_)`). The
//! `break value` and `continue` keywords propagate via
//! `EvalError::BreakSignal(Value)` and
//! `EvalError::ContinueSignal` -- the loop driver catches
//! both. Any other error short-circuits the loop and bubbles
//! up to the caller.
//!
//! The expression evaluates to:
//! - the `break value` if the loop exited via `break value`
//! - scalar `0` if the loop exited via bare `break`
//! - scalar `0` if the loop exited because `cond` went falsy

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::eval::eval_expr;
use crate::value::{Value, value_kind};

/// Drive a `while cond { body }` loop. Truthiness of `cond`
/// matches the `if` rule: rank-0 array (non-zero is true) or
/// `Result` (`Ok` is true). Anything else is a type error.
pub(crate) fn eval_while(
    cond: &Expr,
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    loop {
        let cond_val = eval_expr(cond, env, trace)?;
        let truthy = match &cond_val {
            Value::Array(a) if a.rank() == 0 => a.data()[0] != 0.0,
            Value::Result { ok, .. } => *ok,
            other => {
                return Err(EvalError::Unsupported(format!(
                    "while condition must be a scalar or Result, got {}",
                    value_kind(other)
                )));
            }
        };
        if !truthy {
            return Ok(Value::Array(DenseArray::from_scalar(0.0)));
        }
        match run_body(body, env, trace) {
            Ok(()) => continue,
            Err(EvalError::ContinueSignal) => continue,
            Err(EvalError::BreakSignal(v)) => return Ok(*v),
            Err(e) => return Err(e),
        }
    }
}

fn run_body(
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(), EvalError> {
    for stmt in body {
        eval_expr(stmt, env, trace)?;
    }
    Ok(())
}
