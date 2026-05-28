//! Saga 33 step 004: scalar-extraction helpers used by every
//! `eval_*` model-constructor dispatcher. Centralized so each
//! dispatcher file imports a tiny named helper instead of
//! duplicating an `into_array()? + rank check` pair.

use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;

pub(crate) fn scalar_f64(expr: &Expr, env: &mut Environment, func: &str) -> Result<f64, EvalError> {
    let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "{func}: expected a scalar argument"
        )));
    }
    Ok(arr.data()[0])
}

pub(crate) fn scalar_usize(
    expr: &Expr,
    env: &mut Environment,
    func: &str,
) -> Result<usize, EvalError> {
    let v = scalar_f64(expr, env, func)?;
    if v < 0.0 || v.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "{func}: dimension must be a non-negative integer"
        )));
    }
    Ok(v as usize)
}
