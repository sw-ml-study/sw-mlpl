//! `and` / `or`. On a scalar left side they short-circuit -- the right
//! side is not evaluated when the left decides -- so `i < n and a[i] > 0`
//! style conditions are safe. Otherwise both sides are evaluated and the
//! result is the elementwise 0/1 mask (nonzero = true, NumPy-style
//! broadcasting). `not` is `eq(x, 0)`, desugared by the parser.

use mlpl_array::DenseArray;
use mlpl_parser::{BinOpKind, Expr};
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value};

/// Evaluate `lhs and rhs` / `lhs or rhs`.
pub(crate) fn eval_logical(
    op: &BinOpKind,
    lhs: &Expr,
    rhs: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let l = eval_expr(lhs, env, trace)?.into_array()?;
    if l.rank() == 0 {
        let truthy = l.data()[0] != 0.0;
        if truthy == matches!(op, BinOpKind::Or) {
            return Ok(Value::Array(DenseArray::from_scalar(f64::from(u8::from(
                truthy,
            )))));
        }
    }
    let r = eval_expr(rhs, env, trace)?.into_array()?;
    Ok(Value::Array(mask(op, l, r)?))
}

/// The elementwise 0/1 result of `l and r` / `l or r` (broadcasting).
/// Shared with `grad`, where the result is a stop-gradient constant.
pub(crate) fn mask(op: &BinOpKind, l: DenseArray, r: DenseArray) -> Result<DenseArray, EvalError> {
    let f: fn(f64, f64) -> f64 = match op {
        BinOpKind::And => |a, b| f64::from(u8::from(a != 0.0 && b != 0.0)),
        _ => |a, b| f64::from(u8::from(a != 0.0 || b != 0.0)),
    };
    Ok(mlpl_array_ops_element::ApplyBinopExt::apply_binop(
        &l, &r, f,
    )?)
}
