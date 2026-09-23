//! Scalar / shape argument resolution shared by the grad call handlers:
//! arity checks, integer scalars, and shape dimensions -- each resolved
//! through the traced scope so function-local bindings work.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::grad::eval_tensor_expr;
use mlpl_eval_types::EvalError;

/// Arity check shared by the per-branch helpers. Lifted out
/// of the original eval_tensor_fncall's local closure so
/// callers in grad_calls_basic / grad_calls_shape can use it.
pub(crate) fn arity_check(args: &[Expr], expected: usize, func: &str) -> Result<(), EvalError> {
    if args.len() == expected {
        return Ok(());
    }
    Err(EvalError::BadArity {
        func: func.into(),
        expected,
        got: args.len(),
    })
}

pub(crate) fn tape_scalar_usize(
    arg: &Expr,
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
    what: &str,
) -> Result<usize, EvalError> {
    // Resolve through the traced scope (eval_tensor_expr checks the function's
    // local bindings before the global env), so an axis/index bound to a
    // function argument works inside an inlined user function (finding F20).
    let arr = eval_tensor_expr(arg, env, tape, params)?.value();
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "{what} must be a scalar, got rank {}",
            arr.rank()
        )));
    }
    let v = arr.data()[0];
    if v < 0.0 || v.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "{what} must be a non-negative integer, got {v}"
        )));
    }
    Ok(v as usize)
}

pub(crate) fn eval_shape_dims(
    shape: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Vec<usize>, EvalError> {
    let mut dims = Vec::with_capacity(shape.len());
    for dim_expr in shape {
        // Resolve each dim through the traced scope (eval_tensor_expr checks the
        // function's local bindings before the global env), so a reshape/windows
        // dimension bound to a function argument resolves and the gradient flows
        // through the reshaped value (finding F23).
        let arr = eval_tensor_expr(dim_expr, env, tape, params)?.value();
        if arr.rank() != 0 {
            return Err(EvalError::InvalidShapeDim);
        }
        let v = arr.data()[0];
        if v < 0.0 || v.fract() != 0.0 {
            return Err(EvalError::InvalidShapeDim);
        }
        dims.push(v as usize);
    }
    Ok(dims)
}
