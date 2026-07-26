//! Saga 33 step 021: per-branch helpers for
//! `grad::eval_tensor_fncall`. The "shape" group: patchify,
//! concat, take, reshape -- all shape-manipulating ops that
//! take one or more scalar dimension args.

use std::collections::HashMap;

use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::grad::{arity_check, eval_shape_dims, eval_tensor_expr, tape_scalar_usize};
use mlpl_eval_types::EvalError;

pub(crate) fn call_patchify(
    args: &[Expr],
    env: &mut Environment,
    tape: &std::rc::Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    arity_check(args, 2, "patchify")?;
    let x = eval_tensor_expr(&args[0], env, tape, params)?;
    let p = tape_scalar_usize(&args[1], env, "patchify: patch_size")?;
    Ok(x.patchify(p))
}

pub(crate) fn call_concat(
    args: &[Expr],
    env: &mut Environment,
    tape: &std::rc::Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    arity_check(args, 3, "concat")?;
    let a = eval_tensor_expr(&args[0], env, tape, params)?;
    let b = eval_tensor_expr(&args[1], env, tape, params)?;
    let axis = tape_scalar_usize(&args[2], env, "concat: axis")?;
    Ok(a.concat(&b, axis))
}

pub(crate) fn call_take(
    args: &[Expr],
    env: &mut Environment,
    tape: &std::rc::Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    arity_check(args, 3, "take")?;
    let x = eval_tensor_expr(&args[0], env, tape, params)?;
    let axis = tape_scalar_usize(&args[1], env, "take: axis")?;
    let idx = tape_scalar_usize(&args[2], env, "take: idx")?;
    Ok(x.take(axis, idx))
}

/// `rotate(x, k, axis)` on the tape. `k` may be negative (spelled
/// `0 - 1` in MLPL) so it is extracted here as a signed integer
/// rather than through `tape_scalar_usize`.
pub(crate) fn call_rotate(
    args: &[Expr],
    env: &mut Environment,
    tape: &std::rc::Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    arity_check(args, 3, "rotate")?;
    let x = eval_tensor_expr(&args[0], env, tape, params)?;
    let k_arr = crate::eval::eval_expr(&args[1], env, &mut None)?.into_array()?;
    if k_arr.rank() != 0 || k_arr.data()[0].fract() != 0.0 {
        return Err(EvalError::Unsupported(
            "rotate: k must be an integer scalar".into(),
        ));
    }
    let k = k_arr.data()[0] as i64;
    let axis = tape_scalar_usize(&args[2], env, "rotate: axis")?;
    Ok(x.rotate(k, axis))
}

pub(crate) fn call_reshape(
    args: &[Expr],
    env: &mut Environment,
    tape: &std::rc::Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    arity_check(args, 2, "reshape")?;
    let x = eval_tensor_expr(&args[0], env, tape, params)?;
    let Expr::ArrayLit(elems, _) = &args[1] else {
        return Err(EvalError::Unsupported(
            "grad: reshape's second argument must be an [int, ...] literal".into(),
        ));
    };
    let dims = eval_shape_dims(elems, env)?;
    Ok(x.reshape(mlpl_array::Shape::new(dims)))
}
