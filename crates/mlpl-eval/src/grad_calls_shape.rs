//! Saga 33 step 021: per-branch helpers for
//! `grad::eval_tensor_fncall`. The "shape" group: patchify,
//! concat, take, reshape -- all shape-manipulating ops that
//! take one or more scalar dimension args.

use std::collections::HashMap;

use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::grad::{arity_check, eval_shape_dims, eval_tensor_expr, tape_scalar_usize};

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
