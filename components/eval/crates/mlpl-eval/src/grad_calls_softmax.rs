//! `softmax(x[, axis])` inside `grad()`. RS2 (../reasoning-from-scratch).
//!
//! The tape softmax (NodeKind::Softmax { axis }) is axis-aware for any rank;
//! this threads the optional axis argument (default: the last axis) so
//! `softmax(x, axis)` differentiates along the requested axis instead of
//! silently using the last one.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor};
use mlpl_eval_types::EvalError;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::grad::{eval_tensor_expr, tape_scalar_usize};

pub(crate) fn call_softmax(
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    if args.is_empty() || args.len() > 2 {
        return Err(EvalError::Unsupported(
            "grad: softmax takes (x[, axis])".into(),
        ));
    }
    let x = eval_tensor_expr(&args[0], env, tape, params)?;
    let rank = x.value().rank();
    let axis = match args.get(1) {
        Some(a) => tape_scalar_usize(a, env, tape, params, "softmax: axis")?,
        None => rank.saturating_sub(1),
    };
    if axis >= rank {
        return Err(EvalError::Unsupported(format!(
            "softmax: axis {axis} out of range for rank {rank}"
        )));
    }
    Ok(x.softmax_axis(axis))
}
