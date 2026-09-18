//! `transpose_axes(x, perm)` inside `grad()`. RS3 (../reasoning-from-scratch).
//!
//! A general axis permutation on the tape (NodeKind::Transpose { perm }),
//! differentiable via the inverse-permutation backward. The reverse-axes
//! `transpose` keeps its own 1-arg unary dispatch.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor};
use mlpl_eval_types::EvalError;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::grad::{arity_check, eval_tensor_expr};

pub(crate) fn call_transpose_axes(
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    arity_check(args, 2, "transpose_axes")?;
    let x = eval_tensor_expr(&args[0], env, tape, params)?;
    let rank = x.value().rank();
    let perm = const_perm(&args[1], env, rank)?;
    Ok(x.transpose_axes(perm))
}

/// Resolve `expr` to a constant permutation of `0..rank` (each axis exactly
/// once), or return a clean error.
fn const_perm(expr: &Expr, env: &mut Environment, rank: usize) -> Result<Vec<usize>, EvalError> {
    let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
    let bad = || {
        EvalError::Unsupported(format!(
            "transpose_axes: perm must name each of the {rank} axes exactly once (0-based)"
        ))
    };
    let mut perm = Vec::with_capacity(arr.data().len());
    for &p in arr.data() {
        if p.fract() != 0.0 || p < 0.0 || p as usize >= rank {
            return Err(bad());
        }
        perm.push(p as usize);
    }
    let mut seen = vec![false; rank];
    if perm.len() != rank || perm.iter().any(|&a| std::mem::replace(&mut seen[a], true)) {
        return Err(bad());
    }
    Ok(perm)
}
