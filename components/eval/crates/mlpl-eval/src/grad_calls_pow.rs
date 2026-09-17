//! `pow(base, exponent)` inside `grad()`. RS1-pow (../reasoning-from-scratch).
//!
//! `pow` is a binary elementwise builtin. For a CONSTANT positive-integer
//! exponent the differentiable form is the exact repeated product
//! `base * base * ...`, which reuses the tape's `Mul` backward (so the
//! gradient is exact: `d/dx x^k = k*x^(k-1)`). Fractional, zero, negative,
//! large, or differentiable exponents are rejected with a loud, actionable
//! error -- the general constant-exponent case (a `PowConst` tape node)
//! awaits the `mlpl-autograd` crate split, which is at its module ceiling.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor};
use mlpl_eval_types::EvalError;
use mlpl_parser::Expr;

use crate::env::Environment;

/// Largest exponent unrolled into repeated products (keeps the tape small).
const MAX_POW: i64 = 64;

pub(crate) fn call_pow(
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    crate::grad::arity_check(args, 2, "pow")?;
    if crate::grad_const::differentiably_uses_param(&args[1], params) {
        return Err(EvalError::Unsupported(
            "grad: pow with a differentiable exponent is not supported; \
             a constant integer exponent differentiates"
                .into(),
        ));
    }
    let base = crate::grad::eval_tensor_expr(&args[0], env, tape, params)?;
    let k = const_pos_int_exponent(&args[1], env)?;
    let mut acc = base.clone();
    for _ in 1..k {
        acc = acc.mul(&base);
    }
    Ok(acc)
}

/// Resolve the exponent to a constant positive integer, or return a loud
/// error naming the differentiable alternative.
fn const_pos_int_exponent(expr: &Expr, env: &mut Environment) -> Result<i64, EvalError> {
    let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
    let v = if arr.rank() == 0 {
        arr.data()[0]
    } else {
        f64::NAN
    };
    if v.fract() != 0.0 || v < 1.0 || v > MAX_POW as f64 {
        return Err(EvalError::Unsupported(format!(
            "grad: pow(x, {v}) is not differentiable; write x*x for squares, \
             sqrt(x) for 0.5, 1/x for -1 (integer exponents 1..={MAX_POW} differentiate)"
        )));
    }
    Ok(v as i64)
}
