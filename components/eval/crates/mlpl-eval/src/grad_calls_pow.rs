//! `pow(base, exponent)` inside `grad()` (RS1-pow). For a CONSTANT exponent
//! (any real k), `pow` differentiates via the tape's `PowConst` node:
//! `d/dx x^k = k * x^(k-1)`. A differentiable exponent (one that depends on a
//! parameter) is rejected -- the two-sided rule is not supported.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor};
use mlpl_eval_types::EvalError;
use mlpl_parser::Expr;

use crate::env::Environment;

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
             the exponent must be a constant"
                .into(),
        ));
    }
    let base = crate::grad::eval_tensor_expr(&args[0], env, tape, params)?;
    let exp = const_exponent(&args[1], env)?;
    Ok(base.pow_const(exp))
}

/// Resolve the exponent to a constant scalar (any real value).
fn const_exponent(expr: &Expr, env: &mut Environment) -> Result<f64, EvalError> {
    let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(
            "grad: pow's exponent must be a scalar constant".into(),
        ));
    }
    Ok(arr.data()[0])
}
