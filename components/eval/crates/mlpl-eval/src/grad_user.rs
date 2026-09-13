//! Inline a user function's body onto the grad tape (finding F2), so a loss
//! written as `def u:loss(...) { ... }` differentiates end-to-end instead of
//! having to be spelled inline. Backward-compatible: this is reached ONLY for
//! `u:` calls, which `eval_tensor_fncall` previously rejected outright, so no
//! existing grad expression changes behavior.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::env_api::*;
use crate::grad::eval_tensor_expr;
use mlpl_eval_types::EvalError;

/// Depth guard so a runaway recursive `u:` function traced inside grad errors
/// instead of overflowing the stack.
const MAX_GRAD_FN_DEPTH: usize = 250;

/// Trace a `u:name(args)` call by binding the function's parameters to the
/// traced argument tensors and tracing its body onto the same tape.
pub(crate) fn call_user_fn_grad(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    let f = env
        .get_fn(name)
        .cloned()
        .ok_or_else(|| EvalError::Unsupported(format!("grad: unknown user function '{name}'")))?;
    if args.len() != f.params.len() {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: f.params.len(),
            got: args.len(),
        });
    }
    // The body's scope: the outer grad params (so a global param the body
    // references stays differentiable) with the function's parameters bound to
    // the traced arguments layered on top (shadowing).
    let mut local = params.clone();
    for (p, arg) in f.params.iter().zip(args) {
        let t = eval_tensor_expr(arg, env, tape, params)?;
        local.insert(p.clone(), t);
    }
    env.call_depth += 1;
    let out = trace_body(&f.body, env, tape, &mut local);
    env.call_depth -= 1;
    out
}

/// Trace a user-function body (a statement list) onto the tape, returning the
/// last statement's tensor. Assignments bind locals; a leading docstring (a
/// bare string) is discarded, matching normal evaluation.
fn trace_body(
    body: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    local: &mut HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    if env.call_depth > MAX_GRAD_FN_DEPTH {
        return Err(EvalError::Unsupported(
            "grad: user-function recursion too deep".into(),
        ));
    }
    let mut result = None;
    for stmt in body {
        match stmt {
            Expr::StrLit(_, _) => {}
            Expr::Assign { name, value, .. } => {
                let t = eval_tensor_expr(value, env, tape, local)?;
                local.insert(name.clone(), t.clone());
                result = Some(t);
            }
            // A `repeat N { .. }` is unrolled onto the tape (finding F6): the
            // body's assignments thread through `local` across iterations, so
            // bounded recurrence depth trains without hand-nested apply calls.
            Expr::Repeat { count, body, .. } => {
                let n = repeat_count(count, env)?;
                for _ in 0..n {
                    result = Some(trace_body(body, env, tape, local)?);
                }
            }
            other => result = Some(eval_tensor_expr(other, env, tape, local)?),
        }
    }
    result.ok_or_else(|| {
        EvalError::Unsupported("grad: user function body has no result expression".into())
    })
}

/// Evaluate a `repeat` count to a non-negative integer. The count is a plain
/// scalar (not differentiable), so it is evaluated eagerly like normal `repeat`.
fn repeat_count(count: &Expr, env: &mut Environment) -> Result<usize, EvalError> {
    let n = crate::eval::eval_expr(count, env, &mut None)?.into_array()?;
    if n.rank() != 0 {
        return Err(EvalError::InvalidRepeatCount);
    }
    Ok(n.data()[0] as usize)
}
