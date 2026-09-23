//! Entry points onto the grad tape: `grad(expr, wrt)` and the one-tape batch
//! the optimizer steps use. Both trace the loss once through
//! `eval_tensor_expr` and backward once.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::DenseArray;
use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::env_api::*;
use mlpl_eval_types::EvalError;

/// Evaluate a `grad(expr, wrt)` call and return the gradient array of
/// the scalar expression `expr` with respect to the parameter `wrt`.
pub(crate) fn eval_grad(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    let wrt_name = wrt_param(args, env)?;
    let (_, params) = trace_loss(&args[0], env)?;
    // A `None` gradient means the loss subgraph never reached this parameter
    // -- the loss does not depend on `wrt` (finding D1). Returning zeros here
    // reads as a broken training step, so fail loudly instead of silently.
    params[&wrt_name].grad().ok_or_else(|| {
        EvalError::Unsupported(format!(
            "grad: the loss does not depend on '{wrt_name}' (no gradient flows \
             to it) -- was the loss computed eagerly before grad, or is this \
             the wrong parameter?"
        ))
    })
}

/// Validate `grad`'s arity and resolve its second argument to a tracked
/// parameter name.
fn wrt_param(args: &[Expr], env: &Environment) -> Result<String, EvalError> {
    crate::grad::arity_check(args, 2, "grad")?;
    let Expr::Ident(name, _) = &args[1] else {
        return Err(EvalError::Unsupported(
            "grad: second argument must be a parameter identifier".into(),
        ));
    };
    if !env.is_param(name) {
        return Err(EvalError::Unsupported(format!(
            "grad: '{name}' is not a tracked parameter"
        )));
    }
    Ok(name.clone())
}

/// Seed every tracked param onto a fresh tape, trace `loss`, and backward.
/// Under device("mlx") the tape keeps forward intermediates RESIDENT on the
/// registered backend (saga E4 step 003).
fn trace_loss(
    loss: &Expr,
    env: &mut Environment,
) -> Result<(Tensor, HashMap<String, Tensor>), EvalError> {
    let tape = Tape::new();
    if env.device() == "mlx" {
        crate::device::enable_resident_tape(&tape);
    }
    let params: HashMap<String, Tensor> = env
        .params()
        .map(|(name, value)| (name.clone(), Tensor::param(Rc::clone(&tape), value.clone())))
        .collect();
    let root = crate::grad::eval_tensor_expr(loss, env, &tape, &params)?;
    root.backward();
    Ok((root, params))
}

/// One tape for the whole step: evaluate `loss` once, backward once, and
/// return every tracked parameter's gradient (zeros for params the loss never
/// touched and the step did not ask to train). Every gradient is taken at the
/// SAME step-start weights (saga E4 step 006). A param in `train` that got no
/// gradient is an error (see `require_reached`).
pub(crate) fn eval_grads_batch(
    loss: &Expr,
    train: &[String],
    env: &mut Environment,
) -> Result<(f64, HashMap<String, DenseArray>), EvalError> {
    let (root, params) = trace_loss(loss, env)?;
    let mut unreached = Vec::new();
    let grads = params
        .into_iter()
        .map(|(n, t)| {
            let g = t.grad().unwrap_or_else(|| {
                unreached.push(n.clone());
                DenseArray::zeros(t.value().shape().clone())
            });
            (n, g)
        })
        .collect();
    crate::grad_errors::require_reached(train, env, |n| !unreached.iter().any(|u| u == n))?;
    // The step loss the optimizers return (train records it as the
    // per-step curve).
    let loss_val = root.value().data().first().copied().unwrap_or(0.0);
    Ok((loss_val, grads))
}
