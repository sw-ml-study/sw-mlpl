//! Saga 33 step 021: per-branch helpers for
//! `grad::eval_tensor_fncall`. The "basic" group: unary ops,
//! matmul, apply (model forward), cross_entropy. Each helper
//! takes the same (args, env, tape, params) shape and is a
//! direct lift from the corresponding `if name == ...` branch
//! in the original 86-LOC eval_tensor_fncall.

use crate::env_api::*;
use std::collections::HashMap;

use mlpl_array::DenseArray;
use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::grad::{arity_check, eval_tensor_expr};
use mlpl_eval_types::EvalError;

pub(crate) fn call_unary(
    op: fn(&Tensor) -> Tensor,
    args: &[Expr],
    env: &mut Environment,
    tape: &std::rc::Rc<Tape>,
    params: &HashMap<String, Tensor>,
    name: &str,
) -> Result<Tensor, EvalError> {
    arity_check(args, 1, name)?;
    Ok(op(&eval_tensor_expr(&args[0], env, tape, params)?))
}

pub(crate) fn call_matmul(
    args: &[Expr],
    env: &mut Environment,
    tape: &std::rc::Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    arity_check(args, 2, "matmul")?;
    let a = eval_tensor_expr(&args[0], env, tape, params)?;
    let b = eval_tensor_expr(&args[1], env, tape, params)?;
    Ok(a.matmul(&b))
}

pub(crate) fn call_apply(
    args: &[Expr],
    env: &mut Environment,
    tape: &std::rc::Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    arity_check(args, 2, "apply")?;
    let Expr::Ident(m, _) = &args[0] else {
        return Err(EvalError::Unsupported(
            "grad: apply's first argument must be a model identifier".into(),
        ));
    };
    let model = env
        .get_model(m)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(m.clone()))?;
    let x = eval_tensor_expr(&args[1], env, tape, params)?;
    mlpl_models_tape::apply_model_tape(&model, x, tape, params).map_err(EvalError::from)
}

pub(crate) fn call_cross_entropy(
    args: &[Expr],
    env: &mut Environment,
    tape: &std::rc::Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    arity_check(args, 2, "cross_entropy")?;
    let l = eval_tensor_expr(&args[0], env, tape, params)?;
    let t: DenseArray = crate::eval::eval_expr(&args[1], env, &mut None)?.into_array()?;
    let idx = mlpl_models_tape::validate_cross_entropy_targets(&l.value(), &t)?;
    Ok(l.cross_entropy(idx))
}
