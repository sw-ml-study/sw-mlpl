//! The `grad(expr, wrt)` built-in: reverse-mode autograd over
//! a tree-walked mini-evaluator that lifts array-valued operations
//! onto an autograd tape. This module holds the tracer itself;
//! entry points, arithmetic, argument resolution, and optimizer
//! state live in the `grad_*` siblings and are re-exported here.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::env_api::*;
use mlpl_eval_types::EvalError;

pub(crate) use crate::grad_arith::unary_tensor_op;
use crate::grad_arith::{call_flatten, tensor_binop};
pub(crate) use crate::grad_entry::{eval_grad, eval_grads_batch};
pub(crate) use crate::grad_optim_state::eval_reset_optimizer;
pub use crate::grad_optim_state::{OptimizerState, optim_state, optim_state_mut};
pub(crate) use crate::grad_params::{collect_params, set_trained_param};
pub(crate) use crate::grad_scalars::{arity_check, eval_shape_dims, tape_scalar_usize};

pub(crate) fn eval_tensor_expr(
    expr: &Expr,
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    let leaf = |v: DenseArray| Tensor::leaf(Rc::clone(tape), v, false);
    match expr {
        Expr::IntLit(n, _) => Ok(leaf(DenseArray::from_scalar(*n as f64))),
        Expr::FloatLit(f, _) => Ok(leaf(DenseArray::from_scalar(*f))),
        Expr::Ident(name, _) => {
            if let Some(t) = params.get(name) {
                return Ok(t.clone());
            }
            let arr = env.get(name).cloned();
            Ok(leaf(arr.ok_or_else(|| {
                crate::grad_errors::unbound_ident(name, env)
            })?))
        }
        Expr::ArrayLit(elems, _) => {
            let arr = crate::eval_ops::eval_array_lit(elems, env, &mut None)?;
            Ok(leaf(arr))
        }
        Expr::UnaryNeg { operand, .. } => Ok(eval_tensor_expr(operand, env, tape, params)?.neg()),
        Expr::BinOp { op, lhs, rhs, .. } => {
            let l = eval_tensor_expr(lhs, env, tape, params)?;
            let r = eval_tensor_expr(rhs, env, tape, params)?;
            tensor_binop(op, &l, &r)
        }
        Expr::FnCall { name, args, .. } => {
            crate::grad_const::fncall_or_fold(expr, name, args, env, tape, params)
        }
        Expr::TensorCtor { shape, .. } => {
            let dims = eval_shape_dims(shape, env, tape, params)?;
            Ok(leaf(DenseArray::zeros(Shape::new(dims))))
        }
        // Scoped forms, records, and string literals never have a tensor
        // analogue inside `grad(expr, wrt)` -- the differentiable
        // surface is array-valued ops only.
        _ => Err(crate::grad_errors::unsupported_form(expr)),
    }
}

pub(crate) fn eval_tensor_fncall(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    if let Some(op) = unary_tensor_op(name) {
        return crate::grad_calls_basic::call_unary(op, args, env, tape, params, name);
    }
    if crate::grad_purity::is_constant_leaf_builtin(name) {
        return crate::grad_const::eval_stop_gradient(name, args, env, tape, params);
    }
    match name {
        "softmax" => crate::grad_calls_softmax::call_softmax(args, env, tape, params),
        "transpose_axes" => {
            crate::grad_calls_transpose::call_transpose_axes(args, env, tape, params)
        }
        "pow" => crate::grad_calls_pow::call_pow(args, env, tape, params),
        "matmul" => crate::grad_calls_basic::call_matmul(args, env, tape, params),
        "apply" => crate::grad_calls_basic::call_apply(args, env, tape, params),
        "apply_engram" => crate::grad_calls_engram::call_apply_engram(args, env, tape, params),
        "gather_rows" => crate::grad_calls_engram::call_gather_rows(args, env, tape, params),
        "cross_entropy" => crate::grad_calls_basic::call_cross_entropy(args, env, tape, params),
        "patchify" => crate::grad_calls_shape::call_patchify(args, env, tape, params),
        "concat" => crate::grad_calls_shape::call_concat(args, env, tape, params),
        "take" => crate::grad_calls_shape::call_take(args, env, tape, params),
        "rotate" => crate::grad_calls_shape::call_rotate(args, env, tape, params),
        "reshape" => crate::grad_calls_shape::call_reshape(args, env, tape, params),
        "windows" => crate::grad_calls_shape::call_windows(args, env, tape, params),
        "reduce" | "reduce_add" => {
            crate::grad_calls_shape::call_reduce_grad(name, args, env, tape, params)
        }
        "flatten" => call_flatten(args, env, tape, params),
        // User-defined functions: inline the body onto the tape (F2), so a
        // loss written as `def u:loss(...)` differentiates.
        _ if name.starts_with("u:") => {
            crate::grad_user::call_user_fn_grad(name, args, env, tape, params)
        }
        _ => Err(EvalError::Unsupported(format!(
            "grad: function '{name}' not supported inside grad()"
        ))),
    }
}
