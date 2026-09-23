//! Arithmetic and unary ops on the grad tape: binary operators (with
//! broadcast/label validation), the unary builtin table, and `flatten`.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::{BinOpKind, Expr};

use crate::env::Environment;
use crate::grad::{arity_check, eval_tensor_expr};
use mlpl_eval_types::EvalError;

/// Combine two tape tensors under a binary operator. The four
/// arithmetic ops record a differentiable node; the six comparison
/// ops produce a 0/1 mask with zero gradient almost everywhere, so
/// inside `grad(...)` they are rejected as non-differentiable rather
/// than silently returning a zero-gradient constant.
pub(crate) fn tensor_binop(op: &BinOpKind, l: &Tensor, r: &Tensor) -> Result<Tensor, EvalError> {
    match op {
        BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul | BinOpKind::Div => {
            checked_arith(op, l, r)
        }
        BinOpKind::Lt
        | BinOpKind::Gt
        | BinOpKind::Le
        | BinOpKind::Ge
        | BinOpKind::Eq
        | BinOpKind::Ne => Err(EvalError::Unsupported(format!(
            "grad: comparison operator `{op}` is not differentiable"
        ))),
    }
}

/// Build a differentiable arithmetic node after validating broadcast/label
/// compatibility on the forward values -- so an incompatible shape or label is
/// a clean error, not a panic in the tape's `push_binary` (finding F18; F10
/// covers the label half). Eager evaluation already errors here; this keeps the
/// tape consistent.
fn checked_arith(op: &BinOpKind, l: &Tensor, r: &Tensor) -> Result<Tensor, EvalError> {
    mlpl_array_ops_element::check_binop_compat(&l.value(), &r.value())
        .map_err(EvalError::ArrayError)?;
    Ok(match op {
        BinOpKind::Add => l.add(r),
        BinOpKind::Sub => l.sub(r),
        BinOpKind::Mul => l.mul(r),
        _ => l.div(r),
    })
}

pub(crate) fn unary_tensor_op(name: &str) -> Option<fn(&Tensor) -> Tensor> {
    Some(match name {
        "sum" => Tensor::sum,
        "mean" => Tensor::mean,
        "exp" => Tensor::exp,
        "log" => Tensor::log,
        "sqrt" => Tensor::sqrt,
        "sin" => Tensor::sin,
        "cos" => Tensor::cos,
        "relu" => Tensor::relu,
        // `tanh_fn` is the surface-MLPL spelling (`tanh` itself
        // is reserved by the `tanh_layer()` model layer); both
        // names map to the same tape op.
        "tanh" | "tanh_fn" => Tensor::tanh,
        "sigmoid" => Tensor::sigmoid,
        "transpose" => Tensor::transpose,
        _ => return None,
    })
}

/// `flatten(x)` on the tape: reshape to 1-D, reusing the Reshape backward.
pub(crate) fn call_flatten(
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    arity_check(args, 1, "flatten")?;
    let x = eval_tensor_expr(&args[0], env, tape, params)?;
    let total = x.value().shape().elem_count();
    Ok(x.reshape(mlpl_array::Shape::new(vec![total])))
}
