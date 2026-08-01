//! The `grad(expr, wrt)` built-in: reverse-mode autograd over
//! a tree-walked mini-evaluator that lifts array-valued operations
//! onto an autograd tape.

use crate::env_api::*;
use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::{BinOpKind, Expr};

use crate::env::Environment;
use mlpl_eval_types::EvalError;

/// Evaluate a `grad(expr, wrt)` call and return the gradient array of
/// the scalar expression `expr` with respect to the parameter `wrt`.
pub(crate) fn eval_grad(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "grad".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let wrt_name = match &args[1] {
        Expr::Ident(n, _) => n.clone(),
        _ => {
            return Err(EvalError::Unsupported(
                "grad: second argument must be a parameter identifier".into(),
            ));
        }
    };
    if !env.is_param(&wrt_name) {
        return Err(EvalError::Unsupported(format!(
            "grad: '{wrt_name}' is not a tracked parameter"
        )));
    }
    let tape = Tape::new();
    // Saga E4 step 003: under device("mlx") the tape keeps forward
    // intermediates RESIDENT on the registered backend -- leaves
    // upload once, ops build one lazy graph -- replacing the old
    // second-forward materialize pass.
    if env.device() == "mlx" {
        crate::device::enable_resident_tape(&tape);
    }
    let mut params: HashMap<String, Tensor> = HashMap::new();
    for (name, value) in env.params() {
        params.insert(name.clone(), Tensor::param(Rc::clone(&tape), value.clone()));
    }
    let root = eval_tensor_expr(&args[0], env, &tape, &params)?;
    root.backward();
    let wrt_tensor = params
        .get(&wrt_name)
        .expect("wrt param present in params map");
    Ok(wrt_tensor
        .grad()
        .unwrap_or_else(|| DenseArray::zeros(wrt_tensor.value().shape().clone())))
}

/// One tape for the whole step: evaluate `loss` once, backward
/// once, and return every tracked parameter's gradient (zeros for
/// params the loss never touched). Saga E4 step 006: the optimizer
/// steps use this so every parameter's gradient is taken at the
/// SAME step-start weights (standard batched semantics -- the old
/// per-param rebuild updated earlier params before later params'
/// gradients were computed) and the forward runs once instead of
/// once per parameter.
pub(crate) fn eval_grads_batch(
    loss: &Expr,
    env: &mut Environment,
) -> Result<HashMap<String, DenseArray>, EvalError> {
    let tape = Tape::new();
    if env.device() == "mlx" {
        crate::device::enable_resident_tape(&tape);
    }
    let mut params: HashMap<String, Tensor> = HashMap::new();
    for (name, value) in env.params() {
        params.insert(name.clone(), Tensor::param(Rc::clone(&tape), value.clone()));
    }
    let root = eval_tensor_expr(loss, env, &tape, &params)?;
    root.backward();
    Ok(params
        .into_iter()
        .map(|(n, t)| {
            let g = t
                .grad()
                .unwrap_or_else(|| DenseArray::zeros(t.value().shape().clone()));
            (n, g)
        })
        .collect())
}

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
            let arr = env
                .get(name)
                .cloned()
                .ok_or_else(|| EvalError::UndefinedVariable(name.clone()))?;
            Ok(leaf(arr))
        }
        Expr::ArrayLit(elems, _) => {
            let arr = crate::eval_ops::eval_array_lit(elems, env, &mut None)?;
            Ok(leaf(arr))
        }
        Expr::UnaryNeg { operand, .. } => Ok(eval_tensor_expr(operand, env, tape, params)?.neg()),
        Expr::BinOp { op, lhs, rhs, .. } => {
            let l = eval_tensor_expr(lhs, env, tape, params)?;
            let r = eval_tensor_expr(rhs, env, tape, params)?;
            Ok(match op {
                BinOpKind::Add => l.add(&r),
                BinOpKind::Sub => l.sub(&r),
                BinOpKind::Mul => l.mul(&r),
                BinOpKind::Div => l.div(&r),
            })
        }
        Expr::FnCall { name, args, .. } => eval_tensor_fncall(name, args, env, tape, params),
        Expr::TensorCtor { shape, .. } => {
            let dims = eval_shape_dims(shape, env)?;
            Ok(leaf(DenseArray::zeros(Shape::new(dims))))
        }
        // Scoped forms and string literals never have a tensor
        // analogue inside `grad(expr, wrt)` -- the differentiable
        // surface is array-valued ops only.
        _ => Err(EvalError::Unsupported(
            "grad: expression form not supported inside grad()".into(),
        )),
    }
}

fn eval_tensor_fncall(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    if let Some(op) = unary_tensor_op(name) {
        return crate::grad_calls_basic::call_unary(op, args, env, tape, params, name);
    }
    match name {
        "matmul" => crate::grad_calls_basic::call_matmul(args, env, tape, params),
        "apply" => crate::grad_calls_basic::call_apply(args, env, tape, params),
        "apply_engram" => crate::grad_calls_engram::call_apply_engram(args, env, tape, params),
        "cross_entropy" => crate::grad_calls_basic::call_cross_entropy(args, env, tape, params),
        "patchify" => crate::grad_calls_shape::call_patchify(args, env, tape, params),
        "concat" => crate::grad_calls_shape::call_concat(args, env, tape, params),
        "take" => crate::grad_calls_shape::call_take(args, env, tape, params),
        "rotate" => crate::grad_calls_shape::call_rotate(args, env, tape, params),
        "reshape" => crate::grad_calls_shape::call_reshape(args, env, tape, params),
        _ => Err(EvalError::Unsupported(format!(
            "grad: function '{name}' not supported inside grad()"
        ))),
    }
}

/// Arity check shared by the per-branch helpers. Lifted out
/// of the original eval_tensor_fncall's local closure so
/// callers in grad_calls_basic / grad_calls_shape can use it.
pub(crate) fn arity_check(args: &[Expr], expected: usize, func: &str) -> Result<(), EvalError> {
    if args.len() == expected {
        return Ok(());
    }
    Err(EvalError::BadArity {
        func: func.into(),
        expected,
        got: args.len(),
    })
}

pub(crate) fn tape_scalar_usize(
    arg: &Expr,
    env: &mut Environment,
    what: &str,
) -> Result<usize, EvalError> {
    let arr = crate::eval::eval_expr(arg, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "{what} must be a scalar, got rank {}",
            arr.rank()
        )));
    }
    let v = arr.data()[0];
    if v < 0.0 || v.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "{what} must be a non-negative integer, got {v}"
        )));
    }
    Ok(v as usize)
}

pub(crate) fn unary_tensor_op(name: &str) -> Option<fn(&Tensor) -> Tensor> {
    Some(match name {
        "sum" => Tensor::sum,
        "mean" => Tensor::mean,
        "exp" => Tensor::exp,
        "log" => Tensor::log,
        "relu" => Tensor::relu,
        // `tanh_fn` is the surface-MLPL spelling (`tanh` itself
        // is reserved by the `tanh_layer()` model layer); both
        // names map to the same tape op.
        "tanh" | "tanh_fn" => Tensor::tanh,
        "sigmoid" => Tensor::sigmoid,
        "softmax" => Tensor::softmax,
        "transpose" => Tensor::transpose,
        _ => return None,
    })
}

pub(crate) fn eval_shape_dims(
    shape: &[Expr],
    env: &mut Environment,
) -> Result<Vec<usize>, EvalError> {
    let mut dims = Vec::with_capacity(shape.len());
    for dim_expr in shape {
        let arr = crate::eval::eval_expr(dim_expr, env, &mut None)?.into_array()?;
        if arr.rank() != 0 {
            return Err(EvalError::InvalidShapeDim);
        }
        let v = arr.data()[0];
        if v < 0.0 || v.fract() != 0.0 {
            return Err(EvalError::InvalidShapeDim);
        }
        dims.push(v as usize);
    }
    Ok(dims)
}
// ----- optimizer state and built-in dispatch (Saga 10) -----
//
// Saga 10 design choice: optimizer state lives on `Environment` as a
// map keyed by `(optimizer_name, param_name, slot_name)` instead of
// in a new crate. The `mlpl-autograd` substrate already lives in its
// own crate, and Adam / momentum-SGD are thin wrappers around `grad`
// plus per-param buffers, so a fresh crate would just trampoline
// through `mlpl-eval` to reach `Environment`. Folding the state and
// dispatch hooks into `grad.rs` keeps the wiring local and respects
// the project's per-module function-count budget.
//
// Step 001 adds only the storage type and stub built-in dispatch.
// Steps 002 and 003 fill in `momentum_sgd` and `adam`.

// The buffer type moved to mlpl-eval-state (env-types-out step);
// re-exported so `crate::grad::OptimizerState` paths keep working.
pub use mlpl_eval_state::OptimizerState;

/// Read-only accessor used by tests and downstream optimizer code.
#[must_use]
pub fn optim_state(env: &Environment) -> &OptimizerState {
    &env.optim_state
}

/// Mutable accessor used by tests and downstream optimizer code.
pub fn optim_state_mut(env: &mut Environment) -> &mut OptimizerState {
    &mut env.optim_state
}

/// Resolve the optimizer's `params` argument into a flat list of
/// parameter identifiers. Accepts:
///
/// - a single param identifier: `adam(loss, W, ...)`
/// - an array literal of param identifiers: `adam(loss, [W, b], ...)`
/// - a model identifier registered via the Saga 11 model DSL:
///   `adam(loss, M, ...)` walks `ModelSpec::params()` and returns its
///   flat, order-stable parameter list.
pub(crate) fn collect_params(
    arg: &Expr,
    env: &Environment,
    func: &str,
) -> Result<Vec<String>, EvalError> {
    match arg {
        Expr::Ident(n, _) => {
            if let Some(model) = env.get_model(n) {
                Ok(model.params())
            } else {
                Ok(vec![n.clone()])
            }
        }
        Expr::ArrayLit(elems, _) => {
            let mut v = Vec::with_capacity(elems.len());
            for e in elems {
                match e {
                    // Saga 29 step 009: walk model params when the
                    // ArrayLit element resolves to a registered model,
                    // matching the lone-Ident path's behavior. This
                    // is what lets the trained ViT demo write
                    // `adam(loss, [linear_p, attn, classifier], ...)`
                    // and have every model's param list flattened in.
                    Expr::Ident(n, _) => {
                        if let Some(model) = env.get_model(n) {
                            v.extend(model.params());
                        } else {
                            v.push(n.clone());
                        }
                    }
                    _ => {
                        return Err(EvalError::Unsupported(format!(
                            "{func}: params list must contain only identifiers"
                        )));
                    }
                }
            }
            Ok(v)
        }
        _ => Err(EvalError::Unsupported(format!(
            "{func}: second argument must be a param identifier, model identifier, or list"
        ))),
    }
}
