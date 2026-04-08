//! The `grad(expr, wrt)` built-in: reverse-mode autograd over
//! a tree-walked mini-evaluator that lifts array-valued operations
//! onto an autograd tape.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};
use mlpl_core::Span;
use mlpl_parser::{BinOpKind, Expr};

use crate::env::Environment;
use crate::error::EvalError;

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

fn eval_tensor_expr(
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
        Expr::Assign { .. } | Expr::Repeat { .. } | Expr::Train { .. } | Expr::StrLit(_, _) => Err(
            EvalError::Unsupported("grad: expression form not supported inside grad()".into()),
        ),
    }
}

fn eval_tensor_fncall(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    tape: &Rc<Tape>,
    params: &HashMap<String, Tensor>,
) -> Result<Tensor, EvalError> {
    let arity = |expected: usize| -> Result<(), EvalError> {
        if args.len() == expected {
            Ok(())
        } else {
            Err(EvalError::BadArity {
                func: name.into(),
                expected,
                got: args.len(),
            })
        }
    };
    if let Some(op) = unary_tensor_op(name) {
        arity(1)?;
        let a = eval_tensor_expr(&args[0], env, tape, params)?;
        return Ok(op(&a));
    }
    if name == "matmul" {
        arity(2)?;
        let a = eval_tensor_expr(&args[0], env, tape, params)?;
        let b = eval_tensor_expr(&args[1], env, tape, params)?;
        return Ok(a.matmul(&b));
    }
    Err(EvalError::Unsupported(format!(
        "grad: function '{name}' not supported inside grad()"
    )))
}

fn unary_tensor_op(name: &str) -> Option<fn(&Tensor) -> Tensor> {
    Some(match name {
        "sum" => Tensor::sum,
        "mean" => Tensor::mean,
        "exp" => Tensor::exp,
        "log" => Tensor::log,
        "relu" => Tensor::relu,
        "tanh" => Tensor::tanh,
        "sigmoid" => Tensor::sigmoid,
        "softmax" => Tensor::softmax,
        "transpose" => Tensor::transpose,
        _ => return None,
    })
}

fn eval_shape_dims(shape: &[Expr], env: &mut Environment) -> Result<Vec<usize>, EvalError> {
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

/// Per-optimizer, per-parameter state buffers (e.g. momentum velocity,
/// Adam first/second moments).
///
/// Step 001 exposes the storage as plain public fields keyed by
/// `(optimizer_name, param_name, slot_name)` so steps 002 and 003 can
/// fill in `momentum_sgd` and `adam` without dragging extra accessor
/// helpers across the per-module function-count budget.
#[derive(Clone, Debug, Default)]
pub struct OptimizerState {
    /// Buffers keyed by `(optimizer_name, param_name, slot_name)`.
    /// `slot_name` lets a single optimizer store multiple buffers per
    /// param (e.g. Adam needs both `m` and `v`).
    pub buffers: HashMap<(String, String, String), DenseArray>,
    /// Per-optimizer step counter (for Adam bias correction).
    pub steps: HashMap<String, u64>,
}

/// Read-only accessor used by tests and downstream optimizer code.
#[must_use]
pub fn optim_state(env: &Environment) -> &OptimizerState {
    &env.optim_state
}

/// Mutable accessor used by tests and downstream optimizer code.
pub fn optim_state_mut(env: &mut Environment) -> &mut OptimizerState {
    &mut env.optim_state
}

/// `momentum_sgd(loss_expr, params, lr, beta)` built-in.
///
/// `params` is either a single param identifier or an array literal of
/// param identifiers. For each param `w`, applies the classical
/// heavy-ball update with per-param velocity buffer `v` stored on the
/// environment under `("momentum_sgd", w, "v")`:
///
/// ```text
///     g       = grad(loss_expr, w)
///     v_new   = beta * v_old + g
///     w_new   = w - lr * v_new
/// ```
///
/// Returns a scalar zero (the call is invoked for its side effects on
/// the parameter bindings and optimizer state).
pub(crate) fn eval_momentum_sgd(
    args: &[Expr],
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    if args.len() != 4 {
        return Err(EvalError::BadArity {
            func: "momentum_sgd".into(),
            expected: 4,
            got: args.len(),
        });
    }
    let loss_expr = args[0].clone();
    let param_names: Vec<String> = match &args[1] {
        Expr::Ident(n, _) => vec![n.clone()],
        Expr::ArrayLit(elems, _) => {
            let mut v = Vec::with_capacity(elems.len());
            for e in elems {
                match e {
                    Expr::Ident(n, _) => v.push(n.clone()),
                    _ => {
                        return Err(EvalError::Unsupported(
                            "momentum_sgd: params list must contain only identifiers".into(),
                        ));
                    }
                }
            }
            v
        }
        _ => {
            return Err(EvalError::Unsupported(
                "momentum_sgd: second argument must be a param identifier or list".into(),
            ));
        }
    };
    let scalar_arg = |expr: &Expr, env: &mut Environment| -> Result<f64, EvalError> {
        let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
        if arr.rank() != 0 {
            return Err(EvalError::Unsupported(
                "momentum_sgd: lr and beta must be scalars".into(),
            ));
        }
        Ok(arr.data()[0])
    };
    let lr = scalar_arg(&args[2], env)?;
    let beta = scalar_arg(&args[3], env)?;

    for name in &param_names {
        if !env.is_param(name) {
            return Err(EvalError::Unsupported(format!(
                "momentum_sgd: '{name}' is not a tracked parameter"
            )));
        }
        let grad_args = [
            loss_expr.clone(),
            Expr::Ident(name.clone(), Span::new(0, 0)),
        ];
        let g = eval_grad(&grad_args, env)?;
        let key = ("momentum_sgd".to_string(), name.clone(), "v".to_string());
        let v_old = env
            .optim_state
            .buffers
            .get(&key)
            .cloned()
            .unwrap_or_else(|| DenseArray::zeros(g.shape().clone()));
        if v_old.shape() != g.shape() {
            return Err(EvalError::Unsupported(format!(
                "momentum_sgd: stored velocity shape mismatch for '{name}'"
            )));
        }
        let v_data: Vec<f64> = v_old
            .data()
            .iter()
            .zip(g.data().iter())
            .map(|(vo, gv)| beta * vo + gv)
            .collect();
        let v_new =
            DenseArray::new(v_old.shape().clone(), v_data).expect("velocity shape matches grad");
        env.optim_state.buffers.insert(key, v_new.clone());

        let w = env.get(name).cloned().expect("param exists in environment");
        let w_data: Vec<f64> = w
            .data()
            .iter()
            .zip(v_new.data().iter())
            .map(|(wv, vv)| wv - lr * vv)
            .collect();
        let w_new =
            DenseArray::new(w.shape().clone(), w_data).expect("weight shape matches velocity");
        env.set(name.clone(), w_new);
    }
    *env.optim_state
        .steps
        .entry("momentum_sgd".into())
        .or_insert(0) += 1;
    Ok(DenseArray::from_scalar(0.0))
}

/// `adam(loss_expr, params, lr, b1, b2, eps)` built-in.
///
/// Standard Adam with bias correction. Per-param first/second moment
/// buffers `m`, `v` and a per-optimizer step counter `t` live in
/// `OptimizerState`. At each call (with `t` post-incremented to start
/// at 1):
///
/// ```text
///     g     = grad(loss_expr, w)
///     m     = b1*m + (1 - b1)*g
///     v     = b2*v + (1 - b2)*g*g
///     mhat  = m / (1 - b1^t)
///     vhat  = v / (1 - b2^t)
///     w     = w - lr * mhat / (sqrt(vhat) + eps)
/// ```
pub(crate) fn eval_adam(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    if args.len() != 6 {
        return Err(EvalError::BadArity {
            func: "adam".into(),
            expected: 6,
            got: args.len(),
        });
    }
    let loss_expr = args[0].clone();
    let param_names: Vec<String> = match &args[1] {
        Expr::Ident(n, _) => vec![n.clone()],
        Expr::ArrayLit(elems, _) => {
            let mut v = Vec::with_capacity(elems.len());
            for e in elems {
                match e {
                    Expr::Ident(n, _) => v.push(n.clone()),
                    _ => {
                        return Err(EvalError::Unsupported(
                            "adam: params list must contain only identifiers".into(),
                        ));
                    }
                }
            }
            v
        }
        _ => {
            return Err(EvalError::Unsupported(
                "adam: second argument must be a param identifier or list".into(),
            ));
        }
    };
    let scalar_arg = |expr: &Expr, env: &mut Environment| -> Result<f64, EvalError> {
        let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
        if arr.rank() != 0 {
            return Err(EvalError::Unsupported(
                "adam: lr/b1/b2/eps must be scalars".into(),
            ));
        }
        Ok(arr.data()[0])
    };
    let lr = scalar_arg(&args[2], env)?;
    let b1 = scalar_arg(&args[3], env)?;
    let b2 = scalar_arg(&args[4], env)?;
    let eps = scalar_arg(&args[5], env)?;

    // Step counter is 1-based: bump first, then read.
    let t = {
        let entry = env.optim_state.steps.entry("adam".into()).or_insert(0);
        *entry += 1;
        *entry
    };
    let bc1 = 1.0 - b1.powi(t as i32);
    let bc2 = 1.0 - b2.powi(t as i32);

    for name in &param_names {
        if !env.is_param(name) {
            return Err(EvalError::Unsupported(format!(
                "adam: '{name}' is not a tracked parameter"
            )));
        }
        let grad_args = [
            loss_expr.clone(),
            Expr::Ident(name.clone(), Span::new(0, 0)),
        ];
        let g = eval_grad(&grad_args, env)?;
        let m_key = ("adam".to_string(), name.clone(), "m".to_string());
        let v_key = ("adam".to_string(), name.clone(), "v".to_string());
        let m_old = env
            .optim_state
            .buffers
            .get(&m_key)
            .cloned()
            .unwrap_or_else(|| DenseArray::zeros(g.shape().clone()));
        let v_old = env
            .optim_state
            .buffers
            .get(&v_key)
            .cloned()
            .unwrap_or_else(|| DenseArray::zeros(g.shape().clone()));
        if m_old.shape() != g.shape() || v_old.shape() != g.shape() {
            return Err(EvalError::Unsupported(format!(
                "adam: stored moment shape mismatch for '{name}'"
            )));
        }
        let m_data: Vec<f64> = m_old
            .data()
            .iter()
            .zip(g.data().iter())
            .map(|(mo, gv)| b1 * mo + (1.0 - b1) * gv)
            .collect();
        let v_data: Vec<f64> = v_old
            .data()
            .iter()
            .zip(g.data().iter())
            .map(|(vo, gv)| b2 * vo + (1.0 - b2) * gv * gv)
            .collect();
        let m_new = DenseArray::new(g.shape().clone(), m_data).expect("m shape matches grad");
        let v_new = DenseArray::new(g.shape().clone(), v_data).expect("v shape matches grad");

        let w = env.get(name).cloned().expect("param exists in environment");
        let w_data: Vec<f64> = w
            .data()
            .iter()
            .zip(m_new.data().iter())
            .zip(v_new.data().iter())
            .map(|((wv, mv), vv)| {
                let mhat = mv / bc1;
                let vhat = vv / bc2;
                wv - lr * mhat / (vhat.sqrt() + eps)
            })
            .collect();
        let w_new = DenseArray::new(w.shape().clone(), w_data).expect("weight shape matches grad");

        env.optim_state.buffers.insert(m_key, m_new);
        env.optim_state.buffers.insert(v_key, v_new);
        env.set(name.clone(), w_new);
    }
    Ok(DenseArray::from_scalar(0.0))
}
