//! The `grad(expr, wrt)` built-in: reverse-mode autograd over
//! a tree-walked mini-evaluator that lifts array-valued operations
//! onto an autograd tape.

use std::collections::HashMap;
use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};
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
    // Saga 14 step 006: when the grad call is inside a
    // `device("mlx") { }` block, re-run the forward pass through
    // `mlpl-mlx-rt` so the tape's per-node values are MLX-rounded.
    // The CPU backward formulas in `mlpl-autograd::ops` then
    // operate on those values, producing gradients that match the
    // all-CPU path within fp32 tolerance. See
    // `device::materialize_tape_on_mlx` for the design rationale.
    if env.device() == "mlx" {
        crate::device::materialize_tape_on_mlx(&tape);
    }
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
    let arity = |e: usize| -> Result<(), EvalError> {
        (args.len() == e).then_some(()).ok_or(EvalError::BadArity {
            func: name.into(),
            expected: e,
            got: args.len(),
        })
    };
    if let Some(op) = unary_tensor_op(name) {
        arity(1)?;
        return Ok(op(&eval_tensor_expr(&args[0], env, tape, params)?));
    }
    if name == "matmul" {
        arity(2)?;
        let a = eval_tensor_expr(&args[0], env, tape, params)?;
        let b = eval_tensor_expr(&args[1], env, tape, params)?;
        return Ok(a.matmul(&b));
    }
    if name == "apply" {
        arity(2)?;
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
        return mlpl_models_tape::apply_model_tape(&model, x, tape, params)
            .map_err(EvalError::from);
    }
    if name == "cross_entropy" {
        arity(2)?;
        let l = eval_tensor_expr(&args[0], env, tape, params)?;
        let t = crate::eval::eval_expr(&args[1], env, &mut None)?.into_array()?;
        let idx = mlpl_models_tape::validate_cross_entropy_targets(&l.value(), &t)?;
        return Ok(l.cross_entropy(idx));
    }
    if name == "patchify" {
        arity(2)?;
        let x = eval_tensor_expr(&args[0], env, tape, params)?;
        let p = tape_scalar_usize(&args[1], env, "patchify: patch_size")?;
        return Ok(x.patchify(p));
    }
    if name == "concat" {
        arity(3)?;
        let a = eval_tensor_expr(&args[0], env, tape, params)?;
        let b = eval_tensor_expr(&args[1], env, tape, params)?;
        let axis = tape_scalar_usize(&args[2], env, "concat: axis")?;
        return Ok(a.concat(&b, axis));
    }
    if name == "take" {
        arity(3)?;
        let x = eval_tensor_expr(&args[0], env, tape, params)?;
        let axis = tape_scalar_usize(&args[1], env, "take: axis")?;
        let idx = tape_scalar_usize(&args[2], env, "take: idx")?;
        return Ok(x.take(axis, idx));
    }
    if name == "reshape" {
        arity(2)?;
        let x = eval_tensor_expr(&args[0], env, tape, params)?;
        let dims = eval_shape_dims(
            match &args[1] {
                Expr::ArrayLit(elems, _) => elems,
                _ => {
                    return Err(EvalError::Unsupported(
                        "grad: reshape's second argument must be an [int, ...] literal".into(),
                    ));
                }
            },
            env,
        )?;
        return Ok(x.reshape(mlpl_array::Shape::new(dims)));
    }
    Err(EvalError::Unsupported(format!(
        "grad: function '{name}' not supported inside grad()"
    )))
}

fn tape_scalar_usize(arg: &Expr, env: &mut Environment, what: &str) -> Result<usize, EvalError> {
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

fn unary_tensor_op(name: &str) -> Option<fn(&Tensor) -> Tensor> {
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
