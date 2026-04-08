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
    match expr {
        Expr::IntLit(n, _) => Ok(const_leaf(tape, DenseArray::from_scalar(*n as f64))),
        Expr::FloatLit(f, _) => Ok(const_leaf(tape, DenseArray::from_scalar(*f))),
        Expr::Ident(name, _) => {
            if let Some(t) = params.get(name) {
                return Ok(t.clone());
            }
            let arr = env
                .get(name)
                .cloned()
                .ok_or_else(|| EvalError::UndefinedVariable(name.clone()))?;
            Ok(const_leaf(tape, arr))
        }
        Expr::ArrayLit(_, _) => {
            let arr = crate::eval_ops::eval_array_lit(array_lit_elems(expr), env, &mut None)?;
            Ok(const_leaf(tape, arr))
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
            Ok(const_leaf(tape, DenseArray::zeros(Shape::new(dims))))
        }
        Expr::Assign { .. } | Expr::Repeat { .. } | Expr::StrLit(_, _) => Err(
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
    if let Some(op) = unary_tensor_op(name) {
        check_arity(name, args, 1)?;
        let a = eval_tensor_expr(&args[0], env, tape, params)?;
        return Ok(op(&a));
    }
    if name == "matmul" {
        check_arity(name, args, 2)?;
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

fn check_arity(name: &str, args: &[Expr], expected: usize) -> Result<(), EvalError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(EvalError::BadArity {
            func: name.into(),
            expected,
            got: args.len(),
        })
    }
}

fn const_leaf(tape: &Rc<Tape>, value: DenseArray) -> Tensor {
    Tensor::leaf(Rc::clone(tape), value, false)
}

fn array_lit_elems(expr: &Expr) -> &[Expr] {
    match expr {
        Expr::ArrayLit(e, _) => e,
        _ => unreachable!(),
    }
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
