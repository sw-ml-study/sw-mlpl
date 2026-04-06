//! AST-walking evaluator.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::{BinOpKind, Expr};

use crate::env::Environment;
use crate::error::EvalError;

/// Evaluate a program (list of statements). Returns the last result.
pub fn eval_program(stmts: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    if stmts.is_empty() {
        return Err(EvalError::EmptyInput);
    }
    let mut result = None;
    for stmt in stmts {
        result = Some(eval_expr(stmt, env)?);
    }
    result.ok_or(EvalError::EmptyInput)
}

/// Evaluate a single expression.
pub(crate) fn eval_expr(expr: &Expr, env: &mut Environment) -> Result<DenseArray, EvalError> {
    match expr {
        Expr::IntLit(n, _) => Ok(DenseArray::from_scalar(*n as f64)),
        Expr::FloatLit(f, _) => Ok(DenseArray::from_scalar(*f)),
        Expr::Ident(name, _) => env
            .get(name)
            .cloned()
            .ok_or_else(|| EvalError::UndefinedVariable(name.clone())),
        Expr::ArrayLit(elems, _) => eval_array_lit(elems, env),
        Expr::UnaryNeg { operand, .. } => {
            let val = eval_expr(operand, env)?;
            let neg = DenseArray::from_scalar(-1.0);
            Ok(neg.apply_binop(&val, |a, b| a * b)?)
        }
        Expr::Assign { name, value, .. } => {
            let val = eval_expr(value, env)?;
            env.set(name.clone(), val.clone());
            Ok(val)
        }
        Expr::BinOp { op, lhs, rhs, .. } => {
            let l = eval_expr(lhs, env)?;
            let r = eval_expr(rhs, env)?;
            let f: fn(f64, f64) -> f64 = match op {
                BinOpKind::Add => |a, b| a + b,
                BinOpKind::Sub => |a, b| a - b,
                BinOpKind::Mul => |a, b| a * b,
                BinOpKind::Div => |a, b| a / b,
            };
            Ok(l.apply_binop(&r, f)?)
        }
        Expr::FnCall { name, args, .. } => {
            let evaluated_args: Vec<mlpl_array::DenseArray> = args
                .iter()
                .map(|a| eval_expr(a, env))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(mlpl_runtime::call_builtin(name, evaluated_args)?)
        }
    }
}

/// Evaluate an array literal.
fn eval_array_lit(elems: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    if elems.is_empty() {
        return Ok(DenseArray::from_vec(vec![]));
    }

    let evaluated: Vec<DenseArray> = elems
        .iter()
        .map(|e| eval_expr(e, env))
        .collect::<Result<Vec<_>, _>>()?;

    // Check if all elements are scalars (flat array)
    if evaluated.iter().all(|a| a.rank() == 0) {
        let data: Vec<f64> = evaluated.iter().map(|a| a.data()[0]).collect();
        return Ok(DenseArray::from_vec(data));
    }

    // Nested: all elements must have the same shape
    let inner_shape = evaluated[0].shape().clone();
    let rows = evaluated.len();
    let mut data = Vec::with_capacity(rows * inner_shape.elem_count());
    for arr in &evaluated {
        data.extend_from_slice(arr.data());
    }
    let mut dims = vec![rows];
    dims.extend_from_slice(inner_shape.dims());
    Ok(DenseArray::new(Shape::new(dims), data)?)
}
