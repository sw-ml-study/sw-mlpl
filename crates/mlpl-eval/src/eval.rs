//! AST-walking evaluator.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::{BinOpKind, Expr};
use mlpl_trace::{Trace, TraceEvent, TraceValue};

use crate::env::Environment;
use crate::error::EvalError;

/// Evaluate a program (list of statements). Returns the last result.
pub fn eval_program(stmts: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    if stmts.is_empty() {
        return Err(EvalError::EmptyInput);
    }
    let mut trace: Option<&mut Trace> = None;
    let mut result = None;
    for stmt in stmts {
        result = Some(eval_expr(stmt, env, &mut trace)?);
    }
    result.ok_or(EvalError::EmptyInput)
}

/// Evaluate a program with tracing enabled.
pub fn eval_program_traced(
    stmts: &[Expr],
    env: &mut Environment,
    trace: &mut Trace,
) -> Result<DenseArray, EvalError> {
    if stmts.is_empty() {
        return Err(EvalError::EmptyInput);
    }
    let mut trace: Option<&mut Trace> = Some(trace);
    let mut result = None;
    for stmt in stmts {
        result = Some(eval_expr(stmt, env, &mut trace)?);
    }
    result.ok_or(EvalError::EmptyInput)
}

fn eval_expr(
    expr: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    let (op_name, inputs, result) = match expr {
        Expr::IntLit(n, _) => ("literal", vec![], DenseArray::from_scalar(*n as f64)),
        Expr::FloatLit(f, _) => ("literal", vec![], DenseArray::from_scalar(*f)),
        Expr::Ident(name, _) => {
            let r = env
                .get(name)
                .cloned()
                .ok_or_else(|| EvalError::UndefinedVariable(name.clone()))?;
            ("ident", vec![], r)
        }
        Expr::ArrayLit(elems, _) => ("array_lit", vec![], eval_array_lit(elems, env, trace)?),
        Expr::UnaryNeg { operand, .. } => {
            let val = eval_expr(operand, env, trace)?;
            let r = DenseArray::from_scalar(-1.0).apply_binop(&val, |a, b| a * b)?;
            ("negate", vec![TraceValue::from_array(&val)], r)
        }
        Expr::Assign { name, value, .. } => {
            let val = eval_expr(value, env, trace)?;
            env.set(name.clone(), val.clone());
            ("assign", vec![TraceValue::from_array(&val)], val)
        }
        Expr::BinOp { op, lhs, rhs, .. } => eval_binop(op, lhs, rhs, env, trace)?,
        Expr::FnCall { name, args, .. } => eval_fncall(name, args, env, trace)?,
        Expr::Repeat { count, body, .. } => {
            let n_arr = eval_expr(count, env, trace)?;
            if n_arr.rank() != 0 {
                return Err(EvalError::InvalidRepeatCount);
            }
            let n = n_arr.data()[0] as usize;
            let mut r = DenseArray::from_scalar(0.0);
            for _ in 0..n {
                for stmt in body {
                    r = eval_expr(stmt, env, trace)?;
                }
            }
            ("repeat", vec![], r)
        }
    };
    record_trace(trace, op_name, expr.span(), inputs, &result);
    Ok(result)
}

fn record_trace(
    trace: &mut Option<&mut Trace>,
    op: &str,
    span: mlpl_core::Span,
    inputs: Vec<TraceValue>,
    result: &DenseArray,
) {
    if let Some(t) = trace.as_mut() {
        let seq = t.events().len() as u64;
        t.push(TraceEvent {
            seq,
            op: op.into(),
            span,
            inputs,
            output: TraceValue::from_array(result),
        });
    }
}

fn eval_binop(
    op: &BinOpKind,
    lhs: &Expr,
    rhs: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let l = eval_expr(lhs, env, trace)?;
    let r = eval_expr(rhs, env, trace)?;
    let (name, f): (&str, fn(f64, f64) -> f64) = match op {
        BinOpKind::Add => ("add", |a, b| a + b),
        BinOpKind::Sub => ("sub", |a, b| a - b),
        BinOpKind::Mul => ("mul", |a, b| a * b),
        BinOpKind::Div => ("div", |a, b| a / b),
    };
    let inputs = vec![TraceValue::from_array(&l), TraceValue::from_array(&r)];
    let result = l.apply_binop(&r, f)?;
    Ok((name, inputs, result))
}

fn eval_fncall(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let evaluated: Vec<DenseArray> = args
        .iter()
        .map(|a| eval_expr(a, env, trace))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs: Vec<TraceValue> = evaluated.iter().map(TraceValue::from_array).collect();
    let result = mlpl_runtime::call_builtin(name, evaluated)?;
    Ok(("fncall", inputs, result))
}

fn eval_array_lit(
    elems: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    if elems.is_empty() {
        return Ok(DenseArray::from_vec(vec![]));
    }
    let evaluated: Vec<DenseArray> = elems
        .iter()
        .map(|e| eval_expr(e, env, trace))
        .collect::<Result<Vec<_>, _>>()?;
    if evaluated.iter().all(|a| a.rank() == 0) {
        let data: Vec<f64> = evaluated.iter().map(|a| a.data()[0]).collect();
        return Ok(DenseArray::from_vec(data));
    }
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
