//! Evaluation helpers for binops, function calls, and array literals.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::{BinOpKind, Expr};
use mlpl_trace::{Trace, TraceValue};

use crate::env::Environment;
use crate::error::EvalError;
use crate::eval::eval_expr;
use crate::value::Value;

pub(crate) fn eval_binop(
    op: &BinOpKind,
    lhs: &Expr,
    rhs: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let l = eval_expr(lhs, env, trace)?.into_array()?;
    let r = eval_expr(rhs, env, trace)?.into_array()?;
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

pub(crate) fn eval_fncall(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let evaluated: Vec<DenseArray> = args
        .iter()
        .map(|a| eval_expr(a, env, trace).and_then(Value::into_array))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs: Vec<TraceValue> = evaluated.iter().map(TraceValue::from_array).collect();
    let result = mlpl_runtime::call_builtin(name, evaluated)?;
    Ok(("fncall", inputs, result))
}

pub(crate) fn eval_array_lit(
    elems: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    if elems.is_empty() {
        return Ok(DenseArray::from_vec(vec![]));
    }
    let evaluated: Vec<DenseArray> = elems
        .iter()
        .map(|e| eval_expr(e, env, trace).and_then(Value::into_array))
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
