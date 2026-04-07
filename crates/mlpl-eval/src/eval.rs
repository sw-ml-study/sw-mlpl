//! AST-walking evaluator.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::{Trace, TraceEvent, TraceValue};

use crate::env::Environment;
use crate::error::EvalError;
use crate::eval_ops::{eval_analysis_helper, eval_array_lit, eval_binop, eval_fncall, eval_svg};
use crate::value::Value;

/// Evaluate a program (list of statements). Returns the last result as an array.
///
/// If the final value is a string, returns `EvalError::ExpectedArray`.
/// Use `eval_program_value` to handle both arrays and strings.
pub fn eval_program(stmts: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    eval_program_value(stmts, env)?.into_array()
}

/// Evaluate a program and return the final value (array or string).
pub fn eval_program_value(stmts: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    run_program(stmts, env, None)
}

/// Evaluate a program with tracing enabled. Returns the final array.
pub fn eval_program_traced(
    stmts: &[Expr],
    env: &mut Environment,
    trace: &mut Trace,
) -> Result<DenseArray, EvalError> {
    run_program(stmts, env, Some(trace))?.into_array()
}

fn run_program(
    stmts: &[Expr],
    env: &mut Environment,
    mut trace: Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if stmts.is_empty() {
        return Err(EvalError::EmptyInput);
    }
    let mut result = None;
    for stmt in stmts {
        result = Some(eval_expr(stmt, env, &mut trace)?);
    }
    result.ok_or(EvalError::EmptyInput)
}

pub(crate) fn eval_expr(
    expr: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if let Expr::StrLit(s, _) = expr {
        return Ok(Value::Str(s.clone()));
    }
    if let Expr::FnCall { name, args, .. } = expr
        && name == "svg"
    {
        return eval_svg(args, env, trace).map(Value::Str);
    }
    if let Expr::FnCall { name, args, .. } = expr
        && let Some(result) = eval_analysis_helper(name, args, env, trace)
    {
        return result.map(Value::Str);
    }
    let (op_name, inputs, result) = match expr {
        Expr::IntLit(n, _) => ("literal", vec![], DenseArray::from_scalar(*n as f64)),
        Expr::FloatLit(f, _) => ("literal", vec![], DenseArray::from_scalar(*f)),
        Expr::StrLit(_, _) => unreachable!(),
        Expr::Ident(name, _) => {
            let r = env
                .get(name)
                .cloned()
                .ok_or_else(|| EvalError::UndefinedVariable(name.clone()))?;
            ("ident", vec![], r)
        }
        Expr::ArrayLit(elems, _) => ("array_lit", vec![], eval_array_lit(elems, env, trace)?),
        Expr::UnaryNeg { operand, .. } => {
            let val = eval_expr(operand, env, trace)?.into_array()?;
            let r = DenseArray::from_scalar(-1.0).apply_binop(&val, |a, b| a * b)?;
            ("negate", vec![TraceValue::from_array(&val)], r)
        }
        Expr::Assign { name, value, .. } => {
            let val = eval_expr(value, env, trace)?.into_array()?;
            env.set(name.clone(), val.clone());
            ("assign", vec![TraceValue::from_array(&val)], val)
        }
        Expr::BinOp { op, lhs, rhs, .. } => eval_binop(op, lhs, rhs, env, trace)?,
        Expr::FnCall { name, args, .. } => eval_fncall(name, args, env, trace)?,
        Expr::Repeat { count, body, .. } => eval_repeat(count, body, env, trace)?,
    };
    if let Some(t) = trace.as_mut() {
        let seq = t.events().len() as u64;
        t.push(TraceEvent {
            seq,
            op: op_name.into(),
            span: expr.span(),
            inputs,
            output: TraceValue::from_array(&result),
        });
    }
    Ok(Value::Array(result))
}

fn eval_repeat(
    count: &Expr,
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let n_arr = eval_expr(count, env, trace)?.into_array()?;
    if n_arr.rank() != 0 {
        return Err(EvalError::InvalidRepeatCount);
    }
    let n = n_arr.data()[0] as usize;
    let mut r = DenseArray::from_scalar(0.0);
    for _ in 0..n {
        for stmt in body {
            r = eval_expr(stmt, env, trace)?.into_array()?;
        }
    }
    Ok(("repeat", vec![], r))
}
