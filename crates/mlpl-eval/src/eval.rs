//! AST-walking evaluator.

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd::{Tape, Tensor};
use mlpl_parser::{Expr, TensorCtorKind};
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
    if let Expr::FnCall { name, args, span } = expr
        && name == "grad"
    {
        let result = crate::grad::eval_grad(args, env)?;
        if let Some(t) = trace.as_mut() {
            let seq = t.events().len() as u64;
            t.push(TraceEvent {
                seq,
                op: "grad".into(),
                span: *span,
                inputs: vec![],
                output: TraceValue::from_array(&result),
            });
        }
        return Ok(Value::Array(result));
    }
    if let Expr::FnCall { name, args, .. } = expr
        && name == "linear"
    {
        let model = crate::model_dispatch::eval_linear(args, env)?;
        return Ok(Value::Model(model));
    }
    if let Expr::FnCall { name, args, .. } = expr
        && name == "chain"
    {
        let model = crate::model_dispatch::eval_chain(args, env)?;
        return Ok(Value::Model(model));
    }
    if let Expr::FnCall { name, args, .. } = expr
        && name == "residual"
    {
        let model = crate::model_dispatch::eval_residual(args, env)?;
        return Ok(Value::Model(model));
    }
    if let Expr::FnCall { name, args, .. } = expr
        && name == "rms_norm"
    {
        let model = crate::model_dispatch::eval_rms_norm(args, env)?;
        return Ok(Value::Model(model));
    }
    if let Expr::FnCall { name, args, .. } = expr
        && name == "attention"
    {
        let model = crate::model_dispatch::eval_attention(args, env)?;
        return Ok(Value::Model(model));
    }
    if let Expr::FnCall { name, args, .. } = expr
        && let Some(kind) = crate::model_dispatch::activation_kind(name)
    {
        if !args.is_empty() {
            return Err(EvalError::BadArity {
                func: name.into(),
                expected: 0,
                got: args.len(),
            });
        }
        return Ok(Value::Model(crate::model::ModelSpec::Activation(kind)));
    }
    if let Expr::FnCall { name, args, .. } = expr
        && name == "apply"
    {
        let result = crate::model_dispatch::eval_apply(args, env, trace)?;
        return Ok(Value::Array(result));
    }
    if let Expr::FnCall { name, args, .. } = expr
        && name == "labels"
    {
        if args.len() != 1 {
            return Err(EvalError::BadArity {
                func: "labels".into(),
                expected: 1,
                got: args.len(),
            });
        }
        let arr = eval_expr(&args[0], env, trace)?.into_array()?;
        // Comma-join the per-axis labels, using an empty string for
        // positional axes. Saga 11.5 Phase 1 intentionally stops at
        // this stringly-typed encoding because MLPL has no
        // string-vector literal yet; Phase 2 introduces one alongside
        // the `label(x, [...])` builtin and can upgrade this return
        // type in lock-step. The current encoding is testable and
        // unambiguous for every rank >= 1: a rank-3 positional array
        // becomes `",,"`, a labeled rank-2 becomes `"seq,d_k"`, etc.
        // Rank-0 and rank-1-positional both render as `""`; the user
        // can call `rank(x)` to disambiguate.
        let parts: Vec<String> = match arr.labels() {
            Some(lbls) => lbls.iter().map(|l| l.clone().unwrap_or_default()).collect(),
            None => (0..arr.rank()).map(|_| String::new()).collect(),
        };
        return Ok(Value::Str(parts.join(",")));
    }
    if let Expr::FnCall { name, args, span } = expr
        && (name == "momentum_sgd" || name == "adam")
    {
        let result = if name == "momentum_sgd" {
            crate::grad::eval_momentum_sgd(args, env)?
        } else {
            crate::grad::eval_adam(args, env)?
        };
        if let Some(t) = trace.as_mut() {
            let seq = t.events().len() as u64;
            t.push(TraceEvent {
                seq,
                op: name.clone(),
                span: *span,
                inputs: vec![],
                output: TraceValue::from_array(&result),
            });
        }
        return Ok(Value::Array(result));
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
            let is_param_ctor = matches!(
                value.as_ref(),
                Expr::TensorCtor {
                    kind: TensorCtorKind::Param,
                    ..
                }
            );
            let v = eval_expr(value, env, trace)?;
            match v {
                Value::Model(m) => {
                    env.models.insert(name.clone(), m);
                    let placeholder = DenseArray::from_scalar(0.0);
                    ("assign_model", vec![], placeholder)
                }
                Value::Str(_) => {
                    return Err(EvalError::Unsupported(
                        "assigning string values is not supported".into(),
                    ));
                }
                Value::Array(val) => {
                    env.set(name.clone(), val.clone());
                    if is_param_ctor {
                        env.mark_param(name);
                    }
                    ("assign", vec![TraceValue::from_array(&val)], val)
                }
            }
        }
        Expr::BinOp { op, lhs, rhs, .. } => eval_binop(op, lhs, rhs, env, trace)?,
        Expr::FnCall { name, args, .. } => eval_fncall(name, args, env, trace)?,
        Expr::TensorCtor { kind, shape, .. } => eval_tensor_ctor(*kind, shape, env, trace)?,
        Expr::Repeat { count, body, .. } => eval_repeat(count, body, env, trace)?,
        Expr::Train { count, body, .. } => eval_train(count, body, env, trace)?,
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

fn eval_tensor_ctor(
    kind: TensorCtorKind,
    shape: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(&'static str, Vec<TraceValue>, DenseArray), EvalError> {
    let mut dims = Vec::with_capacity(shape.len());
    for dim_expr in shape {
        let arr = eval_expr(dim_expr, env, trace)?.into_array()?;
        if arr.rank() != 0 {
            return Err(EvalError::InvalidShapeDim);
        }
        let v = arr.data()[0];
        if v < 0.0 || v.fract() != 0.0 {
            return Err(EvalError::InvalidShapeDim);
        }
        dims.push(v as usize);
    }
    let zeros = DenseArray::zeros(Shape::new(dims));
    // Construct an autograd Tensor on a fresh tape. Step 005 will
    // route this through a tape stored in the environment so that
    // operations on the resulting array are recorded; for now we
    // simply return the underlying zero-initialized array.
    let tape = Tape::new();
    let _tensor = match kind {
        TensorCtorKind::Param => Tensor::param(tape, zeros.clone()),
        TensorCtorKind::Tensor => Tensor::leaf(tape, zeros.clone(), false),
    };
    let op_name = match kind {
        TensorCtorKind::Param => "param_ctor",
        TensorCtorKind::Tensor => "tensor_ctor",
    };
    Ok((op_name, vec![], zeros))
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

fn eval_train(
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
    let mut losses: Vec<f64> = Vec::with_capacity(n);
    let mut last = DenseArray::from_scalar(0.0);
    for i in 0..n {
        env.set("step".into(), DenseArray::from_scalar(i as f64));
        let mut step_val = DenseArray::from_scalar(0.0);
        for stmt in body {
            step_val = eval_expr(stmt, env, trace)?.into_array()?;
        }
        // Capture the body's final value as the per-step loss; if it
        // is non-scalar (e.g. a vector), reduce by mean for the loss
        // curve so callers can still rely on a scalar history.
        let scalar_loss = if step_val.rank() == 0 {
            step_val.data()[0]
        } else {
            let s: f64 = step_val.data().iter().sum();
            s / (step_val.data().len().max(1) as f64)
        };
        losses.push(scalar_loss);
        last = step_val;
    }
    let losses_arr = DenseArray::new(mlpl_array::Shape::new(vec![losses.len()]), losses)
        .expect("losses shape matches data");
    env.set("last_losses".into(), losses_arr);
    Ok(("train", vec![], last))
}
