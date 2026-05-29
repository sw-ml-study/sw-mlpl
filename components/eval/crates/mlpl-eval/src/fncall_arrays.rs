//! FnCall dispatch family: array-valued builtins with inline
//! logic (matmul / cross_entropy / labeled reductions / load_images).
//!
//! Lifted out of `eval::eval_expr` for saga 33 step 023. Each
//! helper either calls into the appropriate `crate::*` module or
//! drives `mlpl_runtime::call_builtin` directly when the runtime
//! op is generic enough that a dedicated wrapper would be cruft.

use mlpl_array::{ArrayError, DenseArray};
use mlpl_array_ops_matmul::prelude::*;
use mlpl_core::Span;
use mlpl_parser::Expr;
use mlpl_trace::{Trace, TraceValue};

use crate::env::Environment;
use crate::eval::eval_expr;
use crate::eval_ops::labeled_shape_of;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    span: &Span,
) -> Option<Result<Value, EvalError>> {
    if name == "cross_entropy" && args.len() == 2 {
        return Some(eval_cross_entropy(args, env, trace));
    }
    if name == "matmul" && args.len() == 2 {
        return Some(eval_matmul(args, env, trace, span));
    }
    if matches!(name, "reduce_add" | "reduce_mul" | "argmax" | "softmax")
        && args.len() == 2
        && matches!(&args[1], Expr::StrLit(_, _))
    {
        return Some(eval_reduce_labeled(name, args, env, trace));
    }
    if name == "load_images" {
        return Some(eval_load_images(args, env));
    }
    None
}

fn eval_cross_entropy(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    crate::type_errors::check_logit_consumer("cross_entropy", &args[0], env)?;
    let logits = eval_expr(&args[0], env, trace)?.into_array()?;
    let targets = eval_expr(&args[1], env, trace)?.into_array()?;
    mlpl_models_tape::validate_cross_entropy_targets(&logits, &targets)?;
    let result = mlpl_runtime::call_builtin("cross_entropy", vec![logits, targets])?;
    Ok(Value::Array(result))
}

fn eval_matmul(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    span: &Span,
) -> Result<Value, EvalError> {
    let l = eval_expr(&args[0], env, trace)?.into_array()?;
    let r = eval_expr(&args[1], env, trace)?.into_array()?;
    let result = l.matmul(&r).map_err(|e| match e {
        ArrayError::ShapeMismatch { .. } | ArrayError::LabelMismatch { .. } => {
            EvalError::ShapeMismatch {
                op: "matmul".into(),
                expected: labeled_shape_of(&l),
                actual: labeled_shape_of(&r),
            }
        }
        other => other.into(),
    })?;
    let inputs = vec![TraceValue::from_array(&l), TraceValue::from_array(&r)];
    crate::fncall_trace::push_array_event(trace, "matmul", span, inputs, &result);
    Ok(Value::Array(result))
}

fn eval_reduce_labeled(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let Expr::StrLit(axis_name, _) = &args[1] else {
        unreachable!("try_dispatch matched StrLit");
    };
    let err =
        |reason| EvalError::Unsupported(format!("{name}: axis name \"{axis_name}\" {reason}"));
    let arr = eval_expr(&args[0], env, trace)?.into_array()?;
    let labels = arr
        .labels()
        .ok_or_else(|| err("requires a labeled array"))?;
    let axis = labels
        .iter()
        .position(|l| l.as_deref() == Some(axis_name.as_str()))
        .ok_or_else(|| err("not found in labels"))?;
    let axis_arr = DenseArray::from_scalar(axis as f64);
    let result = mlpl_runtime::call_builtin(name, vec![arr, axis_arr])?;
    Ok(Value::Array(result))
}

fn eval_load_images(args: &[Expr], env: &mut Environment) -> Result<Value, EvalError> {
    let err = |detail: &str| EvalError::Unsupported(format!("load_images: {detail}"));
    let [a0, a1] = args else {
        return Err(EvalError::BadArity {
            func: "load_images".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let Expr::StrLit(dir, _) = a0 else {
        return Err(err("arg 0 must be a directory string"));
    };
    let Expr::ArrayLit(dims, _) = a1 else {
        return Err(err("arg 1 must be a [H, W] array literal"));
    };
    if dims.len() != 2 {
        return Err(err(&format!(
            "expected [H, W], got {} elements",
            dims.len()
        )));
    }
    let parse_dim = |e: &Expr| match e {
        Expr::IntLit(n, _) if *n >= 0 => Ok(*n as usize),
        _ => Err(err("[H, W] entries must be non-negative integers")),
    };
    crate::loader::eval_load_images(env, dir, parse_dim(&dims[0])?, parse_dim(&dims[1])?)
}
