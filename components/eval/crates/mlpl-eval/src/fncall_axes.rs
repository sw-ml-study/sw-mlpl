//! FnCall dispatch family: axis-aware ops.
//!
//! `reshape_labeled(x, dims, labels)`, `label(x, labels)` /
//! `relabel(x, labels)`, and `labels(x)` -- axis-label constructors.
//! Lifted out of `eval::eval_expr` for saga 33 step 023.
//!
//! The name argument is EVALUATED and resolved through the shared
//! `crate::axis_adapter::axis_names_of`, so a bracketed list of names
//! (`["a", "b"]`), an equivalent comma-string (`"a,b"`), or a variable
//! holding either are all accepted -- the same forms `reduce` takes.

use mlpl_array::Shape;
use mlpl_array_ops_shape::prelude::*;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "reshape_labeled" => Some(eval_reshape_labeled(args, env, trace)),
        "label" | "relabel" => Some(eval_label_relabel(name, args, env, trace)),
        "labels" => Some(eval_labels(args, env, trace)),
        "disp" => Some(eval_disp(args, env, trace)),
        _ => None,
    }
}

/// `disp(a)` -- ASCII box diagram of `a` showing its rank, shape, and
/// depth. Returns a `Value::Str` the REPL prints (or the web playground
/// renders) verbatim.
fn eval_disp(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "disp".into(),
            expected: 1,
            got: args.len(),
        });
    }
    // Arrays render boxed; strings, lists, records, Results, etc. render
    // via their own Display so `disp` never rejects a non-array value
    // (user report 2026-08-13).
    match eval_expr(&args[0], env, trace)? {
        Value::Array(a) => Ok(Value::Str(mlpl_array::box_display(&a))),
        other => Ok(Value::Str(format!("{other}"))),
    }
}

fn eval_reshape_labeled(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 3 {
        return Err(EvalError::BadArity {
            func: "reshape_labeled".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let source = eval_expr(&args[0], env, trace)?.into_array()?;
    let shape_arr = eval_expr(&args[1], env, trace)?.into_array()?;
    let names =
        crate::axis_adapter::axis_names_of(&eval_expr(&args[2], env, trace)?, "reshape_labeled")?;
    let dims: Vec<usize> = shape_arr.data().iter().map(|&d| d as usize).collect();
    let reshaped = source.reshape(Shape::new(dims))?;
    Ok(Value::Array(reshaped.with_labels(names.0)?))
}

fn eval_label_relabel(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    let arr = eval_expr(&args[0], env, trace)?.into_array()?;
    let names = crate::axis_adapter::axis_names_of(&eval_expr(&args[1], env, trace)?, name)?;
    Ok(Value::Array(arr.with_labels(names.0)?))
}

fn eval_labels(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "labels".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let arr = eval_expr(&args[0], env, trace)?.into_array()?;
    let parts: Vec<String> = match arr.labels() {
        Some(lbls) => lbls.iter().map(|l| l.clone().unwrap_or_default()).collect(),
        None => (0..arr.rank()).map(|_| String::new()).collect(),
    };
    Ok(Value::Str(parts.join(",")))
}
