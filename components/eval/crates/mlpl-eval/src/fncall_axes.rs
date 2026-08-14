//! FnCall dispatch family: axis-aware ops.
//!
//! `reshape_labeled(x, dims, labels)`, `label(x, labels)` /
//! `relabel(x, labels)`, and `labels(x)` -- all walk axis labels
//! from a bracketed string-literal list. Lifted out of
//! `eval::eval_expr` for saga 33 step 023.
//!
//! `parse_axis_names` is the shared helper used by the labelling
//! constructors; lifting it here keeps the two call sites from
//! duplicating the per-element type check.

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
    let Expr::ArrayLit(label_elems, _) = &args[2] else {
        return Err(EvalError::Unsupported(
            "reshape_labeled: third argument must be a bracketed list of string literals".into(),
        ));
    };
    let labels = parse_axis_names(label_elems, "reshape_labeled")?;
    let source = eval_expr(&args[0], env, trace)?.into_array()?;
    let shape_arr = eval_expr(&args[1], env, trace)?.into_array()?;
    let dims: Vec<usize> = shape_arr.data().iter().map(|&d| d as usize).collect();
    let reshaped = source.reshape(Shape::new(dims))?;
    Ok(Value::Array(reshaped.with_labels(labels)?))
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
    let Expr::ArrayLit(label_elems, _) = &args[1] else {
        return Err(EvalError::Unsupported(format!(
            "{name}: second argument must be a bracketed list of string literals"
        )));
    };
    let labels = parse_axis_names(label_elems, name)?;
    let arr = eval_expr(&args[0], env, trace)?.into_array()?;
    Ok(Value::Array(arr.with_labels(labels)?))
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

fn parse_axis_names(elems: &[Expr], func: &str) -> Result<Vec<Option<String>>, EvalError> {
    let mut labels = Vec::with_capacity(elems.len());
    for e in elems {
        let Expr::StrLit(s, _) = e else {
            return Err(EvalError::Unsupported(format!(
                "{func}: axis names must be string literals"
            )));
        };
        labels.push(Some(s.clone()));
    }
    Ok(labels)
}
