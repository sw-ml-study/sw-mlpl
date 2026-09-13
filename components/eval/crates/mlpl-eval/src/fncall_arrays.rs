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
    if name == "dedupe_rows" {
        return Some(crate::fncall_arrays_data::eval_dedupe_rows(
            args, env, trace,
        ));
    }
    if name == "kg_split" {
        return Some(crate::fncall_arrays_data::eval_kg_split(args, env, trace));
    }
    if name == "matmul" && args.len() == 2 {
        return Some(eval_matmul(args, env, trace, span));
    }
    if matches!(name, "reduce_add" | "reduce_mul" | "argmax" | "softmax")
        && args.len() == 2
        && axis_name_list(&args[1]).is_some()
    {
        return Some(eval_reduce_labeled(name, args, env, trace));
    }
    if name == "load_images" {
        return Some(crate::fncall_arrays_data::eval_load_images(args, env));
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

/// The axis NAME(s) in a labeled-reduce argument: a bracketed list of string
/// literals (`["c", "ky"]`) or a comma-string (`"c"`, `"c,ky"`), split into
/// names. `None` if it is not a name literal at all (a numeric `ArrayLit` like
/// `[1]` takes the positional path). The name spellings match `reduce`.
fn axis_name_list(arg: &Expr) -> Option<Vec<String>> {
    match arg {
        Expr::StrLit(s, _) => Some(s.split(',').map(|p| p.trim().to_string()).collect()),
        Expr::ArrayLit(elems, _) if !elems.is_empty() => {
            let mut out = Vec::with_capacity(elems.len());
            for e in elems {
                let Expr::StrLit(s, _) = e else { return None };
                out.push(s.clone());
            }
            Some(out)
        }
        _ => None,
    }
}

/// Resolve axis names to positions against the array's labels.
fn resolve_named_axes(
    name: &str,
    names: &[String],
    arr: &DenseArray,
) -> Result<Vec<usize>, EvalError> {
    let labels = arr.labels().ok_or_else(|| {
        EvalError::Unsupported(format!("{name}: named axes require a labeled array"))
    })?;
    names
        .iter()
        .map(|n| {
            labels
                .iter()
                .position(|l| l.as_deref() == Some(n.as_str()))
                .ok_or_else(|| EvalError::Unsupported(format!("{name}: no axis labeled \"{n}\"")))
        })
        .collect()
}

fn eval_reduce_labeled(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    // argmax/softmax operate on ONE axis; reduce_add/reduce_mul reduce over
    // every named axis (high-index first), matching reduce(:add/:mul, ...).
    let names = axis_name_list(&args[1]).expect("gate matched an axis-name literal");
    let arr = eval_expr(&args[0], env, trace)?.into_array()?;
    let mut axes = resolve_named_axes(name, &names, &arr)?;
    let scalar = |a: usize| DenseArray::from_scalar(a as f64);
    if matches!(name, "argmax" | "softmax") {
        let [ax] = axes[..] else {
            let n = axes.len();
            return Err(EvalError::Unsupported(format!(
                "{name}: takes a single axis, got {n}"
            )));
        };
        return Ok(Value::Array(mlpl_runtime::call_builtin(
            name,
            vec![arr, scalar(ax)],
        )?));
    }
    axes.sort_unstable();
    axes.dedup();
    let out = axes.into_iter().rev().try_fold(arr, |acc, ax| {
        mlpl_runtime::call_builtin(name, vec![acc, scalar(ax)])
    })?;
    Ok(Value::Array(out))
}
