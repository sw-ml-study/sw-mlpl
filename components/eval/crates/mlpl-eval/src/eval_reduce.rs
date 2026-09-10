//! Higher-order `reduce(:op, x[, axis])` dispatch.
//!
//! The first argument is a `Value::BuiltinRef` naming a
//! binary operation: `:add`, `:mul` (or `:+` / `:*`),
//! `:min`, `:max`, `:and`, `:or`. The fixed-name builtins
//! `reduce_add` and `reduce_mul` continue to exist as
//! direct shorthands.
//!
//! The optional third argument selects which axes collapse:
//! a scalar (`2`) or numeric vector (`[2, 3]`) names axis
//! POSITIONS, and a string names axis LABELS -- one, or
//! several comma-separated (`"channel"`,
//! `"channel,kernel_y,kernel_x"`). Multiple axes are removed
//! high-index first, so a convolution can contract its whole
//! receptive field in one call. With no third argument the
//! array reduces to a scalar.
//!
//! Why `:op` instead of a string or bare-name reference?
//! MLPL has no first-class functions in v0.19; the colon-
//! prefixed BuiltinRef occupies a separate syntactic
//! namespace from variables, so a user's `add = 42` cannot
//! shadow `:add`. The same surface stays valid when
//! first-class functions land -- `:foo` will lift to a
//! `Value::Function::Builtin("foo")` cleanly.

use mlpl_array::DenseArray;
use mlpl_array_ops_reduce::prelude::*;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

type BinOp = fn(f64, f64) -> f64;

/// Map an operator name to its `(identity, binary op)`. The ops are
/// non-capturing closures (they coerce to `BinOp` fn pointers), so the
/// whole reduce-operator table lives in one place.
fn dispatch(op_name: &str) -> Option<(f64, BinOp)> {
    match op_name {
        "add" | "+" => Some((0.0, |a, b| a + b)),
        "mul" | "*" => Some((1.0, |a, b| a * b)),
        "min" => Some((f64::INFINITY, |a, b| if b < a { b } else { a })),
        "max" => Some((f64::NEG_INFINITY, |a, b| if b > a { b } else { a })),
        "and" => Some((1.0, |a, b| if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 })),
        "or" => Some((0.0, |a, b| if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 })),
        _ => None,
    }
}

pub(crate) fn eval_reduce(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(EvalError::BadArity {
            func: "reduce".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let op_name = match eval_expr(&args[0], env, trace)? {
        Value::BuiltinRef { name } => name,
        _ => {
            return Err(EvalError::Unsupported(
                "reduce: first argument must be a builtin reference like :add, :max, :+, :* (use the colon-prefixed form)".into(),
            ));
        }
    };
    let (identity, op) = dispatch(&op_name).ok_or_else(|| {
        EvalError::Unsupported(format!(
            "reduce: unknown op ':{op_name}' (curated set: :add/:+, :mul/:*, :min, :max, :and, :or)"
        ))
    })?;
    let arr = eval_expr(&args[1], env, trace)?.into_array()?;
    let result = if args.len() == 3 {
        let axes = resolve_axes(eval_expr(&args[2], env, trace)?, &arr)?;
        reduce_over(arr, axes, identity, op)?
    } else {
        DenseArray::from_scalar(arr.data().iter().copied().fold(identity, op))
    };
    Ok(Value::Array(result))
}

/// Resolve the axis argument to positional axes. A numeric scalar or
/// vector is a list of positions directly (`2`, `[2, 3]`); a string is
/// one or more comma-separated axis LABELS (`"channel"`,
/// `"channel,kernel_y"`), each looked up in the array's labels.
fn resolve_axes(axis: Value, arr: &DenseArray) -> Result<Vec<usize>, EvalError> {
    let s = match axis {
        Value::Str(s) => s,
        other => {
            return Ok(other
                .into_array()?
                .data()
                .iter()
                .map(|&v| v as usize)
                .collect());
        }
    };
    let labels = arr.labels().ok_or_else(|| {
        EvalError::Unsupported("reduce: named axis but the array has no labels".into())
    })?;
    s.split(',')
        .map(|raw| {
            let name = raw.trim();
            labels
                .iter()
                .position(|l| l.as_deref() == Some(name))
                .ok_or_else(|| {
                    EvalError::Unsupported(format!("reduce: no axis labeled \"{name}\""))
                })
        })
        .collect()
}

/// Reduce `arr` over every axis in `axes`, removing them high-index
/// first so earlier removals do not shift the remaining axis positions.
fn reduce_over(
    arr: DenseArray,
    mut axes: Vec<usize>,
    identity: f64,
    op: BinOp,
) -> Result<DenseArray, EvalError> {
    axes.sort_unstable();
    axes.dedup();
    axes.into_iter()
        .rev()
        .try_fold(arr, |acc, ax| acc.reduce_axis(ax, identity, op))
        .map_err(EvalError::from)
}
