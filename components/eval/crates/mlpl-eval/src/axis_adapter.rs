//! Adapter from a runtime `Value` to the shared `mlpl_axes::AxisSpec`, so
//! every axis-selecting builtin resolves axes the same way and the accepted
//! forms cannot drift. The three interchangeable forms:
//!
//! - `["channel", "kernel_y"]` -> a `Value::StrList` -> axes by NAME
//! - `"channel,kernel_y"`      -> a `Value::Str`     -> axes by NAME (sugar)
//! - `[2, 3]`                  -> a `Value::Array`   -> axes by POSITION

use mlpl_array::DenseArray;
use mlpl_axes::{AxisError, AxisSpec};
use mlpl_eval_types::{EvalError, Value, value_kind};

/// Turn an evaluated axis argument into an [`AxisSpec`]. Exhaustive over the
/// value kinds that can name axes; anything else is a typed error.
pub(crate) fn axis_spec_of(v: &Value, func: &str) -> Result<AxisSpec, EvalError> {
    match v {
        Value::StrList { items } => Ok(AxisSpec::Names(items.clone())),
        Value::Str(s) => Ok(AxisSpec::Names(
            s.split(',').map(|n| n.trim().to_string()).collect(),
        )),
        Value::Array(a) => Ok(AxisSpec::Indices(
            a.data().iter().map(|&v| v as usize).collect(),
        )),
        other => Err(EvalError::Unsupported(format!(
            "{func}: axis selector must be names [\"a\",\"b\"], a comma-string \"a,b\", or indices [0,1], got a {}",
            value_kind(other)
        ))),
    }
}

/// Resolve an evaluated axis argument to concrete axis indices against
/// `arr`, mapping the shared `AxisError` into this crate's `EvalError`.
pub(crate) fn resolve_axes(
    v: &Value,
    arr: &DenseArray,
    func: &str,
) -> Result<Vec<usize>, EvalError> {
    axis_spec_of(v, func)?
        .resolve(arr)
        .map_err(|e| map_axis_error(&e, func))
}

fn map_axis_error(e: &AxisError, func: &str) -> EvalError {
    let reason = match e {
        AxisError::NamedAxisButNoLabels => {
            format!("{func}: named axis but the array has no labels")
        }
        AxisError::NoAxisNamed(n) => format!("{func}: no axis labeled \"{n}\""),
        AxisError::IndexOutOfRank { index, rank } => {
            format!("{func}: axis {index} is out of range for rank {rank}")
        }
        AxisError::DuplicateAxis(a) => format!("{func}: axis {a} selected more than once"),
        _ => format!("{func}: invalid axis selection"),
    };
    EvalError::Unsupported(reason)
}
