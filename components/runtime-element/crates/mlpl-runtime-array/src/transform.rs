//! Structural transforms: builtins that return a re-shaped or
//! re-ordered array. Split out of `shape.rs`, which now holds only
//! the pure introspection queries (`shape`, `rank`, `depth`, `size`,
//! `tally`). Introspection asks about structure; transforms change
//! it -- two responsibilities, two modules.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_compose::prelude::*;
use mlpl_array_ops_shape::prelude::*;
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;

pub(crate) fn reshape(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    let dims: Vec<usize> = args[1].data().iter().map(|&d| d as usize).collect();
    Ok(args[0].reshape(Shape::new(dims))?)
}

pub(crate) fn transpose(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    Ok(args[0].transpose())
}

pub(crate) fn transpose_axes(
    name: &str,
    args: Vec<DenseArray>,
) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    let bad = |reason: String| RuntimeError::InvalidArgument {
        func: name.into(),
        reason,
    };
    let rank = args[0].rank();
    let perm: Vec<usize> = args[1]
        .data()
        .iter()
        .map(|&p| {
            if p.fract() != 0.0 || p < 0.0 {
                Err(bad(format!("perm entries must be axis numbers 0..{rank}")))
            } else {
                Ok(p as usize)
            }
        })
        .collect::<Result<_, _>>()?;
    args[0].transpose_axes(&perm).map_err(|e| {
        bad(format!(
            "perm must name each of the {rank} axes exactly once (0-based) -- {e}"
        ))
    })
}

pub(crate) fn rotate(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(arity_err(name, 3, args.len()));
    }
    let k = scalar_i64(name, &args[1], "k")?;
    let axis = crate::slice::scalar_usize(name, &args[2], "axis")?;
    Ok(args[0].rotate(k, axis)?)
}

/// Signed-integer scalar argument (rotate's `k` may be negative;
/// MLPL spells that `0 - 1` since there is no unary minus).
fn scalar_i64(name: &str, arr: &DenseArray, what: &str) -> Result<i64, RuntimeError> {
    if arr.rank() != 0 || arr.data()[0].fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("{what} must be an integer scalar"),
        });
    }
    Ok(arr.data()[0] as i64)
}
