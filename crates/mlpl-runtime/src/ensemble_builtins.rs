//! Saga 20 step 003: `argtop_k` + `scatter` array-utility builtins.
//!
//! These two builtins close the ensemble loop of the Neural
//! Thickets workflow: `scatter` accumulates per-variant scores
//! into a rank-1 buffer inside a `repeat N { ... }` loop, and
//! `argtop_k` picks the K best indices to ensemble.
//!
//! See `contracts/eval-contract/argtop-k.md` and
//! `contracts/eval-contract/scatter.md`.

use std::cmp::Ordering;

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;

/// Dispatch `argtop_k` and `scatter`. Returns `None` when `name` is
/// not one of them, so the outer `call_builtin` dispatch chain can
/// continue to the next module.
pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "argtop_k" => Some(builtin_argtop_k(args)),
        "scatter" => Some(builtin_scatter(args)),
        _ => None,
    }
}

/// `argtop_k(values, k)` -- return the k indices of the largest
/// entries in a rank-1 `values` vector. Sorted by descending value,
/// ties broken by lower index first (stable).
fn builtin_argtop_k(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: "argtop_k".into(),
            expected: 2,
            got: args.len(),
        });
    }
    if args[0].rank() != 1 {
        return Err(RuntimeError::InvalidArgument {
            func: "argtop_k".into(),
            reason: format!(
                "values must be a rank-1 vector, got rank {}",
                args[0].rank()
            ),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: "argtop_k".into(),
            reason: "k must be a scalar".into(),
        });
    }
    let k_f = args[1].data()[0];
    if !k_f.is_finite() || k_f < 0.0 || k_f.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: "argtop_k".into(),
            reason: format!("k must be a non-negative integer, got {k_f}"),
        });
    }
    let k = k_f as usize;
    let values = args[0].data();
    if k > values.len() {
        return Err(RuntimeError::InvalidArgument {
            func: "argtop_k".into(),
            reason: format!("k = {k} exceeds the vector length {}", values.len()),
        });
    }
    // Stable descending sort by value: tie-break defaults to lower
    // original index first because `sort_by` is stable and we pair
    // indices with values in ascending-index order before sorting.
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let out_data: Vec<f64> = indexed.iter().take(k).map(|(i, _)| *i as f64).collect();
    Ok(DenseArray::new(Shape::new(vec![k]), out_data)?)
}

/// `scatter(buffer, index, value)` -- return a copy of `buffer`
/// with the single entry at `index` replaced by `value`.
fn builtin_scatter(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(RuntimeError::ArityMismatch {
            func: "scatter".into(),
            expected: 3,
            got: args.len(),
        });
    }
    if args[0].rank() != 1 {
        return Err(RuntimeError::InvalidArgument {
            func: "scatter".into(),
            reason: format!(
                "buffer must be a rank-1 vector, got rank {}",
                args[0].rank()
            ),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: "scatter".into(),
            reason: "index must be a scalar".into(),
        });
    }
    if args[2].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: "scatter".into(),
            reason: "value must be a scalar".into(),
        });
    }
    let len = args[0].data().len();
    let idx_f = args[1].data()[0];
    if !idx_f.is_finite() || idx_f < 0.0 || idx_f.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: "scatter".into(),
            reason: format!("index must be a non-negative integer, got {idx_f}"),
        });
    }
    let idx = idx_f as usize;
    if idx >= len {
        return Err(RuntimeError::InvalidArgument {
            func: "scatter".into(),
            reason: format!("index {idx} out of bounds for buffer of length {len}"),
        });
    }
    let mut out_data = args[0].data().to_vec();
    out_data[idx] = args[2].data()[0];
    Ok(DenseArray::new(args[0].shape().clone(), out_data)?)
}
