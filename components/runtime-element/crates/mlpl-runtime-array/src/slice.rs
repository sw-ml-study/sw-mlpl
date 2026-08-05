use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_compose::prelude::*;
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;

pub(crate) fn patchify(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    let p = scalar_usize(name, &args[1], "patch_size")?;
    Ok(args[0].patchify(p)?)
}

pub(crate) fn concat(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(arity_err(name, 3, args.len()));
    }
    let axis = scalar_usize(name, &args[2], "axis")?;
    Ok(args[0].concat(&args[1], axis)?)
}

pub(crate) fn take(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(arity_err(name, 3, args.len()));
    }
    let axis = scalar_usize(name, &args[1], "axis")?;
    let idx = scalar_usize(name, &args[2], "idx")?;
    Ok(args[0].take(axis, idx)?)
}

pub(crate) fn scalar_usize(
    name: &str,
    arr: &DenseArray,
    what: &str,
) -> Result<usize, RuntimeError> {
    if arr.rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("{what} must be a scalar, got rank {}", arr.rank()),
        });
    }
    let v = arr.data()[0];
    if v < 0.0 || v.fract() != 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("{what} must be a non-negative integer, got {v}"),
        });
    }
    Ok(v as usize)
}

/// `grade_up(v)` / `grade_down(v)` -- the stable argsort of a
/// rank-1 vector (ascending / descending); ties keep original
/// order in both directions. `gather_rows(X, grade_up(d))` is the
/// curriculum-ordering idiom.
pub(crate) fn grade(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    if args[0].rank() > 1 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("input must be rank 1, got rank {}", args[0].rank()),
        });
    }
    let data = args[0].data();
    let mut idx: Vec<usize> = (0..data.len()).collect();
    if name == "grade_down" {
        idx.sort_by(|&a, &b| data[b].total_cmp(&data[a]).then(a.cmp(&b)));
    } else {
        idx.sort_by(|&a, &b| data[a].total_cmp(&data[b]).then(a.cmp(&b)));
    }
    Ok(DenseArray::from_vec(
        idx.into_iter().map(|i| i as f64).collect(),
    ))
}

/// `compress(mask, a[, axis])` -- keep the slices of `a` along
/// `axis` (default 0) where rank-1 `mask` is nonzero (APL
/// compress). `compress(gt(scores, t), C)` is the rejection-
/// sampling keep step.
pub(crate) fn compress(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 && args.len() != 3 {
        return Err(arity_err(name, 2, args.len()));
    }
    let axis = if args.len() == 3 {
        args[2].data()[0] as usize
    } else {
        0
    };
    let (mask, a) = (&args[0], &args[1]);
    let dims = a.shape().dims().to_vec();
    if mask.rank() > 1 || axis >= dims.len() || mask.data().len() != dims[axis] {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!(
                "mask must be rank 1 with length {} (axis {axis} of shape {dims:?})",
                dims.get(axis).copied().unwrap_or(0)
            ),
        });
    }
    let keep: Vec<usize> = (0..dims[axis]).filter(|&i| mask.data()[i] != 0.0).collect();
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();
    let mut out = Vec::with_capacity(outer * keep.len() * inner);
    for o in 0..outer {
        for &k in &keep {
            let base = (o * dims[axis] + k) * inner;
            out.extend_from_slice(&a.data()[base..base + inner]);
        }
    }
    let mut new_dims = dims;
    new_dims[axis] = keep.len();
    Ok(DenseArray::new(Shape::new(new_dims), out)?)
}
