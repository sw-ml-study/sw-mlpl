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

/// `pareto_front(P, dirs)`: the `[n]` 0/1 mask of non-dominated
/// rows of the `[n, k]` metric matrix `P`. `dirs` is `[k]` with
/// `1` = maximize the column, `-1` = minimize it. Row `i` is
/// dominated when some row is at least as good on every column
/// and strictly better on one. Duplicates dominate neither way,
/// so both stay. O(n^2 * k) -- experiment logs are small.
pub(crate) fn pareto_front(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    let (p, dirs) = (&args[0], &args[1]);
    let bad = |reason: String| RuntimeError::InvalidArgument {
        func: name.into(),
        reason,
    };
    if p.rank() != 2 {
        return Err(bad(format!(
            "P must be a rank-2 [n, k] metric matrix, got rank {}",
            p.rank()
        )));
    }
    let (n, k) = (p.shape().dims()[0], p.shape().dims()[1]);
    if dirs.rank() != 1 || dirs.shape().dims()[0] != k {
        return Err(bad(format!(
            "dirs needs one direction per column: [{k}] expected"
        )));
    }
    if !dirs.data().iter().all(|d| *d == 1.0 || *d == -1.0) {
        return Err(bad(
            "each dirs entry must be 1 (maximize) or -1 (minimize)".into()
        ));
    }
    let adj: Vec<f64> = p
        .data()
        .iter()
        .enumerate()
        .map(|(i, v)| v * dirs.data()[i % k])
        .collect();
    let row = |i: usize| &adj[i * k..(i + 1) * k];
    let dominated = |i: usize| {
        (0..n).any(|j| {
            j != i
                && row(j).iter().zip(row(i)).all(|(a, b)| a >= b)
                && row(j).iter().zip(row(i)).any(|(a, b)| a > b)
        })
    };
    let mask: Vec<f64> = (0..n).map(|i| f64::from(u8::from(!dominated(i)))).collect();
    Ok(DenseArray::new(Shape::new(vec![n]), mask)?)
}
