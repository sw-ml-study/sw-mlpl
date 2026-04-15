//! Dataset-prep builtins: `shuffle`, `batch`, `batch_mask`, `split`,
//! `val_split` (Saga 12 step 002).
//!
//! Row-axis (axis 0) operations. All five share a common shape
//! discipline: axis 0 is the "sample" dimension; labels on axis 0
//! are dropped in the result (rows are being permuted, batched, or
//! split, so the original axis-0 name no longer refers to the
//! same semantic axis) while labels on axes 1..r are preserved.

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;
use crate::prng::Xorshift64;

/// Dispatch dataset builtins. Returns None if not matched.
pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "shuffle" => Some(builtin_shuffle(name, args)),
        "batch" => Some(builtin_batch(name, args, false)),
        "batch_mask" => Some(builtin_batch(name, args, true)),
        "split" => Some(builtin_split(name, args, true)),
        "val_split" => Some(builtin_split(name, args, false)),
        _ => None,
    }
}

/// Shuffle rows (axis 0) using a seeded Fisher-Yates permutation.
fn builtin_shuffle(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "seed must be a scalar".into(),
        });
    }
    let seed = args[1].data()[0] as i64 as u64;
    let perm = permutation(args[0].shape().dims()[0], seed);
    gather_rows(&args[0], &perm)
}

/// Batch rows along axis 0 into contiguous groups of `size`, with
/// zero-padding for the final short batch. When `as_mask` is true,
/// returns a rank-2 [B, size] 0/1 mask instead of the data-padded
/// array.
fn builtin_batch(
    name: &str,
    args: Vec<DenseArray>,
    as_mask: bool,
) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    let bad_arg = |reason: String| RuntimeError::InvalidArgument {
        func: name.into(),
        reason,
    };
    if args[1].rank() != 0 {
        return Err(bad_arg("size must be a scalar".into()));
    }
    let size = args[1].data()[0] as usize;
    if size == 0 {
        return Err(bad_arg("size must be > 0".into()));
    }
    if args[0].shape().dims().is_empty() {
        return Err(bad_arg("rank >= 1 required".into()));
    }
    let x = &args[0];
    let n = x.shape().dims()[0];
    let batches = n.div_ceil(size);
    if as_mask {
        let mut data = vec![0.0f64; batches * size];
        for slot in data.iter_mut().take(n) {
            *slot = 1.0;
        }
        return Ok(DenseArray::new(Shape::new(vec![batches, size]), data)?);
    }
    batch_rows(x, batches, size)
}

/// Allocate a zero-padded `[batches, size, ...row_shape]` result
/// and copy `x`'s rows in place, preserving non-axis-0 labels
/// (prepending two `None` label slots for the new `[batches, size]`
/// prefix axes).
fn batch_rows(x: &DenseArray, batches: usize, size: usize) -> Result<DenseArray, RuntimeError> {
    let dims = x.shape().dims();
    let n = dims[0];
    let row_stride: usize = dims[1..].iter().product::<usize>().max(1);
    let mut data = vec![0.0f64; batches * size * row_stride];
    let src = x.data();
    for i in 0..n {
        let dst_start = i * row_stride;
        data[dst_start..dst_start + row_stride]
            .copy_from_slice(&src[i * row_stride..(i + 1) * row_stride]);
    }
    let mut out_dims = vec![batches, size];
    out_dims.extend_from_slice(&dims[1..]);
    let mut out = DenseArray::new(Shape::new(out_dims), data)?;
    if let Some(src_labels) = x.labels() {
        let mut labels = vec![None, None];
        labels.extend_from_slice(&src_labels[1..]);
        out = out.with_labels(labels)?;
    }
    Ok(out)
}

/// Split shuffled rows into training and validation chunks. `take_train=true`
/// returns the training chunk, `take_train=false` returns the validation
/// chunk. Both use the same seed, so a caller doing
/// `train = split(x, 0.8, 7); val = val_split(x, 0.8, 7)` gets disjoint
/// rows that together cover `x`.
fn builtin_split(
    name: &str,
    args: Vec<DenseArray>,
    take_train: bool,
) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 3,
            got: args.len(),
        });
    }
    if args[1].rank() != 0 || args[2].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "train_frac and seed must be scalars".into(),
        });
    }
    let frac = args[1].data()[0];
    if !(frac > 0.0 && frac < 1.0) {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("train_frac must be in (0, 1), got {frac}"),
        });
    }
    let seed = args[2].data()[0] as i64 as u64;
    let x = &args[0];
    let n = x.shape().dims()[0];
    let n_train = ((n as f64) * frac) as usize;
    let n_train = n_train.clamp(1, n.saturating_sub(1));
    let perm = permutation(n, seed);
    let chunk: Vec<usize> = if take_train {
        perm[..n_train].to_vec()
    } else {
        perm[n_train..].to_vec()
    };
    gather_rows(x, &chunk)
}

/// Fisher-Yates permutation of 0..n using a seeded xorshift64.
fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    let mut rng = Xorshift64::new(seed);
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        perm.swap(i, j);
    }
    perm
}

/// Build a new array whose rows are `x`'s rows at positions given by
/// `indices`. Preserves every non-axis-0 label; axis-0 label is
/// dropped because the row identity has changed.
fn gather_rows(x: &DenseArray, indices: &[usize]) -> Result<DenseArray, RuntimeError> {
    let dims = x.shape().dims();
    if dims.is_empty() {
        return Err(RuntimeError::InvalidArgument {
            func: "gather_rows".into(),
            reason: "rank >= 1 required".into(),
        });
    }
    let row_stride: usize = dims[1..].iter().product::<usize>().max(1);
    let mut data = Vec::with_capacity(indices.len() * row_stride);
    let src = x.data();
    for &i in indices {
        data.extend_from_slice(&src[i * row_stride..(i + 1) * row_stride]);
    }
    let mut out_dims = vec![indices.len()];
    out_dims.extend_from_slice(&dims[1..]);
    let mut out = DenseArray::new(Shape::new(out_dims), data)?;
    if let Some(src_labels) = x.labels() {
        let mut labels = vec![None];
        labels.extend_from_slice(&src_labels[1..]);
        out = out.with_labels(labels)?;
    }
    Ok(out)
}
