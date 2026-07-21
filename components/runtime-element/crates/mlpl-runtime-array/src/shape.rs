use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;

pub(crate) fn shape(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    let dims: Vec<f64> = args[0].shape().dims().iter().map(|&d| d as f64).collect();
    Ok(DenseArray::from_vec(dims))
}

pub(crate) fn rank(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    Ok(DenseArray::from_scalar(args[0].rank() as f64))
}

/// Nesting depth of a value (APL2 sense).
///
/// A simple scalar has depth 0; any simple non-scalar array has
/// depth 1. MLPL's `DenseArray` is always flat, so depth is `0` or
/// `1` today. When boxed/nested arrays land (staging plan Stage 6)
/// this builtin will report higher depths for nested values.
pub(crate) fn depth(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    let d = f64::from(u8::from(args[0].rank() != 0));
    Ok(DenseArray::from_scalar(d))
}

/// Total number of elements (numel): the product of the shape.
///
/// A scalar (empty shape) has size 1; a vector's size is its length;
/// a matrix's size is `rows * cols`. Returns a rank-0 scalar so it
/// composes with arithmetic.
pub(crate) fn size(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    let numel: usize = args[0].shape().dims().iter().product();
    Ok(DenseArray::from_scalar(numel as f64))
}

/// Length of the leading axis: the number of major cells (APL's
/// monadic `#` / `≢`).
///
/// A scalar tallies to 1; a rank >= 1 array tallies to `shape[0]`.
/// Contrast with `size`, which counts every element. Returns a
/// rank-0 scalar.
pub(crate) fn tally(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    let n = args[0].shape().dims().first().copied().unwrap_or(1);
    Ok(DenseArray::from_scalar(n as f64))
}
