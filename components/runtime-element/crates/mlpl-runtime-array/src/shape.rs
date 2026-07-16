use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_shape::prelude::*;
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
