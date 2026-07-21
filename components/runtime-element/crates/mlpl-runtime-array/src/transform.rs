//! Structural transforms: builtins that return a re-shaped or
//! re-ordered array. Split out of `shape.rs`, which now holds only
//! the pure introspection queries (`shape`, `rank`, `depth`, `size`,
//! `tally`). Introspection asks about structure; transforms change
//! it -- two responsibilities, two modules.

use mlpl_array::{DenseArray, Shape};
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
