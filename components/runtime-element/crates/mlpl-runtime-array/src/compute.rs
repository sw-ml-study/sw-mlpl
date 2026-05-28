use mlpl_array::DenseArray;
use mlpl_array_ops_matmul::prelude::*;
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;

pub(crate) fn dot(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    Ok(args[0].dot(&args[1])?)
}

pub(crate) fn matmul(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    Ok(args[0].matmul(&args[1])?)
}

pub(crate) fn iota(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    if args[0].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("expected scalar, got rank {}", args[0].rank()),
        });
    }
    let n = args[0].data()[0] as usize;
    Ok(DenseArray::from_vec((0..n).map(|i| i as f64).collect()))
}
