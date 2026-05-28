use mlpl_array::DenseArray;
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

fn scalar_usize(name: &str, arr: &DenseArray, what: &str) -> Result<usize, RuntimeError> {
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
