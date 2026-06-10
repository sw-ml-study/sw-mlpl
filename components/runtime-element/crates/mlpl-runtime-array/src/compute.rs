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

/// `linspace(start, stop, n)` -- `n` evenly spaced values from `start`
/// to `stop` inclusive (a 1-D vector). `n = 1` yields `[start]`; `n = 0`
/// yields the empty vector.
pub(crate) fn linspace(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(arity_err(name, 3, args.len()));
    }
    let scalar = |a: &DenseArray| -> Result<f64, RuntimeError> {
        if a.rank() != 0 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("expected scalar, got rank {}", a.rank()),
            });
        }
        Ok(a.data()[0])
    };
    let (start, stop) = (scalar(&args[0])?, scalar(&args[1])?);
    let n = scalar(&args[2])? as usize;
    let data: Vec<f64> = match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (n as f64 - 1.0);
            (0..n).map(|i| start + step * i as f64).collect()
        }
    };
    Ok(DenseArray::from_vec(data))
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
