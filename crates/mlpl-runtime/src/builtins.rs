//! Built-in function implementations.

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;
use crate::math_builtins;

macro_rules! check_arity {
    ($name:expr, $expected:expr, $args:expr) => {
        if $args.len() != $expected {
            return Err(RuntimeError::ArityMismatch {
                func: $name.into(),
                expected: $expected,
                got: $args.len(),
            });
        }
    };
}

/// Dispatch a built-in function call by name.
pub fn call_builtin(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    // Try math/constructor builtins first
    if let Some(result) = math_builtins::try_call(name, args.clone()) {
        return result;
    }
    if let Some(result) = crate::random_builtins::try_call(name, args.clone()) {
        return result;
    }
    if let Some(result) = crate::dataset_builtins::try_call(name, args.clone()) {
        return result;
    }
    if let Some(result) = crate::ml_builtins::try_call(name, args.clone()) {
        return result;
    }
    if let Some(result) = crate::ensemble_builtins::try_call(name, args.clone()) {
        return result;
    }
    match name {
        "iota" => builtin_iota(name, args),
        "shape" => builtin_shape(name, args),
        "rank" => builtin_rank(name, args),
        "reshape" => builtin_reshape(name, args),
        "transpose" => builtin_transpose(name, args),
        "reduce_add" | "reduce_mul" => builtin_reduce(name, args),
        "argmax" => builtin_argmax(name, args),
        "dot" => {
            check_arity!(name, 2, args);
            Ok(args[0].dot(&args[1])?)
        }
        "matmul" => {
            check_arity!(name, 2, args);
            Ok(args[0].matmul(&args[1])?)
        }
        "grid" => crate::grid_builtin::builtin_grid(name, args),
        _ => Err(RuntimeError::UnknownFunction(name.into())),
    }
}

fn builtin_iota(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 1, args);
    if args[0].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("expected scalar, got rank {}", args[0].rank()),
        });
    }
    let n = args[0].data()[0] as usize;
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    Ok(DenseArray::from_vec(data))
}

fn builtin_shape(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 1, args);
    let dims: Vec<f64> = args[0].shape().dims().iter().map(|&d| d as f64).collect();
    Ok(DenseArray::from_vec(dims))
}

fn builtin_rank(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 1, args);
    Ok(DenseArray::from_scalar(args[0].rank() as f64))
}

fn builtin_reshape(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 2, args);
    let shape_data = args[1].data();
    let dims: Vec<usize> = shape_data.iter().map(|&d| d as usize).collect();
    Ok(args[0].reshape(Shape::new(dims))?)
}

fn builtin_transpose(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 1, args);
    Ok(args[0].transpose())
}

fn builtin_argmax(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() == 1 {
        // Flat argmax over all elements, returned as a scalar index.
        let data = args[0].data();
        if data.is_empty() {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: "argmax of empty array".into(),
            });
        }
        let mut best_idx = 0usize;
        let mut best_val = data[0];
        for (i, &v) in data.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        return Ok(DenseArray::from_scalar(best_idx as f64));
    }
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
            reason: format!("axis must be scalar, got rank {}", args[1].rank()),
        });
    }
    let axis = args[1].data()[0] as usize;
    Ok(args[0].argmax_axis(axis)?)
}

fn builtin_reduce(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 && args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    let (identity, op): (f64, fn(f64, f64) -> f64) = match name {
        "reduce_add" => (0.0, |a, b| a + b),
        "reduce_mul" => (1.0, |a, b| a * b),
        _ => unreachable!(),
    };
    if args.len() == 2 {
        if args[1].rank() != 0 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("axis must be scalar, got rank {}", args[1].rank()),
            });
        }
        let axis = args[1].data()[0] as usize;
        Ok(args[0].reduce_axis(axis, identity, op)?)
    } else {
        let result = args[0].data().iter().copied().fold(identity, op);
        Ok(DenseArray::from_scalar(result))
    }
}
