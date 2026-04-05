//! Built-in function implementations.

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;

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
    match name {
        "iota" => builtin_iota(name, args),
        "shape" => builtin_shape(name, args),
        "rank" => builtin_rank(name, args),
        "reshape" => builtin_reshape(name, args),
        "transpose" => builtin_transpose(name, args),
        "reduce_add" | "reduce_mul" => builtin_reduce(name, args),
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

fn builtin_reduce(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    check_arity!(name, 1, args);
    let data = args[0].data();
    let result = match name {
        "reduce_add" => data.iter().sum(),
        "reduce_mul" => data.iter().product(),
        _ => unreachable!(),
    };
    Ok(DenseArray::from_scalar(result))
}
