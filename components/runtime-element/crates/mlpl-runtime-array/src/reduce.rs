use mlpl_array::DenseArray;
use mlpl_array_ops_reduce::prelude::*;
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;

pub(crate) fn argmax(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    match args.len() {
        1 => argmax_flat(name, &args[0]),
        2 => argmax_axis(name, &args),
        got => Err(arity_err(name, 2, got)),
    }
}

fn argmax_flat(name: &str, arr: &DenseArray) -> Result<DenseArray, RuntimeError> {
    let data = arr.data();
    if data.is_empty() {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "argmax of empty array".into(),
        });
    }
    let (idx, _) = data
        .iter()
        .enumerate()
        .reduce(|a, b| if b.1 > a.1 { b } else { a })
        .unwrap();
    Ok(DenseArray::from_scalar(idx as f64))
}

fn argmax_axis(name: &str, args: &[DenseArray]) -> Result<DenseArray, RuntimeError> {
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("axis must be scalar, got rank {}", args[1].rank()),
        });
    }
    Ok(args[0].argmax_axis(args[1].data()[0] as usize)?)
}

/// `cumprod(v)` -- running product along a 1-D vector: out[i] = prod of
/// v[0..=i]. Same length as the input (a noise-schedule alpha-bar is
/// `cumprod(alphas)`).
pub(crate) fn cumprod(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    let mut acc = 1.0;
    let data: Vec<f64> = args[0]
        .data()
        .iter()
        .map(|&x| {
            acc *= x;
            acc
        })
        .collect();
    Ok(DenseArray::from_vec(data))
}

pub(crate) fn reduce(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let got = args.len();
    if got != 1 && got != 2 {
        return Err(arity_err(name, 1, got));
    }
    let (identity, op): (f64, fn(f64, f64) -> f64) = match name {
        "reduce_add" => (0.0, |a, b| a + b),
        "reduce_mul" => (1.0, |a, b| a * b),
        _ => unreachable!(),
    };
    if got == 1 {
        return Ok(DenseArray::from_scalar(
            args[0].data().iter().copied().fold(identity, op),
        ));
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("axis must be scalar, got rank {}", args[1].rank()),
        });
    }
    Ok(args[0].reduce_axis(args[1].data()[0] as usize, identity, op)?)
}
