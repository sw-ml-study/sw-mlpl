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

/// `running_sum(v)` -- running sum along a 1-D vector: out[i] = sum of
/// v[0..=i]. The additive scan beside `running_product`; `scan(:op)`
/// is the planned general form.
pub(crate) fn running_sum(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    rank1_guard(name, &args[0])?;
    let mut acc = 0.0;
    let data: Vec<f64> = args[0]
        .data()
        .iter()
        .map(|&x| {
            acc += x;
            acc
        })
        .collect();
    Ok(DenseArray::from_vec(data))
}

/// `running_product(v)` -- running product along a 1-D vector: out[i] =
/// prod of v[0..=i]. Same length as the input (a noise-schedule
/// alpha-bar is `running_product(alphas)`). `cumprod` is the
/// deprecated alias.
pub(crate) fn running_product(
    name: &str,
    args: Vec<DenseArray>,
) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    rank1_guard(name, &args[0])?;
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

/// Scans are rank-1 only: reject higher ranks with a hint at the
/// two focusing lenses instead of silently scanning the ravel.
fn rank1_guard(name: &str, a: &DenseArray) -> Result<(), RuntimeError> {
    if a.rank() <= 1 {
        return Ok(());
    }
    Err(RuntimeError::InvalidArgument {
        func: name.into(),
        reason: format!(
            "input must be rank 1 (got rank {}); focus a row/column with \
             take(a, axis, i) or scan the whole array explicitly with flatten(a)",
            a.rank()
        ),
    })
}
