//! Bitwise LOGIC dispatch plus the ops that are not plain binary
//! combiners: `bnot` (complement within a width) and `popcount`
//! (docs/bit-ops-design.md). The band/bor/bxor machinery lives in
//! `logic_binary`; the shared value-domain helpers in `bit_domain`.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;
use crate::bit_domain::{as_uint, mask_of, width_arg};
use crate::logic_binary::binary;

/// Dispatch the logic builtins. `None` if not matched.
pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "band" => Some(binary(name, args, |a, b| a & b)),
        "bor" => Some(binary(name, args, |a, b| a | b)),
        "bxor" => Some(binary(name, args, |a, b| a ^ b)),
        "bnot" => Some(bnot(name, args)),
        "popcount" => Some(popcount(name, args)),
        _ => None,
    }
}

fn bnot(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    let width = width_arg(name, &args[1])?;
    let mask = mask_of(width);
    let data = args[0]
        .data()
        .iter()
        .map(|x| -> Result<f64, RuntimeError> { Ok(((!as_uint(name, *x)?) & mask) as f64) })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DenseArray::new(
        Shape::new(args[0].shape().dims().to_vec()),
        data,
    )?)
}

fn popcount(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    let data = args[0]
        .data()
        .iter()
        .map(|x| -> Result<f64, RuntimeError> { Ok(f64::from(as_uint(name, *x)?.count_ones())) })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DenseArray::new(
        Shape::new(args[0].shape().dims().to_vec()),
        data,
    )?)
}
