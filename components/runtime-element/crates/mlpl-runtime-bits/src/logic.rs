//! Bitwise LOGIC over fixed-width unsigned integers held as
//! exact f64 (docs/bit-ops-design.md): band / bor / bxor
//! (element-wise, scalar broadcast), bnot (complement within a
//! width), popcount. The domain is non-negative integers in
//! 0..2^53; anything else is a loud error.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;

/// Largest integer f64 represents exactly.
pub(crate) const MAX_EXACT: f64 = 9_007_199_254_740_992.0; // 2^53

/// Validate + convert one element to a u64 bit pattern.
pub(crate) fn as_uint(name: &str, v: f64) -> Result<u64, RuntimeError> {
    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v >= MAX_EXACT {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("operands must be non-negative integers below 2^53, got {v}"),
        });
    }
    Ok(v as u64)
}

/// A width parameter, checked to `1..=53`.
pub(crate) fn width_arg(name: &str, arr: &DenseArray) -> Result<u32, RuntimeError> {
    if arr.rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "width must be a scalar".into(),
        });
    }
    let w = arr.data()[0];
    if !w.is_finite() || w.fract() != 0.0 || !(1.0..=53.0).contains(&w) {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("width must be an integer in 1..=53, got {w}"),
        });
    }
    Ok(w as u32)
}

/// Element-wise bitwise binary op with scalar broadcast.
fn zip_bits(
    name: &str,
    a: &DenseArray,
    b: &DenseArray,
    op: fn(u64, u64) -> u64,
) -> Result<DenseArray, RuntimeError> {
    let (data, shape) = if a.shape() == b.shape() {
        (broadcast_none(name, a, b, op)?, a.shape().clone())
    } else if a.rank() == 0 {
        (
            broadcast_scalar(name, a.data()[0], b, op, true)?,
            b.shape().clone(),
        )
    } else if b.rank() == 0 {
        (
            broadcast_scalar(name, b.data()[0], a, op, false)?,
            a.shape().clone(),
        )
    } else {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!(
                "shape mismatch: {:?} vs {:?}",
                a.shape().dims(),
                b.shape().dims()
            ),
        });
    };
    Ok(DenseArray::new(shape, data)?)
}

fn broadcast_none(
    name: &str,
    a: &DenseArray,
    b: &DenseArray,
    op: fn(u64, u64) -> u64,
) -> Result<Vec<f64>, RuntimeError> {
    a.data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| Ok(op(as_uint(name, *x)?, as_uint(name, *y)?) as f64))
        .collect()
}

fn broadcast_scalar(
    name: &str,
    s: f64,
    arr: &DenseArray,
    op: fn(u64, u64) -> u64,
    scalar_left: bool,
) -> Result<Vec<f64>, RuntimeError> {
    let su = as_uint(name, s)?;
    arr.data()
        .iter()
        .map(|x| {
            let xu = as_uint(name, *x)?;
            let (l, r) = if scalar_left { (su, xu) } else { (xu, su) };
            Ok(op(l, r) as f64)
        })
        .collect()
}

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

fn binary(
    name: &str,
    args: Vec<DenseArray>,
    op: fn(u64, u64) -> u64,
) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    zip_bits(name, &args[0], &args[1], op)
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

/// The low-`width`-bits mask, `2^width - 1` (width <= 53).
pub(crate) fn mask_of(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}
