//! Shifts, masking, and bit-vector VIEWS (docs/bit-ops-design.md):
//! shl(x, k, width) [width-aware, masks], shr(x, k) [logical],
//! bmask(x, width) [keep low width bits], bits(x, width) [scalar
//! -> [width] LSB-first 0/1 vector], from_bits(v) [pack back].

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;
use crate::logic::{as_uint, mask_of, width_arg};

/// A non-negative integer scalar parameter (shift count k).
fn count_arg(name: &str, arr: &DenseArray) -> Result<u32, RuntimeError> {
    if arr.rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "shift count must be a scalar".into(),
        });
    }
    let k = arr.data()[0];
    if !k.is_finite() || k.fract() != 0.0 || !(0.0..=53.0).contains(&k) {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("shift count must be an integer in 0..=53, got {k}"),
        });
    }
    Ok(k as u32)
}

pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "shl" => Some(shl(name, args)),
        "shr" => Some(shr(name, args)),
        "bmask" => Some(bmask(name, args)),
        "bits" => Some(bits(name, args)),
        "from_bits" => Some(from_bits(name, args)),
        _ => None,
    }
}

/// Map a validated per-element transform over the first arg,
/// preserving shape.
fn map_uint(
    name: &str,
    arr: &DenseArray,
    op: impl Fn(u64) -> u64,
) -> Result<DenseArray, RuntimeError> {
    let data = arr
        .data()
        .iter()
        .map(|x| -> Result<f64, RuntimeError> { Ok(op(as_uint(name, *x)?) as f64) })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DenseArray::new(
        Shape::new(arr.shape().dims().to_vec()),
        data,
    )?)
}

fn shl(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(arity_err(name, 3, args.len()));
    }
    let k = count_arg(name, &args[1])?;
    let mask = mask_of(width_arg(name, &args[2])?);
    map_uint(name, &args[0], |x| (x << k) & mask)
}

fn shr(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    let k = count_arg(name, &args[1])?;
    map_uint(name, &args[0], |x| x >> k)
}

fn bmask(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    let mask = mask_of(width_arg(name, &args[1])?);
    map_uint(name, &args[0], |x| x & mask)
}

/// `bits(x, width)` -- scalar `x` to a `[width]` 0/1 vector,
/// LSB-first (index i holds bit i).
fn bits(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(arity_err(name, 2, args.len()));
    }
    if args[0].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "bits expands a SCALAR integer".into(),
        });
    }
    let x = as_uint(name, args[0].data()[0])?;
    let width = width_arg(name, &args[1])?;
    let data: Vec<f64> = (0..width).map(|i| ((x >> i) & 1) as f64).collect();
    Ok(DenseArray::new(Shape::new(vec![width as usize]), data)?)
}

/// `from_bits(v)` -- pack a rank-1 0/1 vector to a scalar
/// integer (LSB-first, inverse of `bits`).
fn from_bits(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(arity_err(name, 1, args.len()));
    }
    let v = &args[0];
    if v.rank() != 1 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "from_bits takes a rank-1 vector of 0/1".into(),
        });
    }
    if v.data().len() > 53 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!(
                "bit vector too wide for an exact integer: {} > 53",
                v.data().len()
            ),
        });
    }
    let mut acc: u64 = 0;
    for (i, b) in v.data().iter().enumerate() {
        match *b {
            0.0 => {}
            1.0 => acc |= 1u64 << i,
            other => {
                return Err(RuntimeError::InvalidArgument {
                    func: name.into(),
                    reason: format!("bit entries must be 0 or 1, got {other} at index {i}"),
                });
            }
        }
    }
    Ok(DenseArray::from_scalar(acc as f64))
}
