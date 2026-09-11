//! Value-domain helpers shared by every bit op (docs/bit-ops-design.md):
//! the exact-integer validation, the width-parameter check, the low-bits
//! mask, and the per-element uint map. Pure; the domain is non-negative
//! integers in 0..2^53.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

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

/// The low-`width`-bits mask, `2^width - 1` (width <= 53).
pub(crate) fn mask_of(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Map a validated per-element transform over the first arg,
/// preserving shape.
pub(crate) fn map_uint(
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
