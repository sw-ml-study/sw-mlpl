//! Bit SHIFTS (docs/bit-ops-design.md): shl(x, k, width) [width-aware,
//! masks] and shr(x, k) [logical]. The masking / views (bmask, bits,
//! from_bits) live in `bit_views`; the shared uint map in `bit_domain`.

use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

use crate::arity_err;
use crate::bit_domain::{map_uint, mask_of, width_arg};

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
        _ => None,
    }
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
