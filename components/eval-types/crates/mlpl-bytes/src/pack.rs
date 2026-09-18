//! Pack an f64 array into a canonical little-endian packed buffer.
//!
//! Integer dtypes reject fractional or out-of-range values; float
//! dtypes narrow (`f32`) or copy (`f64`) the IEEE-754 bit pattern.
//! The error is a message string; the caller (mlpl-eval) wraps it in
//! its own `EvalError` -- keeping this crate dependency-free.

use crate::dtype::ByteDtype;

/// Pack `values` into a little-endian `dtype` buffer.
///
/// # Errors
/// Returns an `Err(message)` when a value is not representable in the
/// target dtype: a fractional or out-of-range value for an integer
/// dtype.
pub fn pack_f64s(values: &[f64], dtype: ByteDtype) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(values.len() * dtype.width());
    for &v in values {
        push_le(&mut out, v, dtype)?;
    }
    Ok(out)
}

/// Append one value to `out` in little-endian `dtype` form.
//
// The `int!` macro is generic over every integer width, so its f64
// <-> integer conversions are intentional: each value is bounds- and
// fract-checked against the target type BEFORE the `as` cast, so the
// cast cannot silently truncate or lose sign. The bound literals cross
// the f64 boundary via `as` for the same reason. Hence the scoped
// cast allows -- they are correct here, not overlooked.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
fn push_le(out: &mut Vec<u8>, v: f64, dtype: ByteDtype) -> Result<(), String> {
    macro_rules! int {
        ($t:ty) => {{
            if v.fract() != 0.0 || v < <$t>::MIN as f64 || v > <$t>::MAX as f64 {
                return Err(not_representable(v, dtype));
            }
            out.extend_from_slice(&(v as $t).to_le_bytes());
        }};
    }
    match dtype {
        ByteDtype::U8 => int!(u8),
        ByteDtype::I8 => int!(i8),
        ByteDtype::U16 => int!(u16),
        ByteDtype::I16 => int!(i16),
        ByteDtype::U32 => int!(u32),
        ByteDtype::I32 => int!(i32),
        ByteDtype::U64 => int!(u64),
        ByteDtype::I64 => int!(i64),
        ByteDtype::F32 | ByteDtype::F64 | ByteDtype::Bf16 | ByteDtype::F16 => {
            return push_float_le(out, v, dtype);
        }
    }
    Ok(())
}

/// Append a float value: `f32` narrows, `f64` copies. The half formats
/// (`bf16`/`f16`) are decode-only -- model weights arrive as those bytes from
/// a file and are reinterpreted + read, never packed here.
#[allow(clippy::cast_possible_truncation)]
fn push_float_le(out: &mut Vec<u8>, v: f64, dtype: ByteDtype) -> Result<(), String> {
    match dtype {
        ByteDtype::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
        ByteDtype::F64 => out.extend_from_slice(&v.to_le_bytes()),
        _ => {
            return Err(format!(
                "pack: {dtype} is decode-only (reinterpret model-weight bytes and read); \
                 packing to it is not supported"
            ));
        }
    }
    Ok(())
}

fn not_representable(v: f64, dtype: ByteDtype) -> String {
    format!("pack: value {v} is not representable as {dtype}")
}
