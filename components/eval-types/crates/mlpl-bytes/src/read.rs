//! Decode one little-endian value from a packed buffer, widened to
//! `f64` for the interpreter's numeric domain. `None` when the read
//! would run past the end of the buffer.

use crate::dtype::ByteDtype;

/// Read a `dtype` value at byte `offset`, little-endian, as `f64`.
/// `None` if `offset + width` exceeds `data.len()`.
///
/// u64/i64 above 2^53 lose precision in the f64 result -- acceptable
/// for the interpreter's single numeric type; use the raw buffer if
/// exact 64-bit integers are needed.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
pub fn read_le(data: &[u8], offset: usize, dtype: ByteDtype) -> Option<f64> {
    let end = offset.checked_add(dtype.width())?;
    let slice = data.get(offset..end)?;
    macro_rules! rd {
        ($t:ty) => {
            <$t>::from_le_bytes(slice.try_into().ok()?) as f64
        };
    }
    Some(match dtype {
        ByteDtype::U8 => rd!(u8),
        ByteDtype::I8 => rd!(i8),
        ByteDtype::U16 => rd!(u16),
        ByteDtype::I16 => rd!(i16),
        ByteDtype::U32 => rd!(u32),
        ByteDtype::I32 => rd!(i32),
        ByteDtype::U64 => rd!(u64),
        ByteDtype::I64 => rd!(i64),
        ByteDtype::F32 => rd!(f32),
        ByteDtype::F64 => rd!(f64),
        ByteDtype::Bf16 => bf16_to_f64(u16::from_le_bytes(slice.try_into().ok()?)),
        ByteDtype::F16 => f16_to_f64(u16::from_le_bytes(slice.try_into().ok()?)),
    })
}

/// bfloat16 -> f64. bf16 is the top 16 bits of an f32, so widening the bits
/// back into an f32 is exact.
#[must_use]
fn bf16_to_f64(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

/// IEEE-754 half (1 sign, 5 exponent, 10 mantissa) -> f64, covering zero,
/// subnormals, normals, and inf/NaN.
#[must_use]
fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let frac = f64::from(bits & 0x3ff);
    let mag = match exp {
        0 => frac * 2f64.powi(-24),           // subnormal (and zero)
        0x1f if frac == 0.0 => f64::INFINITY, // inf
        0x1f => f64::NAN,                     // NaN
        _ => (1.0 + frac / 1024.0) * 2f64.powi(i32::from(exp) - 15),
    };
    sign * mag
}
