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
    })
}
