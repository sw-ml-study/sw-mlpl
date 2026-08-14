//! Wire codec for `ExtDtype`: encode f64 elements to little-endian
//! dtype bytes and decode them back, plus row-major byte strides. The
//! host carries array elements as f64; the wire uses the dtype width.

use crate::dtype::ExtDtype;

impl ExtDtype {
    /// Encode f64 elements as little-endian bytes of this dtype.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn encode_le(self, data: &[f64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len() * self.width());
        for &x in data {
            match self {
                Self::U8 => out.push(x as u8),
                Self::I64 => out.extend_from_slice(&(x as i64).to_le_bytes()),
                Self::F32 => out.extend_from_slice(&(x as f32).to_le_bytes()),
                Self::F64 => out.extend_from_slice(&x.to_le_bytes()),
            }
        }
        out
    }

    /// Decode little-endian dtype `bytes` back to f64 elements. `bytes`
    /// must be a whole number of elements (checked by the caller).
    #[must_use]
    #[allow(clippy::cast_lossless, clippy::cast_precision_loss)]
    pub fn decode_le(self, bytes: &[u8]) -> Vec<f64> {
        bytes
            .chunks_exact(self.width())
            .map(|c| match self {
                Self::U8 => f64::from(c[0]),
                Self::I64 => i64::from_le_bytes(c.try_into().unwrap()) as f64,
                Self::F32 => f64::from(f32::from_le_bytes(c.try_into().unwrap())),
                Self::F64 => f64::from_le_bytes(c.try_into().unwrap()),
            })
            .collect()
    }

    /// Row-major (C-order) BYTE strides for `shape`.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn byte_strides(self, shape: &[usize]) -> Vec<isize> {
        let mut strides = vec![0isize; shape.len()];
        let mut acc = self.width();
        for i in (0..shape.len()).rev() {
            strides[i] = acc as isize;
            acc *= shape[i];
        }
        strides
    }
}
