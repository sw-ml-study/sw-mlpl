//! The dense-array element dtype crossed at the extension boundary,
//! and its wire encoding: f64 elements to little-endian dtype bytes,
//! row-major byte strides, and the V1 wire tag. The four dtypes of the
//! V1 ABI (`dense-array-views.md`): u8, i64, f32, f64.

/// Element type of a dense array at the boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtDtype {
    U8,
    I64,
    F32,
    F64,
}

impl ExtDtype {
    /// Bytes per element on the wire.
    #[must_use]
    pub fn width(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// The V1 wire dtype code. MUST mirror `mlpl_extension_cabi`'s
    /// `DTypeTag` (U8=1, I64=2, F32=3, F64=4); a mismatch is pinned by
    /// `wire_tag_matches_dtype_tag` in the cabi tests.
    #[must_use]
    pub fn wire_tag(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::I64 => 2,
            Self::F32 => 3,
            Self::F64 => 4,
        }
    }

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
