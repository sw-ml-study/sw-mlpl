//! The dense-array element dtype crossed at the extension boundary:
//! its identity (width, V1 wire code). The four dtypes of the V1 ABI
//! (`dense-array-views.md`): u8, i64, f32, f64. Wire encode/decode +
//! strides live in `dtype_codec`.

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
    /// `wire_tag_mirrors_the_dtype_tag_codes` in the cabi tests.
    #[must_use]
    pub fn wire_tag(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::I64 => 2,
            Self::F32 => 3,
            Self::F64 => 4,
        }
    }

    /// The dtype for a V1 wire code, or `None` if unrecognized.
    #[must_use]
    pub fn from_wire_tag(tag: u32) -> Option<Self> {
        match tag {
            1 => Some(Self::U8),
            2 => Some(Self::I64),
            3 => Some(Self::F32),
            4 => Some(Self::F64),
            _ => None,
        }
    }
}
