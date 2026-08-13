//! Element dtypes for packed byte buffers (`Value::Bytes`).
//!
//! Little-endian is the canonical in-memory layout; big-endian is an
//! explicit reader family, not a stored property. Bounded and
//! index/offset-only -- no pointer arithmetic crosses into MLPL.

use std::fmt;

/// The element type of a packed byte buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteDtype {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
}

impl ByteDtype {
    /// Width in bytes of one element.
    #[must_use]
    pub fn width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Parse a dtype name (`"u8"`, `"f32"`, ...); `None` if unknown.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "u8" => Self::U8,
            "i8" => Self::I8,
            "u16" => Self::U16,
            "i16" => Self::I16,
            "u32" => Self::U32,
            "i32" => Self::I32,
            "u64" => Self::U64,
            "i64" => Self::I64,
            "f32" => Self::F32,
            "f64" => Self::F64,
            _ => return None,
        })
    }

    /// The canonical name, used in `Display` and error messages.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

impl fmt::Display for ByteDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
