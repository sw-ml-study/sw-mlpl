//! Compiled-program I/O helpers over `CVal`: stdout writing and
//! command-line argument access. Kept in a sibling module so the
//! crate root (the `CVal` type + its impls) stays within the
//! function-count budget.

use std::io::Write;

use mlpl_array::DenseArray;

use crate::CVal;

/// `write_stdout(bytes)` -- write a `CVal` to process stdout and
/// flush, returning `CVal::Arr` scalar count of bytes written. A
/// string writes its UTF-8 bytes; an array writes its cells as
/// bytes (each truncated to `0..=255`); a string list writes its
/// items newline-joined.
#[must_use]
pub fn write_stdout(v: &CVal) -> CVal {
    let bytes: Vec<u8> = match v {
        CVal::Str(s) => s.clone().into_bytes(),
        CVal::StrList(items) => items.join("\n").into_bytes(),
        CVal::Arr(a) => a.data().iter().map(|&x| x as u8).collect(),
    };
    let count = bytes.len();
    let mut out = std::io::stdout();
    let _ = out.write_all(&bytes).and_then(|()| out.flush());
    CVal::Arr(DenseArray::from_scalar(count as f64))
}

/// `args()` -- the process command-line arguments (excluding
/// argv[0]) as a `CVal::StrList`.
#[must_use]
pub fn cli_args() -> CVal {
    CVal::StrList(std::env::args().skip(1).collect())
}

/// `arg(i)` -- the i-th command-line argument (0-based, excluding
/// argv[0]) as a `CVal::Str`, or the empty string if out of range.
#[must_use]
pub fn arg(idx: &CVal) -> CVal {
    let i = idx.arr().data()[0] as usize;
    CVal::Str(std::env::args().nth(i + 1).unwrap_or_default())
}
