//! Compiled-program I/O helpers over `CVal`: stdout writing and
//! command-line argument access. Kept in a sibling module so the
//! crate root (the `CVal` type + its impls) stays within the
//! function-count budget.

use std::io::Write;

use mlpl_array::DenseArray;

use crate::CVal;

/// Validate a rank-<=1 array as raw bytes: every cell an integer in
/// `0..=255`. LOUD reject (no silent truncation) on any out-of-range
/// or non-integer cell, mirroring the interpreter's `array_to_bytes`
/// (mlpl-eval `fs_bytes.rs`) so compiled and interpreted byte writers
/// share semantics. `who` names the builtin in the error. Shared by
/// `write_stdout` (and, later, `write_bytes` / `append_bytes`).
pub fn array_to_bytes(who: &str, arr: &DenseArray) -> Result<Vec<u8>, String> {
    if arr.rank() > 1 {
        return Err(format!(
            "{who}: expected rank <= 1, got rank {}",
            arr.rank()
        ));
    }
    arr.data()
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if !(0.0..=255.0).contains(&v) || v.fract() != 0.0 {
                Err(format!(
                    "{who}: cell {i} = {v} is not an integer in 0..=255"
                ))
            } else {
                Ok(v as u8)
            }
        })
        .collect()
}

/// `write_stdout(bytes)` -- write a `CVal` to process stdout and
/// flush, returning `ok(count)` / `err(msg)` (interpreter parity). An
/// array is validated cell-by-cell as bytes (rank <= 1, each an
/// integer in `0..=255`) and REJECTED on violation -- never truncated
/// via `as u8`. A string writes its UTF-8 bytes; a string list writes
/// its items newline-joined. Write / flush failures propagate as an
/// `err` Result rather than being discarded.
#[must_use]
pub fn write_stdout(v: &CVal) -> CVal {
    let bytes: Result<Vec<u8>, String> = match v {
        CVal::Str(s) => Ok(s.clone().into_bytes()),
        CVal::StrList(items) => Ok(items.join("\n").into_bytes()),
        CVal::Arr(a) => array_to_bytes("write_stdout", a),
        other => Err(format!("write_stdout: cannot write {other:?} as bytes")),
    };
    let written = bytes.and_then(|b| {
        let count = b.len();
        let mut out = std::io::stdout();
        out.write_all(&b)
            .and_then(|()| out.flush())
            .map_err(|e| format!("write_stdout: {e}"))?;
        Ok(count)
    });
    match written {
        Ok(count) => CVal::result(true, CVal::Arr(DenseArray::from_scalar(count as f64))),
        Err(msg) => CVal::result(false, CVal::Str(msg)),
    }
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
