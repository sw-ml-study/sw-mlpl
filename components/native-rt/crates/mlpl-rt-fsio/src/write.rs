//! Sandboxed file writes: `write_bytes` (truncate) and `append_bytes`.
//! Both validate their byte array with the shared loud-reject
//! `array_to_bytes` and propagate I/O failures as `err` Results,
//! mirroring mlpl-eval's `fs_bytes.rs` / `fs_append.rs`.

use std::io::Write;

use mlpl_array::DenseArray;
use mlpl_rt_value::{CVal, array_to_bytes};

use crate::sandbox::contained;

/// Extract `(path, validated bytes)` or an `err` CVal. The path must
/// be a string and the value a byte array (each cell an integer in
/// `0..=255`, loud-rejected -- never truncated).
fn prepare(who: &str, path: &CVal, bytes: &CVal) -> Result<(String, Vec<u8>), CVal> {
    let CVal::Str(rel) = path else {
        return Err(CVal::result(
            false,
            CVal::Str(format!("{who}: path must be a string")),
        ));
    };
    let CVal::Arr(a) = bytes else {
        return Err(CVal::result(
            false,
            CVal::Str(format!("{who}: bytes must be an array")),
        ));
    };
    array_to_bytes(who, a)
        .map(|b| (rel.clone(), b))
        .map_err(|e| CVal::result(false, CVal::Str(e)))
}

/// `write_bytes(path, bytes)` -- truncate + write. `ok(1)` / `err`.
#[must_use]
pub fn write_bytes(path: &CVal, bytes: &CVal) -> CVal {
    let (rel, b) = match prepare("write_bytes", path, bytes) {
        Ok(pair) => pair,
        Err(e) => return e,
    };
    match contained(&rel).and_then(|p| std::fs::write(p, b).map_err(|e| e.to_string())) {
        Ok(()) => CVal::result(true, CVal::Arr(DenseArray::from_scalar(1.0))),
        Err(e) => CVal::result(false, CVal::Str(format!("write_bytes: {e}"))),
    }
}

/// `append_bytes(path, bytes)` -- create-or-append. `ok(count)` / `err`.
#[must_use]
pub fn append_bytes(path: &CVal, bytes: &CVal) -> CVal {
    let (rel, b) = match prepare("append_bytes", path, bytes) {
        Ok(pair) => pair,
        Err(e) => return e,
    };
    let count = b.len();
    let done = contained(&rel).and_then(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .and_then(|mut f| f.write_all(&b))
            .map_err(|e| e.to_string())
    });
    match done {
        Ok(()) => CVal::result(true, CVal::Arr(DenseArray::from_scalar(count as f64))),
        Err(e) => CVal::result(false, CVal::Str(format!("append_bytes: {e}"))),
    }
}
