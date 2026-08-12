//! Sandboxed file reads: `read_bytes` (whole + range) and `file_size`.

use std::io::{Read, Seek, SeekFrom};

use mlpl_array::{DenseArray, Shape};
use mlpl_rt_value::CVal;

use crate::sandbox::{contained, nonneg};

/// Wrap a read result: bytes -> `ok(rank-1 array)`, error -> `err`.
fn bytes_result(read: Result<Vec<u8>, String>) -> CVal {
    match read {
        Ok(bytes) => {
            let data: Vec<f64> = bytes.iter().map(|&b| f64::from(b)).collect();
            match DenseArray::new(Shape::vector(data.len()), data) {
                Ok(arr) => CVal::result(true, CVal::Arr(arr)),
                Err(e) => CVal::result(false, CVal::Str(format!("read_bytes: {e}"))),
            }
        }
        Err(e) => CVal::result(false, CVal::Str(format!("read_bytes: {e}"))),
    }
}

/// `read_bytes(path)` -- read the whole file as a rank-1 byte array.
#[must_use]
pub fn read_bytes(path: &CVal) -> CVal {
    let CVal::Str(rel) = path else {
        return CVal::result(false, CVal::Str("read_bytes: path must be a string".into()));
    };
    bytes_result(contained(rel).and_then(|p| std::fs::read(p).map_err(|e| e.to_string())))
}

/// `read_bytes(path, offset, length)` -- seek to `offset` and read up
/// to `length` bytes (clamped at EOF). Non-negative integer args are
/// enforced (a hard panic on violation, matching the interpreter).
#[must_use]
pub fn read_bytes_range(path: &CVal, offset: &DenseArray, length: &DenseArray) -> CVal {
    let CVal::Str(rel) = path else {
        return CVal::result(false, CVal::Str("read_bytes: path must be a string".into()));
    };
    let (off, len) = (nonneg(offset, "offset"), nonneg(length, "length") as usize);
    bytes_result(contained(rel).and_then(|path| {
        (|| -> std::io::Result<Vec<u8>> {
            let mut f = std::fs::File::open(&path)?;
            f.seek(SeekFrom::Start(off))?;
            let mut buf = Vec::new();
            f.take(len as u64).read_to_end(&mut buf)?;
            Ok(buf)
        })()
        .map_err(|e| e.to_string())
    }))
}

/// `file_size(path)` -- the file's byte length as `ok(scalar)` / `err`.
#[must_use]
pub fn file_size(path: &CVal) -> CVal {
    let CVal::Str(rel) = path else {
        return CVal::result(false, CVal::Str("file_size: path must be a string".into()));
    };
    match contained(rel).and_then(|p| {
        std::fs::metadata(p)
            .map(|m| m.len())
            .map_err(|e| e.to_string())
    }) {
        Ok(n) => CVal::result(true, CVal::Arr(DenseArray::from_scalar(n as f64))),
        Err(e) => CVal::result(false, CVal::Str(format!("file_size: {e}"))),
    }
}
