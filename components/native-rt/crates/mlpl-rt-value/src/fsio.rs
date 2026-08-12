//! Compiled-program filesystem READS, sandboxed to a root: the
//! `MLPL_FS_ROOT` environment variable, else the process's current
//! working directory. A compiled binary has no interpreter
//! `Environment`, so the root comes from the process instead of
//! `--source-dir`; otherwise this mirrors mlpl-eval's `read_bytes` /
//! `read_range` / `file_size` + the `contained` sandbox check, and
//! returns the same `ok(..)` / `err(..)` Results.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use mlpl_array::{DenseArray, Shape};

use crate::CVal;

/// Resolve `rel` inside the sandbox root (`MLPL_FS_ROOT` or the cwd)
/// and reject escapes, mirroring the interpreter's `contained`:
/// canonicalize the longest existing prefix, re-append the missing
/// tail, and require the result to stay under the canonical root.
pub(crate) fn contained(rel: &str) -> Result<PathBuf, String> {
    let root = match std::env::var_os("MLPL_FS_ROOT") {
        Some(r) => PathBuf::from(r),
        None => std::env::current_dir().map_err(|e| format!("cwd: {e}"))?,
    };
    let canon_root = root
        .canonicalize()
        .map_err(|e| format!("sandbox root {}: {e}", root.display()))?;
    let mut probe = root.join(rel);
    let mut popped = Vec::new();
    let canon = loop {
        match probe.canonicalize() {
            Ok(c) => break c,
            Err(_) => match (probe.parent(), probe.file_name()) {
                (Some(parent), Some(name)) => {
                    popped.push(name.to_owned());
                    probe = parent.to_path_buf();
                }
                _ => return Err(format!("{rel}: outside the sandbox")),
            },
        }
    };
    let mut resolved = canon;
    resolved.extend(popped.iter().rev());
    if resolved.starts_with(&canon_root) {
        Ok(resolved)
    } else {
        Err(format!("{rel}: outside the sandbox"))
    }
}

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
/// enforced (a hard panic on violation, matching the interpreter's
/// hard error).
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

/// A scalar non-negative integer, or a hard panic (interpreter
/// parity: an invalid offset/length is a hard error, not an `err`).
fn nonneg(a: &DenseArray, who: &str) -> u64 {
    let x = a.data()[0];
    assert!(
        a.rank() == 0 && x >= 0.0 && x.fract() == 0.0,
        "read_bytes: {who} must be a non-negative integer"
    );
    x as u64
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
