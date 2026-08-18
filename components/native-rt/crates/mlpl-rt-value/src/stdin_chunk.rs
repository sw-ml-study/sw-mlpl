//! `read_stdin_chunk(max_bytes)` for the compile-to-Rust path: the
//! bounded, incremental raw-byte stdin source. One call issues a single
//! `read` of up to `max_bytes` bytes and returns
//! `ok({bytes, eof})`, where `bytes` is a rank-1 array of the byte
//! values read (0..=255 as f64) and `eof` is `1` ONLY on the terminal
//! empty read (`read` -> 0 bytes), `0` for any non-empty chunk. Short
//! non-empty chunks are normal; repeated calls after EOF stay
//! `{bytes: [], eof: 1}` and non-blocking (stdin EOF is sticky).
//!
//! `max_bytes` must be a rank-0 positive integer, validated BEFORE
//! stdin is touched, so a bad budget is a clean `err` with no input
//! consumed. Interpreter parity: `mlpl-eval` `stdin_chunk.rs` (same
//! byte / EOF / validation / error semantics). A compiled CLI reads a
//! terminal like `cat`, matching its own `read_stdin`.

use std::io::Read;

use mlpl_array::DenseArray;

use crate::CVal;

/// `read_stdin_chunk(max_bytes)` -- read up to `max_bytes` bytes from
/// stdin (one `read`) into `ok({bytes, eof})`, or `err(msg)` when
/// `max_bytes` is not a positive integer or the read fails.
#[must_use]
pub fn read_stdin_chunk(max_bytes: &CVal) -> CVal {
    let max = match max_scalar(max_bytes) {
        Ok(m) => m,
        Err(msg) => return CVal::result(false, CVal::Str(msg)),
    };
    let mut buf = vec![0u8; max];
    match std::io::stdin().read(&mut buf) {
        Ok(n) => CVal::result(true, chunk_record(&buf[..n])),
        Err(e) => CVal::result(false, CVal::Str(format!("read_stdin_chunk: {e}"))),
    }
}

/// The `{bytes, eof}` record for a freshly read slice (`eof` iff empty).
fn chunk_record(bytes: &[u8]) -> CVal {
    let arr = DenseArray::from_vec(bytes.iter().map(|&b| f64::from(b)).collect());
    let eof = f64::from(u8::from(bytes.is_empty()));
    CVal::record(vec![
        ("bytes".into(), CVal::Arr(arr)),
        ("eof".into(), CVal::Arr(DenseArray::from_scalar(eof))),
    ])
}

/// Validate `max_bytes` is a rank-0 positive integer; returns the count
/// or an error message (checked BEFORE stdin is read).
fn max_scalar(v: &CVal) -> Result<usize, String> {
    match v {
        CVal::Arr(a) if a.rank() == 0 => {
            let m = a.data()[0];
            if m.is_finite() && m.fract() == 0.0 && m >= 1.0 {
                Ok(m as usize)
            } else {
                Err(format!(
                    "read_stdin_chunk: max_bytes must be a positive integer, got {m}"
                ))
            }
        }
        _ => Err("read_stdin_chunk: max_bytes must be a scalar positive integer".into()),
    }
}
