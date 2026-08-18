//! `read_stdin_chunk(max_bytes)` -- the bounded, incremental raw-byte
//! stdin source (../demo-file-processing bounded-stdin pipeline). One
//! call issues a single `read` of up to `max_bytes` bytes and returns
//! `ok({bytes, eof})`: `bytes` is a rank-1 array of the byte values
//! read (0..=255 as f64), and `eof` is `1` ONLY on the terminal empty
//! read (`read` -> 0 bytes) and `0` for any non-empty chunk. Short
//! non-empty chunks are normal. Repeated calls after EOF keep returning
//! `{bytes: [], eof: 1}` without blocking (stdin EOF is sticky).
//!
//! `max_bytes` must be a rank-0 positive integer; it is validated
//! BEFORE stdin is touched, so a bad budget is a clean `err` that
//! consumes no input. A terminal stdin is refused, exactly like
//! `read_stdin`. The compile-to-Rust path (`mlpl-rt-value`
//! `stdin_chunk.rs`) mirrors these byte / EOF / validation / error
//! semantics; only the TTY policy differs per surface (a compiled CLI
//! reads a TTY like `cat`, matching its `read_stdin`).

use std::collections::BTreeMap;
use std::io::{IsTerminal, Read};

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use crate::result_str::ok_or_err;
use mlpl_eval_types::{EvalError, Value};

/// `read_stdin_chunk(max_bytes)` -> `ok({bytes, eof})` / `err`.
pub(crate) fn eval_read_stdin_chunk(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [arg] = args else {
        return Err(EvalError::BadArity {
            func: "read_stdin_chunk".into(),
            expected: 1,
            got: args.len(),
        });
    };
    Ok(ok_or_err(
        max_scalar(&eval_expr(arg, env, trace)?).and_then(read_chunk),
    ))
}

/// Read up to `max` bytes (one `read`) into a `{bytes, eof}` record, or
/// an error message on a TTY / read failure. Validation already ran, so
/// this only fails on I/O.
fn read_chunk(max: usize) -> Result<Value, String> {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return Err(
            "read_stdin_chunk: stdin is a terminal; pipe input or use args() instead".into(),
        );
    }
    let mut buf = vec![0u8; max];
    let n = stdin
        .lock()
        .read(&mut buf)
        .map_err(|e| format!("read_stdin_chunk: read failed: {e}"))?;
    Ok(chunk_record(&buf[..n]))
}

/// The `{bytes, eof}` record for a freshly read slice (`eof` iff empty).
fn chunk_record(bytes: &[u8]) -> Value {
    let arr = DenseArray::from_vec(bytes.iter().map(|&b| f64::from(b)).collect());
    let eof = f64::from(u8::from(bytes.is_empty()));
    let mut fields = BTreeMap::new();
    fields.insert("bytes".into(), Value::Array(arr));
    fields.insert("eof".into(), Value::Array(DenseArray::from_scalar(eof)));
    Value::Record { fields }
}

/// Validate `max_bytes` is a rank-0 positive integer; returns the count
/// or an error message (checked BEFORE stdin is read).
fn max_scalar(v: &Value) -> Result<usize, String> {
    match v {
        Value::Array(a) if a.rank() == 0 => {
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
