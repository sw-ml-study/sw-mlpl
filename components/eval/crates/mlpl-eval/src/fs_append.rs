//! `append_bytes(path, bytes)` -- the bounded incremental SINK
//! (the output half of the ByteSource/ByteSink contract).
//! Appends a rank-<=1 byte array (`0..=255`) to a file, creating
//! it if absent, and returns `ok(count)` / `err`. Position is
//! `file_size(path)` and the flush is implicit per append, so a
//! program can emit output chunk-by-chunk without holding the
//! whole result in memory. Sandboxed and Result-based like the
//! other byte fs ops; the byte validator is shared with
//! `write_bytes` / `decode_bytes`.

use std::io::Write;

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_fs::{contained, fs_err, fs_ok};
use mlpl_eval_types::{EvalError, Value};

/// Dispatch `append_bytes`: path string, sandbox root (err Result
/// if the surface has none), then the append.
pub(crate) fn eval_append_bytes(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [path_arg, value_arg] = args else {
        return Err(EvalError::BadArity {
            func: "append_bytes".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let v0 = crate::eval::eval_expr(path_arg, env, trace)?;
    let Value::Str(rel) = v0 else {
        return Err(EvalError::Unsupported(format!(
            "append_bytes: the first argument is a path string -- got {}",
            mlpl_eval_types::value_kind(&v0)
        )));
    };
    let value = crate::eval::eval_expr(value_arg, env, trace)?;
    let Some(root) = env.fs_root.clone() else {
        return Ok(fs_err(
            "append_bytes: no filesystem sandbox on this surface".into(),
        ));
    };
    Ok(append_bytes(&root, &rel, &value))
}

/// A byte-array value to `Vec<u8>`; invalid input (wrong-typed
/// value or an out-of-range/non-integer cell) is an err MESSAGE
/// the caller surfaces as an err Result.
fn value_to_bytes(value: &Value) -> Result<Vec<u8>, String> {
    match value {
        Value::Array(a) => {
            crate::fs_bytes::array_to_bytes("append_bytes", a).map_err(|e| format!("{e}"))
        }
        other => Err(format!(
            "append_bytes: bytes must be an array -- got {}",
            mlpl_eval_types::value_kind(other)
        )),
    }
}

/// `ok(count)` on success; invalid input is an err Result (the fs
/// API speaks Results, never a hard error for these outcomes).
fn append_bytes(root: &std::path::Path, rel: &str, value: &Value) -> Value {
    let bytes = match value_to_bytes(value) {
        Ok(b) => b,
        Err(e) => return fs_err(e),
    };
    let count = bytes.len();
    let done = contained(root, rel).and_then(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .and_then(|mut f| f.write_all(&bytes))
            .map_err(|e| e.to_string())
    });
    match done {
        Ok(()) => fs_ok(Value::Array(DenseArray::from_scalar(count as f64))),
        Err(e) => fs_err(format!("append_bytes: {e}")),
    }
}
