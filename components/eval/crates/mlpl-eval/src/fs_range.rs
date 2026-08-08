//! Bounded seek reads for large files: `read_bytes(path, offset,
//! length)` reads at most `length` bytes from `offset` WITHOUT
//! materializing the rest, and `file_size(path)` reports the byte
//! count from metadata (no read). Sandboxed and Result-returning
//! like the whole-file `read_bytes` in `fs_bytes`.

use std::io::{Read, Seek, SeekFrom};

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_fs::{contained, fs_err, fs_ok};
use mlpl_eval_types::{EvalError, Value};

/// `read_bytes(path, offset, length)` (the 3-arg overload):
/// bounded slice read. Dispatch guarantees three arguments.
pub(crate) fn eval_read_range(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let Value::Str(rel) = crate::eval::eval_expr(&args[0], env, trace)? else {
        return Err(EvalError::Unsupported(
            "read_bytes: the first argument is a path string".into(),
        ));
    };
    let offset = nonneg_int(
        "read_bytes",
        "offset",
        &crate::eval::eval_expr(&args[1], env, trace)?,
    )?;
    let length = nonneg_int(
        "read_bytes",
        "length",
        &crate::eval::eval_expr(&args[2], env, trace)?,
    )?;
    let Some(root) = env.fs_root.clone() else {
        return Ok(fs_err(
            "read_bytes: no filesystem sandbox on this surface".into(),
        ));
    };
    Ok(read_range(&root, &rel, offset, length as usize))
}

/// `file_size(path)` -> `ok(byte count)` / `err`, from metadata.
pub(crate) fn eval_file_size(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [arg] = args else {
        return Err(EvalError::BadArity {
            func: "file_size".into(),
            expected: 1,
            got: args.len(),
        });
    };
    let Value::Str(rel) = crate::eval::eval_expr(arg, env, trace)? else {
        return Err(EvalError::Unsupported(
            "file_size: the argument is a path string".into(),
        ));
    };
    let Some(root) = env.fs_root.clone() else {
        return Ok(fs_err(
            "file_size: no filesystem sandbox on this surface".into(),
        ));
    };
    let sized = contained(&root, &rel).and_then(|p| {
        std::fs::metadata(p)
            .map(|m| m.len())
            .map_err(|e| e.to_string())
    });
    Ok(match sized {
        Ok(n) => fs_ok(Value::Array(DenseArray::from_scalar(n as f64))),
        Err(e) => fs_err(format!("file_size: {e}")),
    })
}

/// A scalar non-negative integer argument, or a hard error.
fn nonneg_int(who: &str, name: &str, v: &Value) -> Result<u64, EvalError> {
    match v {
        Value::Array(a) if a.rank() == 0 && a.data()[0] >= 0.0 && a.data()[0].fract() == 0.0 => {
            Ok(a.data()[0] as u64)
        }
        _ => Err(EvalError::Unsupported(format!(
            "{who}: {name} must be a non-negative integer"
        ))),
    }
}

/// Seek to `offset` and read up to `length` bytes (clamped at
/// EOF), returning a rank-1 byte array.
fn read_range(root: &std::path::Path, rel: &str, offset: u64, length: usize) -> Value {
    let read = contained(root, rel).and_then(|path| {
        (|| -> std::io::Result<Vec<u8>> {
            let mut f = std::fs::File::open(&path)?;
            f.seek(SeekFrom::Start(offset))?;
            let mut buf = Vec::new();
            f.take(length as u64).read_to_end(&mut buf)?;
            Ok(buf)
        })()
        .map_err(|e| e.to_string())
    });
    match read {
        Ok(bytes) => {
            let data: Vec<f64> = bytes.iter().map(|&b| f64::from(b)).collect();
            match DenseArray::new(Shape::vector(data.len()), data) {
                Ok(arr) => fs_ok(Value::Array(arr)),
                Err(e) => fs_err(format!("read_bytes: {e}")),
            }
        }
        Err(e) => fs_err(format!("read_bytes: {e}")),
    }
}
