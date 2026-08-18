//! `read_bytes_packed(path[, offset, length])` -- a sandboxed read that
//! returns a u8-packed `Value::Bytes` (1x memory) rather than the 8x
//! f64 `Array` from `read_bytes`, so retained byte tables (e.g.
//! ../demo-ml-utils catalog names) cost 8x less. Whole-file (1-arg) or
//! bounded range (3-arg), sandboxed like `read_bytes`.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_fs::{contained, fs_err, fs_ok};
use mlpl_eval_types::{ByteDtype, EvalError, Value};

/// `read_bytes_packed(path)` / `read_bytes_packed(path, offset, length)`
/// -> `ok(Bytes)` / `err`.
pub(crate) fn eval_read_packed(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let (path, range) = match args {
        [p] => (p, None),
        [p, o, l] => (p, Some((o, l))),
        _ => {
            return Err(EvalError::Unsupported(
                "read_bytes_packed: expects (path) or (path, offset, length)".into(),
            ));
        }
    };
    let Value::Str(rel) = crate::eval::eval_expr(path, env, trace)? else {
        return Err(EvalError::Unsupported(
            "read_bytes_packed: the first argument is a path string".into(),
        ));
    };
    let bounds = match range {
        None => None,
        Some((o, l)) => Some((
            nonneg(&crate::eval::eval_expr(o, env, trace)?)?,
            nonneg(&crate::eval::eval_expr(l, env, trace)?)?,
        )),
    };
    let Some(root) = env.fs_root.clone() else {
        return Ok(fs_err(
            "read_bytes_packed: no filesystem sandbox on this surface".into(),
        ));
    };
    Ok(match read_u8(&root, &rel, bounds) {
        Ok(data) => fs_ok(Value::Bytes {
            dtype: ByteDtype::U8,
            data,
        }),
        Err(e) => fs_err(format!("read_bytes_packed: {e}")),
    })
}

/// Read the whole file, or the `offset,length` slice (EOF-clamped),
/// into a byte buffer.
fn read_u8(root: &Path, rel: &str, bounds: Option<(u64, u64)>) -> Result<Vec<u8>, String> {
    let mut file = std::fs::File::open(contained(root, rel)?).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    match bounds {
        None => {
            file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        }
        Some((off, len)) => {
            file.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
            file.take(len)
                .read_to_end(&mut buf)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(buf)
}

/// A scalar non-negative integer argument, or a hard error.
fn nonneg(v: &Value) -> Result<u64, EvalError> {
    match v {
        Value::Array(a) if a.rank() == 0 && a.data()[0] >= 0.0 && a.data()[0].fract() == 0.0 => {
            Ok(a.data()[0] as u64)
        }
        _ => Err(EvalError::Unsupported(
            "read_bytes_packed: offset and length must be non-negative integers".into(),
        )),
    }
}
