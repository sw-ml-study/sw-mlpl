//! Atomic sandboxed write behind `fncall_fs`'s dispatch:
//! `write_atomic(path, value)` -> `ok(1)` / `err`. `value` is a
//! string (its UTF-8 bytes) or a rank-<=1 byte array (`0..=255`,
//! validated by the shared `fs_bytes::array_to_bytes`). The bytes
//! go to a hidden temp file in the SAME directory, which is then
//! `rename`d over the target -- atomic on POSIX same-filesystem,
//! so a concurrent reader sees the old file or the whole new one,
//! never a torn write. The temp is removed if either step fails.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_fs::{contained, fs_err, fs_ok, one};
use mlpl_eval_types::{EvalError, Value};

/// Dispatch `write_atomic`: arity, path string, value bytes
/// (string or byte array), sandbox root, then the temp+rename.
pub(crate) fn eval_write_atomic(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [path_arg, value_arg] = args else {
        return Err(EvalError::BadArity {
            func: "write_atomic".into(),
            expected: 2,
            got: args.len(),
        });
    };
    let v0 = crate::eval::eval_expr(path_arg, env, trace)?;
    let Value::Str(rel) = v0 else {
        return Err(EvalError::Unsupported(format!(
            "write_atomic: the first argument is a path string -- got {}",
            mlpl_eval_types::value_kind(&v0)
        )));
    };
    let value = crate::eval::eval_expr(value_arg, env, trace)?;
    let bytes = value_to_bytes(value)?;
    let Some(root) = env.fs_root.clone() else {
        return Ok(fs_err(
            "write_atomic: no filesystem sandbox on this surface (script mode sets \
             one from --source-dir or the script directory)"
                .into(),
        ));
    };
    Ok(write_atomic(&root, &rel, &bytes))
}

/// A string contributes its UTF-8 bytes; a byte array is
/// validated through the shared `array_to_bytes`; anything else
/// is a LOUD hard error.
fn value_to_bytes(value: Value) -> Result<Vec<u8>, EvalError> {
    match value {
        Value::Str(s) => Ok(s.into_bytes()),
        Value::Array(a) => crate::fs_bytes::array_to_bytes("write_atomic", &a),
        other => Err(EvalError::Unsupported(format!(
            "write_atomic: value must be a string or byte array -- got {}",
            mlpl_eval_types::value_kind(&other)
        ))),
    }
}

/// Write `bytes` to a hidden sibling temp, then rename it over
/// the sandbox-contained target. Cleans up the temp on failure.
fn write_atomic(root: &std::path::Path, rel: &str, bytes: &[u8]) -> Value {
    let target = match contained(root, rel) {
        Ok(p) => p,
        Err(e) => return fs_err(format!("write_atomic: {e}")),
    };
    let Some(name) = target.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return fs_err("write_atomic: path has no file name".into());
    };
    let tmp = target.with_file_name(format!(".{name}.tmp"));
    let done = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, &target));
    match done {
        Ok(()) => fs_ok(one()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            fs_err(format!("write_atomic: {e}"))
        }
    }
}
