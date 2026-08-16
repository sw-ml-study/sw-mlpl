//! `file_metadata(path)` -- a file's sandbox-confined metadata as a
//! record `{kind, size, modified_unix_ms}`. The timestamp is the
//! last-MODIFIED time as an exact UTC Unix-millisecond integer (never
//! access time, creation/birth time, local time, or the current
//! clock); a platform with no modification time is an `err`, never a
//! silent 0. Sandboxed and symlink-confined exactly like `file_size`
//! (`env.fs_root` + `contained`).

use std::collections::BTreeMap;
use std::fs::Metadata;
use std::path::Path;
use std::time::UNIX_EPOCH;

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_fs::{contained, fs_err, fs_ok};
use mlpl_eval_types::{EvalError, Value};

/// `file_metadata(path)` -> `ok({kind, size, modified_unix_ms})` /
/// `err`. Dispatch guarantees the name; arity is checked here.
pub(crate) fn eval_file_metadata(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let [arg] = args else {
        return Err(EvalError::BadArity {
            func: "file_metadata".into(),
            expected: 1,
            got: args.len(),
        });
    };
    let Value::Str(rel) = crate::eval::eval_expr(arg, env, trace)? else {
        return Err(EvalError::Unsupported(
            "file_metadata: the argument is a path string".into(),
        ));
    };
    Ok(read_metadata(env.fs_root.as_deref(), &rel)
        .map(fs_ok)
        .unwrap_or_else(|e| fs_err(format!("file_metadata: {e}"))))
}

/// Resolve `rel` inside the sandbox and read its metadata into a
/// `{kind, size, modified_unix_ms}` record, or a descriptive error
/// (including an unconfigured surface). `kind` is "dir"/"file"/"other"
/// (metadata follows symlinks, so the target's kind is reported); the
/// numeric fields are scalar cells.
fn read_metadata(root: Option<&Path>, rel: &str) -> Result<Value, String> {
    let root = root.ok_or("no filesystem sandbox on this surface")?;
    let path = contained(root, rel)?;
    let meta = std::fs::metadata(&path).map_err(|e| e.to_string())?;
    let ms = modified_unix_ms(&meta)?;
    let kind = if meta.is_dir() {
        "dir"
    } else if meta.is_file() {
        "file"
    } else {
        "other"
    };
    let mut fields = BTreeMap::new();
    fields.insert("kind".into(), Value::Str(kind.into()));
    fields.insert("size".into(), scalar(meta.len() as f64));
    fields.insert("modified_unix_ms".into(), scalar(ms as f64));
    Ok(Value::Record { fields })
}

/// The last-modified time as exact Unix milliseconds. An unavailable
/// modification time (or one preceding the epoch) is a hard error --
/// never a silent substitute.
fn modified_unix_ms(meta: &Metadata) -> Result<u128, String> {
    meta.modified()
        .map_err(|e| format!("modification time unavailable: {e}"))?
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .map_err(|e| format!("modification time precedes the Unix epoch: {e}"))
}

/// A scalar numeric field value (exact integers stay exact in f64).
fn scalar(n: f64) -> Value {
    Value::Array(DenseArray::from_scalar(n))
}
