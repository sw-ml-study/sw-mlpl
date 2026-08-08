//! Sandboxed filesystem builtins (the language-native-runner
//! contract): every operation returns a RESULT value (never a
//! hard error for I/O outcomes), resolves paths against the
//! environment's sandbox root, and refuses to touch anything
//! outside it. `fs_walk` returns root-relative paths in lexical
//! order so results feed straight back into the other calls.

use std::path::{Path, PathBuf};

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    if matches!(name, "read_bytes" | "write_bytes") {
        return Some(crate::fs_bytes::eval_fs_bytes(name, args, env, trace));
    }
    if !matches!(name, "fs_walk" | "read_text" | "write_text" | "remove_path") {
        return None;
    }
    Some(crate::fs_ops::eval_fs(name, args, env, trace))
}

/// `ok(payload)` / `err("message")` -- the fs builtins speak
/// Result values so runner code composes with `?`.
pub(crate) fn fs_ok(payload: Value) -> Value {
    Value::Result {
        ok: true,
        payload: Box::new(payload),
    }
}

pub(crate) fn fs_err(message: String) -> Value {
    Value::Result {
        ok: false,
        payload: Box::new(Value::Str(message)),
    }
}

pub(crate) fn one() -> Value {
    Value::Array(DenseArray::from_scalar(1.0))
}

/// Resolve `rel` against the sandbox root and verify containment:
/// the nearest EXISTING ancestor of the candidate must
/// canonicalize inside the canonical root (so writes to
/// not-yet-existing files still check their parent).
pub(crate) fn contained(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let canon_root = root
        .canonicalize()
        .map_err(|e| format!("sandbox root {}: {e}", root.display()))?;
    let candidate = root.join(rel);
    let mut probe = candidate.clone();
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
    for name in popped.iter().rev() {
        resolved.push(name);
    }
    if resolved.starts_with(&canon_root) {
        Ok(resolved)
    } else {
        Err(format!("{rel}: outside the sandbox"))
    }
}
