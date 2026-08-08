//! Sandboxed raw-byte fs builtins behind `fncall_fs`'s dispatch:
//! `read_bytes(path)` -> `ok(bytes)` and `write_bytes(path,
//! bytes)` -> `ok(1)`, both Result-returning and contained by
//! the environment's sandbox root exactly like the text ops. A
//! byte is an f64 in `0..256` (the `tokenize_bytes` / bit-ops
//! convention), so binary formats round-trip through
//! `tokenize_bytes` / `decode_bytes` and `to_json` / `parse_json`
//! without any new codec surface. The array->`Vec<u8>` validator
//! (`array_to_bytes`) is shared with `decode_bytes`.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_fs::{contained, fs_err, fs_ok, one};
use mlpl_eval_types::{EvalError, Value};

/// Dispatch `read_bytes` / `write_bytes`: arity, path string,
/// sandbox root (err Result if the surface has none), delegate.
pub(crate) fn eval_fs_bytes(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let expected = if name == "read_bytes" { 1 } else { 2 };
    if args.len() != expected {
        return Err(EvalError::BadArity {
            func: name.into(),
            expected,
            got: args.len(),
        });
    }
    let v0 = crate::eval::eval_expr(&args[0], env, trace)?;
    let Value::Str(rel) = v0 else {
        return Err(EvalError::Unsupported(format!(
            "{name}: the first argument is a path string -- got {}",
            mlpl_eval_types::value_kind(&v0)
        )));
    };
    let Some(root) = env.fs_root.clone() else {
        return Ok(fs_err(format!(
            "{name}: no filesystem sandbox on this surface (script mode sets one \
             from --source-dir or the script directory)"
        )));
    };
    if name == "read_bytes" {
        return Ok(read_bytes(&root, &rel));
    }
    let value = crate::eval::eval_expr(&args[1], env, trace)?;
    Ok(write_bytes(&root, &rel, &value))
}

/// Validate a rank-<=1 array as raw bytes: every cell an integer
/// in `0..=255`. Shared by `write_bytes` and `decode_bytes`;
/// `who` names the builtin in the error. A LOUD hard error on any
/// out-of-range or non-integer cell (no silent truncation).
pub(crate) fn array_to_bytes(who: &str, arr: &DenseArray) -> Result<Vec<u8>, EvalError> {
    if arr.rank() > 1 {
        return Err(EvalError::Unsupported(format!(
            "{who}: expected rank <= 1, got rank {}",
            arr.rank()
        )));
    }
    arr.data()
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if !(0.0..=255.0).contains(&v) || v.fract() != 0.0 {
                Err(EvalError::Unsupported(format!(
                    "{who}: cell {i} = {v} is not an integer in 0..=255"
                )))
            } else {
                Ok(v as u8)
            }
        })
        .collect()
}

/// `read_bytes(path)` -> `ok([bytes])` (rank-1, 0..256) / `err`.
fn read_bytes(root: &std::path::Path, rel: &str) -> Value {
    match contained(root, rel).and_then(|p| std::fs::read(p).map_err(|e| e.to_string())) {
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

/// `write_bytes(path, bytes)` -> `ok(1)` / `err`. Invalid input
/// -- a wrong-typed value or an out-of-range/non-integer cell --
/// is an `err` Result (the whole fs API speaks Results), not a
/// hard error.
fn write_bytes(root: &std::path::Path, rel: &str, value: &Value) -> Value {
    let bytes = match value {
        Value::Array(a) => match array_to_bytes("write_bytes", a) {
            Ok(b) => b,
            Err(e) => return fs_err(format!("{e}")),
        },
        other => {
            return fs_err(format!(
                "write_bytes: bytes must be an array -- got {}",
                mlpl_eval_types::value_kind(other)
            ));
        }
    };
    match contained(root, rel).and_then(|p| std::fs::write(p, bytes).map_err(|e| e.to_string())) {
        Ok(()) => fs_ok(one()),
        Err(e) => fs_err(format!("write_bytes: {e}")),
    }
}
