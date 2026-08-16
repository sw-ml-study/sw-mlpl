//! The `load_extension` builtin: dynamically load a native extension by
//! logical name (resolved via `MLPL_EXTENSION_PATH`) or by an explicit
//! shared-library path. On success it returns the registered namespace
//! as a string, so `namespace:function` calls then resolve; on failure
//! it returns an `err(...)` Result rather than raising.

use std::path::PathBuf;

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    _span: &mlpl_core::Span,
) -> Option<Result<Value, EvalError>> {
    match name {
        "load_extension" => Some(eval_load(args, env, trace)),
        _ => None,
    }
}

/// `load_extension(name_or_path)` -- a bare name is resolved via
/// `MLPL_EXTENSION_PATH`; a path (has a separator or the dylib suffix)
/// loads directly. Returns the namespace `Str` or an `err(...)` Result.
fn eval_load(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "load_extension".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let Value::Str(spec) = eval_expr(&args[0], env, trace)? else {
        return Err(EvalError::ExtensionError {
            function: "load_extension".into(),
            message: "argument must be a string (extension name or path)".into(),
        });
    };
    Ok(to_value(load(&spec)))
}

/// Resolve `spec` to a shared library and load it, returning the
/// registered namespace name. A spec that contains a path separator or
/// ends in the shared-library suffix is a path; otherwise it is a
/// logical name resolved via `MLPL_EXTENSION_PATH`.
fn load(spec: &str) -> Result<String, String> {
    let is_path = spec.contains('/')
        || spec.contains(std::path::MAIN_SEPARATOR)
        || spec.ends_with(&format!(".{}", std::env::consts::DLL_EXTENSION));
    let path = if is_path {
        PathBuf::from(spec)
    } else {
        mlpl_extension_loader::resolve_extension_path(spec)?
    };
    unsafe { mlpl_extension_loader::load_c_extension(&path) }
}

/// Map the load result to a value: the namespace `Str` on success, an
/// `err(message)` Result on failure (so a bad load never raises).
fn to_value(result: Result<String, String>) -> Value {
    match result {
        Ok(namespace) => Value::Str(namespace),
        Err(message) => Value::Result {
            ok: false,
            payload: Box::new(Value::Str(message)),
        },
    }
}
