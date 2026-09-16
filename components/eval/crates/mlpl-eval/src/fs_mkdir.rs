//! `make_dir(path)` -- create a directory (and any missing parents) inside the
//! filesystem sandbox (demo-coding-agent CA4). `write_text`/`write_bytes` do
//! not create parent directories, so a program laying out a new subtree calls
//! `make_dir` first. Sandboxed exactly like the other fs builtins: relative
//! paths only, resolved against `env.fs_root`, refused outside it.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_fs::{contained, fs_err, fs_ok, one};
use mlpl_eval_types::{EvalError, Value};

/// `make_dir(path)` -> `ok(1)` on success (including when the directory already
/// exists), `err(...)` on a sandbox violation or filesystem error.
pub(crate) fn eval_make_dir(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "make_dir".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let rel = match crate::eval::eval_expr(&args[0], env, trace)? {
        Value::Str(s) => s,
        other => {
            return Err(EvalError::Unsupported(format!(
                "make_dir: the argument must be a path string -- got {}",
                mlpl_eval_types::value_kind(&other)
            )));
        }
    };
    let Some(root) = env.fs_root.clone() else {
        return Ok(fs_err(
            "make_dir: no filesystem sandbox on this surface".into(),
        ));
    };
    let made =
        contained(&root, &rel).and_then(|p| std::fs::create_dir_all(p).map_err(|e| e.to_string()));
    Ok(match made {
        Ok(()) => fs_ok(one()),
        Err(e) => fs_err(format!("make_dir: {e}")),
    })
}
