//! `run_script(path, {source_dir, data_dir, capture})` -- the
//! isolated-script-execution half of the language-native-runner
//! contract. The evaluator resolves and sandbox-checks the
//! paths, then dispatches to the host hook (registered by the
//! CLI, which owns the source loader); surfaces without a hook
//! answer with a plain err value. Same-process execution:
//! `exit()` inside the child terminates the whole process --
//! process-level isolation stays the runner's separate mode.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::{Environment, RunScriptOpts};
use crate::fncall_fs::{contained, fs_err};
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn try_dispatch(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Option<Result<Value, EvalError>> {
    (name == "run_script").then(|| eval_run_script(args, env, trace))
}

fn eval_run_script(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let (path_expr, opts_expr) = match args {
        [p] => (p, None),
        [p, o] => (p, Some(o)),
        _ => {
            return Err(EvalError::BadArity {
                func: "run_script".into(),
                expected: 2,
                got: args.len(),
            });
        }
    };
    let Value::Str(rel) = crate::eval::eval_expr(path_expr, env, trace)? else {
        return Err(EvalError::Unsupported(
            "run_script: the first argument is the script path (a string)".into(),
        ));
    };
    let opts_val = match opts_expr {
        Some(e) => Some(crate::eval::eval_expr(e, env, trace)?),
        None => None,
    };
    let Some(hook) = env.run_script_hook else {
        return Ok(fs_err(
            "run_script: no script execution on this surface".into(),
        ));
    };
    let Some(root) = env.fs_root.clone() else {
        return Ok(fs_err("run_script: no filesystem sandbox".into()));
    };
    let opts = match build_opts(&root, opts_val.as_ref()) {
        Ok(o) => o,
        Err(msg) => return Ok(fs_err(format!("run_script: {msg}"))),
    };
    match contained(&root, &rel) {
        Ok(path) => Ok(hook(&path, &opts)),
        Err(msg) => Ok(fs_err(format!("run_script: {msg}"))),
    }
}

/// Resolve option paths against the sandbox; capture is a
/// scalar 1. Unknown fields pass (additive tolerance).
fn build_opts(root: &std::path::Path, opts: Option<&Value>) -> Result<RunScriptOpts, String> {
    let mut out = RunScriptOpts::default();
    let Some(Value::Record { fields }) = opts else {
        return match opts {
            None => Ok(out),
            Some(v) => Err(format!(
                "options must be a record -- got {}",
                mlpl_eval_types::value_kind(v)
            )),
        };
    };
    for key in ["source_dir", "data_dir"] {
        if let Some(Value::Str(rel)) = fields.get(key) {
            let resolved = contained(root, rel)?;
            if key == "source_dir" {
                out.source_dir = Some(resolved);
            } else {
                out.data_dir = Some(resolved);
            }
        }
    }
    out.capture = matches!(fields.get("capture"), Some(Value::Array(a)) if a.data() == [1.0]);
    Ok(out)
}
