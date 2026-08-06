//! The four sandboxed fs operations behind `fncall_fs`'s
//! dispatch: argument handling here, containment in
//! `fncall_fs::contained`, walking in `fs_walk_impl`.

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::fncall_fs::{contained, fs_err, fs_ok, one};
use mlpl_eval_types::{EvalError, Value};

pub(crate) fn eval_fs(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let mut vals = Vec::with_capacity(args.len());
    for a in args {
        vals.push(crate::eval::eval_expr(a, env, trace)?);
    }
    let Some(root) = env.fs_root.clone() else {
        return Ok(fs_err(format!(
            "{name}: no filesystem sandbox on this surface (script mode sets one from \
             --source-dir or the script directory)"
        )));
    };
    let path_arg = |i: usize| -> Result<String, EvalError> {
        match vals.get(i) {
            Some(Value::Str(s)) => Ok(s.clone()),
            other => Err(EvalError::Unsupported(format!(
                "{name}: argument {} must be a path string -- got {}",
                i + 1,
                other.map_or("nothing", |v| mlpl_eval_types::value_kind(v))
            ))),
        }
    };
    match (name, vals.len()) {
        ("read_text", 1) => Ok(read_text(&root, &path_arg(0)?)),
        ("write_text", 2) => {
            let Some(Value::Str(text)) = vals.get(1) else {
                return Err(EvalError::Unsupported(
                    "write_text: the second argument is the text (a string)".into(),
                ));
            };
            Ok(write_text(&root, &path_arg(0)?, text))
        }
        ("remove_path", 1) => Ok(remove_path(&root, &path_arg(0)?)),
        ("fs_walk", 2) => {
            let Some(Value::Record { fields }) = vals.get(1) else {
                return Err(EvalError::Unsupported(
                    "fs_walk: the second argument is an options record".into(),
                ));
            };
            Ok(crate::fs_walk_impl::walk(&root, &path_arg(0)?, fields))
        }
        _ => Err(EvalError::BadArity {
            func: name.into(),
            expected: if name == "read_text" || name == "remove_path" {
                1
            } else {
                2
            },
            got: vals.len(),
        }),
    }
}

fn read_text(root: &std::path::Path, rel: &str) -> Value {
    match contained(root, rel).and_then(|p| std::fs::read_to_string(p).map_err(|e| e.to_string())) {
        Ok(text) => fs_ok(Value::Str(text)),
        Err(e) => fs_err(format!("read_text: {e}")),
    }
}

fn write_text(root: &std::path::Path, rel: &str, text: &str) -> Value {
    match contained(root, rel).and_then(|p| std::fs::write(p, text).map_err(|e| e.to_string())) {
        Ok(()) => fs_ok(one()),
        Err(e) => fs_err(format!("write_text: {e}")),
    }
}

fn remove_path(root: &std::path::Path, rel: &str) -> Value {
    let removed = contained(root, rel).and_then(|p| {
        if p.is_dir() {
            std::fs::remove_dir_all(&p).map_err(|e| e.to_string())
        } else {
            std::fs::remove_file(&p).map_err(|e| e.to_string())
        }
    });
    match removed {
        Ok(()) => fs_ok(one()),
        Err(e) => fs_err(format!("remove_path: {e}")),
    }
}
