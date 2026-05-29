//! Saga 31 step 006: script-mode builtins for stdin and
//! process control.
//!
//! - `read_stdin()` blocks until EOF and returns a
//!   `Value::Str` with the contents. Refuses to hang when
//!   stdin is a TTY: returns `Err("read_stdin: stdin is a
//!   terminal; pipe input or use args() instead")`.
//! - `read_stdin_lines()` reads to EOF then splits on `\n`,
//!   returning a `Value::StrList`. A trailing empty entry
//!   (from a final `\n`) is stripped so `"a\nb\n"` and
//!   `"a\nb"` both yield `["a", "b"]`.
//! - `exit(code)` terminates the process with the given
//!   integer exit code. Valid range is [0, 255]; out-of-
//!   range values are an eval error before the exit fires.
//!
//! Lives in its own module so `eval_intercepts.rs` stays
//! near the function-count budget (docs/code_metrics.md
//! file-naming convention: behavior in named files).

use std::io::{IsTerminal, Read};

use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::eval::eval_expr;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

/// `read_stdin()` -- block until EOF, return all bytes as a
/// `Value::Str`. Refuses to read from a TTY.
pub(crate) fn eval_read_stdin(args: &[Expr]) -> Result<Value, EvalError> {
    check_no_args("read_stdin", args)?;
    let contents = read_stdin_to_string("read_stdin")?;
    Ok(Value::Str(contents))
}

/// `read_stdin_lines()` -- read to EOF then split on `\n`.
/// Trailing empty entry stripped so `"a\nb\n"` and `"a\nb"`
/// both yield `["a", "b"]`.
pub(crate) fn eval_read_stdin_lines(args: &[Expr]) -> Result<Value, EvalError> {
    check_no_args("read_stdin_lines", args)?;
    let contents = read_stdin_to_string("read_stdin_lines")?;
    let mut items: Vec<String> = contents.split('\n').map(str::to_owned).collect();
    if items.last().is_some_and(String::is_empty) {
        items.pop();
    }
    Ok(Value::StrList { items })
}

/// `exit(code)` -- terminate the script with `code`. Never
/// returns when the code is valid; on invalid range,
/// returns an `EvalError` so the REPL surfaces the message
/// instead of doing a confusing silent exit.
pub(crate) fn eval_exit(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::BadArity {
            func: "exit".into(),
            expected: 1,
            got: args.len(),
        });
    }
    let arr = eval_expr(&args[0], env, trace)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "exit: code must be a scalar, got rank {}",
            arr.rank()
        )));
    }
    let v = arr.data()[0];
    if !v.is_finite() || v.fract() != 0.0 || !(0.0..=255.0).contains(&v) {
        return Err(EvalError::Unsupported(format!(
            "exit: code must be an integer in 0..=255, got {v}"
        )));
    }
    std::process::exit(v as i32);
}

fn check_no_args(name: &str, args: &[Expr]) -> Result<(), EvalError> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(EvalError::BadArity {
            func: name.into(),
            expected: 0,
            got: args.len(),
        })
    }
}

fn read_stdin_to_string(caller: &str) -> Result<String, EvalError> {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return Err(EvalError::Unsupported(format!(
            "{caller}: stdin is a terminal; pipe input or use args() instead"
        )));
    }
    let mut buf = String::new();
    stdin
        .lock()
        .read_to_string(&mut buf)
        .map_err(|e| EvalError::Unsupported(format!("{caller}: read failed: {e}")))?;
    Ok(buf)
}
