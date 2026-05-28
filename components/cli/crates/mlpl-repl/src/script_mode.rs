//! `mlpl-repl -f script.mlpl` script-mode driver. Saga 31
//! step 006 wired Unix-style exit codes: a final `Err(...)`
//! exits 1 with the message on stderr, `Ok(_)` or any
//! non-`Result` value exits 0, and `exit(code)` short-
//! circuits with the caller-chosen code. Separate from
//! `eval_line` (interactive REPL) so the script contract
//! does not leak into the prompt loop.

use mlpl_eval::{Environment, Value};
use mlpl_trace::Trace;

use crate::print_trace_summary;
use crate::svg_out::SvgOut;

/// Run a script in `-f` mode. Returns the process exit code:
/// `0` on success, `1` on parse / eval failure, `1` when the
/// script's final expression is `Err(...)`. `exit(code)`
/// short-circuits via `std::process::exit` and never reaches
/// this return.
pub(crate) fn run_script(
    content: &str,
    env: &mut Environment,
    tracing: bool,
    svg_out: &mut SvgOut,
) -> i32 {
    let cleaned: Vec<&str> = content
        .lines()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with('#') { "" } else { line }
        })
        .collect();
    let source = cleaned.join("\n");
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return 0;
    }
    for line in content.lines() {
        let t = line.trim();
        if !t.is_empty() && !t.starts_with('#') {
            println!("> {t}");
        }
    }
    let mut last_trace: Option<Trace> = None;
    let code = eval_script_value(trimmed, env, tracing, &mut last_trace, svg_out);
    if let Some(trace) = &last_trace {
        println!();
        print_trace_summary(trace);
    }
    code
}

fn eval_script_value(
    input: &str,
    env: &mut Environment,
    tracing: bool,
    last_trace: &mut Option<Trace>,
    svg_out: &mut SvgOut,
) -> i32 {
    let tokens = match mlpl_parser::lex(input) {
        Ok(t) => t,
        Err(e) => return report_script_err(input, &e),
    };
    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return 0,
        Ok(s) => s,
        Err(e) => return report_script_err(input, &e),
    };
    let result = if tracing {
        let mut trace = Trace::new(input.into());
        let r = mlpl_eval::eval_program_value_traced(&stmts, env, &mut trace);
        if r.is_ok() {
            *last_trace = Some(trace);
        }
        r
    } else {
        mlpl_eval::eval_program_value(&stmts, env)
    };
    match result {
        Ok(v) => finish_script_value(v, svg_out),
        Err(e) => report_script_err(input, &e),
    }
}

fn finish_script_value(value: Value, svg_out: &mut SvgOut) -> i32 {
    if let Value::Result { ok: false, payload } = &value {
        let msg = match payload.as_ref() {
            Value::Str(s) => s.clone(),
            other => format!("{other}"),
        };
        eprintln!("{msg}");
        return 1;
    }
    let formatted = format!("{value}");
    let display = mlpl_cli::viz_cache::transform_value(&formatted, svg_out.dir.as_deref());
    println!("{display}");
    0
}

fn report_script_err(input: &str, e: &dyn std::fmt::Display) -> i32 {
    eprintln!("  {input}");
    eprintln!("  error: {e}");
    1
}

/// Pick the script path from CLI args. Priority order: explicit
/// `-f` / `--file` flag, then a positional path at `args[1]`
/// (the slot right after the binary name) so shebang
/// invocations work. Returns `None` for interactive REPL mode.
///
/// Saga 31 step 007: positional form is restricted to `args[1]`
/// so it does not collide with a flag's positional value
/// (`--svg-out /tmp/out` etc).
pub(crate) fn resolve_script_path<F>(args: &[String], flag: &F) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    flag("-f")
        .or_else(|| flag("--file"))
        .or_else(|| args.get(1).filter(|a| !a.starts_with('-')).cloned())
}
