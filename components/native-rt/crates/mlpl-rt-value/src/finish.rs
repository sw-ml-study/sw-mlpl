//! Program completion for compiled MLPL binaries: how a lowered
//! program's final value becomes process output + exit status.
//!
//! A standalone binary's stdout is PRISTINE -- only what the program
//! explicitly emits (via `print` / `write_stdout`) appears. The wrapper
//! never echoes a `Result` the program computed, so binary
//! `write_stdout` output is not followed by an `ok(N)` line.
//!
//! This is a deliberate divergence from the interpreter's `-f` script
//! echo (which prints every non-error final value): a compiled CLI
//! shows a value only when the program's final value is a plain value
//! (not a `Result`), and a final `err(...)` sets a non-zero exit like a
//! well-behaved CLI. To surface a `Result` on stdout, render it with
//! `disp(...)` (a plain string, which is echoed).

use crate::CVal;

/// Finish a compiled program from its final value:
/// - `err(msg)` -> print `msg` to stderr and exit `1`;
/// - `ok(_)` -> suppressed (any effect already wrote its own output);
/// - any other (non-`Result`) value -> render to stdout, exit `0`.
pub fn finish_program(v: &CVal) {
    match v {
        CVal::Result { ok: false, payload } => {
            eprintln!("{payload}");
            std::process::exit(1);
        }
        CVal::Result { ok: true, .. } => {}
        other => println!("{other}"),
    }
}
