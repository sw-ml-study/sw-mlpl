//! Process output primitives for the compile-to-Rust path: `print` /
//! `eprint` mirror the interpreter's return-the-argument output
//! builtins (eval_intercepts.rs `eval_print`). Each writes the value's
//! Display -- with a trailing newline, `println!`-style -- to stdout /
//! stderr and returns the value UNCHANGED, so it composes:
//! `x = print(v)` both binds and shows, and `print(v)` in expression
//! position yields `v`. (Variadic space-joined printing is an
//! interpreter-only convenience for now; the compiler lowers the
//! single-argument form.)

use crate::CVal;

/// `print(v)` -- write the value's Display + newline to stdout and
/// return it unchanged.
#[must_use]
pub fn print(v: &CVal) -> CVal {
    println!("{v}");
    v.clone()
}

/// `eprint(v)` -- write the value's Display + newline to stderr and
/// return it unchanged.
#[must_use]
pub fn eprint(v: &CVal) -> CVal {
    eprintln!("{v}");
    v.clone()
}

/// `exit(code)` -- end the process with `code` (a scalar integer in
/// `0..=255`). Flushes stdout first (`process::exit` skips destructors,
/// so buffered output would otherwise be lost), matching the
/// interpreter's `eval_exit`. Never returns.
pub fn exit(v: &CVal) -> ! {
    let code = exit_code(v);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    std::process::exit(code);
}

/// Read + validate the exit status from a `CVal`: a rank-0 integer in
/// `0..=255`, else a hard error (interpreter parity).
fn exit_code(v: &CVal) -> i32 {
    let CVal::Arr(a) = v else {
        panic!("exit: code must be a scalar integer, got {v:?}");
    };
    let n = a.data().first().copied().unwrap_or(f64::NAN);
    assert!(
        a.rank() == 0 && n.is_finite() && n.fract() == 0.0 && (0.0..=255.0).contains(&n),
        "exit: code must be an integer in 0..=255, got {n}"
    );
    n as i32
}
