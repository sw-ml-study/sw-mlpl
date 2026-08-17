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
