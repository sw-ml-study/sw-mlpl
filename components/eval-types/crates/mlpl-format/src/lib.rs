//! `format(template, args...)` for MLPL: Python `str.format`-style
//! replacement fields with a format-spec subset. Pure (no interpreter
//! state), so the interpreter and the compiled-program runtime share it.

mod float;
mod number;
mod render;
mod spec;
mod template;

pub use template::{FmtArg, format};
