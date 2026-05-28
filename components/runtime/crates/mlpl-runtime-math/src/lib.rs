//! Math and constructor built-in functions extracted from
//! mlpl-runtime.
//!
//! Exposes a `try_call(name, args)` dispatcher that returns
//! `Some(Result<DenseArray, RuntimeError>)` when the name is
//! one of the math/constructor builtins and `None` otherwise.

mod array_ops;
mod dispatch;
mod elementwise;

pub use dispatch::{NAMES, try_call};
