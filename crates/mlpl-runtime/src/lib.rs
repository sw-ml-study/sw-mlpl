//! Built-in function registry and dispatch for MLPL.

mod builtins;
mod error;
mod math_builtins;

pub use builtins::call_builtin;
pub use error::RuntimeError;
