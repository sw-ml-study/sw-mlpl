//! Built-in function registry and dispatch for MLPL.

mod builtins;
mod error;

pub use builtins::call_builtin;
pub use error::RuntimeError;
