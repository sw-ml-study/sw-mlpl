//! Built-in function registry and dispatch for MLPL.

mod builtins;
mod error;
mod grid_builtin;
mod math_builtins;
mod prng;
mod random_builtins;

pub use builtins::call_builtin;
pub use error::RuntimeError;
