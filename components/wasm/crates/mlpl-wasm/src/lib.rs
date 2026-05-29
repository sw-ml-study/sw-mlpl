//! WASM bindings for MLPL.

mod eval_impl;
mod session;
mod upload;

use mlpl_eval::Environment;
use wasm_bindgen::prelude::*;

pub use eval_impl::EvalResult;
pub use session::WasmSession;

/// Evaluate a single MLPL expression without persistent state.
#[wasm_bindgen]
pub fn eval_line(input: &str) -> String {
    let mut env = Environment::new();
    eval_impl::eval_input(input, &mut env)
}
