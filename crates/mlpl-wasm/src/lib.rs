//! WASM bindings for MLPL.

use std::cell::RefCell;

use mlpl_eval::Environment;
use wasm_bindgen::prelude::*;

/// Evaluate a single MLPL expression without persistent state.
#[wasm_bindgen]
pub fn eval_line(input: &str) -> String {
    let mut env = Environment::new();
    eval_input(input, &mut env)
}

/// A persistent MLPL session with variable bindings.
#[wasm_bindgen]
pub struct WasmSession {
    env: RefCell<Environment>,
}

impl Default for WasmSession {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmSession {
    /// Create a new session with an empty environment.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            env: RefCell::new(Environment::new()),
        }
    }

    /// Evaluate an expression within this session's environment.
    pub fn eval(&self, input: &str) -> String {
        eval_input(input, &mut self.env.borrow_mut())
    }

    /// Reset the session's environment.
    pub fn clear(&self) {
        *self.env.borrow_mut() = Environment::new();
    }
}

fn eval_input(input: &str, env: &mut Environment) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let tokens = match mlpl_parser::lex(trimmed) {
        Ok(t) => t,
        Err(e) => return format!("error: {e}"),
    };

    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return String::new(),
        Ok(s) => s,
        Err(e) => return format!("error: {e}"),
    };

    match mlpl_eval::eval_program(&stmts, env) {
        Ok(arr) => format!("{arr}"),
        Err(e) => format!("error: {e}"),
    }
}
