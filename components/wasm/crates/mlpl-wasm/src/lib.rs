//! WASM bindings for MLPL.

mod upload;

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

    /// Saga 29 step 011 follow-up: bind a `[1, 3, h, w]` image
    /// tensor under `name` in the session's environment. The
    /// JS side decodes the user-selected file (any source size,
    /// any PNG / JPEG / WebP format the browser supports) into
    /// a Canvas at the target `h`x`w` size, then passes the
    /// resulting RGB pixels in channel-first f64 layout
    /// (`[3, h, w]` row-major, values in `[-1, 1]`) here. This
    /// keeps image decoding + resizing in JS/Canvas where it
    /// is fast and dependency-free, while letting the trained
    /// MLPL model treat the result identically to a
    /// `load_preloaded("pets_tiny")` slice.
    pub fn bind_image(&self, name: &str, rgb_chw: &[f64], h: u32, w: u32) -> Result<(), JsValue> {
        use mlpl_array::{DenseArray, Shape};
        let h = h as usize;
        let w = w as usize;
        let expected = 3 * h * w;
        if rgb_chw.len() != expected {
            return Err(JsValue::from_str(&format!(
                "bind_image: expected {expected} floats for [3, {h}, {w}], got {}",
                rgb_chw.len()
            )));
        }
        let arr = DenseArray::new(Shape::new(vec![1, 3, h, w]), rgb_chw.to_vec())
            .map_err(|e| JsValue::from_str(&format!("{e}")))?
            .with_labels(vec![
                Some("batch".into()),
                Some("channel".into()),
                Some("y".into()),
                Some("x".into()),
            ])
            .map_err(|e| JsValue::from_str(&format!("{e}")))?;
        self.env.borrow_mut().set(name.into(), arr);
        Ok(())
    }
}

impl WasmSession {
    pub fn eval_with_values(&self, input: &str) -> EvalResult {
        eval_input_with_values(input, &mut self.env.borrow_mut())
    }

    pub(crate) fn env_mut(&self) -> std::cell::RefMut<'_, Environment> {
        self.env.borrow_mut()
    }
}

pub struct EvalResult {
    pub display: String,
    pub values: Option<Vec<f64>>,
    pub shape: Vec<usize>,
}

fn eval_input_with_values(input: &str, env: &mut Environment) -> EvalResult {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return empty_result();
    }
    if let Some(out) = mlpl_eval::inspect(env, trimmed) {
        return text_result(out);
    }
    let tokens = match mlpl_parser::lex(trimmed) {
        Ok(t) => t,
        Err(e) => return text_result(format!("error: {e}")),
    };
    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return empty_result(),
        Ok(s) => s,
        Err(e) => return text_result(format!("error: {e}")),
    };
    match mlpl_eval::eval_program_value(&stmts, env) {
        Ok(value) => value_to_result(value),
        Err(e) => text_result(format!("error: {e}")),
    }
}

fn empty_result() -> EvalResult {
    EvalResult { display: String::new(), values: None, shape: vec![] }
}

fn text_result(display: String) -> EvalResult {
    EvalResult { display, values: None, shape: vec![] }
}

fn value_to_result(value: mlpl_eval::Value) -> EvalResult {
    match value {
        mlpl_eval::Value::Array(ref arr) => EvalResult {
            display: format!("{}", mlpl_eval::Value::Array(arr.clone())),
            values: Some(arr.data().to_vec()),
            shape: arr.shape().dims().to_vec(),
        },
        other => text_result(format!("{other}")),
    }
}

fn eval_input(input: &str, env: &mut Environment) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Introspection commands (`:vars`, `:models`, `:fns`, `:wsid`,
    // `:describe <name>`) short-circuit evaluation so they work
    // identically in the terminal and web REPLs.
    if let Some(out) = mlpl_eval::inspect(env, trimmed) {
        return out;
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

    match mlpl_eval::eval_program_value(&stmts, env) {
        Ok(value) => format!("{value}"),
        Err(e) => format!("error: {e}"),
    }
}
