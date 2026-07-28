//! Saga 76: `WasmSession` extracted from lib.rs to keep
//! the crate root module function count under the limit.

use mlpl_eval::env_api::*;
use std::cell::RefCell;

use mlpl_eval::Environment;
use wasm_bindgen::prelude::*;

use crate::eval_impl::{EvalResult, eval_input, eval_input_with_values};

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
