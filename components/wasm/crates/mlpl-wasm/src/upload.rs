//! Upload-result binding helpers on `WasmSession`. Saga 29
//! step 016.
//!
//! `bind_upload_result_ok` / `bind_upload_result_err` are
//! the Result-aware counterparts to the older `bind_image`.
//! Lifted out of `lib.rs` so the top-level module stays under
//! the sw-checklist 7-function budget.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::Value;
use mlpl_eval::env_api::*;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::WasmSession;

#[wasm_bindgen]
impl WasmSession {
    /// Saga 29 step 016: bind an uploaded photo as a
    /// `Value::Result` so a cancelled / failed upload becomes
    /// `Err("...")` rather than an undefined variable. On
    /// success the payload is a `Value::Record` with fields
    /// `pixels: [1, 3, h, w]`, `h: scalar`, `w: scalar`. Used
    /// by both the `:upload x` REPL command and the legacy
    /// "Upload Image" button (which uses `name = "uploaded"`).
    pub fn bind_upload_result_ok(
        &self,
        name: &str,
        rgb_chw: &[f64],
        h: u32,
        w: u32,
    ) -> Result<(), JsValue> {
        let hu = h as usize;
        let wu = w as usize;
        let expected = 3 * hu * wu;
        if rgb_chw.len() != expected {
            return Err(JsValue::from_str(&format!(
                "bind_upload_result_ok: expected {expected} floats for [3, {hu}, {wu}], got {}",
                rgb_chw.len()
            )));
        }
        let pixels = DenseArray::new(Shape::new(vec![1, 3, hu, wu]), rgb_chw.to_vec())
            .map_err(|e| JsValue::from_str(&format!("{e}")))?
            .with_labels(vec![
                Some("batch".into()),
                Some("channel".into()),
                Some("y".into()),
                Some("x".into()),
            ])
            .map_err(|e| JsValue::from_str(&format!("{e}")))?;
        let mut fields = std::collections::BTreeMap::new();
        fields.insert("pixels".to_string(), Value::Array(pixels));
        fields.insert(
            "h".to_string(),
            Value::Array(DenseArray::from_scalar(f64::from(h))),
        );
        fields.insert(
            "w".to_string(),
            Value::Array(DenseArray::from_scalar(f64::from(w))),
        );
        self.env_mut()
            .set_result(name.into(), true, Value::Record { fields });
        Ok(())
    }

    /// Saga 29 step 016: bind the Err side of an upload Result.
    /// `message` is the user-visible error string (e.g.
    /// "cancelled", "decode failed: ...").
    pub fn bind_upload_result_err(&self, name: &str, message: &str) {
        self.env_mut()
            .set_result(name.into(), false, Value::Str(message.into()));
    }
}
