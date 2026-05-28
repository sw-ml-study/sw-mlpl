//! Saga 33 step 006: scroll-and-focus side-effect helper
//! extracted from `main.rs`. Called from the `App`'s
//! `use_effect_with(active_history, ...)` so the output pane
//! sticks to the bottom and the REPL input keeps focus after
//! every history append.

use wasm_bindgen::JsCast;
use web_sys::HtmlInputElement;

/// Scroll the `#output` panel to its bottom and focus the
/// `#repl-input`. Best-effort: missing elements (e.g. when the
/// tutorial pane is mounted alone) are silently skipped.
pub fn scroll_and_focus() {
    if let Some(window) = web_sys::window()
        && let Some(document) = window.document()
    {
        if let Some(el) = document.get_element_by_id("output") {
            el.set_scroll_top(el.scroll_height());
        }
        if let Some(el) = document.get_element_by_id("repl-input")
            && let Ok(input) = el.dyn_into::<HtmlInputElement>()
        {
            let _ = input.focus();
        }
    }
}
