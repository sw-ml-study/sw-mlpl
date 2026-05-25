use wasm_bindgen::JsCast;
use web_sys::{HtmlInputElement, KeyboardEvent};
use yew::prelude::*;

/// Unified entry point for completion-related key handling.
/// Checks the Ctrl+Space trigger first, then popup navigation
/// if the popup is open. Returns true if the key was consumed.
pub fn handle_completion_keys(
    e: &KeyboardEvent,
    input_value: &UseStateHandle<String>,
    candidates: &UseStateHandle<Vec<String>>,
    selected: &UseStateHandle<usize>,
) -> bool {
    if crate::completion::is_completion_trigger(e.ctrl_key(), e.code().as_str()) {
        e.prevent_default();
        fire_completion(e, input_value, candidates, selected);
        return true;
    }
    if !candidates.is_empty() {
        return handle_popup_key(e, input_value, candidates, selected);
    }
    false
}

fn handle_popup_key(
    e: &KeyboardEvent,
    input_value: &UseStateHandle<String>,
    candidates: &UseStateHandle<Vec<String>>,
    selected: &UseStateHandle<usize>,
) -> bool {
    let len = candidates.len();
    match e.key().as_str() {
        "ArrowDown" => {
            e.prevent_default();
            selected.set(crate::completion::next_index(**selected, len));
            true
        }
        "ArrowUp" => {
            e.prevent_default();
            selected.set(crate::completion::prev_index(**selected, len));
            true
        }
        "Enter" => {
            e.prevent_default();
            accept_selected(input_value, candidates, selected);
            true
        }
        "ArrowRight" => handle_right(e, input_value, candidates, selected),
        "Escape" => {
            e.prevent_default();
            candidates.set(Vec::new());
            selected.set(0);
            true
        }
        _ => false,
    }
}

fn handle_right(
    e: &KeyboardEvent,
    input_value: &UseStateHandle<String>,
    candidates: &UseStateHandle<Vec<String>>,
    selected: &UseStateHandle<usize>,
) -> bool {
    let cursor = cursor_position(e);
    if crate::completion::should_accept_right(true, cursor, input_value.len()) {
        e.prevent_default();
        accept_selected(input_value, candidates, selected);
        true
    } else {
        false
    }
}

fn accept_selected(
    input_value: &UseStateHandle<String>,
    candidates: &UseStateHandle<Vec<String>>,
    selected: &UseStateHandle<usize>,
) {
    let idx = (**selected).min(candidates.len().saturating_sub(1));
    if let Some(chosen) = candidates.get(idx) {
        let cur = input_value.len();
        let (out, _) = crate::completion::apply_completion(input_value, cur, chosen);
        input_value.set(out);
    }
    candidates.set(Vec::new());
    selected.set(0);
}

fn fire_completion(
    e: &KeyboardEvent,
    input_value: &UseStateHandle<String>,
    candidates: &UseStateHandle<Vec<String>>,
    selected: &UseStateHandle<usize>,
) {
    let cursor = cursor_position(e);
    let value = (**input_value).clone();
    #[cfg(target_arch = "wasm32")]
    let builtins: Vec<&str> = mlpl_eval::runtime_builtin_names().collect();
    #[cfg(not(target_arch = "wasm32"))]
    let builtins: Vec<&str> = Vec::new();
    apply_tab_match(&value, cursor, &builtins, input_value, candidates, selected);
}

fn apply_tab_match(
    value: &str,
    cursor: usize,
    builtins: &[&str],
    input_value: &UseStateHandle<String>,
    candidates: &UseStateHandle<Vec<String>>,
    selected: &UseStateHandle<usize>,
) {
    match crate::completion::compute_tab_match(value, cursor, builtins.iter().copied()) {
        crate::completion::TabMatch::None => {
            candidates.set(Vec::new());
            selected.set(0);
        }
        crate::completion::TabMatch::Apply { input, cursor: _ } => {
            input_value.set(input);
            candidates.set(Vec::new());
            selected.set(0);
        }
        crate::completion::TabMatch::Popup(v) => {
            candidates.set(v);
            selected.set(0);
        }
    }
}

fn cursor_position(e: &KeyboardEvent) -> usize {
    e.target()
        .and_then(|t| t.dyn_into::<HtmlInputElement>().ok())
        .and_then(|el| el.selection_start().ok().flatten())
        .map(|p| p as usize)
        .unwrap_or(0)
}
