//! Generic UI helper: a button callback that flips a Yew
//! `UseStateHandle<bool>` to a fixed value. Extracted from the
//! original handlers.rs facade during saga 82.

use yew::prelude::*;

pub fn toggle_bool(handle: UseStateHandle<bool>, value: bool) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| handle.set(value))
}
