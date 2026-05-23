//! Saga 33 step 007: per-render callback derivations extracted
//! from `render()`. These take state handles + already-built
//! callbacks and produce input-event handlers (oninput,
//! onkeydown) + dialog open/close. Pure: no Yew hooks.

use yew::prelude::*;

use crate::app_callbacks::AppCallbacks;
use crate::app_state::UiState;
use crate::handlers::{make_keydown, make_oninput, toggle_bool};

/// Bundle of input-row event callbacks + dialog toggles needed
/// by the outer html! shell.
pub struct InputCallbacks {
    pub on_input: Callback<InputEvent>,
    pub on_keydown: Callback<web_sys::KeyboardEvent>,
    pub open_dialog: Callback<MouseEvent>,
    pub close_dialog: Callback<MouseEvent>,
}

/// Build the input-row event handlers + the dialog open/close
/// togglers in one shot. Pure builder; safe to call from the
/// component body.
pub fn build_input_callbacks(callbacks: &AppCallbacks, ui: &UiState) -> InputCallbacks {
    InputCallbacks {
        on_input: make_oninput(ui.input_value.clone()),
        on_keydown: make_keydown(
            callbacks.on_submit.clone(),
            ui.input_value.clone(),
            ui.cmd_history.clone(),
            ui.cmd_index.clone(),
        ),
        open_dialog: toggle_bool(ui.dialog_open.clone(), true),
        close_dialog: toggle_bool(ui.dialog_open.clone(), false),
    }
}
