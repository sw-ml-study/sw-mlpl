//! `build_input_callbacks` -- per-render derivation of the
//! input-row event handlers + dialog open/close. Pure; no Yew
//! hooks. Saga 82 moved the `InputCallbacks` struct definition
//! into `mlpl-web-render-types::args` so the shell crate (which
//! also needs the struct) can import it without depending on
//! render-core.

use mlpl_web_handlers_input::input::{make_keydown, make_oninput};
use mlpl_web_handlers_input::toggle::toggle_bool;
use mlpl_web_render_types::app_callbacks::AppCallbacks;
use mlpl_web_render_types::args::InputCallbacks;
use mlpl_web_render_types::state::UiState;

pub fn build_input_callbacks(callbacks: &AppCallbacks, ui: &UiState) -> InputCallbacks {
    InputCallbacks {
        on_input: make_oninput(ui.input_value.clone()),
        on_keydown: make_keydown(
            callbacks.on_submit.clone(),
            ui.input_value.clone(),
            ui.cmd_history.clone(),
            ui.cmd_index.clone(),
            ui.completion_candidates.clone(),
            ui.completion_selected.clone(),
        ),
        open_dialog: toggle_bool(ui.dialog_open.clone(), true),
        close_dialog: toggle_bool(ui.dialog_open.clone(), false),
    }
}
