//! mlpl-web App component + `start()` entrypoint. The App body
//! is intentionally a 1-to-4-line sequence of init / process /
//! render phases; each phase is a named helper in a sibling
//! file:
//!
//! - `app_hooks` -- custom hooks (`use_sessions`, `use_ui_state`,
//!   `use_upload_state`, `use_scroll_effect`).
//! - `app_active` -- pure `active_context(sessions, ui)`
//!   derivation.
//! - `app_callbacks` -- `build_callbacks(active, upload, ui)`.
//! - `app_log` -- side-effect `log_connect_mode()`.
//!
//! Saga 33 step 007.

use yew::prelude::*;

use crate::app_active::active_context;
use crate::app_callbacks::build_callbacks;
use crate::app_hooks::{
    use_escape_closes_dialogs, use_onboarding_state, use_scroll_effect, use_sessions, use_ui_state,
    use_upload_state,
};
use crate::app_log::log_connect_mode;
use crate::render::{RenderArgs, render};

/// Mount the Yew application. Called from `src/main_wasm_body.rs`
/// via `include!()` from `src/main.rs`.
pub fn start() {
    yew::Renderer::<App>::new().render();
}

#[function_component(App)]
fn app() -> Html {
    let sessions = use_sessions();
    let ui = use_ui_state();
    let upload = use_upload_state();
    let onboarding = use_onboarding_state();
    log_connect_mode();
    let active = active_context(&sessions, &ui);
    let callbacks = build_callbacks(&active, &upload, &ui);
    use_scroll_effect(active.history.clone());
    use_escape_closes_dialogs(ui.dialog_open.clone(), ui.lesson_idx.clone());
    render(RenderArgs::from_parts(
        callbacks, ui, upload, active, onboarding,
    ))
}
