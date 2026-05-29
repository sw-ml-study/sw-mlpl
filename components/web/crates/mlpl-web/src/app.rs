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
use crate::app_hooks::{
    use_global_keydown, use_onboarding_state, use_scroll_effect, use_sessions, use_ui_state,
    use_upload_state,
};
use crate::app_log::log_connect_mode;
use mlpl_web_render_shell::shell::render;
use mlpl_web_render_types::app_callbacks::build_callbacks;
use mlpl_web_render_types::args::RenderArgs;

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
    use_global_keydown(
        ui.dialog_open.clone(),
        ui.lesson_idx.clone(),
        onboarding.show_tour.clone(),
        onboarding.show_splash.clone(),
        ui.show_3d.clone(),
    );
    // Saga 82: only this crate's build.rs emits the BUILD_*
    // env vars; format them here and thread down via
    // RenderArgs.
    let build_info = format!(
        "v{}.{} \u{00b7} {} \u{00b7} {} \u{00b7} {}",
        env!("CARGO_PKG_VERSION"),
        env!("BUILD_COMMIT_COUNT"),
        env!("BUILD_HOST"),
        env!("BUILD_SHA"),
        env!("BUILD_TIMESTAMP"),
    );
    let version_label = format!(
        "v{}.{}",
        env!("CARGO_PKG_VERSION"),
        env!("BUILD_COMMIT_COUNT"),
    );
    render(RenderArgs::from_parts(
        callbacks,
        ui,
        upload,
        active,
        onboarding,
        build_info.into(),
        version_label.into(),
    ))
}
