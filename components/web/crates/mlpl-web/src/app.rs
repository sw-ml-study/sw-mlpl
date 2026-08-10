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
use crate::app_hooks::{use_onboarding_state, use_sessions, use_ui_state, use_upload_state};
use crate::app_log::{log_connect_mode, use_chrome_hooks};
use mlpl_web_render_shell::shell::render;
use mlpl_web_render_types::app_callbacks::build_callbacks;
use mlpl_web_render_types::args::{BuildLabels, RenderArgs};

/// Mount the Yew application. Called from `src/main_wasm_body.rs`
/// via `include!()` from `src/main.rs`. Same-origin autoconnect
/// runs BEFORE the mount so every component sees the final
/// `?connect=` state (a same-origin probe resolves in tens of ms;
/// static hosts fail it immediately).
pub fn start() {
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async {
        crate::app_log::apply_same_origin_autoconnect().await;
        yew::Renderer::<App>::new().render();
    });
    #[cfg(not(target_arch = "wasm32"))]
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
    use_chrome_hooks(&active, &ui, &onboarding, &callbacks);
    render(RenderArgs::from_parts(
        callbacks,
        ui,
        upload,
        active,
        onboarding,
        build_labels(),
    ))
}

/// Format the build-stamp labels from the BUILD_* env vars that
/// only this crate's build.rs emits (Saga 82); threaded down via
/// RenderArgs.
fn build_labels() -> BuildLabels {
    let info = format!(
        "v{}.{} \u{00b7} {} \u{00b7} {} \u{00b7} {}",
        env!("CARGO_PKG_VERSION"),
        env!("BUILD_COMMIT_COUNT"),
        env!("BUILD_HOST"),
        env!("BUILD_SHA"),
        env!("BUILD_TIMESTAMP"),
    );
    let version = format!(
        "v{}.{}",
        env!("CARGO_PKG_VERSION"),
        env!("BUILD_COMMIT_COUNT")
    );
    BuildLabels {
        info: info.into(),
        version: version.into(),
        time: AttrValue::from(env!("BUILD_TIMESTAMP")),
    }
}
