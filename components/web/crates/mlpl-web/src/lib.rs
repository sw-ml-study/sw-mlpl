//! mlpl-web library crate. Saga 33 step 007 converted this
//! package into a lib+bin hybrid so `src/main.rs` is a 3-line
//! entrypoint that `include!()`s a target-specific body. All
//! module declarations live HERE, not in main.rs.

// On native (non-wasm) builds, the entire Yew/wasm-bindgen
// frontend is dead code -- only the `main_not_wasm_body` stub
// runs. Suppress the resulting lints so `cargo build` and
// `cargo clippy -D warnings` stay clean for both targets.
#![cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, unused_imports, unused_variables)
)]

pub mod app;
pub mod app_active;
pub use mlpl_web_render_types::app_callbacks;
pub mod app_hooks;
pub mod app_log;
pub use mlpl_web_completion as completion;
pub use mlpl_web_render_types::state as app_state;
/// Yew components facade. The chrome (header / footer /
/// GithubCorner) lives in `mlpl-web-components-chrome`; the
/// content widgets (doc dialog, input row, mode bar, welcome)
/// live in `mlpl-web-components-content`; tutorial pieces stay
/// in `crate::tutorial` for now. Saga 82 moved everything but
/// tutorial.
pub mod components {
    pub use mlpl_web_components_chrome::footer::{Footer, FooterProps};
    pub use mlpl_web_components_chrome::github_corner::{GithubCorner, UrlProps};
    pub use mlpl_web_components_chrome::header::{Header, HeaderMode, HeaderProps};

    pub use mlpl_web_components_content::doc_dialog::{DocDialog, DocDialogProps};
    pub use mlpl_web_components_content::input_row::{InputRow, InputRowProps};
    pub use mlpl_web_components_content::mode_bar::{ModeBar, ModeBarProps};
    pub use mlpl_web_components_content::welcome::Welcome;

    pub use crate::tutorial::{TutorialPanel, TutorialPanelProps, TutorialView};
}
pub use mlpl_web_demos as demos;
pub use mlpl_web_glossary::popup as glossary_popup;
pub use mlpl_web_glossary::view as glossary_view;
pub use mlpl_web_paths::diagrams as diagrams_view;
pub use mlpl_web_render_aux::editor_panel;
pub use mlpl_web_render_aux::entry as entry_render;
/// Handlers facade. Saga 82 split the original handlers cluster
/// into three sub-crates: input (keyboard + popup + toggle),
/// upload (image upload + upload_cmd + EvalDeps), and eval
/// (submit + demo + clear + help + running). External call sites
/// keep using `crate::handlers::*`.
pub mod handlers {
    pub use mlpl_web_handlers_eval::clear::make_clear;
    pub use mlpl_web_handlers_eval::demo::make_run_demo;
    pub use mlpl_web_handlers_eval::submit::{make_submit, make_submit_batch};

    pub use mlpl_web_handlers_input::input::{make_keydown, make_oninput};
    pub use mlpl_web_handlers_input::toggle::toggle_bool;

    pub use mlpl_web_handlers_upload::eval_deps::EvalDeps;
}
pub use mlpl_web_mode::callbacks as mode_callbacks;
pub use mlpl_web_mode::path as mode_path;
pub use mlpl_web_mode::select as mode_select;
pub use mlpl_web_onboarding::splash as onboarding_splash;
pub use mlpl_web_onboarding::storage as onboarding_storage;
pub use mlpl_web_onboarding::tour as onboarding_tour;
pub use mlpl_web_onboarding::whats_new as onboarding_whats_new;
pub use mlpl_web_paths::view as paths_view;
pub use mlpl_web_paths_data as paths;
pub use mlpl_web_render_aux::plotly_panel;
#[cfg(test)]
mod readme_counts;
pub use mlpl_web_render_aux::resize_handle;
pub use mlpl_web_render_aux::tutorial as render_tutorial;
pub use mlpl_web_render_core::callbacks as render_callbacks;
pub use mlpl_web_render_core::modes as render_modes;
pub use mlpl_web_render_core::panel as render_main;
pub use mlpl_web_render_shell::chrome as render_shell_chrome;
pub use mlpl_web_render_shell::footer as render_shell_footer;
pub use mlpl_web_render_shell::header as render_shell_header;
pub use mlpl_web_render_shell::modebar as render_shell_modebar;
pub use mlpl_web_render_shell::overlays as render_shell_overlays;
pub use mlpl_web_render_shell::shell as render_shell;
/// Render entry alias. Saga 82 split the original `render.rs`
/// into `mlpl_web_render_shell::shell::render` (entry) +
/// `mlpl_web_render_types::args::RenderArgs` (struct).
pub mod render {
    pub use mlpl_web_render_shell::shell::render;
    pub use mlpl_web_render_types::args::RenderArgs;
}
pub mod scroll;
pub use mlpl_web_handlers_upload::upload;
pub use mlpl_web_handlers_upload::upload_cmd;
pub use mlpl_web_tutorial as tutorial;
pub use mlpl_web_viz3d::events as viz3d_events;
pub use mlpl_web_viz3d::panel as viz3d_panel;
pub use mlpl_web_viz3d::toggle as viz3d_toggle;
