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
pub mod app_callbacks;
pub mod app_hooks;
pub mod app_log;
pub mod app_state;
pub use mlpl_web_completion as completion;
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
pub use mlpl_web_paths::diagrams as diagrams_view;
pub mod editor_panel;
pub mod entry_render;
pub use mlpl_web_glossary::popup as glossary_popup;
pub use mlpl_web_glossary::view as glossary_view;
pub mod handlers;
pub mod handlers_demo;
pub mod handlers_input;
pub mod handlers_popup;
pub mod handlers_running;
pub mod handlers_submit;
pub mod help;
pub mod mode_callbacks;
pub mod mode_path;
pub mod mode_select;
pub use mlpl_web_onboarding::splash as onboarding_splash;
pub use mlpl_web_onboarding::storage as onboarding_storage;
pub use mlpl_web_onboarding::tour as onboarding_tour;
pub use mlpl_web_onboarding::whats_new as onboarding_whats_new;
pub use mlpl_web_paths::view as paths_view;
pub use mlpl_web_paths_data as paths;
mod plotly_panel;
#[cfg(test)]
mod readme_counts;
pub mod render;
pub mod render_callbacks;
pub mod render_main;
pub mod render_modes;
pub mod render_shell;
pub mod render_shell_chrome;
pub mod render_shell_footer;
pub mod render_shell_header;
pub mod render_shell_modebar;
pub mod render_shell_overlays;
pub mod render_tutorial;
pub mod resize_handle;
pub mod scroll;
pub mod tutorial;
pub mod upload;
pub mod upload_cmd;
pub use mlpl_web_viz3d::events as viz3d_events;
pub use mlpl_web_viz3d::panel as viz3d_panel;
pub use mlpl_web_viz3d::toggle as viz3d_toggle;
