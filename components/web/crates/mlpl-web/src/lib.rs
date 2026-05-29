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
pub mod component_doc_dialog;
pub mod component_footer;
pub mod component_header;
pub mod component_input_row;
pub mod component_mode_bar;
pub mod component_welcome;
pub mod components;
pub mod demos;
pub mod demos_attention;
pub mod demos_autoencoder;
pub mod demos_basics;
pub mod demos_cnn;
pub mod demos_dim_reduction;
pub mod demos_gan;
pub mod demos_lm;
pub mod demos_models;
pub mod demos_rnn;
pub mod demos_udf;
pub mod demos_vit;
pub mod diagrams_view;
pub mod editor_panel;
pub mod entry_render;
pub mod glossary_popup;
pub mod glossary_view;
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
pub mod onboarding_splash;
pub mod onboarding_storage;
pub mod onboarding_tour;
pub mod onboarding_whats_new;
pub mod paths;
mod paths_architectures;
mod paths_history;
mod paths_skills;
pub mod paths_view;
mod paths_visual;
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
