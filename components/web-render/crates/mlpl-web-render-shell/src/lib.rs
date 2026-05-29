//! Render-shell: the outer Yew layout + the top-level
//! `render()` entry point. Composes `render_main` (in
//! render-core) with the chrome / footer / overlays
//! sub-renderers. mlpl-web's `app.rs` calls this crate's
//! `shell::render` as the single render entrypoint.

pub mod chrome;
pub mod footer;
pub mod header;
pub mod modebar;
pub mod overlays;
pub mod shell;
