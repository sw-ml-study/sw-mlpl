//! Auxiliary render helpers: per-entry rendering
//! (`entry::render_entry` + `split_inline_comment` re-export),
//! the tutorial-panel wrapper, and the resize-handle drag
//! component. Lifted out of render-core so render-core's
//! module count stays under the limit.

pub mod editor_panel;
pub mod entry;
pub mod plotly_panel;
pub mod resize_handle;
pub mod telemetry_panel;
pub mod tutorial;
