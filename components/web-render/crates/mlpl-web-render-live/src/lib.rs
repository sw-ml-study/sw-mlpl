//! Live-during-eval panels for the connect-telemetry saga: the pure
//! SVG builder for the live loss chart, and the Yew panel that polls
//! `mlpl_web_eval::loss_trace` while a streamed train block runs.

pub mod loss_panel;
pub mod loss_svg;
