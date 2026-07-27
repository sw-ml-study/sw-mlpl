//! Content-area Yew components for the web playground:
//! the docs dialog, the REPL input row, the mode bar
//! (paths / tutorial / REPL switcher), and the splash welcome
//! card. Extracted from mlpl-web during saga 82.
//!
//! Each module's deps:
//! - `doc_dialog` -- yew, mlpl-web-glossary, mlpl-web-eval.
//! - `input_row` -- yew, web-sys KeyboardEvent.
//! - `mode_bar` -- yew, mlpl-web-demos, web-sys HtmlSelectElement.
//! - `welcome` -- yew.

pub mod demo_gating;
pub mod doc_dialog;
pub mod input_row;
pub mod mode_bar;
pub mod peer_probe;
pub mod welcome;
