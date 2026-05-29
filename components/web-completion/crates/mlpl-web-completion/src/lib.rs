//! REPL completion popup helpers. Pure functions — prefix
//! extraction, candidate matching, popup navigation, the
//! tab-completion engine — extracted from mlpl-web during
//! saga 79. Re-exported by mlpl-web as the `crate::completion`
//! namespace so callers keep using `crate::completion::*`.
//!
//! Trigger: **Ctrl+Space** (IDE standard -- VS Code, IntelliJ,
//! Emacs). Tab was the original trigger but the browser's
//! focus-traversal default beat `preventDefault`; Ctrl+Space
//! takes over and Tab is reserved for browser element nav.

pub mod engine;
pub mod nav;
pub mod prefix;

pub use engine::{KEYWORDS, REPL_COMMANDS, TabMatch, compute_tab_match, is_completion_trigger};
pub use nav::{next_index, prev_index, should_accept_right};
pub use prefix::{apply_completion, extract_prefix, match_candidates};
