//! `:ask` prompt construction for the web REPL's connect mode.
//!
//! Extracted from `mlpl-web-handlers-eval`'s `connect.rs` (connect-
//! telemetry step 002) so the prompt-engineering surface -- grounding
//! rules, the compact MLPL reference, page-context readers, endpoint
//! resolution -- lives in one place with room to grow.

pub(crate) mod context;
pub mod prompt;
