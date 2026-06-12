//! REPL evaluation handlers: the submit pipeline (`make_submit` and
//! `make_submit_batch`), the demo runner (`make_run_demo`), the
//! running-marker spinner helpers, the `:help` text const, and the
//! REPL clear callback (`make_clear`).
//!
//! Depends on mlpl-web-handlers-upload for `EvalDeps` +
//! `upload_cmd` (which the submit pipeline routes
//! `:upload <name>` lines through).

pub mod clear;
pub mod connect;
pub mod demo;
pub mod help;
pub mod running;
pub mod submit;
