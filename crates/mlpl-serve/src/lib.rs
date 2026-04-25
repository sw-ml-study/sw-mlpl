//! `mlpl-serve` -- long-running MLPL interpreter
//! exposed as a REST API. Saga 21 step 001 MVP:
//! sessions + eval + health. Inspect endpoint lands
//! in step 002; LLM proxy / SSE / cancellation /
//! persistence are all explicit non-goals.
//!
//! Library + binary: integration tests
//! (`tests/api_tests.rs`) construct routers via
//! `server::build_app(...)` and serve them on
//! random ports; the `mlpl-serve` binary is a thin
//! shell around `server::run(addr, auth_mode)`.

pub mod auth;
pub mod handlers;
pub mod server;
pub mod sessions;
