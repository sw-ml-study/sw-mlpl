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

pub use mlpl_serve_core::router_layers;
pub use mlpl_serve_core::{auth, devices, eval_viz, sessions, tls};
pub use mlpl_serve_state::{config, handlers_inspect, ollama, peers, persist};
pub mod handlers;
pub mod handlers_eval_task;
pub mod server;
pub mod sse;

/// Compatibility facade: the store half lives in mlpl-serve-core,
/// the HTTP handlers in mlpl-serve-state.
pub mod viz_storage {
    pub use mlpl_serve_core::store::*;
    pub use mlpl_serve_state::viz_handlers::*;
}
