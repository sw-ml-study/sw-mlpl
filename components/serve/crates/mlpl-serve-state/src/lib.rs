//! Server-state layer of the mlpl-serve crate family: AppState,
//! config, the GPU peer registry, persistence, the Ollama proxy, and
//! the viz HTTP handlers.

pub mod config;
pub mod handlers_inspect;
pub mod ollama;
pub mod peers;
pub mod persist;
pub mod viz_handlers;

/// Path-compat alias: `AppState` lives in `config` (merged module).
pub use config as app_state;
