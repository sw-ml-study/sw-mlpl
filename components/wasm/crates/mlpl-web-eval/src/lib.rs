//! Web-side eval transports (native blocking + wasm fetch +
//! SSE streaming). Spike step 015 split the stack into three
//! crates -- mlpl-web-eval-core (state, wire types, url/guard,
//! loss_trace), mlpl-web-trace (telemetry/frame stores,
//! summarizer, fetchers), and this crate (the transports) --
//! with the module re-exports below keeping every historical
//! `mlpl_web_eval::X` (and internal `crate::X`) path valid.

pub use mlpl_web_eval_core::{
    connect_guard, devices, eval_url, loss_trace, narration, state, wire,
};
pub use mlpl_web_trace::{frame_trace, summary, telemetry_trace};
// The fetchers are wasm-only modules (inner #![cfg]), so their
// re-exports carry the same gate.
#[cfg(target_arch = "wasm32")]
pub use mlpl_web_trace::{connect_viz, ollama_fetch, stats_fetch};

pub mod eval;
mod eval_native_stream;
pub mod eval_sse;
pub mod eval_wasm;
pub(crate) mod eval_wasm_helpers;
pub(crate) mod eval_wasm_stream;
