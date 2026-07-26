//! Saga 32 step 004: web-side eval pipeline extracted from
//! `apps/mlpl-web`.
//!
//! Hosts the four eval-shaped modules (eval, eval_url,
//! eval_wasm, eval_sse) plus shared state (HistoryEntry,
//! DocTab) and the numeric-output summarizer. All depend
//! only on external crates (mlpl-wasm, gloo, web-sys,
//! serde_json, wasm-bindgen-futures) so this crate is a
//! clean leaf in the dep DAG.

pub mod connect_guard;
pub mod connect_viz;
pub mod devices;
pub mod eval;
mod eval_native;
mod eval_native_stream;
pub mod eval_sse;
pub mod eval_url;
pub mod eval_wasm;
pub(crate) mod eval_wasm_helpers;
pub(crate) mod eval_wasm_stream;
pub mod frame_trace;
pub mod loss_trace;
pub mod ollama_fetch;
pub mod state;
pub mod stats_fetch;
pub mod summary;
mod summary_stats;
pub mod telemetry_trace;
