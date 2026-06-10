//! Shared wire types for backend resource telemetry.
//!
//! `Snapshot` is the body of `GET /v1/stats`: mlpl-serve gathers it
//! from the host's platform sources and the web REPL renders it as
//! live CPU/RAM/GPU/VRAM sparklines (and a `:status` self-test). This
//! crate is a pure leaf -- no platform code -- so both the server and
//! any client can depend on the same contract.

pub mod render;
pub mod snapshot;
pub mod spark;

pub use snapshot::{Gpu, Snapshot};
