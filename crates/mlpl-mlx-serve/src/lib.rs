//! `mlpl-mlx-serve` -- long-running MLX-bound MLPL
//! interpreter exposed as a REST API. Saga R1 step
//! 002: eval-on-device + transfer + release-handle
//! now ship real implementations on top of the wire
//! format module + per-peer handle store. Tensors
//! returned by eval blocks live on this peer's heap;
//! the orchestrator only holds opaque handles.
//! Orchestrator-side peer routing
//! (`mlpl-serve --peer mlx=<url>`) lands in step 003.
//!
//! Reuses `mlpl-serve`'s auth + session storage
//! primitives via path dep + pub imports -- the
//! same `AuthMode`, the same `SessionMap` shape,
//! the same constant-time bearer-token machinery.

pub mod handlers;
pub mod handles;
pub mod server;
pub mod wire;
