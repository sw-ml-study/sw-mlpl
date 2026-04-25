//! `mlpl-mlx-serve` -- long-running MLX-bound MLPL
//! interpreter exposed as a REST API. Saga R1 step
//! 001: skeleton + sessions + health + an
//! `/eval-on-device` STUB that returns 501. Real
//! wire-format encoding + `eval-on-device`
//! implementation land in step 002; orchestrator
//! peer routing (`mlpl-serve --peer mlx=<url>`) lands
//! in step 003.
//!
//! Reuses `mlpl-serve`'s auth + session storage
//! primitives via path dep + pub imports -- the
//! same `AuthMode`, the same `SessionMap` shape,
//! the same constant-time bearer-token machinery.

pub mod handlers;
pub mod server;
