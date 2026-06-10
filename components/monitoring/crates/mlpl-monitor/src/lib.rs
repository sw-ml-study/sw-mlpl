//! Backend resource-telemetry facade.
//!
//! `snapshot()` returns a [`Snapshot`] gathered from this host's
//! platform sources -- Linux `/proc` + `nvidia-smi` today, a macOS
//! sibling (`mlpl-monitor-macos`) later for the MLX server -- so
//! callers such as mlpl-serve's `/v1/stats` handler stay
//! platform-agnostic. On a target with no source crate wired, every
//! field is `None` (a valid, honest "unknown" reading).

mod gather;
mod host;

pub use gather::snapshot;
pub use host::hostname;
pub use mlpl_monitor_types::Snapshot;
