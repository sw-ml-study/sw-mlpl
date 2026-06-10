use serde::{Deserialize, Serialize};

/// One GPU on a backend host. Every field is `Option` so a source that
/// cannot report it (no `nvidia-smi`, a partial query) degrades to
/// `null` rather than failing the whole reading.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Gpu {
    /// Device name / type, e.g. `"NVIDIA GeForce RTX 5080"`.
    pub name: Option<String>,
    /// Utilization percentage (0-100).
    pub pct: Option<f64>,
    /// GPU memory in use, megabytes.
    pub vram_used_mb: Option<u64>,
    /// Total GPU memory, megabytes.
    pub vram_total_mb: Option<u64>,
}

/// A point-in-time reading of ONE backend host's resource utilization,
/// returned by that host's `GET /v1/stats` and rendered as the REPL's
/// `:status` report and live sparklines.
///
/// CPU/RAM fields are `Option` (a host lacking `/proc` reports `null`);
/// `gpus` is a list so a host with 0, 1, or 2+ GPUs all serialize
/// through one shape. A future proxy aggregates several of these (one
/// per connected mlpl-serve) into the `:status` backend list. Memory is
/// megabytes; percentages 0-100.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Busy CPU percentage across a short sampling window.
    pub cpu_pct: Option<f64>,
    /// System RAM in use (`MemTotal` - `MemAvailable`), megabytes.
    pub ram_used_mb: Option<u64>,
    /// Total system RAM, megabytes.
    pub ram_total_mb: Option<u64>,
    /// Every GPU on this host (empty on a GPU-less / non-NVIDIA host).
    #[serde(default)]
    pub gpus: Vec<Gpu>,
}
