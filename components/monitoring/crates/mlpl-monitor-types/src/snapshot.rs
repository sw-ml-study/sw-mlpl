use serde::{Deserialize, Serialize};

/// A point-in-time reading of backend resource utilization, returned by
/// `GET /v1/stats` and rendered as the REPL's live sparklines.
///
/// Every field is `Option`: a source the host lacks (no `/proc` on
/// macOS, no `nvidia-smi` without an NVIDIA GPU) serializes to `null`
/// rather than failing the whole reading, so one JSON shape serves a
/// CUDA Linux peer and an MLX macOS peer alike. Memory is in megabytes;
/// percentages are 0-100.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Busy CPU percentage across a short sampling window.
    pub cpu_pct: Option<f64>,
    /// System RAM in use (`MemTotal` - `MemAvailable`), megabytes.
    pub ram_used_mb: Option<u64>,
    /// Total system RAM, megabytes.
    pub ram_total_mb: Option<u64>,
    /// GPU utilization percentage (first GPU).
    pub gpu_pct: Option<f64>,
    /// GPU memory in use, megabytes.
    pub vram_used_mb: Option<u64>,
    /// Total GPU memory, megabytes.
    pub vram_total_mb: Option<u64>,
}
