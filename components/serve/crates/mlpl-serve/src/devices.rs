//! `GET /v1/devices` -- report the compute devices this server can run
//! `device("...")` blocks on, in-process. The web client probes this on
//! connect to gate GPU demos by the peer's REAL capability (CUDA on a
//! Linux peer, MLX on an Apple peer) rather than a static guess.

use axum::Json;
use axum::response::IntoResponse;
use serde::Serialize;

/// `GET /v1/devices` response body.
#[derive(Serialize)]
pub struct DevicesResponse {
    pub devices: Vec<&'static str>,
}

/// The devices this build can dispatch: `cpu` always, plus `mlx` /
/// `cuda` when compiled with that feature on its target (the same
/// triple-gate the eval dispatch uses).
#[must_use]
pub fn compiled_devices() -> Vec<&'static str> {
    let mut devices = vec!["cpu"];
    if cfg!(all(
        feature = "mlx",
        target_os = "macos",
        target_arch = "aarch64"
    )) {
        devices.push("mlx");
    }
    if cfg!(all(
        feature = "cuda",
        target_os = "linux",
        target_arch = "x86_64"
    )) {
        devices.push("cuda");
    }
    devices
}

/// Report this server's in-process device set.
pub async fn devices_handler() -> impl IntoResponse {
    Json(DevicesResponse {
        devices: compiled_devices(),
    })
}

/// `GET /v1/stats` -- backend resource telemetry (CPU%, system RAM,
/// GPU%, VRAM) for the REPL's live "evaluating..." sparklines and the
/// `:status` self-test. Delegates to the platform-pluggable
/// `mlpl-monitor` crate, so on a CUDA Linux peer it reports real
/// numbers and on a host without a given source the field is `null`.
pub async fn stats_handler() -> impl IntoResponse {
    Json(mlpl_monitor::snapshot().await)
}
