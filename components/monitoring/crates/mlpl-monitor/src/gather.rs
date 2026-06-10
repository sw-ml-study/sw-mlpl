//! Assemble a [`Snapshot`] from the host's platform sources.

use mlpl_monitor_types::Snapshot;

/// CPU sampling window: long enough for a stable utilization read,
/// short enough to keep `/v1/stats` responsive under ~2-3s polling.
#[cfg(target_os = "linux")]
const CPU_WINDOW_MS: u64 = 120;

/// Gather a resource snapshot from this host's platform sources.
///
/// Linux: `/proc/stat` (CPU), `/proc/meminfo` (RAM), `nvidia-smi`
/// (GPU). Any source that is unavailable leaves its fields `None`.
#[cfg(target_os = "linux")]
pub async fn snapshot() -> Snapshot {
    use mlpl_monitor_linux::{gpu, mem};
    use mlpl_monitor_types::Gpu;
    let cpu_pct = cpu_percent().await;
    let ram = mem::usage();
    // nvidia-smi can block for a while under heavy GPU load (e.g. while a
    // large Ollama model loads), so run it off the async runtime to keep
    // /v1/stats responsive for the live telemetry pollers.
    let gpu_rows = tokio::task::spawn_blocking(gpu::query)
        .await
        .unwrap_or_default();
    let gpus = gpu_rows
        .into_iter()
        .map(|(name, pct, used, total)| Gpu {
            name: Some(name),
            pct: Some(pct),
            vram_used_mb: Some(used),
            vram_total_mb: Some(total),
        })
        .collect();
    Snapshot {
        cpu_pct,
        ram_used_mb: ram.map(|r| r.0),
        ram_total_mb: ram.map(|r| r.1),
        gpus,
    }
}

/// Busy CPU% across [`CPU_WINDOW_MS`] from two `/proc/stat` jiffy
/// samples. Deltas over ~120ms are small, so `u32` conversion is exact
/// and dodges `f64`'s precision-loss lint. `None` if either sample or
/// the conversion fails (idle window, counter wrap).
#[cfg(target_os = "linux")]
async fn cpu_percent() -> Option<f64> {
    use mlpl_monitor_linux::cpu;
    let (t0, i0) = cpu::sample()?;
    tokio::time::sleep(std::time::Duration::from_millis(CPU_WINDOW_MS)).await;
    let (t1, i1) = cpu::sample()?;
    let busy = u32::try_from(t1.checked_sub(t0)?.checked_sub(i1.checked_sub(i0)?)?).ok()?;
    let total = u32::try_from(t1.checked_sub(t0)?).ok()?;
    (total > 0).then(|| f64::from(busy) / f64::from(total) * 100.0)
}

/// Off-Linux there is no source crate wired yet (the MLX macOS server
/// gets `mlpl-monitor-macos` later), so every field reads `None`.
#[cfg(not(target_os = "linux"))]
pub async fn snapshot() -> Snapshot {
    Snapshot::default()
}
