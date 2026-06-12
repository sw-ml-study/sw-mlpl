//! Assemble a [`Snapshot`] from the host's platform sources.

use mlpl_monitor_types::Snapshot;

/// The platform source crate, selected at compile time. Linux and macOS
/// expose the same `cpu::sample` / `mem::usage` / `gpu::query` surface, so
/// `snapshot` is written once against this alias.
#[cfg(target_os = "linux")]
use mlpl_monitor_linux as sources;
#[cfg(target_os = "macos")]
use mlpl_monitor_macos as sources;

/// CPU sampling window: long enough for a stable utilization read,
/// short enough to keep `/v1/stats` responsive under ~2-3s polling.
#[cfg(target_os = "linux")]
const CPU_WINDOW_MS: u64 = 120;

/// Gather a resource snapshot from this host's platform sources.
///
/// Linux: `/proc/stat` (CPU), `/proc/meminfo` (RAM), `nvidia-smi` (GPU).
/// macOS: Mach `host_statistics64` (CPU), `vm_stat`+`sysctl` (RAM),
/// `ioreg` (GPU). Any source that is unavailable leaves its fields `None`.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub async fn snapshot() -> Snapshot {
    use mlpl_monitor_types::Gpu;
    let cpu_pct = cpu_percent().await;
    let ram = sources::mem::usage();
    // The GPU query shells out (nvidia-smi / ioreg) and can block under
    // load, so run it off the async runtime to keep /v1/stats responsive.
    let gpu_rows = tokio::task::spawn_blocking(sources::gpu::query)
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

/// Busy CPU% across [`CPU_WINDOW_MS`] from two `/proc/stat` tick samples.
/// Deltas over ~120ms are small, so `u32` conversion is exact and dodges
/// `f64`'s precision-loss lint. `None` if either sample or the conversion
/// fails (idle window, counter wrap).
#[cfg(target_os = "linux")]
async fn cpu_percent() -> Option<f64> {
    let (t0, i0) = mlpl_monitor_linux::cpu::sample()?;
    tokio::time::sleep(std::time::Duration::from_millis(CPU_WINDOW_MS)).await;
    let (t1, i1) = mlpl_monitor_linux::cpu::sample()?;
    let busy = u32::try_from(t1.checked_sub(t0)?.checked_sub(i1.checked_sub(i0)?)?).ok()?;
    let total = u32::try_from(t1.checked_sub(t0)?).ok()?;
    (total > 0).then(|| f64::from(busy) / f64::from(total) * 100.0)
}

/// Busy CPU% on macOS via `top`'s own interval sample. It blocks ~1s, so
/// run it off the async runtime to keep `/v1/stats` responsive.
#[cfg(target_os = "macos")]
async fn cpu_percent() -> Option<f64> {
    tokio::task::spawn_blocking(mlpl_monitor_macos::cpu::percent)
        .await
        .ok()
        .flatten()
}

/// Off Linux/macOS there is no source crate wired, so every field reads
/// `None`.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub async fn snapshot() -> Snapshot {
    Snapshot::default()
}
