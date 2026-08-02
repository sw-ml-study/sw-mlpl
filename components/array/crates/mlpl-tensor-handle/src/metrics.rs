//! Seam instrumentation (saga E4 step 8, per
//! docs/project-direction.txt): process-global counters for the
//! events that explain device performance -- uploads, downloads
//! (graph forces), device op submissions, and CPU fallbacks taken
//! by a resident tape. Wasm-clean (plain atomics); the bench and
//! telemetry read snapshots, hot paths pay one relaxed increment.

use std::sync::atomic::{AtomicU64, Ordering};

/// One countable seam event.
#[derive(Debug, Clone, Copy)]
pub enum SeamEvent {
    /// Host array moved onto the device.
    Upload,
    /// Resident value materialized on the host (forces the lazy
    /// graph up to that node).
    Download,
    /// One lazy device op submitted (graph node built).
    Submit,
    /// A resident tape took the CPU path for an op (no device
    /// kernel, backend error, or unsupported case).
    CpuFallback,
}

static UPLOADS: AtomicU64 = AtomicU64::new(0);
static DOWNLOADS: AtomicU64 = AtomicU64::new(0);
static SUBMITS: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Record one seam event (relaxed; counters are diagnostics).
pub fn bump(event: SeamEvent) {
    let c = match event {
        SeamEvent::Upload => &UPLOADS,
        SeamEvent::Download => &DOWNLOADS,
        SeamEvent::Submit => &SUBMITS,
        SeamEvent::CpuFallback => &FALLBACKS,
    };
    c.fetch_add(1, Ordering::Relaxed);
}

/// Conditionally record one seam event (e.g. count a CPU fallback
/// only when the tape is resident).
pub fn bump_if(cond: bool, event: SeamEvent) {
    if cond {
        bump(event);
    }
}

/// `(uploads, downloads, submits, cpu_fallbacks)` since the last
/// reset.
#[must_use]
pub fn seam_snapshot() -> (u64, u64, u64, u64) {
    (
        UPLOADS.load(Ordering::Relaxed),
        DOWNLOADS.load(Ordering::Relaxed),
        SUBMITS.load(Ordering::Relaxed),
        FALLBACKS.load(Ordering::Relaxed),
    )
}

/// Zero all counters (bench/test setup).
pub fn seam_reset() {
    for c in [&UPLOADS, &DOWNLOADS, &SUBMITS, &FALLBACKS] {
        c.store(0, Ordering::Relaxed);
    }
}
