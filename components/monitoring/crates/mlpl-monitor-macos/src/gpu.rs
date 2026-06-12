//! Apple GPU utilization + memory + name via `ioreg -c IOAccelerator`.
//!
//! The `IOAccelerator` node's `PerformanceStatistics` dictionary exposes
//! `Device Utilization %` and `In use system memory` without sudo (the
//! macOS analog of `nvidia-smi`). On Apple Silicon memory is unified, so
//! "VRAM total" is reported as the machine's total RAM and "VRAM used" as
//! the GPU's in-use system memory.

use std::process::Command;

/// One parsed GPU row: `(name, gpu_pct, vram_used_mb, vram_total_mb)`.
pub type GpuRow = (String, f64, u64, u64);

/// Parse `ioreg` output (with `total_bytes` = unified-memory size) into a
/// `GpuRow`. Pulls `Device Utilization %`, `In use system memory` (the
/// non-driver key), and the `model` string. `None` if the utilization
/// figure is absent (no `IOAccelerator` node).
#[must_use]
pub fn parse_ioreg(text: &str, total_bytes: u64) -> Option<GpuRow> {
    let after_digits = |key: &str| -> Option<u64> {
        text.split(key)
            .nth(1)?
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .ok()
    };
    let pct = after_digits("\"Device Utilization %\"=")?;
    let used = after_digits("In use system memory\"=").unwrap_or(0);
    let name = text
        .split("\"model\" = \"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .unwrap_or("Apple GPU")
        .to_string();
    Some((
        name,
        f64::from(u32::try_from(pct).unwrap_or(u32::MAX)),
        used / (1024 * 1024),
        total_bytes / (1024 * 1024),
    ))
}

/// Shell out to `ioreg` (+ `sysctl hw.memsize` for the unified-memory
/// total). Empty when not on an Apple GPU host or the output cannot be
/// parsed.
#[must_use]
pub fn query() -> Vec<GpuRow> {
    let total: u64 = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let Ok(out) = Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8(out.stdout)
        .ok()
        .and_then(|s| parse_ioreg(&s, total))
        .into_iter()
        .collect()
}
