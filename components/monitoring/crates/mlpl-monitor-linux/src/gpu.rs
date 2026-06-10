//! GPU utilization + VRAM via `nvidia-smi`.

use std::process::Command;

/// Parse one CSV row of `nvidia-smi --query-gpu=utilization.gpu,
/// memory.used,memory.total --format=csv,noheader,nounits` into
/// `(gpu_pct, vram_used_mb, vram_total_mb)`. Reads only the first GPU
/// row. `None` if the row has fewer than three numeric columns.
#[must_use]
pub fn parse_smi(text: &str) -> Option<(f64, u64, u64)> {
    let mut cols = text.lines().next()?.split(',').map(str::trim);
    let pct: f64 = cols.next()?.parse().ok()?;
    let used: u64 = cols.next()?.parse().ok()?;
    let total: u64 = cols.next()?.parse().ok()?;
    Some((pct, used, total))
}

/// Shell out to `nvidia-smi` for `(gpu_pct, vram_used_mb,
/// vram_total_mb)`. `None` when the binary is absent (no NVIDIA GPU /
/// non-Linux) or the output cannot be parsed.
#[must_use]
pub fn query() -> Option<(f64, u64, u64)> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    parse_smi(&String::from_utf8(out.stdout).ok()?)
}
