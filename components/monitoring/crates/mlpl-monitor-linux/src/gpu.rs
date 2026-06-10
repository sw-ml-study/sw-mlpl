//! Per-GPU utilization + VRAM + name via `nvidia-smi`.

use std::process::Command;

/// One parsed GPU row: `(name, gpu_pct, vram_used_mb, vram_total_mb)`.
pub type GpuRow = (String, f64, u64, u64);

/// Parse the CSV output of `nvidia-smi --query-gpu=name,utilization.gpu,
/// memory.used,memory.total --format=csv,noheader,nounits` into one
/// `GpuRow` per GPU. Rows with fewer than four columns are skipped, so
/// a malformed line never drops the whole list.
#[must_use]
pub fn parse_smi(text: &str) -> Vec<GpuRow> {
    text.lines().filter_map(parse_row).collect()
}

/// Parse a single `name, util, used, total` CSV row. `None` if any of
/// the three numeric columns is missing or unparseable.
fn parse_row(line: &str) -> Option<GpuRow> {
    let mut cols = line.split(',').map(str::trim);
    let name = cols.next()?.to_string();
    let pct: f64 = cols.next()?.parse().ok()?;
    let used: u64 = cols.next()?.parse().ok()?;
    let total: u64 = cols.next()?.parse().ok()?;
    Some((name, pct, used, total))
}

/// Shell out to `nvidia-smi` for every GPU. Empty when the binary is
/// absent (no NVIDIA GPU / non-Linux) or the output cannot be parsed.
#[must_use]
pub fn query() -> Vec<GpuRow> {
    let Ok(out) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8(out.stdout)
        .map(|s| parse_smi(&s))
        .unwrap_or_default()
}
