//! Pure rendering of a [`Snapshot`] -- shared by the `:status` REPL
//! command and (later) the live sparklines, in both the web client and
//! the CLI. Kept here, beside the wire type, so there is one canonical
//! formatting and it is unit-testable without a browser.

use crate::{Gpu, Snapshot};

/// Format an optional megabyte count as gibibytes with one decimal, or
/// `"n/a"` when absent. Converts through `u32` (megabytes well under
/// `u32::MAX` for any realistic host) to dodge the `f64` precision-loss
/// lint; a value too large to fit renders `"n/a"`.
#[must_use]
pub fn gb(mb: Option<u64>) -> String {
    mb.and_then(|m| u32::try_from(m).ok()).map_or_else(
        || "n/a".to_string(),
        |m| format!("{:.1}", f64::from(m) / 1024.0),
    )
}

/// Format an optional percentage (0-100) as `"42%"`, or `"n/a"`.
#[must_use]
pub fn pct(p: Option<f64>) -> String {
    p.map_or_else(|| "n/a".to_string(), |v| format!("{v:.0}%"))
}

/// One GPU line, e.g. `"    GPU 0: NVIDIA RTX 5080  88%  VRAM 2.1 / 15.9 GB"`.
#[must_use]
pub fn gpu_line(index: usize, g: &Gpu) -> String {
    let name = g.name.as_deref().unwrap_or("GPU");
    format!(
        "    GPU {index}: {name}  {}  VRAM {} / {} GB",
        pct(g.pct),
        gb(g.vram_used_mb),
        gb(g.vram_total_mb)
    )
}

/// The resource lines of one backend in a `:status` report: CPU, RAM,
/// then one line per GPU (or a single "GPU : none" line on a GPU-less
/// host). The caller frames these with the backend index, connect URL,
/// device list, and Ollama state.
#[must_use]
pub fn snapshot_lines(s: &Snapshot) -> Vec<String> {
    let mut lines = vec![
        format!("    CPU  : {}", pct(s.cpu_pct)),
        format!(
            "    RAM  : {} / {} GB",
            gb(s.ram_used_mb),
            gb(s.ram_total_mb)
        ),
    ];
    if s.gpus.is_empty() {
        lines.push("    GPU  : none".to_string());
    } else {
        lines.extend(s.gpus.iter().enumerate().map(|(i, g)| gpu_line(i, g)));
    }
    lines
}
