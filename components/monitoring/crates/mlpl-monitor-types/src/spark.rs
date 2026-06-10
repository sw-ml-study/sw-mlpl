//! Pure unicode sparkline rendering + Snapshot->percentage derivation
//! for the live "evaluating..." telemetry panel. Integer math in the
//! sparkline (samples are 0..=max percentages) so there are no float
//! casts there; the one unavoidable f64->int rounding is isolated in
//! `round_pct`.

use crate::Snapshot;

/// The eight block glyphs, low to high.
const BARS: [char; 8] = [
    '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}',
];

/// Render `samples` (each in `0..=max`) as a sparkline string, one
/// block glyph per sample. `max` of 0 is treated as 1. Values above
/// `max` clamp to the tallest bar. Empty input yields an empty string.
#[must_use]
pub fn sparkline(samples: &[u32], max: u32) -> String {
    let max = max.max(1);
    samples
        .iter()
        .map(|&v| {
            // v.min(max) * 7 / max maps 0..=max onto bar index 0..=7.
            let level = (v.min(max) * 7 / max) as usize;
            BARS[level]
        })
        .collect()
}

/// `[cpu, ram, gpu, vram]` as 0..=100 percentages for the live
/// sparklines. Missing sources read 0; GPU/VRAM use the first GPU.
#[must_use]
pub fn metric_percents(s: &Snapshot) -> [u32; 4] {
    let g = s.gpus.first();
    [
        round_pct(s.cpu_pct),
        ratio_pct(s.ram_used_mb, s.ram_total_mb),
        g.map_or(0, |x| round_pct(x.pct)),
        g.map_or(0, |x| ratio_pct(x.vram_used_mb, x.vram_total_mb)),
    ]
}

/// Round an optional 0..=100 percentage to `u32`. The clamp bounds the
/// value, so the cast cannot truncate or lose sign.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn round_pct(p: Option<f64>) -> u32 {
    p.unwrap_or(0.0).clamp(0.0, 100.0).round() as u32
}

/// `used/total` as an integer percentage (0..=100), or 0 when total is
/// absent/zero. Integer math -- no float cast.
fn ratio_pct(used: Option<u64>, total: Option<u64>) -> u32 {
    match (used, total) {
        (Some(u), Some(t)) if t > 0 => u32::try_from(u.min(t) * 100 / t).unwrap_or(100),
        _ => 0,
    }
}
