//! Map raw per-edge `widths` to display stroke widths in pixels. Two
//! honest scales: clamped-linear (the caller supplies visual widths,
//! bounded so a raw quantity can't become a monstrous stroke) and
//! log-normalized (extreme quantity ratios -- e.g. 269 MB vs 4 KiB --
//! read as an orders-of-magnitude contrast instead of one edge blowing
//! out the canvas and the other vanishing).

use crate::model::Graph;

/// Display stroke-width band, in pixels.
const W_MIN: f64 = 1.0;
const W_MAX: f64 = 9.0;
/// Stroke width for an edge when the graph carries no `widths`.
pub const W_DEFAULT: f64 = 1.5;

/// One display stroke width per edge, or empty when the graph has no
/// `edge_widths` (callers then fall back to [`W_DEFAULT`]).
pub fn stroke_widths(g: &Graph) -> Vec<f64> {
    if g.edge_widths.is_empty() {
        Vec::new()
    } else if g.width_log {
        log_scaled(&g.edge_widths)
    } else {
        g.edge_widths
            .iter()
            .map(|&w| w.clamp(W_MIN, W_MAX))
            .collect()
    }
}

/// Natural-log-normalize widths into `[W_MIN, W_MAX]`; a flat set (all
/// equal) maps to the band midpoint. Non-positive widths floor at 1.
fn log_scaled(ws: &[f64]) -> Vec<f64> {
    let logs: Vec<f64> = ws.iter().map(|&w| w.max(1.0).ln()).collect();
    let lo = logs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    let mid = (W_MIN + W_MAX) / 2.0;
    logs.iter()
        .map(|&l| {
            if span <= 0.0 {
                mid
            } else {
                W_MIN + (l - lo) / span * (W_MAX - W_MIN)
            }
        })
        .collect()
}
