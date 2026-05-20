//! SVG diagram rendering for MLPL arrays.

mod charts;
mod decision_boundary;
mod gallery;
mod heatmap;
mod heatmap_grid;
mod scatter;

use std::fmt;

use mlpl_array::DenseArray;

pub use charts::{render_bar, render_line};
pub use decision_boundary::render_decision_boundary;
pub use gallery::{render_attention_overlay, render_gallery};
pub use heatmap::render_heatmap;
pub use heatmap_grid::render_heatmap_grid;
pub use scatter::{render_scatter, render_scatter3d};

pub(crate) const W: f64 = 400.0;
pub(crate) const H: f64 = 300.0;
pub(crate) const PAD: f64 = 30.0;

/// Plain min/max of a slice (no expansion when all equal).
pub(crate) fn data_range(values: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// Min/max of a slice, returning (lo-1, hi+1) when all values are equal
/// so the resulting range is non-zero.
pub(crate) fn bounds(values: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if lo == hi {
        (lo - 1.0, hi + 1.0)
    } else {
        (lo, hi)
    }
}

/// Scale a data coordinate to plot pixels. `axis` 0 = x, 1 = y (flipped).
pub(crate) fn scale(v: f64, lo: f64, hi: f64, axis: u8) -> f64 {
    let t = (v - lo) / (hi - lo);
    if axis == 0 {
        PAD + t * (W - 2.0 * PAD)
    } else {
        H - PAD - t * (H - 2.0 * PAD)
    }
}

/// Saga 29 step 019: write min/max scale labels at the
/// corners of the plot area. `xmin/xmax` flank the bottom
/// axis, `ymin/ymax` flank the left axis. Used by scatter,
/// line, and bar to make the plot self-explanatory without
/// a full tick-mark axis system.
pub(crate) fn write_corner_scale_labels(
    out: &mut String,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
) {
    let x0 = PAD;
    let x1 = W - PAD;
    let y0 = PAD;
    let y1 = H - PAD;
    let fmt_label = |out: &mut String, x: f64, y: f64, anchor: &str, v: f64| {
        out.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"{y:.1}\" fill=\"#cdd6f4\" \
             font-size=\"10\" font-family=\"monospace\" \
             text-anchor=\"{anchor}\">{v:.2}</text>"
        ));
    };
    // X-axis labels just below the bottom edge.
    fmt_label(out, x0, y1 + 11.0, "start", xmin);
    fmt_label(out, x1, y1 + 11.0, "end", xmax);
    // Y-axis labels just to the left of the left edge.
    fmt_label(out, x0 - 4.0, y1, "end", ymin);
    fmt_label(out, x0 - 4.0, y0 + 4.0, "end", ymax);
}

pub(crate) fn write_svg_open(out: &mut String) {
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {W} {H}\" width=\"{W}\" height=\"{H}\">"
    ));
    out.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#1e1e2e\"/>");
    let (x0, x1, y0, y1) = (PAD, W - PAD, PAD, H - PAD);
    out.push_str(&format!(
        "<line x1=\"{x0}\" y1=\"{y1}\" x2=\"{x1}\" y2=\"{y1}\" stroke=\"#45475a\" stroke-width=\"1\"/>"
    ));
    out.push_str(&format!(
        "<line x1=\"{x0}\" y1=\"{y0}\" x2=\"{x0}\" y2=\"{y1}\" stroke=\"#45475a\" stroke-width=\"1\"/>"
    ));
}

pub(crate) fn write_svg_close(out: &mut String) {
    out.push_str("</svg>");
}

/// Errors produced by SVG rendering.
#[derive(Clone, Debug, PartialEq)]
pub enum VizError {
    /// The data shape is not valid for the requested diagram.
    InvalidShape(String),
    /// The diagram type name is not recognized.
    UnknownType(String),
}

impl fmt::Display for VizError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(s) => write!(f, "invalid shape for diagram: {s}"),
            Self::UnknownType(s) => write!(f, "unknown svg type: '{s}'"),
        }
    }
}

impl std::error::Error for VizError {}

/// Dispatch on a diagram type name. Returns the rendered SVG string.
pub fn render(data: &DenseArray, type_name: &str) -> Result<String, VizError> {
    render_with_aux(data, type_name, None)
}

/// Dispatch on a diagram type name with an optional auxiliary array.
/// Diagrams that need extra inputs (e.g. `decision_boundary` needs the
/// training points) read them from `aux`.
pub fn render_with_aux(
    data: &DenseArray,
    type_name: &str,
    aux: Option<&DenseArray>,
) -> Result<String, VizError> {
    match type_name {
        "scatter" => render_scatter(data),
        "scatter3d" => render_scatter3d(data),
        "line" => render_line(data),
        "bar" => render_bar(data),
        "heatmap" => render_heatmap(data),
        "heatmap_grid" => render_heatmap_grid(data),
        "gallery" => render_gallery(data, aux),
        "attention_overlay" => render_attention_overlay(data, aux),
        "decision_boundary" => {
            let training = aux.ok_or_else(|| {
                VizError::InvalidShape(
                    "decision_boundary requires a third argument (training points Nx3)".into(),
                )
            })?;
            render_decision_boundary(data, training)
        }
        other => Err(VizError::UnknownType(other.to_string())),
    }
}
