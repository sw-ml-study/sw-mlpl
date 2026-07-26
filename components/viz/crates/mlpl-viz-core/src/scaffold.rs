//! Canvas constants + scaling and SVG document helpers shared by
//! every renderer in the mlpl-viz crate family.

pub const W: f64 = 400.0;
pub const H: f64 = 300.0;
pub const PAD: f64 = 30.0;

/// Plain min/max of a slice (no expansion when all equal).
pub fn data_range(values: &[f64]) -> (f64, f64) {
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
pub fn bounds(values: &[f64]) -> (f64, f64) {
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
pub fn scale(v: f64, lo: f64, hi: f64, axis: u8) -> f64 {
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
pub fn write_corner_scale_labels(out: &mut String, xmin: f64, xmax: f64, ymin: f64, ymax: f64) {
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

pub fn write_svg_open(out: &mut String) {
    write_svg_open_with_size(out, W, H);
}

/// Same as [`write_svg_open`] but with a caller-controlled
/// outer canvas size. The data area axis lines stay at the
/// usual `(PAD, W-PAD) x (PAD, H-PAD)` rectangle so existing
/// point-scaling code keeps working; any extra width / height
/// past `W` / `H` becomes a gutter the caller can use for
/// legends, side panels, etc. Saga 33 step 037b: introduced to
/// let `analysis_scatter_labeled` render its index legend
/// outside the plot area instead of overlapping data points.
pub fn write_svg_open_with_size(out: &mut String, cw: f64, ch: f64) {
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {cw} {ch}\" width=\"{cw}\" height=\"{ch}\">"
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

pub fn write_svg_close(out: &mut String) {
    out.push_str("</svg>");
}
