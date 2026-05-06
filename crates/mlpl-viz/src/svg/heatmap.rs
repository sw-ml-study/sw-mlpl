//! Heatmap (2D matrix) rendering with a viridis-like color ramp.

use mlpl_array::DenseArray;

use super::{H, PAD, VizError, W, write_svg_close, write_svg_open};

/// Width pulled from the cell area to make room for the legend
/// strip + value labels (`hi` / `mid` / `lo`).
const LEGEND_RESERVE: f64 = 60.0;
/// Width of the vertical colorbar itself.
const LEGEND_W: f64 = 14.0;
/// Number of stacked rectangles used to approximate the gradient.
const LEGEND_STEPS: usize = 32;

/// Render an MxN matrix as a colored grid with a viridis
/// colorbar legend on the right (top = max, bottom = min).
pub fn render_heatmap(data: &DenseArray) -> Result<String, VizError> {
    let dims = data.shape().dims();
    if dims.len() != 2 {
        return Err(VizError::InvalidShape(format!(
            "heatmap expects a 2D matrix, got {dims:?}"
        )));
    }
    let rows = dims[0];
    let cols = dims[1];
    let mut out = String::new();
    write_svg_open(&mut out);
    if rows == 0 || cols == 0 {
        write_svg_close(&mut out);
        return Ok(out);
    }
    let raw = data.data();
    let (lo, hi) = data_range(raw);
    let span = if (hi - lo).abs() < f64::EPSILON {
        1.0
    } else {
        hi - lo
    };
    let plot_w = W - 2.0 * PAD - LEGEND_RESERVE;
    let plot_h = H - 2.0 * PAD;
    let cell_w = plot_w / cols as f64;
    let cell_h = plot_h / rows as f64;
    for r in 0..rows {
        for c in 0..cols {
            let v = raw[r * cols + c];
            let t = ((v - lo) / span).clamp(0.0, 1.0);
            let (red, green, blue) = viridis(t);
            let x = PAD + cell_w * c as f64;
            let y = PAD + cell_h * r as f64;
            out.push_str(&format!(
                "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{cell_w:.1}\" height=\"{cell_h:.1}\" fill=\"rgb({red},{green},{blue})\"/>"
            ));
        }
    }
    render_legend(&mut out, plot_w, plot_h, lo, hi);
    write_svg_close(&mut out);
    Ok(out)
}

/// Vertical colorbar + min/mid/max value labels in the right
/// margin reserved by `LEGEND_RESERVE`. Top of the bar is the
/// data max (yellow); bottom is the data min (dark purple).
fn render_legend(out: &mut String, plot_w: f64, plot_h: f64, lo: f64, hi: f64) {
    let legend_x = PAD + plot_w + 8.0;
    let label_x = legend_x + LEGEND_W + 5.0;
    let step_h = plot_h / LEGEND_STEPS as f64;
    for i in 0..LEGEND_STEPS {
        // i=0 is top of the bar; t=1 is yellow (max).
        let t = 1.0 - (i as f64 + 0.5) / LEGEND_STEPS as f64;
        let (red, green, blue) = viridis(t);
        let y = PAD + step_h * i as f64;
        out.push_str(&format!(
            "<rect x=\"{legend_x:.1}\" y=\"{y:.1}\" width=\"{LEGEND_W:.1}\" height=\"{:.2}\" fill=\"rgb({red},{green},{blue})\"/>",
            step_h + 0.5
        ));
    }
    let mid = (lo + hi) * 0.5;
    let label = |out: &mut String, y: f64, v: f64| {
        out.push_str(&format!(
            "<text x=\"{label_x:.1}\" y=\"{y:.1}\" fill=\"#cdd6f4\" font-size=\"11\" font-family=\"monospace\" dominant-baseline=\"middle\">{v:.2}</text>"
        ));
    };
    label(out, PAD + 4.0, hi);
    label(out, PAD + plot_h * 0.5, mid);
    label(out, PAD + plot_h - 4.0, lo);
}

fn data_range(values: &[f64]) -> (f64, f64) {
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

/// Viridis-ish color ramp: dark purple -> teal -> yellow.
/// `t` is clamped to [0, 1].
fn viridis(t: f64) -> (u8, u8, u8) {
    // Three control points (purple, teal, yellow) interpolated linearly.
    const STOPS: [(f64, f64, f64); 3] = [
        (68.0, 1.0, 84.0),    // dark purple
        (33.0, 145.0, 140.0), // teal
        (253.0, 231.0, 37.0), // yellow
    ];
    let (a, b, frac) = if t < 0.5 {
        (STOPS[0], STOPS[1], t * 2.0)
    } else {
        (STOPS[1], STOPS[2], (t - 0.5) * 2.0)
    };
    let r = (a.0 + (b.0 - a.0) * frac).round() as u8;
    let g = (a.1 + (b.1 - a.1) * frac).round() as u8;
    let bl = (a.2 + (b.2 - a.2) * frac).round() as u8;
    (r, g, bl)
}
