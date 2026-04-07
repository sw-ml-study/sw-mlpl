//! Heatmap (2D matrix) rendering with a viridis-like color ramp.

use mlpl_array::DenseArray;

use super::{H, PAD, VizError, W, write_svg_close, write_svg_open};

/// Render an MxN matrix as a colored grid.
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
    let plot_w = W - 2.0 * PAD;
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
    write_svg_close(&mut out);
    Ok(out)
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
