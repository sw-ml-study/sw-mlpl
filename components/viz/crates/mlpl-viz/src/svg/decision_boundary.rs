//! Decision-boundary diagram: heatmap of classifier outputs over a 2D
//! region with the training points overlaid as colored circles.

use mlpl_array::DenseArray;

use super::{H, PAD, VizError, W, write_svg_close, write_svg_open};

/// Render a 2D grid of classifier outputs and overlay training points.
///
/// `grid` is an MxN matrix where each cell holds the classifier output at
/// that grid location (e.g. a probability in [0, 1]).
/// `training` is an Nx3 matrix where each row is `(x, y, label)`. `x` and
/// `y` are mapped to the same plot region as the grid (column 0 -> x,
/// column 1 -> y, in the unit square [0, 1] x [0, 1]).
pub fn render_decision_boundary(
    grid: &DenseArray,
    training: &DenseArray,
) -> Result<String, VizError> {
    let dims = grid.shape().dims();
    if dims.len() != 2 {
        return Err(VizError::InvalidShape(format!(
            "decision_boundary expects a 2D grid, got {dims:?}"
        )));
    }
    let train_dims = training.shape().dims();
    if train_dims.len() != 2 || train_dims[1] != 3 {
        return Err(VizError::InvalidShape(format!(
            "decision_boundary training data must be Nx3 (x, y, label), got {train_dims:?}"
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
    let (lo, hi) = draw_surface(&mut out, grid.data(), rows, cols);
    draw_points(&mut out, training.data(), train_dims[0]);
    // Saga 29 step 019: vertical colorbar on the right edge
    // showing the surface's blue -> beige -> red ramp with the
    // grid's actual lo/mid/hi values. Matches the heatmap's
    // legend pattern.
    draw_legend(&mut out, lo, hi);
    write_svg_close(&mut out);
    Ok(out)
}

fn draw_surface(out: &mut String, raw: &[f64], rows: usize, cols: usize) -> (f64, f64) {
    let (lo, hi) = data_range(raw);
    let span = if (hi - lo).abs() < f64::EPSILON {
        1.0
    } else {
        hi - lo
    };
    let cell_w = (W - 2.0 * PAD) / cols as f64;
    let cell_h = (H - 2.0 * PAD) / rows as f64;
    // Flip r so math ymin (grid row 0) lands at image bottom, matching
    // draw_points's math-y-bottom convention (saga 33 step 025).
    for r in 0..rows {
        for c in 0..cols {
            let t = ((raw[(rows - 1 - r) * cols + c] - lo) / span).clamp(0.0, 1.0);
            let (red, green, blue) = ramp(t);
            let x = PAD + cell_w * c as f64;
            let y = PAD + cell_h * r as f64;
            out.push_str(&format!(
                "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{cell_w:.1}\" height=\"{cell_h:.1}\" fill=\"rgb({red},{green},{blue})\" fill-opacity=\"0.85\"/>"
            ));
        }
    }
    (lo, hi)
}

/// Saga 29 step 019: vertical colorbar matching the
/// blue-beige-red ramp used by `draw_surface`. Sits on the
/// right edge of the plot area.
fn draw_legend(out: &mut String, lo: f64, hi: f64) {
    let bar_w: f64 = 10.0;
    let bar_x: f64 = W - PAD - bar_w - 2.0;
    let bar_top: f64 = PAD;
    let bar_h: f64 = H - 2.0 * PAD;
    let steps: usize = 32;
    let step_h = bar_h / steps as f64;
    for i in 0..steps {
        // i=0 is top of the bar; t=1 corresponds to `hi`.
        let t = 1.0 - (i as f64 + 0.5) / steps as f64;
        let (red, green, blue) = ramp(t);
        let y = bar_top + step_h * i as f64;
        out.push_str(&format!(
            "<rect x=\"{bar_x:.1}\" y=\"{y:.1}\" \
             width=\"{bar_w:.1}\" height=\"{:.2}\" \
             fill=\"rgb({red},{green},{blue})\"/>",
            step_h + 0.5,
        ));
    }
    let label_x = bar_x - 3.0;
    let label = |out: &mut String, y: f64, v: f64| {
        out.push_str(&format!(
            "<text x=\"{label_x:.1}\" y=\"{y:.1}\" fill=\"#cdd6f4\" \
             font-size=\"10\" font-family=\"monospace\" \
             text-anchor=\"end\" dominant-baseline=\"middle\">{v:.2}</text>"
        ));
    };
    label(out, bar_top + 4.0, hi);
    label(out, bar_top + bar_h * 0.5, (lo + hi) * 0.5);
    label(out, bar_top + bar_h - 4.0, lo);
}

fn draw_points(out: &mut String, tdata: &[f64], n: usize) {
    let plot_w = W - 2.0 * PAD;
    let plot_h = H - 2.0 * PAD;
    for i in 0..n {
        let tx = tdata[i * 3];
        let ty = tdata[i * 3 + 1];
        let label = tdata[i * 3 + 2];
        let cx = PAD + tx.clamp(0.0, 1.0) * plot_w;
        let cy = H - PAD - ty.clamp(0.0, 1.0) * plot_h;
        let fill = if label > 0.5 { "#f38ba8" } else { "#89b4fa" };
        out.push_str(&format!(
            "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"5\" fill=\"{fill}\" stroke=\"#1e1e2e\" stroke-width=\"1.5\"/>"
        ));
    }
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

/// Diverging blue -> white -> red ramp emphasizing the 0.5 boundary.
fn ramp(t: f64) -> (u8, u8, u8) {
    const LO: (f64, f64, f64) = (137.0, 180.0, 250.0); // #89b4fa blue
    const MID: (f64, f64, f64) = (245.0, 245.0, 220.0); // beige
    const HI: (f64, f64, f64) = (243.0, 139.0, 168.0); // #f38ba8 red
    let (a, b, frac) = if t < 0.5 {
        (LO, MID, t * 2.0)
    } else {
        (MID, HI, (t - 0.5) * 2.0)
    };
    let r = (a.0 + (b.0 - a.0) * frac).round() as u8;
    let g = (a.1 + (b.1 - a.1) * frac).round() as u8;
    let bl = (a.2 + (b.2 - a.2) * frac).round() as u8;
    (r, g, bl)
}
