//! Multi-heatmap grid renderer. Saga 29 step 014 viz demos.
//!
//! Renders a rank-3 `[N, R, C]` tensor as a grid of N
//! independently-color-scaled heatmaps. Used by the multi-
//! head ViT attention demos to show all H per-head softmax
//! matrices side by side (one cell per head) so per-head
//! specialization is legible at a glance.
//!
//! Grid layout: `cols = ceil(sqrt(N))`, `rows = ceil(N /
//! cols)`. Each cell carries its own min/max colormap so a
//! sharply-focused head and a uniform head both render with
//! visible structure rather than washing one out.
//!
//! The overall SVG is sized to fit the grid -- this renderer
//! does NOT use the shared 400x300 plot dimensions of the
//! single-panel heatmap, because the four-panel layout needs
//! more room to be readable.

use mlpl_array::DenseArray;

use super::VizError;

/// Pixel size of each per-head sub-heatmap (square cell).
const CELL_PX: f64 = 220.0;
/// Padding around the whole grid and between cells.
const GRID_PAD: f64 = 24.0;
/// Vertical room reserved for the "head N" label above each cell.
const LABEL_H: f64 = 18.0;

/// Render `[N, R, C]` as a grid of N heatmaps. Returns
/// `InvalidShape` for any other rank, for `N == 0`, or for
/// degenerate `R == 0` / `C == 0`.
pub fn render_heatmap_grid(data: &DenseArray) -> Result<String, VizError> {
    let dims = data.shape().dims();
    if dims.len() != 3 {
        return Err(VizError::InvalidShape(format!(
            "heatmap_grid expects a 3D [N, R, C] tensor, got {dims:?}"
        )));
    }
    let (n, rows, cols) = (dims[0], dims[1], dims[2]);
    if n == 0 || rows == 0 || cols == 0 {
        return Err(VizError::InvalidShape(format!(
            "heatmap_grid: all dims must be > 0, got {dims:?}"
        )));
    }
    let grid_cols = (n as f64).sqrt().ceil() as usize;
    let grid_rows = n.div_ceil(grid_cols);
    let cell_w_total = CELL_PX + GRID_PAD;
    let cell_h_total = CELL_PX + GRID_PAD + LABEL_H;
    let svg_w = GRID_PAD + grid_cols as f64 * cell_w_total;
    let svg_h = GRID_PAD + grid_rows as f64 * cell_h_total;
    let mut out = String::new();
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" \
         viewBox=\"0 0 {svg_w:.1} {svg_h:.1}\" \
         width=\"{svg_w:.1}\" height=\"{svg_h:.1}\">"
    ));
    out.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#1e1e2e\"/>");
    let plane_elems = rows * cols;
    let raw = data.data();
    for h in 0..n {
        let gx = h % grid_cols;
        let gy = h / grid_cols;
        let x0 = GRID_PAD + gx as f64 * cell_w_total;
        let y0 = GRID_PAD + gy as f64 * cell_h_total;
        write_cell_label(&mut out, x0, y0, h);
        let plane = &raw[h * plane_elems..(h + 1) * plane_elems];
        write_cell_grid(&mut out, x0, y0 + LABEL_H, plane, rows, cols);
    }
    out.push_str("</svg>");
    Ok(out)
}

fn write_cell_label(out: &mut String, x0: f64, y0: f64, h: usize) {
    let tx = x0 + CELL_PX * 0.5;
    let ty = y0 + LABEL_H - 4.0;
    out.push_str(&format!(
        "<text x=\"{tx:.1}\" y=\"{ty:.1}\" fill=\"#cdd6f4\" \
         font-size=\"13\" font-family=\"monospace\" \
         text-anchor=\"middle\">head {h}</text>"
    ));
}

fn write_cell_grid(out: &mut String, x0: f64, y0: f64, plane: &[f64], rows: usize, cols: usize) {
    let (lo, hi) = data_range(plane);
    let span = if (hi - lo).abs() < f64::EPSILON {
        1.0
    } else {
        hi - lo
    };
    let cell_w = CELL_PX / cols as f64;
    let cell_h = CELL_PX / rows as f64;
    for r in 0..rows {
        for c in 0..cols {
            let v = plane[r * cols + c];
            let t = ((v - lo) / span).clamp(0.0, 1.0);
            let (red, green, blue) = viridis(t);
            let x = x0 + cell_w * c as f64;
            let y = y0 + cell_h * r as f64;
            out.push_str(&format!(
                "<rect x=\"{x:.1}\" y=\"{y:.1}\" \
                 width=\"{cell_w:.2}\" height=\"{cell_h:.2}\" \
                 fill=\"rgb({red},{green},{blue})\"/>"
            ));
        }
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

fn viridis(t: f64) -> (u8, u8, u8) {
    const STOPS: [(f64, f64, f64); 3] = [
        (68.0, 1.0, 84.0),
        (33.0, 145.0, 140.0),
        (253.0, 231.0, 37.0),
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
