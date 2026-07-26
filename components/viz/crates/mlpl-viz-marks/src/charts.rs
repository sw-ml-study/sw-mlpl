//! 1D chart renderings: bar charts and line plots.

use mlpl_array::DenseArray;

use mlpl_viz_core::{
    H, PAD, VizError, W, bounds, scale, write_corner_scale_labels, write_svg_close, write_svg_open,
};

/// Render a vector as a bar chart (one bar per element).
pub fn render_bar(data: &DenseArray) -> Result<String, VizError> {
    let dims = data.shape().dims();
    if dims.len() > 1 {
        return Err(VizError::InvalidShape(format!(
            "bar expects a vector, got {dims:?}"
        )));
    }
    let mut out = String::new();
    write_svg_open(&mut out);
    let values = data.data();
    if values.is_empty() {
        write_svg_close(&mut out);
        return Ok(out);
    }
    let (ymin, ymax, yrange) = bar_y_range(values);
    let n = values.len();
    write_bars(&mut out, values, ymin, yrange);
    // Saga 29 step 019: scale labels. X axis carries 0..N-1
    // bar indices (formatted as the actual index range), Y
    // axis carries the value range.
    let xmax = if n == 0 { 0.0 } else { (n - 1) as f64 };
    write_corner_scale_labels(&mut out, 0.0, xmax, ymin, ymax);
    write_svg_close(&mut out);
    Ok(out)
}

/// Render a vector or Nx2 matrix as a polyline plot.
///
/// - Vector input: x = 0..N-1, y = element values (loss curves).
/// - Nx2 matrix input: rows are (x, y) pairs.
pub fn render_line(data: &DenseArray) -> Result<String, VizError> {
    let dims = data.shape().dims();
    let (xs, ys) = match dims {
        [] | [_] => line_extract_vector(data),
        [_, 2] => line_extract_matrix(data),
        other => {
            return Err(VizError::InvalidShape(format!(
                "line expects vector or Nx2 matrix, got {other:?}"
            )));
        }
    };
    let mut out = String::new();
    write_svg_open(&mut out);
    if xs.is_empty() {
        write_svg_close(&mut out);
        return Ok(out);
    }
    let (xmin, xmax) = bounds(&xs);
    let (ymin, ymax) = bounds(&ys);
    out.push_str(&polyline(&xs, &ys, (xmin, xmax), (ymin, ymax)));
    // Saga 29 step 019: corner scale labels.
    write_corner_scale_labels(&mut out, xmin, xmax, ymin, ymax);
    write_svg_close(&mut out);
    Ok(out)
}

fn line_extract_vector(data: &DenseArray) -> (Vec<f64>, Vec<f64>) {
    let ys: Vec<f64> = data.data().to_vec();
    let xs: Vec<f64> = (0..ys.len()).map(|i| i as f64).collect();
    (xs, ys)
}

fn line_extract_matrix(data: &DenseArray) -> (Vec<f64>, Vec<f64>) {
    let raw = data.data();
    let n = raw.len() / 2;
    (0..n).map(|i| (raw[i * 2], raw[i * 2 + 1])).unzip()
}

/// One green rect per value, drawn from the zero baseline.
fn write_bars(out: &mut String, values: &[f64], ymin: f64, yrange: f64) {
    let n = values.len();
    let plot_w = W - 2.0 * PAD;
    let plot_h = H - 2.0 * PAD;
    let bar_slot = plot_w / n as f64;
    let bar_w = (bar_slot * 0.8).max(1.0);
    let baseline = H - PAD - (-ymin / yrange) * plot_h;
    for (i, &v) in values.iter().enumerate() {
        let x = PAD + bar_slot * i as f64 + (bar_slot - bar_w) / 2.0;
        let bar_h = (v / yrange).abs() * plot_h;
        let y = if v >= 0.0 { baseline - bar_h } else { baseline };
        out.push_str(&format!(
            "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{bar_w:.1}\" height=\"{bar_h:.1}\" fill=\"#a6e3a1\"/>"
        ));
    }
}

/// The scaled blue polyline through the (x, y) series.
fn polyline(xs: &[f64], ys: &[f64], (xmin, xmax): (f64, f64), (ymin, ymax): (f64, f64)) -> String {
    let mut points = String::new();
    for i in 0..xs.len() {
        let px = scale(xs[i], xmin, xmax, 0);
        let py = scale(ys[i], ymin, ymax, 1);
        if i > 0 {
            points.push(' ');
        }
        points.push_str(&format!("{px:.1},{py:.1}"));
    }
    format!("<polyline points=\"{points}\" fill=\"none\" stroke=\"#89b4fa\" stroke-width=\"2\"/>")
}

/// Y range for bars: always includes 0 so positive values draw upward
/// from the baseline; degenerate ranges widen to 1.
fn bar_y_range(values: &[f64]) -> (f64, f64, f64) {
    let (mut ymin, ymax) = bounds(values);
    if ymin > 0.0 {
        ymin = 0.0;
    }
    let yrange = if ymax == ymin { 1.0 } else { ymax - ymin };
    (ymin, ymax, yrange)
}
