//! 1D chart renderings: bar charts and line plots.

use mlpl_array::DenseArray;

use super::{H, PAD, VizError, W, bounds, scale, write_svg_close, write_svg_open};

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
    // Y range includes 0 so positive values draw upward from the baseline.
    let (mut ymin, ymax) = bounds(values);
    if ymin > 0.0 {
        ymin = 0.0;
    }
    let yrange = if ymax == ymin { 1.0 } else { ymax - ymin };
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
    let mut points = String::new();
    for i in 0..xs.len() {
        let px = scale(xs[i], xmin, xmax, 0);
        let py = scale(ys[i], ymin, ymax, 1);
        if i > 0 {
            points.push(' ');
        }
        points.push_str(&format!("{px:.1},{py:.1}"));
    }
    out.push_str(&format!(
        "<polyline points=\"{points}\" fill=\"none\" stroke=\"#89b4fa\" stroke-width=\"2\"/>"
    ));
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
