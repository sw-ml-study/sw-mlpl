//! Bar chart rendering.

use mlpl_array::DenseArray;

use super::{H, PAD, VizError, W, bounds, write_svg_close, write_svg_open};

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
