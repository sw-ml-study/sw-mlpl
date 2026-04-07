//! Line plot rendering.

use mlpl_array::DenseArray;

use super::{VizError, bounds, scale, write_svg_close, write_svg_open};

/// Render a vector or Nx2 matrix as a polyline plot.
///
/// - Vector input: x = 0..N-1, y = element values (loss curves).
/// - Nx2 matrix input: rows are (x, y) pairs.
pub fn render_line(data: &DenseArray) -> Result<String, VizError> {
    let dims = data.shape().dims();
    let (xs, ys) = match dims {
        [] | [_] => extract_vector(data),
        [_, 2] => extract_matrix(data),
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

fn extract_vector(data: &DenseArray) -> (Vec<f64>, Vec<f64>) {
    let ys: Vec<f64> = data.data().to_vec();
    let xs: Vec<f64> = (0..ys.len()).map(|i| i as f64).collect();
    (xs, ys)
}

fn extract_matrix(data: &DenseArray) -> (Vec<f64>, Vec<f64>) {
    let raw = data.data();
    let n = raw.len() / 2;
    (0..n).map(|i| (raw[i * 2], raw[i * 2 + 1])).unzip()
}
