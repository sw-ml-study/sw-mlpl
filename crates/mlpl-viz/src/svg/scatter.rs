//! Scatter plot rendering.

use mlpl_array::DenseArray;

use super::{VizError, bounds, scale, write_svg_close, write_svg_open};

/// Render an Nx2 matrix as a 2D scatter plot.
pub fn render_scatter(data: &DenseArray) -> Result<String, VizError> {
    let dims = data.shape().dims();
    if dims.len() != 2 || dims[1] != 2 {
        return Err(VizError::InvalidShape(format!(
            "scatter expects Nx2 matrix, got {dims:?}"
        )));
    }
    let n = dims[0];
    let mut out = String::new();
    write_svg_open(&mut out);

    if n == 0 {
        write_svg_close(&mut out);
        return Ok(out);
    }

    let raw = data.data();
    let (xs, ys): (Vec<f64>, Vec<f64>) = (0..n).map(|i| (raw[i * 2], raw[i * 2 + 1])).unzip();
    let (xmin, xmax) = bounds(&xs);
    let (ymin, ymax) = bounds(&ys);
    for i in 0..n {
        let cx = scale(xs[i], xmin, xmax, 0);
        let cy = scale(ys[i], ymin, ymax, 1);
        out.push_str(&format!(
            "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"3\" fill=\"#89b4fa\"/>"
        ));
    }
    write_svg_close(&mut out);
    Ok(out)
}
