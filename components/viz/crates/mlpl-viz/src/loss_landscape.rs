//! Loss-landscape view: a two-weight loss surface drawn as a heatmap with
//! the optimizer's trajectory walked across it. The point is to turn
//! "gradient descent" from a formula into a picture -- a path rolling
//! downhill into the dark low-loss basin.

use mlpl_array::DenseArray;

use crate::svg::{H, PAD, VizError, W, data_range, write_svg_close, write_svg_open};

/// Heatmap of a `rows x cols` loss surface (dark = low loss) overlaid with
/// an optimizer trajectory. `path` is `[N, 2]`; each row is an (x, y) point
/// already normalized to `[0, 1]` over the swept weight ranges (the demo
/// owns that normalization since it chose the grid bounds). A green dot
/// marks the start, a red dot the end.
pub fn analysis_loss_landscape(
    surface: &DenseArray,
    dims: &DenseArray,
    path: &DenseArray,
) -> Result<String, VizError> {
    let (rows, cols, n) = validate(surface, dims, path)?;
    let mut out = String::new();
    write_svg_open(&mut out);
    draw_surface(&mut out, surface.data(), rows, cols);
    draw_path(&mut out, path.data(), n);
    write_svg_close(&mut out);
    Ok(out)
}

fn validate(
    surface: &DenseArray,
    dims: &DenseArray,
    path: &DenseArray,
) -> Result<(usize, usize, usize), VizError> {
    let bad = |m: &str| Err(VizError::InvalidShape(m.into()));
    if dims.rank() != 1 || dims.data().len() != 2 {
        return bad("loss_landscape dims must be [rows, cols]");
    }
    let (rows, cols) = (dims.data()[0] as usize, dims.data()[1] as usize);
    if surface.data().len() != rows * cols {
        return bad("loss_landscape surface length must equal rows*cols");
    }
    let pdims = path.shape().dims();
    if pdims.len() != 2 || pdims[1] != 2 {
        return bad("loss_landscape path must be [N, 2]");
    }
    Ok((rows, cols, pdims[0]))
}

/// Fill the plot area with one rect per surface cell. Row 0 is the lowest
/// swept weight-1 value, so flip it to the image bottom. Dark = low loss.
fn draw_surface(out: &mut String, raw: &[f64], rows: usize, cols: usize) {
    let (lo, hi) = data_range(raw);
    let span = (hi - lo).abs().max(f64::EPSILON);
    let cw = (W - 2.0 * PAD) / cols as f64;
    let ch = (H - 2.0 * PAD) / rows as f64;
    for r in 0..rows {
        for c in 0..cols {
            let t = ((raw[(rows - 1 - r) * cols + c] - lo) / span).clamp(0.0, 1.0);
            let (rd, gr, bl) = (30.0 + 220.0 * t, 30.0 + 180.0 * t, 46.0 + 50.0 * t);
            let (x, y) = (PAD + cw * c as f64, PAD + ch * r as f64);
            out.push_str(&format!(
                "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{cw:.1}\" height=\"{ch:.1}\" \
                 fill=\"rgb({},{},{})\"/>",
                rd as u8, gr as u8, bl as u8
            ));
        }
    }
}

/// Connect the normalized trajectory points and mark start (green) / end
/// (red). y is flipped so weight-1 increases upward, matching the surface.
fn draw_path(out: &mut String, pts: &[f64], n: usize) {
    if n == 0 {
        return;
    }
    let px = |f: f64| PAD + f.clamp(0.0, 1.0) * (W - 2.0 * PAD);
    let py = |f: f64| PAD + (1.0 - f.clamp(0.0, 1.0)) * (H - 2.0 * PAD);
    let mut poly = String::new();
    for i in 0..n {
        if !poly.is_empty() {
            poly.push(' ');
        }
        poly.push_str(&format!("{:.1},{:.1}", px(pts[i * 2]), py(pts[i * 2 + 1])));
    }
    out.push_str(&format!(
        "<polyline points=\"{poly}\" fill=\"none\" stroke=\"#cdd6f4\" stroke-width=\"2\"/>\
         <circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"5\" fill=\"#a6e3a1\"/>\
         <circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"5\" fill=\"#f38ba8\"/>",
        px(pts[0]),
        py(pts[1]),
        px(pts[(n - 1) * 2]),
        py(pts[(n - 1) * 2 + 1]),
    ));
}
