//! SVG diagram rendering for MLPL arrays.

use std::fmt;

use mlpl_array::DenseArray;

const W: f64 = 400.0;
const H: f64 = 300.0;
const PAD: f64 = 30.0;

/// Errors produced by SVG rendering.
#[derive(Clone, Debug, PartialEq)]
pub enum VizError {
    /// The data shape is not valid for the requested diagram.
    InvalidShape(String),
    /// The diagram type name is not recognized.
    UnknownType(String),
}

impl fmt::Display for VizError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(s) => write!(f, "invalid shape for diagram: {s}"),
            Self::UnknownType(s) => write!(f, "unknown svg type: '{s}'"),
        }
    }
}

impl std::error::Error for VizError {}

/// Dispatch on a diagram type name. Returns the rendered SVG string.
pub fn render(data: &DenseArray, type_name: &str) -> Result<String, VizError> {
    match type_name {
        "scatter" => render_scatter(data),
        other => Err(VizError::UnknownType(other.to_string())),
    }
}

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

/// Min/max of a slice, returning (lo-1, hi+1) when all values are equal
/// so the resulting range is non-zero.
pub(crate) fn bounds(values: &[f64]) -> (f64, f64) {
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
    if lo == hi {
        (lo - 1.0, hi + 1.0)
    } else {
        (lo, hi)
    }
}

/// Scale a data coordinate to plot pixels. `axis` 0 = x, 1 = y (flipped).
pub(crate) fn scale(v: f64, lo: f64, hi: f64, axis: u8) -> f64 {
    let t = (v - lo) / (hi - lo);
    if axis == 0 {
        PAD + t * (W - 2.0 * PAD)
    } else {
        H - PAD - t * (H - 2.0 * PAD)
    }
}

fn write_svg_open(out: &mut String) {
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {W} {H}\" width=\"{W}\" height=\"{H}\">"
    ));
    out.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#1e1e2e\"/>");
    let (x0, x1, y0, y1) = (PAD, W - PAD, PAD, H - PAD);
    out.push_str(&format!(
        "<line x1=\"{x0}\" y1=\"{y1}\" x2=\"{x1}\" y2=\"{y1}\" stroke=\"#45475a\" stroke-width=\"1\"/>"
    ));
    out.push_str(&format!(
        "<line x1=\"{x0}\" y1=\"{y0}\" x2=\"{x0}\" y2=\"{y1}\" stroke=\"#45475a\" stroke-width=\"1\"/>"
    ));
}

fn write_svg_close(out: &mut String) {
    out.push_str("</svg>");
}
