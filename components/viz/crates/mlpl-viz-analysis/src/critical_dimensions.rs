//! "Critical dimensions" heatmap -- per-feature importance
//! view of a PCA loadings matrix. Saga 33 step 032
//! (dim-reduction milestone Phase 1b).
//!
//! Input: `loadings [k, D]` from `pca_components(X, k)`.
//! Optional aux: `variance_explained [k]` from
//! `pca_variance_explained(X, k)`; when present, the legend
//! on the right side labels each row with its variance-
//! explained percentage.
//!
//! Renders a viridis-colored heatmap of `|loadings[i, j]|`
//! (k rows tall, D cols wide). Loadings are signed; the
//! ABSOLUTE value is what matters for "this feature
//! contributes a lot to this component" -- direction is
//! captured by the sign of the projection, not the loading.
//!
//! Feature names are NOT supported in this step (MLPL's
//! `svg()` only accepts one aux array per call). When
//! `Value::StrList` aux is plumbed through `svg()`, the viz
//! will switch to labelling columns with the names.

use mlpl_array::DenseArray;

use mlpl_viz_core::{H, PAD, VizError, W, write_svg_close, write_svg_open};

const LEGEND_RESERVE: f64 = 70.0;

/// Render `[k, D]` loadings as a per-feature importance heatmap.
/// `variance_explained` is an optional `[k]` vector of ratios
/// (each in `[0, 1]`); when present, each row's variance-
/// explained percentage is drawn next to its right-hand edge.
pub fn render_critical_dimensions(
    loadings: &DenseArray,
    variance_explained: Option<&DenseArray>,
) -> Result<String, VizError> {
    let (k, d) = validate_critical_dims_input(loadings, variance_explained)?;
    let mut out = String::new();
    write_svg_open(&mut out);
    if k == 0 || d == 0 {
        write_svg_close(&mut out);
        return Ok(out);
    }
    let plot_w = W - 2.0 * PAD - LEGEND_RESERVE;
    let plot_h = H - 2.0 * PAD;
    let cell_w = plot_w / d as f64;
    let cell_h = plot_h / k as f64;
    let max_abs = loadings_max_abs(loadings);
    draw_loading_cells(&mut out, loadings, k, d, cell_w, cell_h, max_abs);
    draw_component_labels(&mut out, k, cell_h, variance_explained, plot_w);
    write_svg_close(&mut out);
    Ok(out)
}

fn validate_critical_dims_input(
    loadings: &DenseArray,
    variance_explained: Option<&DenseArray>,
) -> Result<(usize, usize), VizError> {
    let dims = loadings.shape().dims();
    if dims.len() != 2 {
        return Err(VizError::InvalidShape(format!(
            "critical_dimensions expects [k, D] loadings, got {dims:?}"
        )));
    }
    let (k, d) = (dims[0], dims[1]);
    if let Some(v) = variance_explained
        && (v.rank() != 1 || v.data().len() != k)
    {
        return Err(VizError::InvalidShape(format!(
            "critical_dimensions variance_explained must be [k={k}], got {:?}",
            v.shape().dims()
        )));
    }
    Ok((k, d))
}

fn loadings_max_abs(loadings: &DenseArray) -> f64 {
    let m = loadings
        .data()
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    if m < f64::EPSILON { 1.0 } else { m }
}

fn draw_loading_cells(
    out: &mut String,
    loadings: &DenseArray,
    k: usize,
    d: usize,
    cell_w: f64,
    cell_h: f64,
    max_abs: f64,
) {
    let raw = loadings.data();
    for r in 0..k {
        for c in 0..d {
            let t = (raw[r * d + c].abs() / max_abs).clamp(0.0, 1.0);
            let (red, green, blue) = viridis(t);
            let x = PAD + cell_w * c as f64;
            let y = PAD + cell_h * r as f64;
            out.push_str(&format!(
                "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{cell_w:.1}\" height=\"{cell_h:.1}\" fill=\"rgb({red},{green},{blue})\"/>"
            ));
        }
    }
}

fn draw_component_labels(
    out: &mut String,
    k: usize,
    cell_h: f64,
    variance_explained: Option<&DenseArray>,
    plot_w: f64,
) {
    let label_x = PAD + plot_w + 6.0;
    for r in 0..k {
        let cy = PAD + cell_h * (r as f64 + 0.5) + 4.0;
        let label = match variance_explained {
            Some(v) => format!("PC{}: {:.1}%", r + 1, v.data()[r] * 100.0),
            None => format!("PC{}", r + 1),
        };
        out.push_str(&format!(
            "<text x=\"{label_x:.1}\" y=\"{cy:.1}\" fill=\"#cdd6f4\" font-family=\"monospace\" font-size=\"11\">{label}</text>"
        ));
    }
}

/// Same viridis ramp used by `render_heatmap` (4-segment
/// dark-purple -> teal -> green -> yellow). Inlined here to
/// avoid a module-fn-count budget hit from a shared helper.
fn viridis(t: f64) -> (u8, u8, u8) {
    const STOPS: &[(f64, f64, f64)] = &[
        (68.0, 1.0, 84.0),    // dark purple
        (59.0, 82.0, 139.0),  // blue
        (33.0, 144.0, 141.0), // teal
        (94.0, 201.0, 98.0),  // green
        (253.0, 231.0, 37.0), // bright yellow
    ];
    let n = (STOPS.len() - 1) as f64;
    let pos = (t.clamp(0.0, 1.0) * n).min(n - 1e-9);
    let i = pos as usize;
    let f = pos - i as f64;
    let a = STOPS[i];
    let b = STOPS[i + 1];
    let r = (a.0 + (b.0 - a.0) * f).round().clamp(0.0, 255.0) as u8;
    let g = (a.1 + (b.1 - a.1) * f).round().clamp(0.0, 255.0) as u8;
    let bl = (a.2 + (b.2 - a.2) * f).round().clamp(0.0, 255.0) as u8;
    (r, g, bl)
}
