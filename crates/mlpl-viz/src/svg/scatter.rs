//! 2-D and 3-D scatter plot rendering.
//!
//! `render_scatter` consumes an `[N, 2]` matrix and
//! produces a flat SVG scatter. `render_scatter3d`
//! (Saga 16 step 003) consumes `[N, 3]` (point cloud)
//! or `[N, 4]` (points + integer cluster id) and
//! produces an orthographic-projection SVG with a fixed
//! azimuth=30 / elevation=20 camera, one `<circle>` per
//! point, small labeled axis gizmos at a fixed canvas
//! position, and an optional legend when the 4th column
//! is present. No rotation, no interactivity -- a
//! static snapshot good for docs and cluster-structure
//! checks.

use mlpl_array::DenseArray;

use super::{PAD, VizError, bounds, scale, write_svg_close, write_svg_open};

/// Fixed azimuth (rotation about the vertical axis) and
/// elevation (tilt above the horizontal) for the
/// `scatter3d` orthographic camera.
const AZIMUTH_RAD: f64 = 30.0 * std::f64::consts::PI / 180.0;
const ELEVATION_RAD: f64 = 20.0 * std::f64::consts::PI / 180.0;

/// Standard 8-color palette matched to
/// `analysis::scatter_labeled` so the 4-column
/// scatter3d variant reads the same as the 2-D analog.
const PALETTE: &[&str] = &[
    "#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7", "#94e2d5", "#f9e2af", "#eba0ac",
];

/// Parsed scatter3d input: the point list plus optional
/// labels. Named so `parse_scatter3d_input`'s return
/// type reads cleanly at the call site and clippy does
/// not flag `type_complexity`.
struct Scatter3dInput {
    points: Vec<(f64, f64, f64)>,
    labels: Option<Vec<usize>>,
}

/// 2-D bounds of the projected point cloud; drives the
/// `scale` helper in the scatter3d renderer.
struct ProjectedBounds {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
}

/// Render an Nx2 matrix as a 2-D scatter plot.
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

/// Render an `[N, 3]` or `[N, 4]` array as a 3-D scatter
/// SVG via orthographic projection. Orchestrator:
/// validate -> project -> write axes -> write points ->
/// write legend.
pub fn render_scatter3d(data: &DenseArray) -> Result<String, VizError> {
    let input = parse_scatter3d_input(data)?;
    let mut out = String::new();
    write_svg_open(&mut out);
    render_scatter3d_axes(&mut out);
    if !input.points.is_empty() {
        let projected: Vec<(f64, f64)> = input
            .points
            .iter()
            .map(|&p| project_orthographic(p))
            .collect();
        let xs: Vec<f64> = projected.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = projected.iter().map(|p| p.1).collect();
        let (xmin, xmax) = bounds(&xs);
        let (ymin, ymax) = bounds(&ys);
        let pb = ProjectedBounds {
            xmin,
            xmax,
            ymin,
            ymax,
        };
        render_scatter3d_points(&mut out, &projected, &pb, input.labels.as_deref());
    }
    if let Some(lbls) = input.labels.as_ref() {
        render_scatter3d_legend(&mut out, lbls);
    }
    write_svg_close(&mut out);
    Ok(out)
}

/// Shape validation for scatter3d: rank-2, last dim 3
/// or 4. Returns the validated points + optional labels
/// as a named struct.
fn parse_scatter3d_input(data: &DenseArray) -> Result<Scatter3dInput, VizError> {
    let dims = data.shape().dims();
    if dims.len() != 2 {
        return Err(VizError::InvalidShape(format!(
            "scatter3d expects rank-2 [N, 3] or [N, 4], got shape {dims:?}"
        )));
    }
    let cols = dims[1];
    if cols != 3 && cols != 4 {
        return Err(VizError::InvalidShape(format!(
            "scatter3d expects 3 or 4 columns, got {cols}"
        )));
    }
    let raw = data.data();
    let n = dims[0];
    let mut points = Vec::with_capacity(n);
    let mut labels = if cols == 4 {
        Some(Vec::with_capacity(n))
    } else {
        None
    };
    for i in 0..n {
        let base = i * cols;
        points.push((raw[base], raw[base + 1], raw[base + 2]));
        if let Some(ls) = labels.as_mut() {
            ls.push(raw[base + 3] as usize);
        }
    }
    Ok(Scatter3dInput { points, labels })
}

/// Pure orthographic projection at the fixed
/// `AZIMUTH_RAD` / `ELEVATION_RAD`.
///
/// ```text
/// x_2d =  x * cos(az) - y * sin(az)
/// y_2d = (x * sin(az) + y * cos(az)) * sin(el) + z * cos(el)
/// ```
fn project_orthographic(p: (f64, f64, f64)) -> (f64, f64) {
    let (x, y, z) = p;
    let ca = AZIMUTH_RAD.cos();
    let sa = AZIMUTH_RAD.sin();
    let ce = ELEVATION_RAD.cos();
    let se = ELEVATION_RAD.sin();
    let x_2d = x * ca - y * sa;
    let y_2d = (x * sa + y * ca) * se + z * ce;
    (x_2d, y_2d)
}

/// Small axis gizmo at a fixed canvas position (bottom-
/// left corner, inside the pad). Three arrows pointing
/// along the projected X/Y/Z axes, each 24 pixels long,
/// labeled with a letter at the tip. Not scaled with
/// the data -- the gizmo indicates camera orientation.
fn render_scatter3d_axes(out: &mut String) {
    let origin_x = PAD + 20.0;
    let origin_y = super::H - PAD - 20.0;
    let len = 24.0;
    for (pt3, label, color) in [
        ((1.0_f64, 0.0, 0.0), "X", "#f38ba8"),
        ((0.0, 1.0, 0.0), "Y", "#a6e3a1"),
        ((0.0, 0.0, 1.0), "Z", "#89b4fa"),
    ] {
        let (dx, dy) = project_orthographic(pt3);
        // SVG y grows downward; our projection's y grows
        // upward -- flip the arrow's vertical component.
        let tip_x = origin_x + dx * len;
        let tip_y = origin_y - dy * len;
        out.push_str(&format!(
            "<line x1=\"{origin_x:.1}\" y1=\"{origin_y:.1}\" \
             x2=\"{tip_x:.1}\" y2=\"{tip_y:.1}\" \
             stroke=\"{color}\" stroke-width=\"1.5\"/>"
        ));
        out.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"{color}\" \
             font-family=\"monospace\" font-size=\"11\">{label}</text>",
            tip_x + 2.0,
            tip_y - 2.0
        ));
    }
}

/// Render one `<circle>` per projected point, using the
/// palette color keyed by cluster id when labels are
/// present and a neutral default otherwise.
fn render_scatter3d_points(
    out: &mut String,
    projected: &[(f64, f64)],
    pb: &ProjectedBounds,
    labels: Option<&[usize]>,
) {
    for (i, &(x2d, y2d)) in projected.iter().enumerate() {
        let cx = scale(x2d, pb.xmin, pb.xmax, 0);
        let cy = scale(y2d, pb.ymin, pb.ymax, 1);
        let fill = match labels {
            Some(ls) => PALETTE[ls[i] % PALETTE.len()],
            None => "#89b4fa",
        };
        out.push_str(&format!(
            "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"3\" fill=\"{fill}\" \
             stroke=\"#1e1e2e\" stroke-width=\"0.8\"/>"
        ));
    }
}

/// Legend: one row per unique cluster id with a color
/// swatch + text, anchored at the top-right of the
/// canvas.
fn render_scatter3d_legend(out: &mut String, labels: &[usize]) {
    let mut unique: Vec<usize> = labels.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let x = super::W - PAD - 60.0;
    let mut y = PAD + 10.0;
    out.push_str(&format!(
        "<g class=\"legend\" transform=\"translate(0,0)\"><text x=\"{x:.1}\" y=\"{:.1}\" \
         fill=\"#cdd6f4\" font-family=\"monospace\" font-size=\"10\">legend</text></g>",
        y - 2.0
    ));
    for id in unique {
        y += 14.0;
        let color = PALETTE[id % PALETTE.len()];
        out.push_str(&format!(
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"4\" fill=\"{color}\" \
             stroke=\"#1e1e2e\" stroke-width=\"0.8\"/>"
        ));
        out.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#cdd6f4\" font-family=\"monospace\" \
             font-size=\"10\">{id}</text>",
            x + 10.0,
            y + 3.0
        ));
    }
}
