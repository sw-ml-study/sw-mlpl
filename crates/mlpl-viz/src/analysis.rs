//! High-level analysis/visualization helpers. Each function takes one
//! or more `DenseArray`s and returns a complete SVG string.

use mlpl_array::DenseArray;

use crate::svg::{H, PAD, VizError, W, bounds, data_range, scale, write_svg_close, write_svg_open};

type CellFn<'a> = dyn Fn(usize, usize) -> (String, Option<String>) + 'a;

const PALETTE: &[&str] = &[
    "#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7", "#94e2d5", "#f9e2af", "#eba0ac",
];

/// Histogram of a vector with `bins` equal-width bins.
pub fn analysis_hist(data: &DenseArray, bins: usize) -> Result<String, VizError> {
    if data.rank() != 1 {
        return Err(VizError::InvalidShape(format!(
            "hist expects a vector, got rank {}",
            data.rank()
        )));
    }
    if bins == 0 {
        return Err(VizError::InvalidShape("hist bins must be > 0".into()));
    }
    let raw = data.data();
    let mut out = String::new();
    write_svg_open(&mut out);
    if raw.is_empty() {
        write_svg_close(&mut out);
        return Ok(out);
    }
    let (lo, hi) = data_range(raw);
    let span = if (hi - lo).abs() < f64::EPSILON {
        1.0
    } else {
        hi - lo
    };
    let mut counts = vec![0_usize; bins];
    for &v in raw {
        let idx = (((v - lo) / span) * bins as f64) as usize;
        counts[idx.min(bins - 1)] += 1;
    }
    let max_c = counts.iter().copied().max().unwrap_or(1).max(1) as f64;
    let bw = (W - 2.0 * PAD) / bins as f64;
    let plot_h = H - 2.0 * PAD;
    for (i, &c) in counts.iter().enumerate() {
        let bh = (c as f64 / max_c) * plot_h;
        let x = PAD + bw * i as f64;
        let y = H - PAD - bh;
        out.push_str(&format!(
            "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{bh:.1}\" fill=\"#89b4fa\"/>",
            (bw - 1.0).max(0.0)
        ));
    }
    write_svg_close(&mut out);
    Ok(out)
}

/// Multi-color scatter plot. `points` is Nx2; `labels` is length-N.
pub fn analysis_scatter_labeled(
    points: &DenseArray,
    labels: &DenseArray,
) -> Result<String, VizError> {
    let dims = points.shape().dims();
    if dims.len() != 2 || dims[1] != 2 {
        return Err(VizError::InvalidShape(format!(
            "scatter_labeled expects Nx2 points, got {dims:?}"
        )));
    }
    let n = dims[0];
    if labels.rank() != 1 || labels.data().len() != n {
        return Err(VizError::InvalidShape(format!(
            "scatter_labeled labels length {} must match {} points",
            labels.data().len(),
            n
        )));
    }
    let mut out = String::new();
    write_svg_open(&mut out);
    draw_labeled_points(&mut out, points.data(), labels.data(), n, 4);
    write_svg_close(&mut out);
    Ok(out)
}

/// Render a loss curve from a vector with axis labels.
pub fn analysis_loss_curve(losses: &DenseArray) -> Result<String, VizError> {
    if losses.rank() != 1 {
        return Err(VizError::InvalidShape(format!(
            "loss_curve expects a vector, got rank {}",
            losses.rank()
        )));
    }
    let raw = losses.data();
    let mut out = String::new();
    write_svg_open(&mut out);
    if raw.is_empty() {
        write_svg_close(&mut out);
        return Ok(out);
    }
    let (ymin, ymax) = bounds(raw);
    let xmax = (raw.len() - 1).max(1) as f64;
    let mut pts = String::new();
    for (i, &v) in raw.iter().enumerate() {
        let cx = scale(i as f64, 0.0, xmax, 0);
        let cy = scale(v, ymin, ymax, 1);
        if !pts.is_empty() {
            pts.push(' ');
        }
        pts.push_str(&format!("{cx:.1},{cy:.1}"));
    }
    out.push_str(&format!(
        "<polyline points=\"{pts}\" fill=\"none\" stroke=\"#a6e3a1\" stroke-width=\"2\"/>"
    ));
    out.push_str(&format!(
        "<text x=\"{PAD:.1}\" y=\"18\" fill=\"#cdd6f4\" font-family=\"monospace\" font-size=\"12\">loss</text>"
    ));
    out.push_str(&format!(
        "<text x=\"4\" y=\"{:.1}\" fill=\"#a6adc8\" font-family=\"monospace\" font-size=\"10\">{ymax:.3}</text>",
        PAD + 4.0,
    ));
    out.push_str(&format!(
        "<text x=\"4\" y=\"{:.1}\" fill=\"#a6adc8\" font-family=\"monospace\" font-size=\"10\">{ymin:.3}</text>",
        H - PAD,
    ));
    write_svg_close(&mut out);
    Ok(out)
}

/// Confusion matrix from two vectors of class ids.
pub fn analysis_confusion_matrix(
    predicted: &DenseArray,
    actual: &DenseArray,
) -> Result<String, VizError> {
    if predicted.rank() != 1 || actual.rank() != 1 {
        return Err(VizError::InvalidShape(
            "confusion_matrix expects two vectors".into(),
        ));
    }
    let p = predicted.data();
    let a = actual.data();
    if p.len() != a.len() {
        return Err(VizError::InvalidShape(format!(
            "confusion_matrix length mismatch: {} vs {}",
            p.len(),
            a.len()
        )));
    }
    let mut out = String::new();
    write_svg_open(&mut out);
    if p.is_empty() {
        write_svg_close(&mut out);
        return Ok(out);
    }
    let k = p
        .iter()
        .chain(a.iter())
        .map(|&v| v as usize + 1)
        .max()
        .unwrap_or(1);
    let mut counts = vec![0_usize; k * k];
    for i in 0..p.len() {
        counts[(a[i] as usize) * k + (p[i] as usize)] += 1;
    }
    let max = counts.iter().copied().max().unwrap_or(1).max(1) as f64;
    draw_grid_cells(&mut out, k, k, &|r, c| {
        let v = counts[r * k + c] as f64;
        let s = (40.0 + (v / max).clamp(0.0, 1.0) * 200.0) as u8;
        (
            format!("rgb({s},{s},255)"),
            Some(format!("{}", counts[r * k + c])),
        )
    });
    write_svg_close(&mut out);
    Ok(out)
}

/// 2D classifier surface with separately-supplied training points.
pub fn analysis_boundary_2d(
    grid_outputs: &DenseArray,
    dims: &DenseArray,
    points: &DenseArray,
    labels: &DenseArray,
) -> Result<String, VizError> {
    let bad = |m: &str| Err(VizError::InvalidShape(m.into()));
    if dims.rank() != 1 || dims.data().len() != 2 {
        return bad("boundary_2d dims must be [rows, cols]");
    }
    let rows = dims.data()[0] as usize;
    let cols = dims.data()[1] as usize;
    let raw = grid_outputs.data();
    if raw.len() != rows * cols {
        return bad("boundary_2d grid_outputs length does not match dims");
    }
    let pdims = points.shape().dims();
    if pdims.len() != 2 || pdims[1] != 2 {
        return bad("boundary_2d points must be Nx2");
    }
    let n = pdims[0];
    if labels.rank() != 1 || labels.data().len() != n {
        return bad("boundary_2d labels length must match points");
    }
    let mut out = String::new();
    write_svg_open(&mut out);
    let (lo, hi) = data_range(raw);
    let span = (hi - lo).abs().max(f64::EPSILON);
    draw_grid_cells(&mut out, rows, cols, &|r, c| {
        let t = ((raw[r * cols + c] - lo) / span).clamp(0.0, 1.0);
        let rd = (137.0 + 106.0 * t) as u8;
        let gr = (180.0 - 41.0 * t) as u8;
        let bl = (250.0 - 82.0 * t) as u8;
        (format!("rgb({rd},{gr},{bl})"), None)
    });
    draw_labeled_points(&mut out, points.data(), labels.data(), n, 5);
    write_svg_close(&mut out);
    Ok(out)
}

fn draw_grid_cells(out: &mut String, rows: usize, cols: usize, cell: &CellFn<'_>) {
    if rows == 0 || cols == 0 {
        return;
    }
    let cw = (W - 2.0 * PAD) / cols as f64;
    let ch = (H - 2.0 * PAD) / rows as f64;
    for r in 0..rows {
        for c in 0..cols {
            let (fill, text) = cell(r, c);
            let x = PAD + cw * c as f64;
            let y = PAD + ch * r as f64;
            out.push_str(&format!(
                "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{cw:.1}\" height=\"{ch:.1}\" fill=\"{fill}\"/>"
            ));
            if let Some(t) = text {
                out.push_str(&format!(
                    "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#1e1e2e\" font-family=\"monospace\" font-size=\"12\" text-anchor=\"middle\">{t}</text>",
                    x + cw / 2.0,
                    y + ch / 2.0 + 4.0,
                ));
            }
        }
    }
}

fn draw_labeled_points(out: &mut String, pts: &[f64], labels: &[f64], n: usize, r: u32) {
    if n == 0 {
        return;
    }
    let xs: Vec<f64> = (0..n).map(|i| pts[i * 2]).collect();
    let ys: Vec<f64> = (0..n).map(|i| pts[i * 2 + 1]).collect();
    let (xmin, xmax) = bounds(&xs);
    let (ymin, ymax) = bounds(&ys);
    for i in 0..n {
        let cx = scale(xs[i], xmin, xmax, 0);
        let cy = scale(ys[i], ymin, ymax, 1);
        let fill = PALETTE[(labels[i] as usize) % PALETTE.len()];
        out.push_str(&format!(
            "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r}\" fill=\"{fill}\" stroke=\"#1e1e2e\" stroke-width=\"1.5\"/>"
        ));
    }
}
