//! Train-vs-validation loss curves on one set of axes. The *gap* is the
//! point: a beginner watches training loss (green) keep falling while
//! validation loss (peach) bottoms out and turns back up -- the canonical
//! picture of overfitting that a single `loss_curve` cannot show.

use mlpl_array::DenseArray;

use mlpl_viz_core::{
    H, PAD, VizError, bounds, scale, write_corner_scale_labels, write_svg_close, write_svg_open,
};

const TRAIN_COLOR: &str = "#a6e3a1";
const VAL_COLOR: &str = "#fab387";

/// Two loss curves (train green, validation peach) sharing one y-axis so
/// the generalization gap is directly visible. Both inputs are 1-D vectors,
/// one entry per recorded checkpoint; they need not be the same length.
pub fn analysis_train_val_curve(train: &DenseArray, val: &DenseArray) -> Result<String, VizError> {
    require_vector(train)?;
    require_vector(val)?;
    let (tr, va) = (train.data(), val.data());
    let mut out = String::new();
    write_svg_open(&mut out);
    if tr.is_empty() && va.is_empty() {
        write_svg_close(&mut out);
        return Ok(out);
    }
    let combined: Vec<f64> = tr.iter().chain(va).copied().collect();
    let (ymin, ymax) = bounds(&combined);
    let xmax = tr.len().max(va.len()).saturating_sub(1).max(1) as f64;
    // Distinct dash patterns (train long-dash, val dotted) so the two lines
    // stay legible even when a well-generalizing model makes them nearly
    // coincide -- you see alternating green dashes and peach dots.
    out.push_str(&loss_line(tr, ymin, ymax, xmax, TRAIN_COLOR, "8 4"));
    out.push_str(&loss_line(va, ymin, ymax, xmax, VAL_COLOR, "2 4"));
    out.push_str(&legend(ymin, ymax));
    write_svg_close(&mut out);
    Ok(out)
}

fn require_vector(a: &DenseArray) -> Result<(), VizError> {
    if a.rank() != 1 {
        return Err(VizError::InvalidShape(format!(
            "train_val_curve expects vectors, got rank {}",
            a.rank()
        )));
    }
    Ok(())
}

/// One `dash`-patterned polyline mapping a loss vector across the shared
/// `[ymin, ymax]` axis.
fn loss_line(raw: &[f64], ymin: f64, ymax: f64, xmax: f64, color: &str, dash: &str) -> String {
    if raw.is_empty() {
        return String::new();
    }
    let mut pts = String::new();
    for (i, &v) in raw.iter().enumerate() {
        if !pts.is_empty() {
            pts.push(' ');
        }
        let (cx, cy) = (scale(i as f64, 0.0, xmax, 0), scale(v, ymin, ymax, 1));
        pts.push_str(&format!("{cx:.1},{cy:.1}"));
    }
    format!(
        "<polyline points=\"{pts}\" fill=\"none\" stroke=\"{color}\" \
         stroke-width=\"2\" stroke-dasharray=\"{dash}\"/>"
    )
}

/// Color-keyed "train"/"val" labels plus the y-axis bound readouts.
fn legend(ymin: f64, ymax: f64) -> String {
    let txt = |x: f64, y: f64, fill: &str, s: &str| {
        format!(
            "<text x=\"{x:.1}\" y=\"{y:.1}\" fill=\"{fill}\" \
             font-family=\"monospace\" font-size=\"12\">{s}</text>"
        )
    };
    let lab = |y: f64, v: f64| {
        format!(
            "<text x=\"4\" y=\"{y:.1}\" fill=\"#a6adc8\" \
             font-family=\"monospace\" font-size=\"10\">{v:.3}</text>"
        )
    };
    format!(
        "{}{}{}{}",
        txt(PAD, 18.0, TRAIN_COLOR, "train"),
        txt(PAD + 48.0, 18.0, VAL_COLOR, "val"),
        lab(PAD + 4.0, ymax),
        lab(H - PAD, ymin),
    )
}

const FRONT_COLOR: &str = "#f38ba8";
const DOMINATED_COLOR: &str = "#89b4fa";

/// Pareto frontier plot: every (x, y) metric pair as a dot --
/// dominated points blue, frontier members pink and larger --
/// plus the staircase line through the frontier, sorted by x.
/// `mask` is `pareto_front`'s 0/1 vector; the eval layer computes
/// it with the builtin, so plot and mask can never disagree.
pub fn analysis_pareto_plot(points: &DenseArray, mask: &DenseArray) -> Result<String, VizError> {
    let (n, xs, ys) = pareto_inputs(points, mask)?;
    let (xmin, xmax) = bounds(&xs);
    let (ymin, ymax) = bounds(&ys);
    let px = |i: usize| (scale(xs[i], xmin, xmax, 0), scale(ys[i], ymin, ymax, 1));
    let mut out = String::new();
    write_svg_open(&mut out);
    staircase(&mut out, mask.data(), &xs, px);
    for i in 0..n {
        let (cx, cy) = px(i);
        let front = mask.data()[i] != 0.0;
        let (fill, r) = if front {
            (FRONT_COLOR, 5)
        } else {
            (DOMINATED_COLOR, 4)
        };
        out.push_str(&format!(
            "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r}\" fill=\"{fill}\" stroke=\"#1e1e2e\" stroke-width=\"1.5\"/>"
        ));
    }
    write_corner_scale_labels(&mut out, xmin, xmax, ymin, ymax);
    write_svg_close(&mut out);
    Ok(out)
}

/// Validate the Nx2 points + length-N mask pair and split the
/// coordinate columns.
#[allow(clippy::type_complexity)]
fn pareto_inputs(
    points: &DenseArray,
    mask: &DenseArray,
) -> Result<(usize, Vec<f64>, Vec<f64>), VizError> {
    let dims = points.shape().dims();
    if dims.len() != 2 || dims[1] != 2 {
        return Err(VizError::InvalidShape(format!(
            "pareto_plot expects Nx2 points, got {dims:?}"
        )));
    }
    let n = dims[0];
    if mask.rank() != 1 || mask.data().len() != n {
        return Err(VizError::InvalidShape(format!(
            "pareto_plot mask length {} must match {n}",
            mask.data().len()
        )));
    }
    let pts = points.data();
    let xs: Vec<f64> = (0..n).map(|i| pts[i * 2]).collect();
    let ys: Vec<f64> = (0..n).map(|i| pts[i * 2 + 1]).collect();
    Ok((n, xs, ys))
}

/// The frontier staircase: frontier points sorted by x, joined by
/// horizontal-then-vertical steps, drawn UNDER the dots.
fn staircase(out: &mut String, mask: &[f64], xs: &[f64], px: impl Fn(usize) -> (f64, f64)) {
    let mut front: Vec<usize> = (0..xs.len()).filter(|&i| mask[i] != 0.0).collect();
    front.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    if front.len() < 2 {
        return;
    }
    let (x0, y0) = px(front[0]);
    let mut d = format!("M {x0:.1} {y0:.1}");
    for w in front.windows(2) {
        let (x2, y2) = px(w[1]);
        let (_, y1) = px(w[0]);
        d.push_str(&format!(" L {x2:.1} {y1:.1} L {x2:.1} {y2:.1}"));
    }
    out.push_str(&format!(
        "<path d=\"{d}\" fill=\"none\" stroke=\"{FRONT_COLOR}\" stroke-width=\"2\" stroke-dasharray=\"5 3\" opacity=\"0.8\"/>"
    ));
}
