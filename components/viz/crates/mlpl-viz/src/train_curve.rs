//! Train-vs-validation loss curves on one set of axes. The *gap* is the
//! point: a beginner watches training loss (green) keep falling while
//! validation loss (peach) bottoms out and turns back up -- the canonical
//! picture of overfitting that a single `loss_curve` cannot show.

use mlpl_array::DenseArray;

use crate::svg::{H, PAD, VizError, bounds, scale, write_svg_close, write_svg_open};

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
