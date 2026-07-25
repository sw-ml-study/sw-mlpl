//! Pure SVG builder for the live loss panel: train (green, solid) and
//! validation (peach, dotted) series on ONE shared y-axis, matching
//! `mlpl-viz`'s static `train_val_curve` palette so the live chart and
//! the final rendered curve read as the same picture. String assembly
//! only -- no browser APIs -- so this tests natively.

const W: f64 = 360.0;
const H: f64 = 140.0;
const PAD: f64 = 24.0;
const TRAIN_COLOR: &str = "#a6e3a1";
const VAL_COLOR: &str = "#fab387";

/// The live chart for the streamed `(train, val)` series so far, or
/// `None` until any series has two points (a single dot reads as
/// noise, and the panel stays hidden for metric-less evals).
#[must_use]
pub fn loss_panel_svg(train: &[f64], val: &[f64]) -> Option<String> {
    let steps = train.len().max(val.len());
    if steps < 2 {
        return None;
    }
    let combined: Vec<f64> = train.iter().chain(val).copied().collect();
    let (ymin, ymax) = bounds(&combined);
    let xmax = (steps - 1) as f64;
    let mut out = format!(
        "<svg viewBox=\"0 0 {W} {H}\" width=\"100%\" height=\"{H}\" \
         xmlns=\"http://www.w3.org/2000/svg\">"
    );
    out.push_str(&polyline(train, (ymin, ymax), xmax, TRAIN_COLOR, "none"));
    out.push_str(&polyline(val, (ymin, ymax), xmax, VAL_COLOR, "2 4"));
    out.push_str(&labels(ymin, ymax));
    out.push_str("</svg>");
    Some(out)
}

/// `(min, max)` over the combined series, padded when the series is
/// constant so the y-scale never divides by zero.
fn bounds(vals: &[f64]) -> (f64, f64) {
    let (min, max) = vals
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if max > min {
        (min, max)
    } else {
        (min - 0.5, min + 0.5)
    }
}

/// One series as a `dash`-patterned polyline across the shared axes.
fn polyline(vals: &[f64], (ymin, ymax): (f64, f64), xmax: f64, color: &str, dash: &str) -> String {
    if vals.is_empty() {
        return String::new();
    }
    let pts: Vec<String> = vals
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let cx = PAD + (i as f64) / xmax * (W - 2.0 * PAD);
            let cy = H - PAD - (v - ymin) / (ymax - ymin) * (H - 2.0 * PAD);
            format!("{cx:.1},{cy:.1}")
        })
        .collect();
    format!(
        "<polyline points=\"{}\" fill=\"none\" stroke=\"{color}\" \
         stroke-width=\"2\" stroke-dasharray=\"{dash}\"/>",
        pts.join(" ")
    )
}

/// Shared-axis bound readouts (top = ymax, bottom = ymin).
fn labels(ymin: f64, ymax: f64) -> String {
    let lab = |y: f64, v: f64| {
        format!(
            "<text x=\"2\" y=\"{y:.1}\" fill=\"#a6adc8\" \
             font-family=\"monospace\" font-size=\"10\">{v:.3}</text>"
        )
    };
    format!("{}{}", lab(PAD - 4.0, ymax), lab(H - PAD + 12.0, ymin))
}
