//! `svg(frames, "life")` -- a Game of Life grid that ANIMATES.
//!
//! Takes a rank-3 `[T, H, W]` frames tensor (rank-2 `[H, W]` is
//! one frame) and emits a single self-contained SVG: one `<g>`
//! per frame, stepped by discrete-mode SMIL opacity animations.
//! No script, CSP-safe, still animates in the downloaded file
//! and on the static pages site. T = 1 renders a static grid.

use mlpl_array::DenseArray;
use mlpl_viz_core::VizError;

/// Seconds each generation stays on screen.
const FRAME_SECS: f64 = 0.35;
/// Cell edge in SVG units. Sized so a small board (7x7) renders
/// at ~270px -- reviewer feedback: the first cut (14.0) was too
/// small to watch.
const CELL: f64 = 36.0;
/// Outer margin in SVG units.
const MARGIN: f64 = 12.0;

/// Render the frames tensor as an SMIL-animated life grid.
pub fn render_life(frames: &DenseArray) -> Result<String, VizError> {
    let (t, h, w) = frame_dims(frames)?;
    let (cw, ch) = (
        w as f64 * CELL + 2.0 * MARGIN,
        h as f64 * CELL + 2.0 * MARGIN,
    );
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {cw} {ch}\" \
         width=\"{cw}\" height=\"{ch}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"#1e1e2e\" rx=\"4\"/>"
    );
    for i in 0..t {
        write_frame(&mut out, frames.data(), (i, t), (h, w));
    }
    out.push_str("</svg>");
    Ok(out)
}

/// Accept `[T, H, W]` or `[H, W]` (as one frame); anything else
/// is a shape error naming what was expected.
fn frame_dims(frames: &DenseArray) -> Result<(usize, usize, usize), VizError> {
    match frames.shape().dims() {
        [t, h, w] => Ok((*t, *h, *w)),
        [h, w] => Ok((1, *h, *w)),
        other => Err(VizError::InvalidShape(format!(
            "life expects [T, H, W] frames or one [H, W] board, got rank {}",
            other.len()
        ))),
    }
}

/// One frame group: alive cells as rects, plus (when animated)
/// the discrete opacity schedule that makes exactly this frame
/// visible during its 1/T slot of the loop.
fn write_frame(out: &mut String, data: &[f64], (i, t): (usize, usize), (h, w): (usize, usize)) {
    let opacity = if i == 0 { 1 } else { 0 };
    out.push_str(&format!("<g class=\"life-frame\" opacity=\"{opacity}\">"));
    for r in 0..h {
        for c in 0..w {
            if data[(i * h + r) * w + c] > 0.5 {
                let x = MARGIN + c as f64 * CELL + 2.0;
                let y = MARGIN + r as f64 * CELL + 2.0;
                let s = CELL - 4.0;
                out.push_str(&format!(
                    "<rect x=\"{x}\" y=\"{y}\" width=\"{s}\" height=\"{s}\" \
                     fill=\"#a6e3a1\" rx=\"2\"/>"
                ));
            }
        }
    }
    if t > 1 {
        out.push_str(&animate_tag(i, t));
    }
    out.push_str("</g>");
}

/// Discrete keyTimes/values pair: opacity 1 exactly during
/// `[i/T, (i+1)/T)` of the `T * FRAME_SECS` loop, 0 elsewhere.
fn animate_tag(i: usize, t: usize) -> String {
    let dur = t as f64 * FRAME_SECS;
    let t0 = i as f64 / t as f64;
    let t1 = (i + 1) as f64 / t as f64;
    let (key_times, values) = if i == 0 {
        (format!("0;{t1:.4};1"), "1;0;0".to_string())
    } else if i + 1 == t {
        (format!("0;{t0:.4};1"), "0;1;1".to_string())
    } else {
        (format!("0;{t0:.4};{t1:.4};1"), "0;1;0;0".to_string())
    };
    format!(
        "<animate attributeName=\"opacity\" dur=\"{dur}s\" \
         repeatCount=\"indefinite\" calcMode=\"discrete\" \
         keyTimes=\"{key_times}\" values=\"{values}\"/>"
    )
}
