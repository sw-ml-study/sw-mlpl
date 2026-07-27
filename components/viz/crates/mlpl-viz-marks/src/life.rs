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
/// Largest cell edge in SVG units -- a small board (7x7) renders
/// at ~270px (reviewer feedback: 14.0 was too small to watch).
const MAX_CELL: f64 = 36.0;
/// Smallest cell edge, so huge boards stay legible dots.
const MIN_CELL: f64 = 8.0;
/// Target edge for the whole grid; cell size adapts so a 40x40
/// gun board is ~600px instead of 1.4k px.
const TARGET_EDGE: f64 = 600.0;
/// Outer margin in SVG units.
const MARGIN: f64 = 12.0;

/// Cell edge for an `h x w` board: fill `TARGET_EDGE`, clamped.
fn cell_px(h: usize, w: usize) -> f64 {
    (TARGET_EDGE / h.max(w).max(1) as f64).clamp(MIN_CELL, MAX_CELL)
}

/// Render the frames tensor as an SMIL-animated life grid.
pub fn render_life(frames: &DenseArray) -> Result<String, VizError> {
    let (t, h, w) = frame_dims(frames)?;
    let cell = cell_px(h, w);
    let (cw, ch) = (
        w as f64 * cell + 2.0 * MARGIN,
        h as f64 * cell + 2.0 * MARGIN,
    );
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {cw} {ch}\" \
         width=\"{cw}\" height=\"{ch}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"#1e1e2e\" rx=\"4\"/>"
    );
    for i in 0..t {
        write_frame(&mut out, frames.data(), (i, t), (h, w), cell);
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
fn write_frame(
    out: &mut String,
    data: &[f64],
    (i, t): (usize, usize),
    (h, w): (usize, usize),
    cell: f64,
) {
    let opacity = if i == 0 { 1 } else { 0 };
    out.push_str(&format!("<g class=\"life-frame\" opacity=\"{opacity}\">"));
    let inset = (cell * 0.06).max(0.5);
    let s = cell - 2.0 * inset;
    let alive = (0..h * w).filter(|k| data[i * h * w + k] > 0.5);
    for k in alive {
        let x = MARGIN + (k % w) as f64 * cell + inset;
        let y = MARGIN + (k / w) as f64 * cell + inset;
        out.push_str(&format!(
            "<rect x=\"{x}\" y=\"{y}\" width=\"{s}\" height=\"{s}\" \
             fill=\"#a6e3a1\" rx=\"2\"/>"
        ));
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
