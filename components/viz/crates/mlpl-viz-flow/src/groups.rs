//! Group bands: a faint rounded rectangle behind the bounding box of
//! each group's nodes, drawn under the edges and boxes so grouped
//! nodes read as one region (the memory-hierarchy / stage contrast).

use std::fmt::Write;

use crate::model::{NODE_H, NODE_W, Positioned};

/// Distinct band colors, cycled by group id.
const BAND: &[&str] = &[
    "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7", "#89dceb", "#f9e2af",
];
/// Padding around a group's node bounding box.
const BAND_PAD: i32 = 12;

/// Draw one band per distinct group id (no-op when `groups` is empty).
pub fn group_bands(s: &mut String, p: &Positioned) {
    if p.graph.groups.is_empty() {
        return;
    }
    let max = p.graph.groups.iter().copied().max().unwrap_or(0);
    for g in 0..=max {
        if let Some(bounds) = bbox(p, g) {
            band(s, bounds, BAND[g % BAND.len()]);
        }
    }
}

/// The `(x, y, w, h)` bounding box (padded) of the nodes in group `g`,
/// or `None` when the group has no members.
fn bbox(p: &Positioned, g: usize) -> Option<(i32, i32, i32, i32)> {
    let mut members = (0..p.graph.groups.len())
        .filter(|&i| p.graph.groups[i] == g)
        .map(|i| p.pos[i]);
    let (fx, fy) = members.next()?;
    let (mut x0, mut y0, mut x1, mut y1) = (fx, fy, fx + NODE_W, fy + NODE_H);
    for (x, y) in members {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x + NODE_W);
        y1 = y1.max(y + NODE_H);
    }
    Some((
        x0 - BAND_PAD,
        y0 - BAND_PAD,
        x1 - x0 + 2 * BAND_PAD,
        y1 - y0 + 2 * BAND_PAD,
    ))
}

/// One faint band rectangle.
fn band(s: &mut String, (x, y, w, h): (i32, i32, i32, i32), color: &str) {
    let _ = write!(
        s,
        "<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" rx=\"12\" fill=\"{color}\" \
         fill-opacity=\"0.10\" stroke=\"{color}\" stroke-opacity=\"0.35\"/>"
    );
}
