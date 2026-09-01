//! Positioned graph -> SVG string. Dynamic canvas (the layout decides
//! the size, and the `<svg>` carries explicit `width`/`height` so it
//! renders inline, not collapsed), theme-matched to the playground
//! palette. Group bands (under everything), directed elbow edges with a
//! shared arrowhead marker + optional width/highlight, and node boxes
//! with optional highlight.

use std::fmt::Write;

use crate::groups::group_bands;
use crate::model::{BACK_LANE, LABEL_CHAR_W, LABEL_PAD, NODE_H, Positioned};
use crate::widths::{W_DEFAULT, stroke_widths};

const BG: &str = "#1e1e2e";
const SURFACE: &str = "#313244";
const EDGE: &str = "#89b4fa";
const TEXT: &str = "#cdd6f4";
const SUB: &str = "#a6adc8";
const HL: &str = "#f9e2af";

/// Emit the full SVG: header + background + arrowhead marker, group
/// bands, then edges (under nodes), then the node boxes.
pub fn render(p: &Positioned) -> String {
    let mut s = String::new();
    open(&mut s, p);
    group_bands(&mut s, p);
    let sw = stroke_widths(&p.graph);
    for (i, &(u, v)) in p.graph.edges.iter().enumerate() {
        let width = sw.get(i).copied().unwrap_or(W_DEFAULT);
        let hl = *p.graph.edge_highlight.get(i).unwrap_or(&false);
        edge(
            &mut s,
            p,
            (u, v),
            p.graph.edge_labels.get(i).map(String::as_str),
            width,
            hl,
            *p.back.get(i).unwrap_or(&false),
        );
    }
    for (id, label) in p.graph.labels.iter().enumerate() {
        node_box(
            &mut s,
            p.pos[id],
            p.node_w[id],
            label,
            *p.graph.node_highlight.get(id).unwrap_or(&false),
        );
    }
    s.push_str("</svg>");
    s
}

/// SVG header (with explicit width/height so it renders inline),
/// background rect, and the arrowhead marker def.
fn open(s: &mut String, p: &Positioned) {
    let _ = write!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" \
         height=\"{h}\" font-family=\"ui-monospace, monospace\" font-size=\"13\">\
         <rect width=\"{w}\" height=\"{h}\" fill=\"{BG}\"/>\
         <defs><marker id=\"aw\" markerUnits=\"userSpaceOnUse\" markerWidth=\"12\" \
         markerHeight=\"12\" refX=\"10\" refY=\"5\" orient=\"auto\">\
         <path d=\"M0,0 L10,5 L0,10 Z\" fill=\"{EDGE}\"/></marker>\
         <marker id=\"awh\" markerUnits=\"userSpaceOnUse\" markerWidth=\"12\" \
         markerHeight=\"12\" refX=\"10\" refY=\"5\" orient=\"auto\">\
         <path d=\"M0,0 L10,5 L0,10 Z\" fill=\"{HL}\"/></marker></defs>",
        w = p.width,
        h = p.height
    );
}

/// One node box with a centered label; highlighted nodes get the accent
/// stroke + a bolder fill.
fn node_box(s: &mut String, (x, y): (i32, i32), w: i32, label: &str, hl: bool) {
    let (cx, cy) = (x + w / 2, y + NODE_H / 2 + 4);
    let (stroke, sw) = if hl { (HL, 2) } else { (EDGE, 1) };
    let _ = write!(
        s,
        "<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{NODE_H}\" rx=\"6\" \
         fill=\"{SURFACE}\" stroke=\"{stroke}\" stroke-width=\"{sw}\"/>\
         <text x=\"{cx}\" y=\"{cy}\" text-anchor=\"middle\" fill=\"{TEXT}\">{}</text>",
        escape(label)
    );
}

/// A directed edge: a forward elbow (source's right side to target's
/// left), or a dashed "rewind" that loops through the bottom lane for a
/// back-edge. Carries the arrowhead, stroke width, optional highlight
/// color, and an optional label at the path's midpoint.
fn edge(
    s: &mut String,
    p: &Positioned,
    (u, v): (usize, usize),
    label: Option<&str>,
    w: f64,
    hl: bool,
    back: bool,
) {
    let (stroke, marker) = if hl { (HL, "awh") } else { (EDGE, "aw") };
    let dash = if back {
        " stroke-dasharray=\"6 4\""
    } else {
        ""
    };
    let (points, (lx, ly)) = edge_path(p, (u, v), back);
    let _ = write!(
        s,
        "<polyline points=\"{points}\" fill=\"none\" stroke=\"{stroke}\" \
         stroke-width=\"{w}\" marker-end=\"url(#{marker})\"{dash}/>"
    );
    if let Some(l) = label.filter(|l| !l.is_empty()) {
        edge_label(s, l, lx, ly);
    }
}

/// A midpoint edge label sitting above the line, on a small rounded
/// background plate so a thick or crossing edge can never obscure it.
fn edge_label(s: &mut String, label: &str, x: i32, y: i32) {
    let w = label.chars().count() as i32 * LABEL_CHAR_W + LABEL_PAD;
    let _ = write!(
        s,
        "<rect x=\"{rx}\" y=\"{ry}\" width=\"{w}\" height=\"15\" rx=\"3\" \
         fill=\"{BG}\" fill-opacity=\"0.85\"/>\
         <text x=\"{x}\" y=\"{y}\" text-anchor=\"middle\" fill=\"{SUB}\" \
         font-size=\"11\">{}</text>",
        escape(label),
        rx = x - w / 2,
        ry = y - 12,
    );
}

/// Polyline points + label anchor for an edge. Forward edges elbow
/// across the column gap; back-edges drop out of the source's bottom,
/// run left through the reserved lane, and rise into the target's
/// bottom so the arrow reads as a rewind.
fn edge_path(p: &Positioned, (u, v): (usize, usize), back: bool) -> (String, (i32, i32)) {
    if back {
        let (sx, tx) = (p.pos[u].0 + p.node_w[u] / 2, p.pos[v].0 + p.node_w[v] / 2);
        let (sy, ty) = (p.pos[u].1 + NODE_H, p.pos[v].1 + NODE_H);
        let lane = p.height - BACK_LANE / 2;
        let pts = format!("{sx},{sy} {sx},{lane} {tx},{lane} {tx},{ty}");
        (pts, ((sx + tx) / 2, lane - 10))
    } else {
        let (sx, sy) = (p.pos[u].0 + p.node_w[u], p.pos[u].1 + NODE_H / 2);
        let (tx, ty) = (p.pos[v].0, p.pos[v].1 + NODE_H / 2);
        let mx = (sx + tx) / 2;
        let pts = format!("{sx},{sy} {mx},{sy} {mx},{ty} {tx},{ty}");
        (pts, (mx, (sy + ty) / 2 - 13))
    }
}

/// Minimal XML text escaping for labels.
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
