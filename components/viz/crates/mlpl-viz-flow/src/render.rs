//! Positioned graph -> SVG string. Dynamic canvas (the layout decides
//! the size, and the `<svg>` carries explicit `width`/`height` so it
//! renders inline, not collapsed), theme-matched to the playground
//! palette. Group bands (under everything), directed elbow edges with a
//! shared arrowhead marker + optional width/highlight, and node boxes
//! with optional highlight.

use std::fmt::Write;

use crate::groups::group_bands;
use crate::model::{NODE_H, NODE_W, Positioned};
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
        );
    }
    for (id, label) in p.graph.labels.iter().enumerate() {
        node_box(
            &mut s,
            p.pos[id],
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
         <defs><marker id=\"aw\" markerWidth=\"9\" markerHeight=\"9\" refX=\"8\" refY=\"3\" \
         orient=\"auto\"><path d=\"M0,0 L8,3 L0,6 Z\" fill=\"{EDGE}\"/></marker>\
         <marker id=\"awh\" markerWidth=\"9\" markerHeight=\"9\" refX=\"8\" refY=\"3\" \
         orient=\"auto\"><path d=\"M0,0 L8,3 L0,6 Z\" fill=\"{HL}\"/></marker></defs>",
        w = p.width,
        h = p.height
    );
}

/// One node box with a centered label; highlighted nodes get the accent
/// stroke + a bolder fill.
fn node_box(s: &mut String, (x, y): (i32, i32), label: &str, hl: bool) {
    let (cx, cy) = (x + NODE_W / 2, y + NODE_H / 2 + 4);
    let (stroke, sw) = if hl { (HL, 2) } else { (EDGE, 1) };
    let _ = write!(
        s,
        "<rect x=\"{x}\" y=\"{y}\" width=\"{NODE_W}\" height=\"{NODE_H}\" rx=\"6\" \
         fill=\"{SURFACE}\" stroke=\"{stroke}\" stroke-width=\"{sw}\"/>\
         <text x=\"{cx}\" y=\"{cy}\" text-anchor=\"middle\" fill=\"{TEXT}\">{}</text>",
        escape(label)
    );
}

/// A directed elbow edge from the source's right side to the target's
/// left side, with an arrowhead, an optional midpoint label, a stroke
/// width, and an optional highlight color.
fn edge(
    s: &mut String,
    p: &Positioned,
    (u, v): (usize, usize),
    label: Option<&str>,
    w: f64,
    hl: bool,
) {
    let (sx, sy) = (p.pos[u].0 + NODE_W, p.pos[u].1 + NODE_H / 2);
    let (tx, ty) = (p.pos[v].0, p.pos[v].1 + NODE_H / 2);
    let mx = (sx + tx) / 2;
    let (stroke, marker) = if hl { (HL, "awh") } else { (EDGE, "aw") };
    let _ = write!(
        s,
        "<polyline points=\"{sx},{sy} {mx},{sy} {mx},{ty} {tx},{ty}\" fill=\"none\" \
         stroke=\"{stroke}\" stroke-width=\"{w}\" marker-end=\"url(#{marker})\"/>"
    );
    if let Some(l) = label.filter(|l| !l.is_empty()) {
        let ly = (sy + ty) / 2 - 4;
        let _ = write!(
            s,
            "<text x=\"{mx}\" y=\"{ly}\" text-anchor=\"middle\" fill=\"{SUB}\" \
             font-size=\"11\">{}</text>",
            escape(l)
        );
    }
}

/// Minimal XML text escaping for labels.
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
