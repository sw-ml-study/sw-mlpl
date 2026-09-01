//! Positioned graph -> SVG string. Dynamic canvas (the layout decides
//! the size), theme-matched to the playground palette. Boxes, directed
//! elbow edges with a shared arrowhead marker, and midpoint edge
//! labels. Groups / widths / highlight are later phases.

use std::fmt::Write;

use crate::model::{NODE_H, NODE_W, Positioned};

const BG: &str = "#1e1e2e";
const SURFACE: &str = "#313244";
const EDGE: &str = "#89b4fa";
const TEXT: &str = "#cdd6f4";
const SUB: &str = "#a6adc8";

/// Emit the full SVG: header + background + arrowhead marker, then the
/// edges (drawn under the nodes), then the node boxes.
pub fn render(p: &Positioned) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" \
         font-family=\"ui-monospace, monospace\" font-size=\"13\">\
         <rect width=\"{w}\" height=\"{h}\" fill=\"{BG}\"/>\
         <defs><marker id=\"aw\" markerWidth=\"9\" markerHeight=\"9\" refX=\"8\" refY=\"3\" \
         orient=\"auto\"><path d=\"M0,0 L8,3 L0,6 Z\" fill=\"{EDGE}\"/></marker></defs>",
        w = p.width,
        h = p.height
    );
    for (i, &(u, v)) in p.graph.edges.iter().enumerate() {
        edge(
            &mut s,
            p,
            u,
            v,
            p.graph.edge_labels.get(i).map(String::as_str),
        );
    }
    for (id, label) in p.graph.labels.iter().enumerate() {
        node_box(&mut s, p.pos[id], label);
    }
    s.push_str("</svg>");
    s
}

/// One node box with a centered label.
fn node_box(s: &mut String, (x, y): (i32, i32), label: &str) {
    let (cx, cy) = (x + NODE_W / 2, y + NODE_H / 2 + 4);
    let _ = write!(
        s,
        "<rect x=\"{x}\" y=\"{y}\" width=\"{NODE_W}\" height=\"{NODE_H}\" rx=\"6\" \
         fill=\"{SURFACE}\" stroke=\"{EDGE}\" stroke-width=\"1\"/>\
         <text x=\"{cx}\" y=\"{cy}\" text-anchor=\"middle\" fill=\"{TEXT}\">{}</text>",
        escape(label)
    );
}

/// A directed elbow edge from the source's right side to the target's
/// left side, with an arrowhead and an optional midpoint label.
fn edge(s: &mut String, p: &Positioned, u: usize, v: usize, label: Option<&str>) {
    let (sx, sy) = (p.pos[u].0 + NODE_W, p.pos[u].1 + NODE_H / 2);
    let (tx, ty) = (p.pos[v].0, p.pos[v].1 + NODE_H / 2);
    let mx = (sx + tx) / 2;
    let _ = write!(
        s,
        "<polyline points=\"{sx},{sy} {mx},{sy} {mx},{ty} {tx},{ty}\" fill=\"none\" \
         stroke=\"{EDGE}\" stroke-width=\"1.5\" marker-end=\"url(#aw)\"/>"
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
