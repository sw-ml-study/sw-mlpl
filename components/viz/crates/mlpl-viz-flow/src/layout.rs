//! Layered (Sugiyama-lite) layout: assign each node a layer by
//! longest-path over the forward edges (back-edges are detected first
//! and excluded so a recurrence never distorts layering), then place
//! layers left-to-right with nodes stacked in input order within a
//! layer. Phase 1 uses input order (barycenter crossing reduction is a
//! later phase).

use crate::model::{BACK_LANE, COL_GAP, Graph, NODE_H, NODE_W, PAD, Positioned, ROW_GAP};

/// Lay a validated graph out into pixel positions + canvas size.
pub fn layout(graph: Graph) -> Positioned {
    let back = back_edges(&graph);
    let ranks = rank(&graph, &back);
    place(graph, &ranks, back)
}

/// Longest-path layer per node: 0 for a source, else `1 + max` over
/// forward predecessors. At most `n` relaxation passes; back-edges are
/// skipped so the pass sees an acyclic graph.
fn rank(graph: &Graph, back: &[bool]) -> Vec<usize> {
    let mut r = vec![0usize; graph.labels.len()];
    for _ in 0..graph.labels.len() {
        let mut changed = false;
        for (i, &(u, v)) in graph.edges.iter().enumerate() {
            if !back[i] && r[v] < r[u] + 1 {
                r[v] = r[u] + 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    r
}

/// Column x by layer, row y by in-column order; derive the canvas size,
/// reserving a bottom lane when any back-edge needs a rewind route.
fn place(graph: Graph, ranks: &[usize], back: Vec<bool>) -> Positioned {
    let max_rank = ranks.iter().copied().max().unwrap_or(0);
    let mut next_row = vec![0i32; max_rank + 1];
    let mut pos = vec![(0i32, 0i32); graph.labels.len()];
    for (id, slot) in pos.iter_mut().enumerate() {
        let col = ranks[id];
        let row = next_row[col];
        next_row[col] += 1;
        *slot = (
            PAD + col as i32 * (NODE_W + COL_GAP),
            PAD + row * (NODE_H + ROW_GAP),
        );
    }
    let (width, height) = canvas(max_rank, &next_row, back.iter().any(|&b| b));
    Positioned {
        graph,
        pos,
        back,
        width,
        height,
    }
}

/// Canvas `(width, height)` for the placed columns/rows, plus the
/// back-edge lane when one is present.
fn canvas(max_rank: usize, next_row: &[i32], has_back: bool) -> (i32, i32) {
    let cols = max_rank as i32 + 1;
    let rows = next_row.iter().copied().max().unwrap_or(1);
    let width = PAD * 2 + cols * NODE_W + (cols - 1).max(0) * COL_GAP;
    let height = PAD * 2
        + rows * NODE_H
        + (rows - 1).max(0) * ROW_GAP
        + if has_back { BACK_LANE } else { 0 };
    (width, height)
}

/// Per-edge flag: true where the edge points back to a DFS ancestor (a
/// recurrence). Ranking ignores these so layering stays a DAG, and the
/// renderer draws them dashed. Teaching graphs are small, so a plain
/// recursive DFS is fine.
fn back_edges(graph: &Graph) -> Vec<bool> {
    let n = graph.labels.len();
    let adj = adjacency(graph, n);
    let mut color = vec![0u8; n]; // 0 = white, 1 = gray (on stack), 2 = black
    let mut back = vec![false; graph.edges.len()];
    for s in 0..n {
        if color[s] == 0 {
            visit(s, &adj, &mut color, &mut back);
        }
    }
    back
}

/// Outgoing `(edge_index, target)` per node.
fn adjacency(graph: &Graph, n: usize) -> Vec<Vec<(usize, usize)>> {
    let mut adj = vec![Vec::new(); n];
    for (i, &(u, v)) in graph.edges.iter().enumerate() {
        adj[u].push((i, v));
    }
    adj
}

/// DFS from `u`, marking every edge into a gray (on-stack) node as back.
fn visit(u: usize, adj: &[Vec<(usize, usize)>], color: &mut [u8], back: &mut [bool]) {
    color[u] = 1;
    for &(e, v) in &adj[u] {
        match color[v] {
            1 => back[e] = true,
            0 => visit(v, adj, color, back),
            _ => {}
        }
    }
    color[u] = 2;
}
