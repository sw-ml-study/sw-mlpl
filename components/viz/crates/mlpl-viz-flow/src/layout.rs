//! Layered (Sugiyama-lite) layout: assign each node a layer by
//! longest-path over the forward edges (back-edges are detected first
//! and excluded so a recurrence never distorts layering), then place
//! layers left-to-right with nodes stacked in input order within a
//! layer. Phase 1 uses input order (barycenter crossing reduction is a
//! later phase).

use crate::model::{
    BACK_LANE, BOX_PAD, CHAR_W, COL_GAP, Graph, LABEL_CHAR_W, LABEL_PAD, NODE_H, NODE_W, PAD,
    Positioned, ROW_GAP,
};

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

/// Place nodes into variable-width columns (each column as wide as its
/// widest label-fitted box), centered within the column, and derive the
/// canvas size -- reserving a bottom lane when a back-edge needs a
/// rewind route.
fn place(graph: Graph, ranks: &[usize], back: Vec<bool>) -> Positioned {
    let max_rank = ranks.iter().copied().max().unwrap_or(0);
    // Fit each box to its label (monospace estimate), never below NODE_W.
    let node_w: Vec<i32> = (graph.labels.iter())
        .map(|l| (l.chars().count() as i32 * CHAR_W + 2 * BOX_PAD).max(NODE_W))
        .collect();
    let mut col_w = vec![NODE_W; max_rank + 1];
    for (id, &c) in ranks.iter().enumerate() {
        col_w[c] = col_w[c].max(node_w[id]);
    }
    let gaps = column_gaps(&graph, ranks, max_rank);
    let mut col_x = vec![PAD; max_rank + 1];
    for c in 1..=max_rank {
        col_x[c] = col_x[c - 1] + col_w[c - 1] + gaps[c - 1];
    }
    let mut next_row = vec![0i32; max_rank + 1];
    let mut pos = vec![(0i32, 0i32); graph.labels.len()];
    for (id, slot) in pos.iter_mut().enumerate() {
        let c = ranks[id];
        let x = col_x[c] + (col_w[c] - node_w[id]) / 2;
        *slot = (x, PAD + next_row[c] * (NODE_H + ROW_GAP));
        next_row[c] += 1;
    }
    let (width, height) = canvas(&col_w, &gaps, &next_row, back.iter().any(|&b| b));
    Positioned {
        graph,
        pos,
        node_w,
        back,
        width,
        height,
    }
}

/// Width of each inter-column gap: at least `COL_GAP`, widened to fit
/// the widest adjacent-edge label plate that sits in it (so a long edge
/// label lands between the columns, not on top of the node boxes). Gap
/// `c` sits between columns `c` and `c + 1`; a skip edge's label is left
/// to its plate, not sized here.
fn column_gaps(graph: &Graph, ranks: &[usize], max_rank: usize) -> Vec<i32> {
    let mut gaps = vec![COL_GAP; max_rank];
    for (i, &(u, v)) in graph.edges.iter().enumerate() {
        let label = graph.edge_labels.get(i).filter(|l| !l.is_empty());
        if let (Some(l), true) = (label, ranks[v] == ranks[u] + 1) {
            let plate = l.chars().count() as i32 * LABEL_CHAR_W + LABEL_PAD;
            gaps[ranks[u]] = gaps[ranks[u]].max(plate + 16);
        }
    }
    gaps
}

/// Canvas `(width, height)` for the placed columns/rows and gaps, plus
/// the back-edge lane when one is present.
fn canvas(col_w: &[i32], gaps: &[i32], next_row: &[i32], has_back: bool) -> (i32, i32) {
    let rows = next_row.iter().copied().max().unwrap_or(1);
    let width = PAD * 2 + col_w.iter().sum::<i32>() + gaps.iter().sum::<i32>();
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
    let mut adj = vec![Vec::new(); n];
    for (i, &(u, v)) in graph.edges.iter().enumerate() {
        adj[u].push((i, v));
    }
    let mut color = vec![0u8; n]; // 0 = white, 1 = gray (on stack), 2 = black
    let mut back = vec![false; graph.edges.len()];
    for s in 0..n {
        if color[s] == 0 {
            visit(s, &adj, &mut color, &mut back);
        }
    }
    back
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
