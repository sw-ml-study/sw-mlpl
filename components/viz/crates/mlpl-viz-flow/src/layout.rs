//! Layered (Sugiyama-lite) layout: assign each node a layer by
//! longest-path from the sources, then place layers left-to-right with
//! nodes stacked in input order within a layer. Phase 1 uses input
//! order (barycenter crossing reduction is a later phase).

use crate::model::{COL_GAP, Graph, NODE_H, NODE_W, PAD, Positioned, ROW_GAP};

/// Lay a validated graph out into pixel positions + canvas size.
pub fn layout(graph: Graph) -> Positioned {
    let ranks = rank(&graph);
    place(graph, &ranks)
}

/// Longest-path layer per node: 0 for a source, else `1 + max` over
/// predecessors. At most `n` relaxation passes, which also bounds any
/// accidental cycle.
fn rank(graph: &Graph) -> Vec<usize> {
    let mut r = vec![0usize; graph.labels.len()];
    for _ in 0..graph.labels.len() {
        let mut changed = false;
        for &(u, v) in &graph.edges {
            if r[v] < r[u] + 1 {
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

/// Column x by layer, row y by in-column order; derive the canvas size.
fn place(graph: Graph, ranks: &[usize]) -> Positioned {
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
    let cols = max_rank as i32 + 1;
    let rows = next_row.iter().copied().max().unwrap_or(1);
    let width = PAD * 2 + cols * NODE_W + (cols - 1).max(0) * COL_GAP;
    let height = PAD * 2 + rows * NODE_H + (rows - 1).max(0) * ROW_GAP;
    Positioned {
        graph,
        pos,
        width,
        height,
    }
}
