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

/// Lay a validated graph out into pixel positions + canvas size. Detects
/// back-edges, ranks nodes by longest-path over the forward edges only
/// (back-edges skipped so the relaxation sees a DAG), then places the
/// columns. At most `n` relaxation passes.
pub fn layout(graph: Graph) -> Positioned {
    let back = back_edges(&graph);
    let mut ranks = vec![0usize; graph.labels.len()];
    for _ in 0..graph.labels.len() {
        let mut changed = false;
        for (i, &(u, v)) in graph.edges.iter().enumerate() {
            if !back[i] && ranks[v] < ranks[u] + 1 {
                ranks[v] = ranks[u] + 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    place(graph, &ranks, back)
}

/// Place nodes into variable-width columns (each column as wide as its
/// widest label-fitted box), centered within the column, and derive the
/// canvas size -- reserving a bottom lane when a back-edge needs a
/// rewind route.
fn place(graph: Graph, ranks: &[usize], back: Vec<bool>) -> Positioned {
    let max_rank = ranks.iter().copied().max().unwrap_or(0);
    let or_default = |v: i32, d: i32| if v > 0 { v } else { d };
    let (col_gap, row_gap) = (
        or_default(graph.col_gap, COL_GAP),
        or_default(graph.row_gap, ROW_GAP),
    );
    // Fit each box to its label (monospace estimate), never below NODE_W.
    let node_w: Vec<i32> = (graph.labels.iter())
        .map(|l| (l.chars().count() as i32 * CHAR_W + 2 * BOX_PAD).max(NODE_W))
        .collect();
    let mut col_w = vec![NODE_W; max_rank + 1];
    for (id, &c) in ranks.iter().enumerate() {
        col_w[c] = col_w[c].max(node_w[id]);
    }
    let gaps = column_gaps(&graph, ranks, col_gap);
    let mut col_x = vec![PAD; max_rank + 1];
    for c in 1..=max_rank {
        col_x[c] = col_x[c - 1] + col_w[c - 1] + gaps[c - 1];
    }
    let mut next_row = vec![0i32; max_rank + 1];
    let mut pos = vec![(0i32, 0i32); graph.labels.len()];
    for (id, slot) in pos.iter_mut().enumerate() {
        let c = ranks[id];
        let x = col_x[c] + (col_w[c] - node_w[id]) / 2;
        *slot = (x, PAD + next_row[c] * (NODE_H + row_gap));
        next_row[c] += 1;
    }
    let (width, height) = canvas(&col_w, &gaps, &next_row, row_gap, back.iter().any(|&b| b));
    let geo = Geo {
        ranks,
        col_x: &col_x,
        col_w: &col_w,
        gaps: &gaps,
        pos: &pos,
        node_w: &node_w,
        back: &back,
        lane: height - BACK_LANE / 2,
    };
    let label_at = label_anchors(&graph.edges, &geo);
    Positioned {
        graph,
        pos,
        node_w,
        label_at,
        back,
        width,
        height,
    }
}

/// The placed geometry a label anchor is derived from.
struct Geo<'a> {
    ranks: &'a [usize],
    col_x: &'a [i32],
    col_w: &'a [i32],
    gaps: &'a [i32],
    pos: &'a [(i32, i32)],
    node_w: &'a [i32],
    back: &'a [bool],
    lane: i32,
}

/// Per-edge label anchor. A forward-edge label sits at the center of the
/// first gap after its source -- always clear of boxes, so a skip-edge
/// label never lands on a box it routes past. Back-edge labels sit above
/// the reserved lane.
fn label_anchors(edges: &[(usize, usize)], g: &Geo) -> Vec<(i32, i32)> {
    (edges.iter().enumerate())
        .map(|(i, &(u, v))| {
            if g.back[i] {
                let mid = (g.pos[u].0 + g.node_w[u] / 2 + g.pos[v].0 + g.node_w[v] / 2) / 2;
                (mid, g.lane - 10)
            } else {
                let r = g.ranks[u];
                (
                    g.col_x[r] + g.col_w[r] + g.gaps[r] / 2,
                    g.pos[u].1 + NODE_H / 2 - 13,
                )
            }
        })
        .collect()
}

/// Width of each inter-column gap: at least `col_gap`, widened to fit
/// the widest label plate of any forward edge whose source is in the
/// column to its left -- so a long or skip-edge label lands in the gap,
/// not on a node box. Gap `c` sits between columns `c` and `c + 1`.
fn column_gaps(graph: &Graph, ranks: &[usize], col_gap: i32) -> Vec<i32> {
    let max_rank = ranks.iter().copied().max().unwrap_or(0);
    let mut gaps = vec![col_gap; max_rank];
    for (i, &(u, v)) in graph.edges.iter().enumerate() {
        let label = graph.edge_labels.get(i).filter(|l| !l.is_empty());
        if let (Some(l), true) = (label, ranks[v] > ranks[u]) {
            let plate = l.chars().count() as i32 * LABEL_CHAR_W + LABEL_PAD;
            gaps[ranks[u]] = gaps[ranks[u]].max(plate + 16);
        }
    }
    gaps
}

/// Canvas `(width, height)` for the placed columns/rows and gaps, plus
/// the back-edge lane when one is present.
fn canvas(
    col_w: &[i32],
    gaps: &[i32],
    next_row: &[i32],
    row_gap: i32,
    has_back: bool,
) -> (i32, i32) {
    let rows = next_row.iter().copied().max().unwrap_or(1);
    let width = PAD * 2 + col_w.iter().sum::<i32>() + gaps.iter().sum::<i32>();
    let height = PAD * 2
        + rows * NODE_H
        + (rows - 1).max(0) * row_gap
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
