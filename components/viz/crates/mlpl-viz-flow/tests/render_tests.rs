//! `render_dataflow` end to end: layout + SVG structure for a chain, a
//! residual skip, and the error shapes (docs/dataflow-renderer-design.md).

use mlpl_viz_flow::render_dataflow;

fn lbls(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn renders_a_chain_with_boxes_edges_and_labels() {
    let svg = render_dataflow(
        &lbls(&["storage", "FIFO", "lanes"]),
        &[0, 1],
        &[1, 2],
        &lbls(&["stream", "issue"]),
    )
    .unwrap();
    assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
    assert_eq!(svg.matches("<rect").count(), 1 + 3); // background + 3 nodes
    assert_eq!(svg.matches("<polyline").count(), 2); // 2 edges
    assert!(svg.contains("storage") && svg.contains("stream"));
    assert!(svg.contains("marker-end=\"url(#aw)\""));
}

#[test]
fn a_residual_skip_edge_spans_two_layers() {
    // 0->1->2 plus a skip 0->2: node 2 lands in layer 2 (longest path),
    // so the skip edge is a long forward arrow.
    let svg = render_dataflow(&lbls(&["a", "b", "c"]), &[0, 1, 0], &[1, 2, 2], &[]).unwrap();
    assert_eq!(svg.matches("<polyline").count(), 3);
}

#[test]
fn errors_are_clean_not_panics() {
    assert!(render_dataflow(&[], &[], &[], &[]).is_err()); // no nodes
    assert!(render_dataflow(&lbls(&["a", "b"]), &[0], &[0, 1], &[]).is_err()); // from/to mismatch
    assert!(render_dataflow(&lbls(&["a"]), &[0], &[5], &[]).is_err()); // endpoint out of range
    assert!(render_dataflow(&lbls(&["a", "b"]), &[0], &[1], &lbls(&["x", "y"])).is_err()); // labels
}
