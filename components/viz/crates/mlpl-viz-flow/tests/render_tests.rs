//! `render_dataflow` end to end: layout + SVG structure for a chain, a
//! residual skip, the Phase-2 channels (groups / widths / highlight),
//! and the error shapes (docs/dataflow-renderer-design.md).

use mlpl_viz_flow::{Dataflow, render_dataflow};

fn lbls(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn renders_a_chain_with_boxes_edges_labels_and_a_sized_svg() {
    let labels = lbls(&["storage", "FIFO", "lanes"]);
    let edge_labels = lbls(&["stream", "issue"]);
    let svg = render_dataflow(&Dataflow {
        labels: &labels,
        from: &[0, 1],
        to: &[1, 2],
        edge_labels: &edge_labels,
        ..Default::default()
    })
    .unwrap();
    assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
    // The bug fix: an explicit width/height so the inline SVG is sized.
    assert!(svg.contains("width=\"") && svg.contains("height=\""));
    assert_eq!(svg.matches("<rect").count(), 1 + 3); // background + 3 nodes
    assert_eq!(svg.matches("<polyline").count(), 2); // 2 edges
    assert!(svg.contains("storage") && svg.contains("stream"));
    assert!(svg.contains("marker-end=\"url(#aw)\""));
}

#[test]
fn a_residual_skip_edge_spans_two_layers() {
    let labels = lbls(&["a", "b", "c"]);
    let svg = render_dataflow(&Dataflow {
        labels: &labels,
        from: &[0, 1, 0],
        to: &[1, 2, 2],
        ..Default::default()
    })
    .unwrap();
    assert_eq!(svg.matches("<polyline").count(), 3);
}

#[test]
fn groups_widths_and_highlight_render() {
    let labels = lbls(&["a", "b", "c"]);
    let svg = render_dataflow(&Dataflow {
        labels: &labels,
        from: &[0, 1],
        to: &[1, 2],
        groups: &[0, 1, 1],
        node_highlight: &[false, false, true],
        edge_widths: &[3.0, 1.0],
        edge_highlight: &[true, false],
        ..Default::default()
    })
    .unwrap();
    // Two distinct group ids -> two band rects (plus bg + 3 nodes = 5).
    assert_eq!(svg.matches("<rect").count(), 2 + 1 + 3);
    assert!(svg.contains("fill-opacity=\"0.10\"")); // a band
    assert!(svg.contains("stroke-width=\"3\"")); // the wide edge
    assert!(svg.contains("url(#awh)")); // the highlighted edge marker
}

#[test]
fn errors_are_clean_not_panics() {
    let two = lbls(&["a", "b"]);
    assert!(render_dataflow(&Dataflow::default()).is_err()); // no nodes
    assert!(
        render_dataflow(&Dataflow {
            labels: &two,
            from: &[0],
            to: &[0, 1],
            ..Default::default()
        })
        .is_err()
    );
    let one = lbls(&["a"]);
    assert!(
        render_dataflow(&Dataflow {
            labels: &one,
            from: &[0],
            to: &[5],
            ..Default::default()
        })
        .is_err()
    );
    // A channel length mismatch is a clean error too.
    assert!(
        render_dataflow(&Dataflow {
            labels: &two,
            groups: &[0],
            ..Default::default()
        })
        .is_err()
    );
}
