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
    // background + 3 nodes + 2 edge-label backing plates.
    assert_eq!(svg.matches("<rect").count(), 1 + 3 + 2);
    assert_eq!(svg.matches("<polyline").count(), 2); // 2 edges
    assert!(svg.contains("storage") && svg.contains("stream"));
    assert!(svg.contains("marker-end=\"url(#aw)\""));
    // Each edge label sits on a backing plate so a thick/crossing edge
    // can't obscure it.
    assert_eq!(svg.matches("fill-opacity=\"0.85\"").count(), 2);
    // Arrowheads are a fixed size, not scaled by stroke width, so a wide
    // edge never grows a monstrous triangle.
    assert!(svg.contains("markerUnits=\"userSpaceOnUse\""));
}

#[test]
fn a_long_label_widens_its_node_box() {
    let labels = lbls(&["x", "sequential store: 269 MB"]);
    let svg = render_dataflow(&Dataflow {
        labels: &labels,
        from: &[0],
        to: &[1],
        ..Default::default()
    })
    .unwrap();
    // The short label keeps the minimum box width; the 24-char label
    // widens its box to 24*8 + 2*14 = 220.
    assert!(svg.contains("width=\"128\""));
    assert!(svg.contains("width=\"220\""));
}

#[test]
fn a_long_edge_label_widens_its_column_gap() {
    let labels = lbls(&["a", "b"]);
    // 27-char label -> plate 27*7 + 8 = 197, gap = 197 + 16 = 213 (> the
    // 96 default). Canvas width = 24*2 + 128 + 128 + 213 = 517.
    let edge_labels = lbls(&["a very long edge label here"]);
    let svg = render_dataflow(&Dataflow {
        labels: &labels,
        from: &[0],
        to: &[1],
        edge_labels: &edge_labels,
        ..Default::default()
    })
    .unwrap();
    assert!(svg.contains("width=\"517\"")); // canvas widened for the label
    // A short label would have left the canvas at 24*2+128+128+96 = 400.
    assert!(!svg.contains("width=\"400\""));
}

#[test]
fn col_gap_override_widens_the_canvas() {
    let labels = lbls(&["a", "b"]);
    let svg = render_dataflow(&Dataflow {
        labels: &labels,
        from: &[0],
        to: &[1],
        col_gap: 200,
        ..Default::default()
    })
    .unwrap();
    // Canvas = 24*2 + 128 + 128 + 200 = 504 (default would be 400).
    assert!(svg.contains("width=\"504\""));
}

#[test]
fn a_skip_edge_label_lands_in_the_gap_not_on_the_passed_box() {
    let labels = lbls(&["a", "b", "c"]);
    // a->b, b->c, and a->c (skips column 1, where box b sits).
    let edge_labels = lbls(&["", "", "skip"]);
    let svg = render_dataflow(&Dataflow {
        labels: &labels,
        from: &[0, 1, 0],
        to: &[1, 2, 2],
        edge_labels: &edge_labels,
        ..Default::default()
    })
    .unwrap();
    // The label sits at the center of the first gap (x = 24 + 128 + 96/2
    // = 200), NOT at the a->c geometric midpoint (312), which is inside
    // box b (248..376).
    assert!(svg.contains(">skip</text>"));
    assert!(svg.contains("x=\"200\" y=\"31\"")); // the skip label anchor
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
fn log_width_scale_tames_extreme_ratios() {
    let labels = lbls(&["dram", "fifo", "sram"]);
    // A 65,000:1 raw ratio (269 MB vs 4 KiB). Linear clamps BOTH to the
    // band max -- indistinguishable; log spreads them across the band.
    let mk = |log| {
        render_dataflow(&Dataflow {
            labels: &labels,
            from: &[0, 1],
            to: &[1, 2],
            edge_widths: &[269_000_000.0, 4096.0],
            width_log: log,
            ..Default::default()
        })
        .unwrap()
    };
    let log = mk(true);
    assert!(log.contains("stroke-width=\"9\"")); // the big edge -> band max
    assert!(log.contains("stroke-width=\"1\"")); // the small edge -> band min
    let lin = mk(false);
    assert!(!lin.contains("269000000")); // never the raw quantity as a stroke
    assert_eq!(lin.matches("stroke-width=\"9\"").count(), 2); // both clamp to max
}

#[test]
fn a_back_edge_is_dashed_and_does_not_distort_layering() {
    let labels = lbls(&["a", "b", "c"]);
    // a -> b -> c forward, plus c -> a: the recurrence (a back-edge).
    let svg = render_dataflow(&Dataflow {
        labels: &labels,
        from: &[0, 1, 2],
        to: &[1, 2, 0],
        ..Default::default()
    })
    .unwrap();
    assert_eq!(svg.matches("<polyline").count(), 3); // all three edges drawn
    assert_eq!(svg.matches("stroke-dasharray").count(), 1); // only the back-edge
    // Layering ignores c -> a: a stays in column 0 (x = PAD = 24) and c
    // in column 2 (x = 24 + 2*(128+96) = 472). If the back-edge were
    // ranked, a would be pushed right and x="24" would not be a node.
    assert!(svg.contains("x=\"24\"")); // node a, column 0
    assert!(svg.contains("x=\"472\"")); // node c, column 2
}

#[test]
fn a_self_loop_is_a_back_edge() {
    let labels = lbls(&["s"]);
    let svg = render_dataflow(&Dataflow {
        labels: &labels,
        from: &[0],
        to: &[0],
        ..Default::default()
    })
    .unwrap();
    assert_eq!(svg.matches("stroke-dasharray").count(), 1);
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
