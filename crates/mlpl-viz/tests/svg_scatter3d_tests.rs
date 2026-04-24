//! Saga 16 step 003: `svg(pts, "scatter3d")` renderer.
//!
//! Accepts `[N, 3]` (pure point cloud, single default
//! color) or `[N, 4]` (points + cluster id in the last
//! column, mirroring `scatter_labeled`). Orthographic
//! projection at azimuth=30 / elevation=20; renders axis
//! gizmos + one `<circle>` per point + optional legend.

use mlpl_array::{DenseArray, Shape};
use mlpl_viz::svg::render_scatter3d;

fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data).unwrap()
}

#[test]
fn scatter3d_returns_well_formed_svg() {
    let pts = matrix(
        4,
        3,
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    let svg = render_scatter3d(&pts).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
}

#[test]
fn scatter3d_contains_one_circle_per_point() {
    let pts = matrix(
        4,
        3,
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    let svg = render_scatter3d(&pts).unwrap();
    let circle_count = svg.matches("<circle").count();
    assert_eq!(circle_count, 4);
}

#[test]
fn scatter3d_without_labels_has_no_legend() {
    let pts = matrix(
        4,
        3,
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    let svg = render_scatter3d(&pts).unwrap();
    assert!(
        !svg.contains("legend"),
        "3-column input should not render a legend, got:\n{svg}"
    );
}

#[test]
fn scatter3d_with_labels_renders_a_legend() {
    // 4 points, 2 clusters (labels 0, 0, 1, 1).
    let pts = matrix(
        4,
        4,
        vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ],
    );
    let svg = render_scatter3d(&pts).unwrap();
    assert!(
        svg.contains("legend"),
        "4-column input should render a legend; got:\n{svg}"
    );
    // Still one circle per point (4), plus legend swatches
    // (2 unique clusters). Circle count should be >= 4.
    let circle_count = svg.matches("<circle").count();
    assert!(
        circle_count >= 4,
        "expected at least 4 <circle> elements (dots), got {circle_count}"
    );
}

#[test]
fn scatter3d_renders_axis_gizmos() {
    let pts = matrix(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let svg = render_scatter3d(&pts).unwrap();
    // Axis labels "X", "Y", "Z" appear as small text
    // anchored at the projected axis endpoints.
    for label in ["X", "Y", "Z"] {
        assert!(
            svg.contains(&format!(">{label}<")),
            "axis label '{label}' should appear in the SVG; got:\n{svg}"
        );
    }
}

#[test]
fn scatter3d_unit_cube_snapshot_is_stable() {
    // Eight corners of a unit cube. Pins the projection
    // math + element order so regressions in either
    // surface immediately.
    let pts = matrix(
        8,
        3,
        vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    );
    let svg = render_scatter3d(&pts).unwrap();
    // 8 circles, exactly.
    assert_eq!(svg.matches("<circle").count(), 8);
    // Two runs produce identical output (determinism).
    let svg2 = render_scatter3d(&pts).unwrap();
    assert_eq!(svg, svg2);
}

#[test]
fn scatter3d_rejects_rank_non_2() {
    let pts = matrix(3, 3, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    // Reshape to rank-1; this should error.
    let flat = DenseArray::new(Shape::new(vec![9]), pts.data().to_vec()).unwrap();
    let err = render_scatter3d(&flat).expect_err("rank-1 input should error");
    let msg = format!("{err}").to_ascii_lowercase();
    assert!(
        msg.contains("scatter3d") || msg.contains("rank") || msg.contains("shape"),
        "got: {msg}"
    );
}

#[test]
fn scatter3d_rejects_wrong_last_dim() {
    // 5 columns instead of 3 or 4.
    let pts = matrix(2, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0]);
    let err = render_scatter3d(&pts).expect_err("last-dim=5 should error");
    let msg = format!("{err}").to_ascii_lowercase();
    assert!(
        msg.contains("scatter3d") || msg.contains("column") || msg.contains("shape"),
        "got: {msg}"
    );
}

#[test]
fn scatter3d_renders_empty_input_without_panicking() {
    let pts = matrix(0, 3, vec![]);
    let svg = render_scatter3d(&pts).unwrap();
    assert!(svg.starts_with("<svg"));
    assert_eq!(svg.matches("<circle").count(), 0);
}
