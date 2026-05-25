//! Tests for the `plotly3d` HTML/JS rendering. Saga 33 step 030.

use mlpl_array::{DenseArray, Shape};
use mlpl_viz::{render_plotly3d, render_with_aux};

fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data).unwrap()
}

fn vec1(data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::vector(data.len()), data).unwrap()
}

#[test]
fn plotly3d_emits_marker_and_div() {
    let pts = matrix(3, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let html = render_plotly3d(&pts, None).unwrap();
    assert!(html.starts_with("<!-- mlpl-plotly3d -->"));
    assert!(html.contains("<div id=\"mlpl-plotly3d-"));
    assert!(html.contains("Plotly.newPlot"));
    assert!(html.contains("plotly_click"));
}

#[test]
fn plotly3d_no_labels_emits_one_trace() {
    let pts = matrix(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let html = render_plotly3d(&pts, None).unwrap();
    // Naive trace count: one `"type":"scatter3d"` per trace.
    assert_eq!(html.matches(r#""type":"scatter3d""#).count(), 1);
}

#[test]
fn plotly3d_with_labels_emits_one_trace_per_unique_label() {
    let pts = matrix(
        4,
        3,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
    );
    let labels = vec1(vec![0.0, 1.0, 0.0, 1.0]);
    let html = render_plotly3d(&pts, Some(&labels)).unwrap();
    assert_eq!(html.matches(r#""type":"scatter3d""#).count(), 2);
    // Cluster map must back-map each trace's points to the
    // original sample index. trace 0 (label 0) holds samples
    // 0 and 2; trace 1 (label 1) holds samples 1 and 3.
    assert!(html.contains("var map=[[0,2],[1,3]];"));
}

#[test]
fn plotly3d_rejects_non_nx3() {
    let bad = matrix(2, 4, vec![0.0; 8]);
    assert!(render_plotly3d(&bad, None).is_err());
}

#[test]
fn plotly3d_rejects_label_count_mismatch() {
    let pts = matrix(3, 3, vec![0.0; 9]);
    let bad_labels = vec1(vec![0.0, 1.0]);
    assert!(render_plotly3d(&pts, Some(&bad_labels)).is_err());
}

#[test]
fn plotly3d_dispatched_via_render_with_aux() {
    let pts = matrix(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let html = render_with_aux(&pts, "plotly3d", None).unwrap();
    assert!(html.starts_with("<!-- mlpl-plotly3d -->"));
}
