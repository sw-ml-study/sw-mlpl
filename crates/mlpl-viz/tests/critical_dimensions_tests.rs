//! Tests for the `critical_dimensions` viz (saga 33 step 032).

use mlpl_array::{DenseArray, Shape};
use mlpl_viz::{render_critical_dimensions, render_with_aux};

fn mat(rows: usize, cols: usize, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data).unwrap()
}

fn vec1(data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::vector(data.len()), data).unwrap()
}

#[test]
fn critical_dimensions_emits_svg() {
    let loadings = mat(2, 4, vec![1.0, 0.0, 0.5, -0.3, 0.0, 1.0, -0.2, 0.4]);
    let svg = render_critical_dimensions(&loadings, None).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
}

#[test]
fn critical_dimensions_emits_k_times_d_cells_plus_chrome() {
    // 2x4 loadings -> 8 plot cells. Background + 2 axis lines
    // come from write_svg_open. No variance legend rects.
    let loadings = mat(2, 4, vec![0.5; 8]);
    let svg = render_critical_dimensions(&loadings, None).unwrap();
    // 8 plot cells + 1 background = 9 rects.
    assert_eq!(svg.matches("<rect").count(), 9);
}

#[test]
fn critical_dimensions_pc_labels_default_to_index() {
    let loadings = mat(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let svg = render_critical_dimensions(&loadings, None).unwrap();
    assert!(svg.contains(">PC1<"));
    assert!(svg.contains(">PC2<"));
    assert!(svg.contains(">PC3<"));
}

#[test]
fn critical_dimensions_variance_legend_shows_percentages() {
    let loadings = mat(2, 3, vec![1.0, 0.5, 0.0, 0.0, 0.5, 1.0]);
    let var = vec1(vec![0.85, 0.12]);
    let svg = render_critical_dimensions(&loadings, Some(&var)).unwrap();
    // 85.0% and 12.0% should appear in the legend text.
    assert!(svg.contains(">PC1: 85.0%<"));
    assert!(svg.contains(">PC2: 12.0%<"));
}

#[test]
fn critical_dimensions_rejects_non_2d() {
    let bad = vec1(vec![1.0, 2.0, 3.0]);
    assert!(render_critical_dimensions(&bad, None).is_err());
}

#[test]
fn critical_dimensions_rejects_variance_length_mismatch() {
    let loadings = mat(3, 2, vec![1.0; 6]);
    let bad_var = vec1(vec![0.5, 0.3]);
    assert!(render_critical_dimensions(&loadings, Some(&bad_var)).is_err());
}

#[test]
fn critical_dimensions_dispatched_via_render_with_aux() {
    let loadings = mat(2, 3, vec![1.0, 0.5, 0.0, 0.0, 0.5, 1.0]);
    let var = vec1(vec![0.7, 0.2]);
    let svg = render_with_aux(&loadings, "critical_dimensions", Some(&var)).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">PC1: 70.0%<"));
}
