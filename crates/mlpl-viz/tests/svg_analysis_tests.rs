use mlpl_array::{DenseArray, Shape};
use mlpl_viz::{
    analysis_boundary_2d, analysis_confusion_matrix, analysis_hist, analysis_loss_curve,
    analysis_scatter_labeled,
};

fn vector(data: Vec<f64>) -> DenseArray {
    DenseArray::from_vec(data)
}

fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data).unwrap()
}

#[test]
fn hist_returns_svg() {
    let v = vector(vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);
    let svg = analysis_hist(&v, 4).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<rect"));
}

#[test]
fn hist_rejects_non_vector() {
    let m = matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(analysis_hist(&m, 4).is_err());
}

#[test]
fn hist_rejects_zero_bins() {
    let v = vector(vec![1.0, 2.0]);
    assert!(analysis_hist(&v, 0).is_err());
}

#[test]
fn scatter_labeled_returns_svg() {
    let pts = matrix(4, 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let labels = vector(vec![0.0, 1.0, 0.0, 1.0]);
    let svg = analysis_scatter_labeled(&pts, &labels).unwrap();
    assert!(svg.starts_with("<svg"));
    assert_eq!(svg.matches("<circle").count(), 4);
}

#[test]
fn scatter_labeled_length_mismatch() {
    let pts = matrix(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    let labels = vector(vec![0.0, 1.0]);
    assert!(analysis_scatter_labeled(&pts, &labels).is_err());
}

#[test]
fn loss_curve_returns_svg() {
    let v = vector(vec![5.0, 3.0, 2.0, 1.0, 0.5, 0.25]);
    let svg = analysis_loss_curve(&v).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<polyline"));
}

#[test]
fn loss_curve_rejects_non_vector() {
    let m = matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(analysis_loss_curve(&m).is_err());
}

#[test]
fn confusion_matrix_returns_svg() {
    let pred = vector(vec![0.0, 1.0, 2.0, 1.0, 0.0]);
    let actual = vector(vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    let svg = analysis_confusion_matrix(&pred, &actual).unwrap();
    assert!(svg.starts_with("<svg"));
    // 1 background + 9 cells (3x3) = 10 rects
    assert!(svg.matches("<rect").count() >= 10);
}

#[test]
fn confusion_matrix_length_mismatch() {
    let pred = vector(vec![0.0, 1.0, 2.0]);
    let actual = vector(vec![0.0, 1.0]);
    assert!(analysis_confusion_matrix(&pred, &actual).is_err());
}

#[test]
fn boundary_2d_returns_svg() {
    // 4x4 grid of outputs (16 values), with separately supplied points
    let grid_outputs = vector((0..16).map(|i| i as f64 / 15.0).collect());
    let dims = vector(vec![4.0, 4.0]);
    let points = matrix(2, 2, vec![0.0, 0.0, 1.0, 1.0]);
    let labels = vector(vec![0.0, 1.0]);
    let svg = analysis_boundary_2d(&grid_outputs, &dims, &points, &labels).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<rect"));
    assert!(svg.contains("<circle"));
}

#[test]
fn boundary_2d_dim_mismatch() {
    let grid_outputs = vector(vec![1.0, 2.0, 3.0]);
    let dims = vector(vec![2.0, 2.0]);
    let points = matrix(1, 2, vec![0.0, 0.0]);
    let labels = vector(vec![0.0]);
    assert!(analysis_boundary_2d(&grid_outputs, &dims, &points, &labels).is_err());
}
