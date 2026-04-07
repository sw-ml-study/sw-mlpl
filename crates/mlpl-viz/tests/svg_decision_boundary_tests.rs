use mlpl_array::{DenseArray, Shape};
use mlpl_viz::{render_decision_boundary, render_with_aux};

fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data).unwrap()
}

#[test]
fn decision_boundary_returns_svg() {
    // 3x3 grid of classifier outputs (probabilities) and 4 training points
    let grid = matrix(3, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    let train = matrix(
        4,
        3,
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    );
    let svg = render_decision_boundary(&grid, &train).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
}

#[test]
fn decision_boundary_emits_grid_cells_and_points() {
    let grid = matrix(2, 2, vec![0.1, 0.9, 0.9, 0.1]);
    let train = matrix(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let svg = render_decision_boundary(&grid, &train).unwrap();
    // background + 4 cells = 5 rects
    assert_eq!(svg.matches("<rect").count(), 5);
    // 2 training-point circles
    assert_eq!(svg.matches("<circle").count(), 2);
}

#[test]
fn decision_boundary_rejects_non_2d_grid() {
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    let train = matrix(1, 3, vec![0.0, 0.0, 0.0]);
    assert!(render_decision_boundary(&v, &train).is_err());
}

#[test]
fn decision_boundary_rejects_bad_training_shape() {
    let grid = matrix(2, 2, vec![0.1, 0.2, 0.3, 0.4]);
    let bad = matrix(2, 2, vec![0.0, 0.0, 1.0, 1.0]);
    assert!(render_decision_boundary(&grid, &bad).is_err());
}

#[test]
fn decision_boundary_via_dispatch() {
    let grid = matrix(2, 2, vec![0.1, 0.9, 0.9, 0.1]);
    let train = matrix(1, 3, vec![0.5, 0.5, 1.0]);
    let svg = render_with_aux(&grid, "decision_boundary", Some(&train)).unwrap();
    assert!(svg.contains("<rect"));
    assert!(svg.contains("<circle"));
}

#[test]
fn decision_boundary_dispatch_requires_aux() {
    let grid = matrix(2, 2, vec![0.1, 0.9, 0.9, 0.1]);
    assert!(render_with_aux(&grid, "decision_boundary", None).is_err());
}
