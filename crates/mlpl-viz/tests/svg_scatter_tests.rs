use mlpl_array::{DenseArray, Shape};
use mlpl_viz::svg::render_scatter;

fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data).unwrap()
}

#[test]
fn scatter_returns_svg_string() {
    let pts = matrix(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 4.0]);
    let svg = render_scatter(&pts).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
}

#[test]
fn scatter_contains_one_circle_per_point() {
    let pts = matrix(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 4.0]);
    let svg = render_scatter(&pts).unwrap();
    let circle_count = svg.matches("<circle").count();
    assert_eq!(circle_count, 3);
}

#[test]
fn scatter_empty_data_renders_empty_plot() {
    let pts = matrix(0, 2, vec![]);
    let svg = render_scatter(&pts).unwrap();
    assert!(svg.starts_with("<svg"));
    assert_eq!(svg.matches("<circle").count(), 0);
}

#[test]
fn scatter_single_point_does_not_panic() {
    let pts = matrix(1, 2, vec![5.0, 5.0]);
    let svg = render_scatter(&pts).unwrap();
    assert_eq!(svg.matches("<circle").count(), 1);
}

#[test]
fn scatter_rejects_vector_input() {
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    let result = render_scatter(&v);
    assert!(result.is_err());
}

#[test]
fn scatter_rejects_wrong_column_count() {
    let m = matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = render_scatter(&m);
    assert!(result.is_err());
}

#[test]
fn scatter_handles_negative_coordinates() {
    let pts = matrix(3, 2, vec![-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]);
    let svg = render_scatter(&pts).unwrap();
    assert_eq!(svg.matches("<circle").count(), 3);
}
