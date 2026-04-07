use mlpl_array::{DenseArray, Shape};
use mlpl_viz::{render, render_bar, render_line};

#[test]
fn line_from_vector_returns_polyline() {
    let v = DenseArray::from_vec(vec![0.5, 0.4, 0.3, 0.2, 0.1]);
    let svg = render_line(&v).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<polyline"));
}

#[test]
fn line_from_matrix_uses_xy_pairs() {
    let pts = DenseArray::new(Shape::new(vec![3, 2]), vec![0.0, 0.0, 1.0, 2.0, 2.0, 1.0]).unwrap();
    let svg = render_line(&pts).unwrap();
    assert!(svg.contains("<polyline"));
}

#[test]
fn line_empty_vector_renders_empty_plot() {
    let v = DenseArray::from_vec(vec![]);
    let svg = render_line(&v).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(!svg.contains("<polyline"));
}

#[test]
fn line_constant_values_does_not_panic() {
    let v = DenseArray::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
    let svg = render_line(&v).unwrap();
    assert!(svg.contains("<polyline"));
}

#[test]
fn bar_from_vector_renders_one_rect_per_element() {
    let v = DenseArray::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
    let svg = render_bar(&v).unwrap();
    assert!(svg.starts_with("<svg"));
    let rect_count = svg.matches("<rect").count();
    // 1 background rect + 8 bars
    assert_eq!(rect_count, 9);
}

#[test]
fn bar_empty_vector_renders_empty_plot() {
    let v = DenseArray::from_vec(vec![]);
    let svg = render_bar(&v).unwrap();
    assert!(svg.starts_with("<svg"));
    // Just background rect
    assert_eq!(svg.matches("<rect").count(), 1);
}

#[test]
fn bar_rejects_matrix() {
    let m = DenseArray::new(Shape::new(vec![2, 3]), vec![1.0; 6]).unwrap();
    let result = render_bar(&m);
    assert!(result.is_err());
}

#[test]
fn line_rejects_3d_array() {
    let arr = DenseArray::new(Shape::new(vec![2, 2, 2]), vec![1.0; 8]).unwrap();
    let result = render_line(&arr);
    assert!(result.is_err());
}

#[test]
fn render_line_via_dispatch() {
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    let svg = render(&v, "line").unwrap();
    assert!(svg.contains("<polyline"));
}

#[test]
fn render_bar_via_dispatch() {
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    let svg = render(&v, "bar").unwrap();
    assert!(svg.contains("<rect"));
}
