use mlpl_array::{DenseArray, Shape};
use mlpl_viz::{render, render_heatmap};

fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data).unwrap()
}

#[test]
fn heatmap_returns_svg() {
    let m = matrix(3, 4, (0..12).map(|i| i as f64).collect());
    let svg = render_heatmap(&m).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
}

#[test]
fn heatmap_emits_one_rect_per_cell_plus_background() {
    let m = matrix(3, 4, (0..12).map(|i| i as f64).collect());
    let svg = render_heatmap(&m).unwrap();
    // 1 background rect + 12 cells
    assert_eq!(svg.matches("<rect").count(), 13);
}

#[test]
fn heatmap_constant_values_does_not_panic() {
    let m = matrix(2, 2, vec![5.0, 5.0, 5.0, 5.0]);
    let svg = render_heatmap(&m).unwrap();
    assert_eq!(svg.matches("<rect").count(), 5);
    // No NaN in fill values
    assert!(!svg.contains("NaN"));
}

#[test]
fn heatmap_single_cell() {
    let m = matrix(1, 1, vec![42.0]);
    let svg = render_heatmap(&m).unwrap();
    assert_eq!(svg.matches("<rect").count(), 2);
}

#[test]
fn heatmap_rejects_vector() {
    let v = DenseArray::from_vec(vec![1.0, 2.0, 3.0]);
    let result = render_heatmap(&v);
    assert!(result.is_err());
}

#[test]
fn heatmap_rejects_3d_array() {
    let arr = DenseArray::new(Shape::new(vec![2, 2, 2]), vec![1.0; 8]).unwrap();
    let result = render_heatmap(&arr);
    assert!(result.is_err());
}

#[test]
fn heatmap_via_dispatch() {
    let m = matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let svg = render(&m, "heatmap").unwrap();
    assert!(svg.contains("<rect"));
}

#[test]
fn heatmap_uses_color_ramp() {
    let m = matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
    let svg = render_heatmap(&m).unwrap();
    // Should contain hex colors (rgb format actually)
    assert!(svg.contains("fill=\"rgb("));
}
