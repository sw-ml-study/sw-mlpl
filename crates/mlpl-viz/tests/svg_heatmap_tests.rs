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

/// 32 stacked rects approximate the gradient on the legend
/// colorbar; tests count `<rect` elements as `1 background +
/// rows*cols cells + LEGEND_STEPS legend strips`.
const LEGEND_STEPS: usize = 32;

#[test]
fn heatmap_emits_one_rect_per_cell_plus_background() {
    let m = matrix(3, 4, (0..12).map(|i| i as f64).collect());
    let svg = render_heatmap(&m).unwrap();
    assert_eq!(svg.matches("<rect").count(), 1 + 12 + LEGEND_STEPS);
}

#[test]
fn heatmap_constant_values_does_not_panic() {
    let m = matrix(2, 2, vec![5.0, 5.0, 5.0, 5.0]);
    let svg = render_heatmap(&m).unwrap();
    assert_eq!(svg.matches("<rect").count(), 1 + 4 + LEGEND_STEPS);
    // No NaN in fill values
    assert!(!svg.contains("NaN"));
}

#[test]
fn heatmap_single_cell() {
    let m = matrix(1, 1, vec![42.0]);
    let svg = render_heatmap(&m).unwrap();
    assert_eq!(svg.matches("<rect").count(), 1 + 1 + LEGEND_STEPS);
}

#[test]
fn heatmap_legend_renders_value_labels() {
    let m = matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
    let svg = render_heatmap(&m).unwrap();
    // Three labels: hi (3.00), mid (1.50), lo (0.00).
    assert!(svg.contains(">3.00<"), "missing hi label in {svg}");
    assert!(svg.contains(">1.50<"), "missing mid label in {svg}");
    assert!(svg.contains(">0.00<"), "missing lo label in {svg}");
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
