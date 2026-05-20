//! Tests for heatmap_grid (Saga 29 step 014) and its
//! per-cell scale labels (Saga 29 step 019).

use mlpl_array::{DenseArray, Shape};
use mlpl_viz::render_heatmap_grid;

fn rank3(n: usize, r: usize, c: usize, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(vec![n, r, c]), data).unwrap()
}

#[test]
fn heatmap_grid_rejects_rank_2_input() {
    let arr = DenseArray::new(Shape::new(vec![2, 2]), vec![0.0, 0.5, 1.0, 0.3]).unwrap();
    assert!(render_heatmap_grid(&arr).is_err());
}

#[test]
fn heatmap_grid_includes_head_labels() {
    let data = rank3(4, 2, 2, vec![0.0; 16]);
    let svg = render_heatmap_grid(&data).unwrap();
    for h in 0..4 {
        let needle = format!("head {h}");
        assert!(svg.contains(&needle), "expected '{needle}' in SVG");
    }
}

#[test]
fn heatmap_grid_per_cell_scale_shows_each_heads_min_max() {
    // Saga 29 step 019: each cell renders its OWN min/max
    // since per-cell colormaps differ between heads. Cell 0
    // has range [0.10, 0.40]; cell 1 has range [0.50, 0.90].
    let data = rank3(
        2,
        2,
        2,
        vec![
            0.10, 0.20, 0.30, 0.40, // head 0: [0.10, 0.40]
            0.50, 0.60, 0.70, 0.90, // head 1: [0.50, 0.90]
        ],
    );
    let svg = render_heatmap_grid(&data).unwrap();
    assert!(svg.contains("0.10"), "expected head-0 min 0.10 in: {svg}");
    assert!(svg.contains("0.40"), "expected head-0 max 0.40 in: {svg}");
    assert!(svg.contains("0.50"), "expected head-1 min 0.50 in: {svg}");
    assert!(svg.contains("0.90"), "expected head-1 max 0.90 in: {svg}");
}
