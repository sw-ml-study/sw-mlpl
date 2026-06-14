use mlpl_array::{DenseArray, Shape};
use mlpl_viz::{
    analysis_boundary_2d, analysis_confusion_matrix, analysis_hist, analysis_loss_curve,
    analysis_loss_landscape, analysis_scatter_labeled, analysis_train_val_curve,
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
    // 4 data point circles + 2 legend swatches (one per unique label).
    assert_eq!(svg.matches("<circle").count(), 6);
    // Legend block + numeric labels rendered.
    assert!(svg.contains("class=\"legend\""), "legend group missing");
    assert!(svg.contains(">legend<"), "legend header missing");
    assert!(
        svg.contains(">0<") && svg.contains(">1<"),
        "label numbers missing"
    );
    // Saga 33 step 037b: legend lives in an extended right gutter
    // (canvas width = W + 90 = 490). ViewBox width and the SVG
    // width attribute reflect that.
    assert!(
        svg.contains("viewBox=\"0 0 490 300\""),
        "extended canvas viewBox missing"
    );
    // The legend's x-coord should be past the data area
    // (W - PAD = 370). Legend swatches at x = W + 10 = 410.
    assert!(
        svg.contains("cx=\"410.0\""),
        "legend swatch should be in the right gutter (x=410)"
    );
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
fn train_val_curve_draws_two_lines() {
    // train keeps falling; val bottoms out and turns up (overfitting).
    let train = vector(vec![2.0, 1.2, 0.7, 0.4, 0.2, 0.1]);
    let val = vector(vec![2.0, 1.3, 0.9, 0.85, 0.95, 1.2]);
    let svg = analysis_train_val_curve(&train, &val).unwrap();
    assert!(svg.starts_with("<svg"));
    // One polyline per series.
    assert_eq!(svg.matches("<polyline").count(), 2, "svg was: {svg}");
    // Color-keyed legend names both series.
    assert!(
        svg.contains(">train<") && svg.contains(">val<"),
        "legend: {svg}"
    );
    assert!(
        svg.contains("#a6e3a1") && svg.contains("#fab387"),
        "colors: {svg}"
    );
}

#[test]
fn train_val_curve_handles_unequal_lengths() {
    // Validation recorded once per epoch, training once per step: lengths differ.
    let train = vector(vec![2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]);
    let val = vector(vec![1.9, 1.1, 0.9]);
    let svg = analysis_train_val_curve(&train, &val).unwrap();
    assert_eq!(svg.matches("<polyline").count(), 2, "svg was: {svg}");
}

#[test]
fn train_val_curve_rejects_non_vector() {
    let m = matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let v = vector(vec![1.0, 2.0]);
    assert!(analysis_train_val_curve(&m, &v).is_err());
    assert!(analysis_train_val_curve(&v, &m).is_err());
}

#[test]
fn loss_landscape_draws_surface_and_trajectory() {
    let surface = vector((0..16).map(|i| i as f64).collect()); // 4x4 bowl-ish
    let dims = vector(vec![4.0, 4.0]);
    let path = matrix(4, 2, vec![0.1, 0.9, 0.3, 0.6, 0.5, 0.3, 0.7, 0.1]);
    let svg = analysis_loss_landscape(&surface, &dims, &path).unwrap();
    assert!(svg.starts_with("<svg"), "svg: {svg}");
    // 16 surface cells, each an rgb() fill (the background rect is a hex fill).
    assert_eq!(
        svg.matches("fill=\"rgb(").count(),
        16,
        "16 surface cells: {svg}"
    );
    assert_eq!(svg.matches("<polyline").count(), 1, "one trajectory: {svg}");
    // Start (green) and end (red) markers.
    assert_eq!(
        svg.matches("<circle").count(),
        2,
        "start+end markers: {svg}"
    );
    assert!(
        svg.contains("#a6e3a1") && svg.contains("#f38ba8"),
        "marker colors: {svg}"
    );
}

#[test]
fn loss_landscape_rejects_bad_shapes() {
    let surface = vector((0..16).map(|i| i as f64).collect());
    let dims = vector(vec![4.0, 4.0]);
    // path must be [N, 2], not a vector.
    assert!(analysis_loss_landscape(&surface, &dims, &vector(vec![0.1, 0.2])).is_err());
    // surface length must equal rows*cols.
    let bad_dims = vector(vec![3.0, 4.0]);
    let path = matrix(1, 2, vec![0.5, 0.5]);
    assert!(analysis_loss_landscape(&surface, &bad_dims, &path).is_err());
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
