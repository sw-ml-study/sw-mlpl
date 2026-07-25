//! Connect-telemetry step 002: pure SVG builder for the live loss panel.
//!
//! Native tests -- the builder is plain string assembly, no browser APIs.

use mlpl_web_render_live::loss_svg::loss_panel_svg;

#[test]
fn too_few_points_renders_nothing() {
    assert!(loss_panel_svg(&[], &[]).is_none());
    assert!(loss_panel_svg(&[1.0], &[]).is_none());
    assert!(loss_panel_svg(&[], &[1.0]).is_none());
}

#[test]
fn train_only_series_renders_one_polyline() {
    let svg = loss_panel_svg(&[3.0, 2.0, 1.0], &[]).expect("svg");
    assert!(svg.starts_with("<svg"), "root element: {svg}");
    assert_eq!(svg.matches("<polyline").count(), 1);
    assert!(svg.contains("#a6e3a1"), "train series uses the train color");
}

#[test]
fn train_and_val_render_two_polylines_on_shared_axes() {
    let svg = loss_panel_svg(&[3.0, 2.0, 1.0], &[3.5, 3.0, 2.8]).expect("svg");
    assert_eq!(svg.matches("<polyline").count(), 2);
    assert!(svg.contains("#a6e3a1"));
    assert!(svg.contains("#fab387"), "val series uses the val color");
}

#[test]
fn y_bounds_cover_both_series() {
    // val max (9.0) and train min (0.5) define the shared axis labels.
    let svg = loss_panel_svg(&[2.0, 0.5], &[9.0, 8.0]).expect("svg");
    assert!(
        svg.contains("9.0") || svg.contains("9.000"),
        "ymax label: {svg}"
    );
    assert!(
        svg.contains("0.5") || svg.contains("0.500"),
        "ymin label: {svg}"
    );
}

#[test]
fn constant_series_does_not_divide_by_zero() {
    let svg = loss_panel_svg(&[1.0, 1.0, 1.0], &[]).expect("svg");
    assert!(!svg.contains("NaN") && !svg.contains("inf"), "{svg}");
}
